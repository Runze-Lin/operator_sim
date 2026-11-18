"""Check device identity, numerical results, kernel identity, and all-core time windows."""
from collections import defaultdict
from pathlib import Path
import math
import sqlite3
import re
from campaign import ROOT, OUT, RUNS, PR, DT, load, sha, write

def checks(log,soc):
    if 'SMOKE_FAIL' in log or 'SMOKE_PASS' not in log or re.findall(r'\bruntime_soc=(\S+)',log)!=[soc]:
        raise ValueError('incorrect/unfinished host or unexpected runtime SKU')
    correct=re.findall(r'\bcorrectness (.+)',log)
    if not correct or any(not re.search(r'\bbad=0\b',r) for r in correct):
        raise ValueError('missing or failed numerical golden')
    return correct

def pr_summary(c,d):
    log_path=d/'run.log';log=log_path.read_text(errors='replace');correct=checks(log,PR)
    axes=c['shape']; line,=re.findall(r'^case=(.+)$', log,re.M)
    dims=dict(re.findall(r'\b(rows|cols|M|N|K|Q|KV|D|B|H)=(\d+)\b',line))
    keys={'quant':('rows','cols'),'matmul':('M','N','K'),'fa':('Q','KV','D','B','H')}[c['op']]
    if [int(dims[k]) for k in keys]!=axes:raise ValueError('host logged different input shape')
    if c['op']=='quant':
        n=math.prod(axes)
        if not any(f'quant_elements={n} scale_elements={n//32} ' in z for z in correct):raise ValueError('incomplete quant/scale golden')
    else:
        n=(axes[0]*axes[1] if c['op']=='matmul' else axes[0]*axes[2]*axes[3]*axes[4])
        if not any(re.search(r'\b(?:elements|count)='+str(n)+r'\b',z) for z in correct):raise ValueError('incomplete output golden')
    for marker in ('All task success.', 'Profiling data parse finished.', 'soc version: Ascend950PR', 'lpddr5_6400M_bailu.ini'):
        if marker not in log: raise ValueError('missing PR completion/DRAM marker: '+marker)
    kernel,=re.findall(r'Start profiling on kernel: (\S+)',log)
    prefix={'quant':'DynamicMxQuant','matmul':'MatMul','fa':'FlashAttentionScore'}[c['op']]
    if not kernel.startswith(prefix): raise ValueError('unexpected launched library kernel: '+kernel)
    trace,=d.glob('profile/OPPROF_*/simulator/trace.json'); cfg,=re.findall(r'Using config from SO dir: (\S+)',log)
    starts={};ends={};counts=defaultdict(int);instructions=defaultdict(int)
    # Use the official combined timeline: per-core presentation files also contain
    # extra tracks and do not have identical populations. Stream at most 2 MiB.
    import json
    def stream():
        decode=json.JSONDecoder().raw_decode
        with trace.open() as fd:
            chunk=fd.read(1024*1024);chunk=chunk.split('"traceEvents":[',1)[1];pos=0
            while True:
                if len(chunk)-pos<100000:
                    chunk=chunk[pos:]+fd.read(1024*1024);pos=0
                if chunk[pos]==']':break
                obj,pos=decode(chunk,pos);yield obj
                if chunk[pos]==',':pos+=1
    phases=defaultdict(lambda:defaultdict(list))
    for e in stream():
        if e['ph']!='X':continue
        core=e['pid'];t=e['ts'];duration=e['dur']
        if not math.isfinite(t) or not math.isfinite(duration) or duration<0 or not re.fullmatch(r'core\d+\.(?:cube|vec)core\d+',core): raise ValueError('malformed instruction')
        starts[core]=min(t,starts.get(core,t));ends[core]=max(t+duration,ends.get(core,t));counts[core]+=1;instructions[e['name']]+=1
        if e['name'] in ('MMAD','FIX_L0C_TO_DST','MOV_OUT_TO_L1_MULTI_ND2NZ') or (e['tid']=='MTE2' and c['op']=='fa'):
            phases[core][e['name']].append([t,duration])
    if c.get('backend')=='official_full_compacted_after_parse':
        with sqlite3.connect(d/'landmarks.db') as db:
            db.execute('create table landmarks(core text, opcode text, ts_us real, duration_us real)')
            db.executemany('insert into landmarks values(?,?,?,?)',((name,op,t,z) for name,opcodes in phases.items() for op,items in opcodes.items() for t,z in items))
    summary={}
    for k,ops in phases.items():
        summary[k]={op:dict(count=len(items),first_start_us=min(t for t,_ in items)-min(starts.values()),last_end_us=max(t+z for t,z in items)-min(starts.values())) for op,items in ops.items()}
    windows=[]
    for core in sorted(starts):
        a,b=starts[core],ends[core]; span=b-a
        timing,=re.findall(r'^'+re.escape(core)+r'\s+([.\d]+)\s+[.\d]+\s*$',log,re.M)
        if span<=0 or abs(span-float(timing))>max(.03,span*.02): raise ValueError('profiler and instruction window disagree for '+core)
        windows.append(dict(core=core,start_us=a-min(starts.values()),end_us=b-min(starts.values()),profiler_span_us=float(timing),events=counts[core]))
    return dict(soc=PR,kernel=kernel,correctness=correct,backend='msopprof',trace=str(trace.relative_to(ROOT)),trace_sha256=sha(trace),log_sha256=sha(log_path),kernel_binary_sha256=sha(trace.parents[1]/'dump/aicore_binary.o'),ca_config_sha256=sha(Path(cfg)),host_binary_sha256=c['host_binary_sha256'],ca_us=max(ends.values())-min(starts.values()),core_windows=windows,phase_landmarks=summary,active_aiv=sum('veccore' in k for k in starts),active_aic=sum('cubecore' in k for k in starts),events=sum(counts.values()),selected_timeline_counts={k:instructions[k] for k in ('MMAD','RVEC','MOV_OUT_TO_UB','MOV_UB_TO_OUT')},memory_model='lpddr5_6400M_bailu.ini')



def qualify(c,f,contract):
    d=RUNS/c['id']; cost=load(d/'cost.json')
    if cost['exit_code']!=0 or cost['stop_reason']:raise ValueError('incomplete CA process')
    pr=pr_summary(c,d);dt=c['dt']
    return dict(**f,pr_ca_us=pr['ca_us'],dt_ca_us=(dt['ca_us'] if dt else None),active_aic=pr['active_aic'],active_aiv=pr['active_aiv'],kernel=pr['kernel'],dt=dt,pr=pr,pr_cost=cost,contract=contract,coverage=c['coverage'],backend=c['backend'])
