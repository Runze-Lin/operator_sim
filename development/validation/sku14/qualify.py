"""Check device identity, numerical results, kernel identity, and all-core time windows."""
from collections import defaultdict
from pathlib import Path
import math
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
    for marker in ('All task success.', 'Profiling data parse finished.', 'lpddr5_6400M_bailu.ini'):
        if marker not in log: raise ValueError('missing PR completion/DRAM marker: '+marker)
    kernel,=re.findall(r'Start profiling on kernel: (\S+)',log)
    prefix={'quant':'DynamicMxQuant','matmul':'MatMul','fa':'FlashAttentionScore'}[c['op']]
    if not kernel.startswith(prefix): raise ValueError('unexpected launched library kernel: '+kernel)
    trace,=d.glob('profile/OPPROF_*/simulator/trace.json'); cfg,=re.findall(r'Using config from SO dir: (\S+)',log)
    starts={};ends={};counts=defaultdict(int);instructions=defaultdict(int)
    events=load(trace)['traceEvents']
    for e in events:
        if e['ph']!='X':continue
        core=e['pid'];t=e['ts'];duration=e['dur']
        if not math.isfinite(t) or not math.isfinite(duration) or duration<0 or not re.fullmatch(r'core\d+\.(?:cube|vec)core\d+',core): raise ValueError('malformed instruction')
        starts[core]=min(t,starts.get(core,t));ends[core]=max(t+duration,ends.get(core,t));counts[core]+=1;instructions[e['name']]+=1
    del events
    windows=[]
    for core in sorted(starts):
        a,b=starts[core],ends[core]; span=b-a
        timing,=re.findall(r'^'+re.escape(core)+r'\s+([.\d]+)\s+[.\d]+\s*$',log,re.M)
        if span<=0 or abs(span-float(timing))>max(.03,span*.02): raise ValueError('profiler and instruction window disagree for '+core)
        windows.append(dict(core=core,start_us=a-min(starts.values()),end_us=b-min(starts.values()),profiler_span_us=float(timing),events=counts[core]))
    return dict(soc=PR,kernel=kernel,correctness=correct,backend='msopprof',trace=str(trace.relative_to(ROOT)),trace_sha256=sha(trace),log_sha256=sha(log_path),kernel_binary_sha256=sha(trace.parents[1]/'dump/aicore_binary.o'),ca_config_sha256=sha(Path(cfg)),host_binary_sha256=c['host_binary_sha256'],ca_us=max(ends.values())-min(starts.values()),core_windows=windows,active_aiv=sum('veccore' in k for k in starts),active_aic=sum('cubecore' in k for k in starts),events=sum(counts.values()),selected_timeline_counts={k:instructions[k] for k in ('MMAD','RVEC','MOV_OUT_TO_UB','MOV_UB_TO_OUT')},memory_model='lpddr5_6400M_bailu.ini')

def dt_summary(c):
    path=ROOT/c['dt_result'];d=load(path);m=d.get('measurement',{});base=path.parent
    if d['status']!='ok':raise ValueError('invalid DT archive')
    if c['dt_backend']=='msopprof':
        log=(base/'run.log').read_text(errors='replace');correct=checks(log,DT)
        if d['identity']['host_binary_sha256']!=c['host_binary_sha256']:raise ValueError('DT host mismatch')
        trace,=base.glob('profile/OPPROF_*/simulator/trace.json'); digest=m['trace_sha256'];kernel=m['kernel'];ca=m['reference_us'];aic=m['active_aic'];aiv=m['active_aiv'];binary=m['binary_sha256']
    else:
        if 'measurement' in d: trace=Path(m['db_path']); digest=m['db_sha256']; kernel=m['kernel'];aic=m['active_aic'];aiv=m['active_aiv']
        else:
            m,=d['kernels'];trace,=base.glob('profile/npusim_*/record/*.db');digest=m['db_sha256']; kernel=m['kernel'];aic=m['active_aic'];aiv=m['active_aiv']
        log=(trace.parents[3]/'run.log').read_text(errors='replace');correct=checks(log,DT);ca=d['reference_us'];binary=None
    if sha(trace)!=digest:raise ValueError('DT archive trace hash mismatch')
    return dict(soc=DT,backend=c['dt_backend'],kernel=kernel,correctness=correct,active_aic=aic,active_aiv=aiv,ca_us=ca,trace=str(trace.relative_to(ROOT)),trace_sha256=digest,kernel_binary_sha256=binary,source_record=c['dt_result'],source_sha256=sha(path))

def qualify(c,f,contract):
    d=RUNS/c['id']; cost=load(d/'cost.json');assert cost['exit_code']==0 and cost['stop_reason'] is None
    pr=pr_summary(c,d);dt=dt_summary(c)
    return dict(**f,pr_ca_us=pr['ca_us'],dt_ca_us=dt['ca_us'],active_aic=pr['active_aic'],active_aiv=pr['active_aiv'],kernel=pr['kernel'],dt=dt,pr=pr,pr_cost=cost,contract=contract,coverage=c['coverage'])
