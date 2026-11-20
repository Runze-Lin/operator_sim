"""Stream full instruction traces and compare roster, multiset fingerprints and terminal times."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

MOD=2**256

def events(path):
    dec=json.JSONDecoder().raw_decode
    with path.open() as f:
        buf=f.read(1<<20).split('"traceEvents":[',1)[1];i=0
        while True:
            if len(buf)-i<100000:
                buf=buf[i:]+f.read(1<<20);i=0
            if buf[i]==']':return
            ev,i=dec(buf,i);yield ev
            if buf[i]==',':i+=1

def fingerprint(path,aggregate):
    rows={};opened={}
    for e in events(path):
        ph=e['ph']
        if ph not in ('B','E','X'):continue
        c=e['pid'] if aggregate else path.parent.name
        if ph!='X':
            key=(c,e['tid'],e['id'],e['name'])
            if ph=='B':
                if key in opened:raise ValueError('duplicate open event')
                opened[key]=e;continue
            hi=round(e['ts']*1650);e=opened.pop(key)
        else:
            hi=round((e['ts']+e['dur'])*1650)
        lo=round(e['ts']*1650)
        if hi<lo:raise ValueError('reversed interval')
        pc=e['args']['pc_addr'];name=e['name'];d=json.loads(e['args']['detail'] or '{}')
        if c not in rows:
            rows[c]=dict(first=lo,last=hi,events=0,multiset=0,end=[],mmad=0,mmad_geometry=Counter(),pipe=Counter())
        r=rows[c];r['first']=min(lo,r['first']);r['last']=max(hi,r['last'])
        sig=(name,pc,lo,hi) if not(d.get('is_sync') or d.get('is_wait')) else (name,pc,hi,d.get('flag_id'),d.get('sync_id'),d.get('target_pipe'),d.get('consumer_pipe'))
        h=int.from_bytes(hashlib.sha256(repr(sig).encode()).digest(),'big')
        r['events']+=1;r['multiset']=(r['multiset']+h)%MOD
        if name=='END':r['end'].append([lo,hi])
        if name=='MMAD':r['mmad']+=1;r['mmad_geometry'][f"{d['m']}x{d['n']}x{d['k']}"]+=1
        if not d.get('is_sync') and not d.get('is_wait'):r['pipe'][e['tid']]+=1
    if opened:raise ValueError('unclosed begin/end event')
    return rows


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def stage(path,name,ordinal,durations):
    merged=fingerprint(path/'simulator/trace.json',True)
    cores={p.parent.name:p for p in path.glob('simulator/core*/trace.json')}
    if set(cores)!=set(merged) or set(cores)!=set(durations) or not cores:raise ValueError('incomplete active core roster')
    for c,p in cores.items():
        r,=fingerprint(p,False).values()
        for f in ('first','last','events','multiset','end','mmad','mmad_geometry'):
            if r[f]!=merged[c][f]:raise ValueError(f'{c}: shard/aggregate {f} mismatch')
        if len(r['end'])!=1 or not r['first']<=r['end'][0][0]<=r['end'][0][1]<=r['last']:raise ValueError(c+' physical END')
        dur=(r['last']-r['first'])/1650
        if abs(float(durations[c])-dur)>.0051:raise ValueError(f'{c} duration {dur} vs vendor {durations[c]}')
        merged[c]['multiset']=format(r['multiset'],'064x')
        merged[c]['shard_sha256']=digest(p)
        merged[c]['official_us']=float(durations[c])
    lo=min(r['first'] for r in merged.values());hi=max(r['last'] for r in merged.values())
    return dict(kernel=name,ordinal=ordinal,trace=str(path/'simulator/trace.json'),trace_sha256=digest(path/'simulator/trace.json'),
                elf_sha256=digest(path/'dump/aicore_binary.o'),first_tick=lo,last_tick=hi,ca_us=(hi-lo)/1650,
                active_aic=sum('cubecore' in c for c in cores),active_aiv=sum('veccore' in c for c in cores),
                mmads=sum(r['mmad'] for r in merged.values()),cores=merged)


def qualify(job,chip,folder,cost,v2reg,root):
    log=(folder/'run.log').read_text(errors='replace');gold=re.findall(r'^correctness [^\r\n]+',log,re.M)
    matched={}
    for line in gold:
        d=dict(re.findall(r'\b(name|count|bad)=(\S+)',line))
        if 'count' not in d or 'bad' not in d:raise ValueError('malformed check')
        name=d.get('name','output')
        if name in matched:raise ValueError('duplicate golden')
        matched[name]=[int(d['count']),int(d['bad'])]
    groups_ok=matched=={k:[v,0] for k,v in job['golden'].items()}
    names=re.findall(r'Start profiling on kernel: (\S+)',log)
    base=dict(qualified=False,ca_us=None,actual=matched,expected=job['golden'],numerical=groups_ok,launched=names,
              log_sha256=digest(folder/'run.log'))
    if cost['exit_code']!=0 or cost['stop_reason'] or not groups_ok:return dict(**base,reason='process/budget/golden')
    needed=('SMOKE_PASS','All task success.','Profiling data parse finished.',('hbm3_8000M.ini' if '950DT' in chip else 'lpddr5_6400M_bailu.ini'),job['contract'])
    if not all(x in log for x in needed) or 'SMOKE_FAIL' in log or re.findall(r'^runtime_soc=(\S+)',log,re.M)!=[chip]:return dict(**base,reason='scope, SKU, memory, host or profiler completion')
    if re.search(r'\[ERROR\].*(execute_(set|wait)_flag|deadlock|invalid instruction)',log):return dict(**base,reason='synchronization/illegal instruction')
    if len(job['prefix'])!=len(names) or any(not n.startswith(prefix) for n,prefix in zip(names,job['prefix'])):return dict(**base,reason='different real launch family/count')
    sections=re.split(r'^core_name\s+duration_time\(us\)\s+running_time\(us\)\s*$',log,flags=re.M)[1:]
    if len(sections)!=len(names):return dict(**base,reason='official table count')
    profiler,=folder.glob('profile/OPPROF_*');stages=[];ords=Counter()
    for ordn,(name,section) in enumerate(zip(names,sections)):
        i=ords[name];ords[name]+=1
        candidates=[profiler/name/str(i),profiler/'device0'/name/str(i)]+([profiler] if len(names)==1 else [])
        path=next(p for p in candidates if (p/'dump/aicore_binary.o').is_file())
        durations=re.findall(r'^(core\d+\.(?:cube|vec)core\d+)\s+([.\d]+)\s+[.\d]+\s*$',section,re.M)
        if len(durations)!=len(dict(durations)):raise ValueError('duplicate core in official table')
        record=stage(path,name,i,dict(durations));stages.append(record)
        if job['id']=='kda-v2':
            from subprocess import check_output
            import tempfile
            with tempfile.TemporaryDirectory() as d:
                obj=path/'dump/aicore_binary.o';text=Path(d)/'text.bin'
                compiler=root/'toolchains/cann/9.2.1/cann-9.2.1/compiler/ccec_compiler/bin/llvm-objcopy'
                # Use the exact text extractor used to qualify the earlier V2 release, including fat ELF handling.
                import sys
                sys.path.insert(0,str(root/'development/validation/kda_v23'))
                from series import probe
                hh=hashlib.sha256(probe.text_bytes(obj)).hexdigest()
            fam=('chunk_kda_fwd_prepare','chunk_fwd_h','chunk_kda_fwd_finalize')[ordn]
            if hh not in v2reg['published_device_text'][fam].values():return dict(**base,reason='unverified official V2 device text')
            record['official_text_sha256']=hh
    if job['id']=='kda-v2':
        lib=str(root/'results/chunk-recovery/fla2691/raw/fla_npu/opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so')
        if dict(re.findall(r'^bound_api=(\S+) library=(\S+)',log,re.M))!={n:lib for n in ('aclnnChunkKdaFwdV2','aclnnChunkKdaFwdV2GetWorkspaceSize')} or 'input_vk_unchanged=1' not in log:return dict(**base,reason='wrong FLA library or modified initial state')
        if any(a['last_tick']>=b['first_tick'] for a,b in zip(stages,stages[1:])):return dict(**base,reason='V2 stage order')
    if job['id']=='moe' and ('stage_complete=grouped_matmul_swiglu_quant_v2' not in log or 'stage_complete=gmm2' not in log):
        return dict(**base,reason='missing synchronized library stage')
    base.update(qualified=True,ca_us=sum(s['ca_us'] for s in stages),stages=stages)
    return base
