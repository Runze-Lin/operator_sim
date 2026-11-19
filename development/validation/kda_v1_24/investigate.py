"""Single-attempt, serial V1 release probes. A sync fault disqualifies timing."""
from collections import Counter
import fcntl, hashlib, json, os, re, signal, subprocess, sys, time
from pathlib import Path
from host import HERE,ROOT,RAW,OUT,BIN,probe,build
sys.path.insert(0,str(HERE.parent/'kda_v23'))
import qualify
from series import DT, PR, MEMORY, when, read, save
from score import tables, checked

REG=OUT/'register.json'
OLD=probe.REGISTER
SOURCES={'chunk_kda_fwd_tiling.cpp':'fla/ops/ascendc/kda/chunk_kda_fwd/op_host/chunk_kda_fwd_tiling.cpp',
         'chunk_kda_fwd_tiling_impl.h':'fla/ops/ascendc/kda/chunk_kda_fwd/op_host/arch35/chunk_kda_fwd_tiling_impl.h',
         'vendor-v1-oracle.py':'tests/atk/chunk_kda_fwd/executor_chunk_kda_fwd.py'}
CASES=[(DT,2,1,'varlen'),(DT,64,2,'dense'),(PR,2,1,'varlen'),(PR,64,2,'dense')]
def register():
    if REG.exists():raise FileExistsError(REG)
    base=read(OLD);tree=read(probe.RAW/'fla-source-tree.json')
    blobs={b['path']:b['sha'] for b in tree['tree']}
    verified={}
    for n,p in SOURCES.items():
        d=(RAW/n).read_bytes()
        sha=hashlib.sha1(b'blob '+str(len(d)).encode()+b'\0'+d).hexdigest()
        checked(sha==blobs[p],'fetched blob mismatch: '+p)
        verified[n]=dict(path=p,git_blob=sha,tree=tree['sha'])
    build()
    cases=[]
    for sku,t,h,m in CASES:
        fixture=RAW/'oracles'/f'{t}-{h}'
        if not fixture.exists():
            subprocess.run([str(ROOT/'toolchains/envs/cann-9.2.1/bin/python'),str(HERE/'golden.py'),str(t),str(h),str(fixture)],check=True)
        cases.append(dict(id=f'{sku[9:11].lower()}-t{t}-h{h}-{m}',shape=[1,t,h,h,128,128],metadata=m,sku=sku,
            physical_memory=MEMORY[sku],oracle=dict(attention=t*h*128,aqk=t*h*64,akk=t*h*64,state_vk=h*128*128)))
    paths=[*HERE.glob('*.py'),*RAW.glob('*.cpp'),*(RAW/k for k in SOURCES),BIN,*((RAW/'oracles').glob('*/*.bin')),
           HERE.parent/'kda_v23/precise_host.hpp',HERE.parent/'kda_v23/qualify.py',HERE.parent/'kda_v23/series.py',
           HERE.parent/'chunk_recovery/fla2691/v2/verify_chrome.py',OLD,probe.SOURCE,probe.RAW/'fla-source-tree.json',
           OLD.parent/'vllm-source.json',probe.VENDOR/'op_api/lib/libcust_opapi.so',ROOT/'smoke/extension_kda.cpp']
    checked(len(set(c['id'] for c in cases))==4,'duplicate registration ID')
    save(REG,dict(utc=when(),role='predeclared cause-discriminating development only',cases=cases,max_concurrency=1,
        budget_seconds=1200,budget_bytes=6*2**30,
        conditional='PR dense runs only after DT dense has verified numerics, full windows, correct FLA identity and zero sync errors.',
        eligibility='all four V1 CPU oracle outputs, initial VK unchanged, successful process, no simulator sync errors, physically bound released host; exactly one monolithic FLA V1 device code among all full official per-core windows with physical END; chronological auxiliary launches.',
        dtype_layout='Q,K,V,raw_gate BF16, beta and VK state FP32. BSND, chunk=64, K=V=128, mixed signs, safe gate -5 including AF/dt_bias.',
        mechanism='Original cu/indices force isVarLen and generic-backend path for little work. Without metadata T=64,H=2 permits dedicated dense FwdH path; on B=1 numerical inputs and results are identical. Dense is separate from the proven vLLM call with metadata.',
        oracle='FLA fused V1 ATK _reference_model_parallel, BF16 operand pairs with last-row shift per 16. Original higher precision mathematical comparator also emitted, with identical original thresholds. Neither threshold nor fixtures may be changed after capture.',
        published_kda_text=base['published_kda_text'],builtin_text=base['cann_builtin_kda_text'],release_asset=base['release_asset'],official_source=verified,source_sha256={probe.rel(p):probe.digest(p) for p in paths}))
    print(probe.digest(REG),flush=True)

def capture(reg,c,p):
    runlog=p/'run.log';out=p/'cost.json'
    if out.exists():return read(out)
    checked(not runlog.exists(),'prior attempt found with no cost receipt')
    cmd=['bash',str(ROOT/'scripts/cann-run.sh'),'bash','-c','source "$1/bin/set_env.bash"; shift; exec "$@"','fla',str(probe.VENDOR),
         'msopprof','simulator','--soc-version='+c['sku'],'--dump=false','--launch-count=32','--timeout=20','--output='+str(p/'profile'),
         str(BIN),str(c['shape'][1]),str(c['shape'][2]),c['metadata']]
    t=time.monotonic();start=when();peak=0;stop=None
    with runlog.open('x') as log:
        child=subprocess.Popen(cmd,cwd=p,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',OPENBLAS_NUM_THREADS='1'))
        while child.poll() is None:
            total=0
            for f in p.rglob('*'):
                try:
                    if f.is_file(): total+=f.stat().st_size
                except FileNotFoundError:pass
            peak=max(peak,total)
            if total>reg['budget_bytes']:stop='disk budget'
            elif time.monotonic()-t>reg['budget_seconds']:stop='wall budget'
            if stop:
                os.killpg(child.pid,signal.SIGKILL);child.wait();break
            time.sleep(2)
    result=dict(started=start,finished=when(),seconds=time.monotonic()-t,peak_bytes=peak,stop=stop,exit_code=child.returncode,command=cmd)
    save(out,result);return result

def qualify_v1(reg,c,p,cost):
    log=(p/'run.log').read_text(errors='replace')
    sku=re.findall(r'^runtime_soc=(\S+)',log,re.M)==[c['sku']] and c['physical_memory'] in log
    api=str(probe.VENDOR/'op_api/lib/libcust_opapi.so')
    bind=dict(re.findall(r'^bound_v1_api=(\S+) library=(\S+)',log,re.M))=={n:api for n in ('aclnnChunkKdaFwd','aclnnChunkKdaFwdGetWorkspaceSize')}
    outputs={n:dict(count=int(cnt),bad=int(b),max_abs=float(err)) for n,cnt,b,err in re.findall(r'^correctness name=(\S+) count=(\d+) bad=(\d+) max_abs_error=(\S+)',log,re.M)}
    golden=all(outputs.get(n,{}).get('count')==cnt and outputs[n]['bad']==0 for n,cnt in c['oracle'].items()) and 'input_vk_unchanged=1' in log and 'SMOKE_PASS' in log and 'SMOKE_FAIL' not in log
    proc=cost['exit_code']==0 and not cost['stop'] and 'Profiling running finished. All task success.' in log and 'Profiling data parse finished.' in log
    faults=[l for l in log.splitlines() if re.search(r'\[ERROR\].*(?:execute_(?:set|wait)_flag|deadlock|invalid instruction)',l,re.I)]
    pc=Counter(re.findall(r'already has same set_flag! pc:(0x[\da-fA-F]+)',log))
    rows=[];names=re.findall(r'Start profiling on kernel: (\S+)',log);seen=Counter();tab=tables(log);roots=list(p.glob('profile/OPPROF_*'))
    for idx,name in enumerate(names):
        ord=seen[name];seen[name]+=1;row=dict(kernel=name,ordinal=ord,ca_us=None,raw_path=None)
        try:
            checked(len(roots)==1,'missing unique profile')
            variants=[roots[0]/name/str(ord),roots[0]/'device0'/name/str(ord),roots[0]]
            path=next((d for d in variants if (d/'dump/aicore_binary.o').exists()),None)
            checked(path is not None,'missing captured ELF')
            row['raw_path']=probe.rel(path)
            row['device_text_sha256']=hashlib.sha256(probe.text_bytes(path/'dump/aicore_binary.o')).hexdigest()
            checked(idx<len(tab),'missing official duration table')
            row.update(qualify.stage(path,name,ord,tab[idx],{}))
        except (ValueError,RuntimeError,KeyError,IndexError,OSError,TypeError) as e:row['reason']=str(e)
        row.pop('family',None);row.pop('label',None)
        if name.startswith('ChunkKdaFwd_'):
            row['published_names']=[n for n,d in reg['published_kda_text'].items() if row.get('device_text_sha256')==d]
        rows.append(row)
    main=[r for r in rows if r['kernel'].startswith('ChunkKdaFwd_')]
    ident=(len(main)==1 and bool(main[0].get('published_names')) and main[0]['device_text_sha256'] not in reg['builtin_text'].values() and not any(any(s in n for s in ('FwdPrepare','FwdH_','FwdFinalize')) for n in names))
    full=bool(rows) and len(rows)==len(tab) and all(r['ca_us'] is not None for r in rows)
    ordered=full and all(a['last_tick']<b['first_tick'] for a,b in zip(rows,rows[1:]))
    good=bool(sku and bind and ident and proc and golden and not faults and full and ordered)
    return dict(qualified=good,actual_sku_verified=sku,published_host=bind,published_monolithic_elf=ident,process_success=proc,golden=golden,
                output_groups=outputs,sync_errors=len(faults),sync_fault_pc_counts=dict(pc),sync_examples=faults[:3],full_windows=full,
                chronological=ordered,stages=rows,official_tables=len(tab),main_ca_us=main[0]['ca_us'] if good else None,
                serialized_ca_us=sum(r['ca_us'] for r in rows) if good else None,log_sha256=probe.digest(p/'run.log'))
def run():
    reg=read(REG)
    for p,h in reg['source_sha256'].items():checked(probe.digest(ROOT/p)==h,'frozen source differs: '+p)
    receipts=[]
    with (RAW/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for c in reg['cases']:
            dest=OUT/(c['id']+'.json');base=RAW/c['id'];base.mkdir(exist_ok=True)
            if not dest.exists():
                save(OUT/'status.json',dict(state='running',updated=when(),current=c['id'],finished=receipts,registration_sha256=probe.digest(REG)))
                if c['sku']==PR and c['metadata']=='dense' and not read(OUT/'dt-t64-h2-dense.json')['qualified']:
                    r=dict(qualified=False,skipped='DT dense did not qualify',cost=None)
                else:
                    cost=capture(reg,c,base)
                    r=dict(cost=cost,**qualify_v1(reg,c,base,cost))
                save(dest,dict(case=c,registration_sha256=probe.digest(REG),**r))
            receipts.append(c['id'])
        save(OUT/'status.json',dict(state='finished',updated=when(),finished=receipts,registration_sha256=probe.digest(REG)))

if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
