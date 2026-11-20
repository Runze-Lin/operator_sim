"""Freeze model-derived BF16 matmul shapes and run bounded, all-launch device profiling."""
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.28/matmul'
RAW=ROOT/'runs/matmul28'
sys.path.insert(0,str(ROOT))
from operator_model import estimate
from development.validation.sku19.collect import nbytes
from campaign_collect import stages
sys.path.insert(0,str(ROOT/'development/validation/pro_prefill23'))
from development.validation.pro_prefill23.evidence import endpoint

DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'
BIN=ROOT/'build/smoke/matmul28'
CATALOG=ROOT/'results/v0.23/workloads/contracts.jsonl.gz'
FLASH,PRO,KIMI='DeepSeek-V4.1-Flash','DeepSeek-V4-Pro','Kimi-K3'


def now():
    return datetime.now(timezone.utc).isoformat()

def read(p):
    return json.loads(p.read_bytes())

def save(p,value):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(p)

def digest(p):
    return sha256(p.read_bytes()).hexdigest()

def geometries():
    with gzip.open(CATALOG,'rt') as src:
        nodes={}
        for line in src:
            row=json.loads(line);s=row['scenario']
            if (s['batch'],s['seq'],s['tp'],s['dp'],s['chunk'],s['step'])!=(8,32768,8,1,0,0):
                continue
            for r in row['contracts']:
                dims=r['tensors'].get('weight_logical',[])
                if len(dims)==2:
                    nodes[s['model'],r['op']]=dims
    result=[]
    def add(lane,model,node,b,q,wall,raw):
        n,k=nodes[model,node]
        result.append(dict(lane=lane,model=model,node=node,requests=b,query_per_request=q,
                           shape=[b*q,n,k],wall_s=wall,raw_mib=raw))
    for m in (1,9,11,16):
        add('decode',PRO,'indexer.weights_proj',m,1,1800,3072)
    add('decode',KIMI,'mla.q_b_proj',1,1,1800,4096)
    add('decode',FLASH,'dsa.wq_a',1,9,2400,4096)
    add('decode',KIMI,'mla.q_b_proj',11,1,2400,4096)
    add('decode',FLASH,'dsa.wkv',16,1,2400,4096)
    add('decode',FLASH,'indexer.weights_proj',9,9,1800,3072)
    add('decode',KIMI,'mla.q_b_proj',11,9,3600,6144)
    add('decode',PRO,'indexer.weights_proj',16,9,1800,3072)
    # A real unchunked 32Ki row geometry starts before the slower wide-K full batch.
    add('prefill',FLASH,'indexer.wk',1,32768,7200,12288)
    add('prefill',KIMI,'mla.q_b_proj',8,512,10800,24576)
    add('prefill',FLASH,'indexer.weights_proj',8,512,7200,12288)
    add('prefill',FLASH,'indexer.wk',8,32768,21600,32768)
    add('prefill',FLASH,'indexer.weights_proj',8,32768,64800,65536)
    return result

def freeze():
    if (OUT/'freeze.json').exists():raise FileExistsError('existing registration')
    seen={(r['chip'],tuple(r['shape'])) for r in read(ROOT/'results/v0.27/replay.json')['rows'] if r['op']=='matmul'}
    rows=geometries()
    assert len({tuple(r['shape']) for r in rows})==len(rows)
    for r in rows:
        sh=r['shape']
        if any((c,tuple(sh)) in seen for c in (DT,PR)):raise ValueError('historically measured '+str(sh))
        r['predictions']={}
        for c in (DT,PR,FUT):
            start=time.perf_counter_ns();p=estimate('matmul',sh,chip=c,dtype='bf16')
            q=(time.perf_counter_ns()-start)/1e6
            assert p.latency_us and q<100
            r['predictions'][c]=dict(us=p.latency_us,active_aic=p.active_aic,active_aiv=p.active_aiv,
                                     query_ms=q,work=p.work,plan=p.projection['target_plan'],stages=p.projection['stages_us'])
    sources=[Path(__file__),Path(__file__).with_name('runner.cpp'),BIN,ROOT/'smoke/large_vector.cpp',CATALOG,
             *[ROOT/'operator_model'/f for f in ('matmul_resource.py','matmul_v15.py','matmul_waves.py')],
             *[ROOT/'configs/models'/(m+'.json') for m in (FLASH,PRO,KIMI)]]
    reg=dict(utc=now(),rows=rows,source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
             interpretation='Per-rank BF16 unquantized ND X[M,K] times W[K,N]. Pinned TP8 geometry, contiguous transposed logical weight. Actual vLLM quantization, format and dispatch remain deployment-specific. Ordinary decode M=B; verify-8 M=9B absent padding. M=9 has both B9 single-step and B1 nine-token interpretations.',
             verification='All M*N outputs, installed ACLNN Matmul; all launches captured without launch-count selection; all active cores physical END, official core duration, SKU and memory model. Separate multiple auxiliary kernels from single Matmul when comparing. 970DT extrapolation only.',
             acquisition='One slot per lane; disjoint CPU masks. Each registered shape has one attempt per 950 SKU. Pause before a run if disk falls below 2x its cap + 16 GiB.')
    save(OUT/'freeze.json',reg)

def measure(job,chip,folder):
    m,n,k=job['shape']
    args=['/usr/bin/time','-f','%U %S','-o',str(folder/'cpu.txt'),'bash',str(ROOT/'scripts/cann-run.sh'),'msopprof','simulator',
          '--soc-version='+chip,'--dump=false','--timeout='+str((job['wall_s']+59)//60),'--output='+str(folder/'profile'),str(BIN),*map(str,(m,n,k))]
    t=time.monotonic();start=now();peak=0;stop=None
    with (folder/'run.log').open('w') as log:
        with subprocess.Popen(args,cwd=folder,stdout=log,stderr=subprocess.STDOUT,start_new_session=True) as proc:
            try:
                while proc.poll() is None:
                    peak=max(peak,nbytes(folder))
                    if time.monotonic()-t>job['wall_s'] or peak>job['raw_mib']*2**20:
                        stop='wall cap' if time.monotonic()-t>job['wall_s'] else 'raw cap'
                        os.killpg(proc.pid,signal.SIGKILL);break
                    time.sleep(2)
            finally:
                if proc.poll() is None and not stop:
                    os.killpg(proc.pid,signal.SIGTERM)
                proc.wait()
    save(folder/'cost.json',dict(seconds=time.monotonic()-t,started_utc=start,finished_utc=now(),exit_code=proc.returncode,
                                 stop_reason=stop,peak_bytes=max(peak,nbytes(folder)),command=args[5:]))

def qualify(job,chip,folder):
    m,n,k=job['shape'];log=(folder/'run.log').read_text(errors='replace')
    if f'case=matmul api=aclnnMatmul M={m} N={n} K={k} dtype=BF16 layout=ND_ND' not in log:
        raise ValueError('incorrect invocation')
    names=re.findall(r'Start profiling on kernel: (\S+)',log)
    if not names:raise ValueError('no profiled kernels')
    ks,gold=stages(dict(kernels=len(names),nchecks=1),chip,folder)
    if gold!=[f'correctness count={m*n} bad=0']:raise ValueError('numerical result incomplete')
    for stage in ks:
        trace=ROOT/stage['trace'];cores={p.parent.name:p for p in trace.parent.glob('core*/trace.json')}
        if set(cores)!=set(stage['windows']):raise ValueError('core roster mismatch')
        origins=[]
        for c,p in cores.items():
            marks=endpoint(p)
            origins.append(max(v['retire'] for v in marks.values())-1650*stage['windows'][c][1])
        if not origins or max(origins)-min(origins)>3:raise ValueError('end marker disagrees with all-core window')
    # Restrict the official comparison to the standalone kernel scope.
    main=[s for s in ks if 'matmul' in s['kernel'].lower() or 'gemm' in s['kernel'].lower() or 'gemv' in s['kernel'].lower()]
    if len(ks)!=1 or len(main)!=1:
        return dict(qualified=False,ca_us=None,stages=ks,correctness=gold,reason='different/multiple installed kernels; keep measured stages separate')
    ref=job['predictions'][chip]['us'];ca=main[0]['ca_us']
    return dict(qualified=True,ca_us=ca,model_us=ref,signed_error_pct=100*(ref/ca-1),stages=ks,correctness=gold)

def run(lane):
    reg=read(OUT/'freeze.json')
    for path,h in reg['source_sha256'].items():
        if digest(ROOT/path)!=h:raise ValueError('frozen input changed: '+path)
    RAW.mkdir(exist_ok=True)
    with (RAW/(lane+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.nice(8);os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
        allowed=sorted(os.sched_getaffinity(0)); split=len(allowed)//2
        cpus=allowed[:split] if lane=='decode' else allowed[split:]
        os.sched_setaffinity(0,cpus[:min(96,len(cpus))])
        for job in (r for r in reg['rows'] if r['lane']==lane):
            for chip,tag in ((DT,'dt'),(PR,'pr')):
                slug='-'.join(map(str,job['shape']))+'-'+tag;folder=RAW/slug
                path=OUT/(slug+'.json')
                if path.exists():continue
                state=dict(utc=now(),lane=lane,pid=os.getpid(),chip=chip,shape=job['shape'],case=slug)
                if not (folder/'cost.json').exists():
                    if folder.exists():raise RuntimeError('incomplete previous run: '+slug)
                    while shutil.disk_usage(ROOT).free<2*job['raw_mib']*2**20+16*2**30:
                        save(OUT/(lane+'-status.json'),dict(state,phase='waiting for free disk'));time.sleep(30)
                    save(OUT/(lane+'-status.json'),dict(state,phase='acquiring'))
                    folder.mkdir();measure(job,chip,folder)
                try:
                    result=qualify(job,chip,folder)
                except (ValueError,KeyError,OSError,AssertionError,IndexError) as exc:
                    result=dict(qualified=False,ca_us=None,reason=repr(exc))
                save(path,dict(state,freeze_sha256=digest(OUT/'freeze.json'),cost=read(folder/'cost.json'),**result))
                subprocess.run([sys.executable,str(Path(__file__).with_name('report.py'))],check=True)
        save(OUT/(lane+'-status.json'),dict(utc=now(),phase='finished',lane=lane))

if __name__=='__main__':
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['freeze','decode','prefill'])
    action=ap.parse_args().action
    if action=='freeze':freeze()
    else:run(action)
