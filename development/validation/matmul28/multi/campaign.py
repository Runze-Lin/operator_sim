"""Register and serially collect two exact new workload geometries after the narrow queue."""
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from operator_model.matmul_multisplit import estimate_matmul_retile
from operator_model.matmul_splitk import estimate_matmul_splitk
from development.validation.matmul28.campaign import DT,PR,FUT,BIN,digest,measure,qualify,now,read,save
from development.validation.matmul28.drain import recover
OUT=ROOT/'results/v0.28/matmul/multi';RAW=ROOT/'runs/matmul28-retile';REG=OUT/'freeze.json'
NARROW=ROOT/'results/v0.28/matmul/narrow/state.json'
SHAPES=[([72,64,7168],'DeepSeek-V4-Pro','indexer.weights_proj',8),([36,1280,5120],'DeepSeek-V4.1-Flash','dsa.wq_a',4)]
def register():
    if REG.exists() or RAW.exists():raise FileExistsError('freeze/raw already exists')
    known=set()
    for f in [ROOT/'results/v0.25/coverage/audit.json',ROOT/'results/v0.28/matmul/freeze.json',*list((OUT.parent/'narrow').glob('*freeze*.json'))]:
        for r in read(f)['rows']:
            if 'op' not in r or r['op']=='matmul':known.add(tuple(r['shape']))
    hits=[];checked=0
    for base in [ROOT/'runs',ROOT/'results']:
        for folder,dirs,fs in os.walk(base):
            dirs[:]=[d for d in dirs if d not in ('simulator','npusim','ca_workspace','profile','record','log','dump','sources') and not d.startswith(('OPPROF_','npusim_','PROF_'))]
            if 'run.log' not in fs:continue
            q=Path(folder)/'run.log'
            if q.stat().st_size>32*2**20:continue
            checked+=1
            for triple in re.findall(r'case=matmul api=aclnnMatmul M=(\d+) N=(\d+) K=(\d+)',q.read_text(errors='replace')):
                if tuple(map(int,triple)) in [tuple(x[0]) for x in SHAPES]:hits.append(str(q.relative_to(ROOT)))
    assert not hits and all(tuple(s) not in known for s,_,_,_ in SHAPES)
    records=[]
    for s,model,node,b in SHAPES:
        records.append(dict(shape=s,model=model,node=node,requests=b,q=9,wall_s=2400,raw_mib=4096,
            predictions={c:dict(us=estimate_matmul_retile(s,chip=c)['latency_us'],new=estimate_matmul_retile(s,chip=c),public=estimate_matmul_splitk(s,chip=c)) for c in (DT,PR,FUT)}))
    deps=[Path(__file__),ROOT/'operator_model/matmul_multisplit.py',ROOT/'operator_model/matmul_splitk.py',ROOT/'operator_model/matmul_resource.py',ROOT/'operator_model/matmul_v15.py',ROOT/'operator_model/matmul_waves.py',ROOT/'results/v0.16/matmul.json',ROOT/'development/validation/matmul28/campaign.py',ROOT/'development/validation/matmul28/drain.py',ROOT/'development/validation/matmul28/runner.cpp',ROOT/'scripts/cann-run.sh',BIN]
    deps.extend(ROOT/'configs/chips'/f'{c}.json' for c in (DT,PR,FUT))
    pkg=next((ROOT/'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_nn/mat_mul_v3').glob('*BF16_BF16_BF16_BF16_high_performance.o'));deps.append(pkg)
    with REG.open('x') as f:
        json.dump(dict(utc=now(),rows=records,source_sha256={str(v.relative_to(ROOT)):digest(v) for v in deps},
            novelty=dict(completed_historic_geometries=len(known),prior_host_logs_checked=checked,matched=[]),
            scope='Two distinct input geometries, two measured SKUs, both new. 970 configuration forward projection only.',
            admission='On each 950 endpoint require full golden, physical retirements and installed MIX key. For each shape and target match output rectangle count, per-rectangle splits, productive AIC and launch counts; MMA count checked separately. Per-SKU two-shape WAPE<=25%, max<=50%, WAPE below previously published schedule; archived original route must retain coverage. These two alone do not establish the ten-shape all-route objective.',
            policy='Single bounded process. Wait for results/v0.28/matmul/narrow/state.json phase=finished. Reserve >=100GiB storage and 64GiB memory, one-minute load under 60% host capacity, use up to 96 cores at low nice. Preserve any failed raw and original verdict. Re-use existing BF16 whole-output host and supplemental async-drain qualification; no retry of interrupted CA.'),f,indent=2);f.write('\n')
def state(phase,**kwargs):save(OUT/'state.json',dict(utc=now(),phase=phase,register_sha256=digest(REG),**kwargs))
def run():
    reg=read(REG)
    assert all(digest(ROOT/f)==h for f,h in reg['source_sha256'].items()),'source change after freeze'
    RAW.mkdir(exist_ok=True)
    with (RAW/'single.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        allowed=sorted(os.sched_getaffinity(0));cap=len(allowed);os.nice(10);os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
        while not NARROW.exists() or read(NARROW).get('phase')!='finished':
            state('queued behind narrow',depends_on=str(NARROW.relative_to(ROOT)));time.sleep(45)
        os.sched_setaffinity(0,allowed[:min(96,cap)])
        for job in reg['rows']:
            for c,tag in ((DT,'dt'),(PR,'pr')):
                slug='-'.join(map(str,job['shape']))+'-'+tag;folder=RAW/slug;orig=OUT/(slug+'-acquisition.json');dest=OUT/(slug+'.json')
                if dest.exists():continue
                if not (folder/'cost.json').exists():
                    if folder.exists():state('interrupted raw preserved',case=slug);return
                    while True:
                        ram=int(re.search(r'^MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text(),re.M)[1])/2**20
                        if os.getloadavg()[0]<.6*cap and shutil.disk_usage(ROOT).free>100*2**30 and ram>64:break
                        state('waiting for headroom',case=slug,load=os.getloadavg()[0]);time.sleep(45)
                    state('acquiring',case=slug);folder.mkdir();measure(job,c,folder)
                state('verifying',case=slug);cost=read(folder/'cost.json')
                assert cost['started_utc']>reg['utc']
                if not orig.exists():
                    try:initial=qualify(job,c,folder)
                    except (ValueError,OSError,AssertionError,KeyError,IndexError)as e:initial=dict(qualified=False,reason=repr(e))
                    save(orig,dict(case=slug,chip=c,shape=job['shape'],cost=cost,**initial))
                result=read(orig)
                if not result.get('qualified'):
                    try:
                        supplemental=recover(job,result,folder,acquisition=orig)
                        if supplemental:result=supplemental
                    except (ValueError,OSError,AssertionError,KeyError,IndexError)as e:result=dict(qualified=False,reason=repr(e))
                save(dest,dict(result,shape=job['shape'],chip=c,cost=cost,register_sha256=digest(REG),original_sha256=digest(orig)))
                from verify import score
                score()
        state('finished')
if __name__=='__main__':{'freeze':register,'collect':run}[sys.argv[1]]()
