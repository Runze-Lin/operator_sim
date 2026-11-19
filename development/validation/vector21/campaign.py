"""Paired, frozen, full-dispatch CA validation of two AIV streaming operators."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from threading import Lock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.sku19.campaign import CHIPS, DT, PR, digest, now, read, write
from development.validation.sku19.collect import capture
from campaign_collect import stages
from operator_model import estimate

OUT = ROOT/'results/v0.21/vector'
RAW = ROOT/'runs/vector21'
OLD = '/tmp/operator-sim-v20-deploy/bin/python'
S, C = 'sigmoid_fp32', 'cast_bf16_fp32'
RUNNERS = {S: 'build/smoke/sigmoid20', C: 'build/smoke/large_cast20'}
SHAPES = {
    S: [(3073,12),(524288,12),(65537,24),(1572864,12),(3,96),(129,12),
        (4545,24),(16385,48),(4097,96),(98304,12)],
    C: [(65,5120),(393216,32),(3072,5120),(1572864,32),(3,7168),(17,32),
        (111,7168),(511,5120),(32769,32),(98305,64)]}
# Multiple layouts with identical element counts exercise the same contiguous
# kernel. Exclude those from the new set even if the tensor shapes differ.
def historical(op):
    if op == S:
        rows = sum((read(OUT.parents[1]/'v0.20/newop'/f'{p}.json') for p in ('holdout','development')),[])
    else:
        prior = OUT.parents[1]/'v0.20/large'
        rows = read(prior/'prior-cast-replay.json')['rows'] + read(prior/'cast-v20-freeze.json')['jobs']
    return {r['shape'][0]*r['shape'][1] for r in rows}


def freeze():
    path = OUT/'freeze.json'
    if path.exists():
        raise FileExistsError(path)
    pairs = [(op,r,c) for i in range(10) for op in (S,C) for r,c in [SHAPES[op][i]]]
    for op in SHAPES:
        seen = historical(op)
        for r,c in SHAPES[op]:
            if r*c in seen:
                raise ValueError((op,r,c,'repeated flat workload'))
            seen.add(r*c)
    # Keep the baseline interpreter isolated from the working source tree.
    query = """import json,sys
from operator_model import estimate,__version__
assert __version__ == '0.20.0'
print(json.dumps([estimate(op,shape,chip=chip).latency_us for op,shape,chip in json.load(sys.stdin)]))"""
    inputs = [(op,[r,c],chip) for op,r,c in pairs for chip in CHIPS.values()]
    old = iter(json.loads(subprocess.check_output([OLD,'-I','-c',query],cwd='/tmp',
                                                   input=json.dumps(inputs),text=True)))
    jobs = []
    for op,r,c in pairs:
        targets={}
        for chip in CHIPS.values():
            tick=perf_counter()
            e=estimate(op,[r,c],chip=chip)
            ms=(perf_counter()-tick)*1000
            targets[chip]=dict(us=e.latency_us,aiv=e.active_aiv,old_us=next(old),
                               query_ms=ms,diagnostics=e.projection['timing_diagnostics'])
        jobs.append(dict(id=f'{op}_{r}_{c}',op=op,shape=[r,c],args=[str(r),str(c)],
                         runner=RUNNERS[op],runner_sha256=digest(ROOT/RUNNERS[op]),targets=targets,
                         batch=False,kernels=1,nchecks=1,wall_s=10800 if r*c>5000000 else 1800,
                         raw_mib=24576 if r*c>5000000 else 4096))
    modules=['operator_model/'+s+'.py' for s in ('vector_memory','sigmoid_resource','cast_resource','rms_resource','backend','chips')]
    modules+=['configs/chips/'+c+'.json' for c in CHIPS.values()]
    write(path,dict(utc=now(),jobs=jobs,hashes={p:digest(ROOT/p) for p in modules},
                    role='post-freeze paired holdout',max_workers=2,
                    rule='Whole all-AIV window, per-core vendor replay and every output correct. v0.20 wheel isolated. 970DT projected only. 10 distinct flattened sizes/operator, 2 long unchunked prefills each.'))


def assert_resource_revision():
    """Analysis that reprices data must use the registered resource implementation."""
    hashes=read(OUT/'freeze.json')['hashes']
    for p in ('operator_model/vector_memory.py','operator_model/sigmoid_resource.py',
              'operator_model/cast_resource.py','operator_model/rms_resource.py','operator_model/chips.py',
              *(f'configs/chips/{c}.json' for c in CHIPS.values())):
        if digest(ROOT/p)!=hashes[p]:
            raise ValueError(f'Frozen resource source changed: {p}')


def collect():
    reg=read(OUT/'freeze.json')
    os.environ.update(OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0')
    records_path=OUT/'holdout.json'
    rows=read(records_path) if records_path.exists() else []
    lock=Lock()
    def one(arg):
        job,chip=arg
        if any(row['id']==job['id'] and row['chip']==chip for row in rows):
            return
        assert digest(ROOT/job['runner'])==job['runner_sha256']
        d=RAW/(job['id']+'_'+chip)
        if (d/'run.log').exists() and not (d/'cost.json').exists():
            raise RuntimeError(f'Existing collection missing final cost; preserve its trace: {d}')
        cost=read(d/'cost.json') if (d/'cost.json').exists() else capture(job,chip,d)
        assert cost['started_utc']>reg['utc']
        row=dict(id=job['id'],op=job['op'],shape=job['shape'],chip=chip,cost=cost,qualified=False)
        try:
            (stage,),correct=stages(job,chip,d)
            r,c=job['shape']
            api,identity=(('aclnnSigmoid',f'T={r} H={c}') if job['op']==S else ('aclnnCast',f'R={r} D={c}'))
            log=(d/'run.log').read_text(errors='replace')
            assert f"case={job['op']} api={api} {identity}" in log and f'correctness count={r*c} bad=0' in log
            assert stage['kernel'].startswith(api[5:]+'_') and stage['active_aiv'] and not stage['active_aic']
            row.update(qualified=True,ca_us=stage['ca_us'],stage=stage,correctness=correct,api=api)
        except (OSError, AssertionError, ValueError, IndexError, KeyError) as exc:
            row['reason']=repr(exc)
        with lock:
            rows.append(row)
            write(records_path,rows)
            print(job['id'],chip,row['qualified'],row.get('ca_us'),row.get('reason',''),flush=True)
    RAW.mkdir(parents=True,exist_ok=True)
    with (RAW/'collection.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with ThreadPoolExecutor(reg['max_workers']) as pool:
            list(pool.map(one,((j,c) for j in reg['jobs'] for c in (DT,PR))))

if __name__=='__main__':
    {'freeze':freeze,'collect':collect}[sys.argv[1]]()
