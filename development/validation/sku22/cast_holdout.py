"""Append four unseen flat workloads to the six previously frozen Cast pairs."""
import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns

ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'results/v0.22'
sys.path.insert(0,str(ROOT))
from operator_model import estimate
from development.validation.sku19.campaign import now, read, write, CHIPS


def digest(path): return sha256(path.read_bytes()).hexdigest()


def freeze(original):
    path=OUT/'cast-freeze.json'
    if path.exists():raise FileExistsError(path)
    shapes=[[9,5120],[255,7168],[479,5120],[131073,64]]
    seen={r['shape'][0]*r['shape'][1] for r in read(original/'results/v0.21/evaluation.json')['rows'] if r['op']=='cast_bf16_fp32'}
    assert all(r*c not in seen for r,c in shapes)
    old='/tmp/operator-sim-v21-deploy/bin/python'
    inputs=[(s,c) for s in shapes for c in CHIPS.values()]
    code="import json,sys;from operator_model import estimate,__version__;assert __version__=='0.21.0';print(json.dumps([estimate('cast_bf16_fp32',s,chip=c).latency_us for s,c in json.load(sys.stdin)]))"
    baseline=json.loads(subprocess.check_output([old,'-I','-c',code],text=True,input=json.dumps(inputs),cwd='/tmp'))
    jobs=[]
    for (s,c),old_us in zip(inputs,baseline):
        start=perf_counter_ns(); e=estimate('cast_bf16_fp32',s,chip=c)
        jobs.append(dict(shape=s,chip=c,us=e.latency_us,old_us=old_us,query_ms=(perf_counter_ns()-start)/1.e6,active_aiv=e.active_aiv))
    files=['operator_model/cast_resource.py','operator_model/vector_memory.py','operator_model/backend.py',*(f'configs/chips/{c}.json' for c in CHIPS.values())]
    write(path,dict(utc=now(),runner_sha256=digest(original/'build/smoke/large_cast20'),prior_candidate_sha256=digest(original/'results/cast-capacity/two-slot-freeze.json'),
                    files={f:digest(ROOT/f) for f in files},jobs=jobs,wall_s=3600,raw_mib=8192,concurrency=1,reason='Six prospective paired inputs were available. Four new distinct sizes cover partial grid, tile transition, three-tile contention and long streaming. No parameter updates.'))


def collect(original):
    # Use the host/profiler wrappers installed in the collection checkout; public forecasts were frozen separately.
    sys.path.insert(0,str(original/'development/validation/sku17'))
    from development.validation.sku19 import collect as collector
    import campaign_collect
    collector.ROOT=original;campaign_collect.ROOT=original
    reg=read(OUT/'cast-freeze.json');runner='build/smoke/large_cast20'
    assert all(digest(ROOT/k)==v for k,v in reg['files'].items()) and digest(original/runner)==reg['runner_sha256']
    raw=original/'runs/sku22-cast';raw.mkdir(exist_ok=True);dest=OUT/'cast-receipts';dest.mkdir(exist_ok=True)
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    with (raw/'collection.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for row in reg['jobs']:
            r,c=row['shape'];chip=row['chip'];name=f'cast_{r}_{c}_{chip}';folder=raw/name;target=dest/(name+'.json')
            if chip==CHIPS['future'] or target.exists():continue
            job=dict(wall_s=reg['wall_s'],raw_mib=reg['raw_mib'],runner=runner,args=list(map(str,row['shape'])),batch=False,kernels=1,nchecks=1)
            if (folder/'cost.json').exists():cost=read(folder/'cost.json')
            elif folder.exists():raise RuntimeError('Incomplete previous capture: '+str(folder))
            else:cost=collector.capture(job,chip,folder)
            assert cost['started_utc']>reg['utc']
            try:
                (stage,),checks=campaign_collect.stages(job,chip,folder);log=(folder/'run.log').read_text()
                if f'case=cast_bf16_fp32 api=aclnnCast R={r} D={c}' not in log or checks!=[f'correctness count={r*c} bad=0'] or not stage['kernel'].startswith('Cast_') or stage['active_aic'] or not stage['active_aiv']:raise ValueError('host/kernel/golden mismatch')
                result=dict(qualified=True,stage=stage,correctness=checks,api='aclnnCast')
            except (OSError,ValueError,IndexError,KeyError) as exc:result=dict(qualified=False,reason=repr(exc))
            write(target,dict(**row,cost=cost,**result));print(name,result['qualified'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['freeze','collect']);parser.add_argument('archive',type=Path);a=parser.parse_args()
    {'freeze':freeze,'collect':collect}[a.mode](a.archive.resolve())
