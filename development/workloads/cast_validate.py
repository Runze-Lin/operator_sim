"""Register, freeze and validate the BF16-to-FP32 Cast on the 950 series."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.collect import capture
from development.validation.sku19.campaign import DT, PR, FUTURE, digest, now, read, write
from campaign_collect import stages

OUT=ROOT/'results/v0.19/workloads'
RAW=ROOT/'runs/workloads19'
TRAIN=[(1,192),(8,2048),(64,4096)]
TEST=[(32,4096),(1,64),(72,4096),(1,3584),(96,7168),(9,4096),(48,7168),(16,7168),(128,5120),(144,6144)]
PROFILE=ROOT/'operator_model/cast_resource.py'
BINARY=ROOT/'build/smoke/cast_bf16_fp32'


def register(stage):
    dest=OUT/f'cast-{stage}-freeze.json'
    if dest.exists():
        raise ValueError('Already frozen')
    chips=[DT,PR] if stage=='holdout' else [DT]
    jobs=[]
    for r,d in (TEST if stage=='holdout' else TRAIN):
        job=dict(id=f'cast_{r}_{d}',shape=[r,d],args=[str(r),str(d)],op='cast_bf16_fp32',batch=False,
                 kernels=1,nchecks=1,runner=str(BINARY.relative_to(ROOT)),wall_s=240,raw_mib=1800)
        if stage=='holdout':
            from operator_model.cast_resource import resource_cast
            from operator_model.chips import load_chip
            job['predictions']={p:resource_cast((r,d),load_chip(p))[0] for p in (DT,PR,FUTURE)}
        jobs.append(job)
    write(dest,dict(utc=now(),chips=chips,jobs=jobs,module_sha256=digest(PROFILE) if stage=='holdout' else None,
                    binary_sha256=digest(BINARY),max_workers=2,
                    omp_wait='PASSIVE' if stage=='holdout' else 'host default',
                    max_physical_seconds=240*len(chips)*len(jobs),raw_mib_cap_per_attempt=1800))


def collect(stage, retry=False):
    reg=read(OUT/f'cast-{stage}-freeze.json')
    assert digest(BINARY)==reg['binary_sha256']
    if stage=='holdout':
        assert digest(PROFILE)==reg['module_sha256']
    if reg.get('omp_wait')=='PASSIVE':
        import os
        os.environ['OMP_WAIT_POLICY']='PASSIVE'
        os.environ['GOMP_SPINCOUNT']='0'
    previous=read(OUT/f'cast-{stage}.json') if retry else []
    failed={(tuple(row['shape']),row['sku']) for row in previous if not row['qualified']}
    def run(arg):
        j,chip=arg
        path=RAW/(j['id']+'_'+chip+('-retry' if retry else ''))
        if not (path/'cost.json').exists():capture(j,chip,path)
        cost=read(path/'cost.json')
        row=dict(shape=j['shape'],sku=chip,cost=cost)
        try:
            evidence,checks=stages(j,chip,path)
            row.update(qualified=True,kernel=evidence[0],correctness=checks)
        except (AssertionError,IndexError,ValueError) as e:
            row.update(qualified=False,reason=str(e))
        print(j['id'],chip,row['qualified'],row.get('reason',''),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=reg['max_workers']) as pool:
        pending=[(j,c) for j in reg['jobs'] for c in reg['chips'] if not retry or (tuple(j['shape']),c) in failed]
        rows=list(pool.map(run,pending))
    write(OUT/(f'cast-{stage}-retry.json' if retry else f'cast-{stage}.json'),rows)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['register','collect'])
    p.add_argument('stage',choices=['calibration','holdout'])
    p.add_argument('--retry',action='store_true',help='Repeat initial failed inputs once, preserving the original attempts')
    args=p.parse_args()
    if args.action=='register':
        register(args.stage)
    else:
        collect(args.stage,args.retry)
