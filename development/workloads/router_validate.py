"""Calibrate/freeze/replay the Kimi FP32 sigmoid top-k leaf with bounded CA."""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'development/validation/sku17'))
from campaign_collect import capture, stages
sys.path.insert(0,str(ROOT))

OUT = ROOT/'results/v0.19/workloads'
RAW = ROOT/'runs/workloads19'
DT, PR, FUT = 'Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'
TRAIN = [(1,128,8),(1,896,16),(129,896,16)]
TEST = [(1,384,6),(2,896,16),(4,256,8),(7,896,16),(9,384,8),(12,896,16),
        (16,512,8),(24,896,16),(33,384,8),(65,128,8)]

def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()

def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def register(stage):
    shapes = TEST if stage=='holdout' else TRAIN
    runner=ROOT/'build/smoke/moe_router19'
    jobs=[]
    for shape in shapes:
        job=dict(id='router_'+'_'.join(map(str,shape)),op='moe_gating_top_k',args=list(map(str,shape)),batch=False,
                 runner=str(runner.relative_to(ROOT)),runner_sha256=digest(runner),nchecks=2,kernels=1,
                 wall_s=240 if stage=='holdout' else 180,raw_mib=(max(180, min(shape[0],64)*38) if stage=='holdout' else 300),shape=shape)
        if stage=='holdout':
            from operator_model.moe_router import resource_router
            from operator_model.chips import load_chip
            job['predictions']={sku:resource_router(shape,load_chip(sku))[0] for sku in [DT,PR,FUT]}
        jobs.append(job)
    p=OUT/f'router-{stage}-freeze.json'
    if p.exists():
        raise ValueError(f'Registration already exists: {p}')
    write(p,dict(utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,chips=[DT,PR] if stage=='holdout' else [DT],
                 model_sha256=digest(ROOT/'operator_model/moe_router.py') if stage=='holdout' else None,
                 max_workers=2))

def collect(stage,attempt):
    reg=json.loads((OUT/f'router-{stage}-freeze.json').read_text())
    if reg['model_sha256'] and digest(ROOT/'operator_model/moe_router.py')!=reg['model_sha256']:
        raise ValueError('Changed frozen model')
    def one(arg):
        j,chip=arg
        d=RAW/(j['id']+'_'+chip+attempt)
        costfile=d/'cost.json'
        cost=json.loads(costfile.read_text()) if costfile.exists() else capture(j,chip,d,reg)
        row=dict(shape=j['shape'],sku=chip,cost=cost)
        try:
            ks,correct=stages(j,chip,d)
            row.update(qualified=True,correctness=correct,kernel=ks[0],predictions=j.get('predictions'))
        except (ValueError,IndexError,AssertionError) as err:
            row.update(qualified=False,reason=str(err))
        print(j['id'],chip,row['qualified'],row.get('reason',''),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=reg['max_workers']) as pool:
        rows=list(pool.map(one,[(j,chip) for j in reg['jobs'] for chip in reg['chips']]))
    write(OUT/f'router-{stage}{attempt}.json',rows)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['register','collect'])
    p.add_argument('stage',choices=['calibration','holdout'])
    p.add_argument('--attempt-suffix', choices=['','-ipc'], default='')
    a=p.parse_args()
    if a.operation=='register':
        register(a.stage)
    else:
        collect(a.stage,a.attempt_suffix)
