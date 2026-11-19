"""A small, paired RMSNorm campaign for the unknown-stream-history model."""
from pathlib import Path
import json
import os
import sys
import subprocess
from time import perf_counter_ns

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate
from development.validation.sku19.campaign import CHIPS,read,write,digest,now
from development.validation.sku19.collect import capture,qualify
from development.validation.sku20.rms_campaign import RUNNER

OUT=ROOT/'results/v0.21/rms'
RAW=ROOT/'runs/rms21'
SHAPES=[(3,6576),(2,5808),(7,6320),(19,7632),(51,1488),(79,2192),(157,3136),
        (495,5040),(1537,5120),(4097,1536)]


def freeze():
    if RAW.exists() or (OUT/'freeze.json').exists():raise FileExistsError(OUT/'freeze.json')
    # Existing explicit history through v0.18 + all subsequently registered 2-D shapes.
    older=read(ROOT/'results/v0.18/sku/excluded-history.json')['shapes']['rmsnorm']
    known={tuple(s) for s in older}
    for version in ('0.18','0.19','0.20'):
        for filename in ('evaluation.json','sku/freeze.json','rms/freeze.json'):
            path=ROOT/f'results/v{version}'/filename
            if path.exists():
                obj=read(path)
                known.update(tuple(r['shape']) for r in obj.get('rows',obj.get('jobs',[])) if r.get('op')=='rmsnorm')
    assert not known.intersection(SHAPES)
    code="""import sys,json;from operator_model import estimate,__version__;assert __version__=='0.20.0'
cs,ss=json.load(sys.stdin);print(json.dumps([{k:estimate('rmsnorm',s,chip=c).latency_us for k,c in cs.items()} for s in ss]))"""
    prior=json.loads(subprocess.check_output(['/tmp/operator-sim-v20-deploy/bin/python','-I','-c',code],
                     text=True,cwd='/tmp',input=json.dumps([CHIPS,SHAPES])))
    jobs=[]
    for s,old in zip(SHAPES,prior):
        y=dict(op='rmsnorm',shape=s,attrs={},indices=[0])
        for side,chip in CHIPS.items():
            t=perf_counter_ns();r=estimate('rmsnorm',s,chip=chip);ms=(perf_counter_ns()-t)/1e6
            assert r.latency_us and ms<100
            y.update({side+'_us':r.latency_us,side+'_ms':ms,side+'_first_us':r.projection['timing_diagnostics']['conditional_us']['first'],
                      'old_'+side+'_us':old[side]})
        jobs.append(dict(id='rmsnorm_'+'_'.join(map(str,s)),op='rmsnorm',shape=s,attrs={},runner=RUNNER,
                         runner_sha256=digest(ROOT/RUNNER),args=['rmsnorm',*map(str,s)],outputs=[y],
                         kernels=1,nchecks=2,wall_s=5400,raw_mib=max(1200,s[0]*s[1]//480)))
    files=[ROOT/'operator_model'/p for p in ('rms_dispatch.py','rms_resource.py','rms_target_cost.py','rms_v12.py','wide_v10.py')]
    files.extend([OUT/'calibration.json',ROOT/RUNNER,Path(__file__),*(ROOT/'configs/chips').glob('*.json')])
    write(OUT/'freeze.json',dict(utc=now(),jobs=jobs,excluded_shapes=len(known),
          refs={str(p.relative_to(ROOT)):digest(p) for p in files},workers=1,
          role='independent: no prelaunch state given to predictor; fresh scope known to harness only'))


def run():
    plan=read(OUT/'freeze.json')
    assert all(digest(ROOT/k)==h for k,h in plan['refs'].items())
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    rec=OUT/'receipts';rec.mkdir(exist_ok=True)
    for j in plan['jobs']:
        for side in ('dt','pr'):
            chip=CHIPS[side];d=RAW/(j['id']+'_'+chip);path=rec/(d.name+'.json')
            if path.exists():continue
            cost=read(d/'cost.json') if (d/'cost.json').exists() else capture(j,chip,d)
            assert cost['started_utc']>plan['utc']
            try:
                obs,=qualify(j,chip,d)
                out=dict(qualified=True,observation=obs)
            except (ValueError,OSError,IndexError,KeyError) as e:
                out=dict(qualified=False,error=str(e),cost=cost)
            write(path,dict(id=j['id'],op=j['op'],shape=j['shape'],chip=chip,**out))
            print(d.name,out['qualified'],flush=True)

if __name__=='__main__':
    {'freeze':freeze,'run':run}[sys.argv[1]]()
