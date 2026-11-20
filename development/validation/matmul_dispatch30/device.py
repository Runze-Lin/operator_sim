"""Run the same physical Flash projection across decode, speculative verification and prefill."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.matmul28 import campaign as ca
from development.validation.matmul28.drain import recover
from operator_model import estimate
from operator_model.matmul_supply30 import estimate_matmul_supply
from operator_model.matmul_placement import matmul_placement

OUT=ROOT/'results/v0.30/matmul-dispatch/device'
RAW=ROOT/'runs/matmul_dispatch30/device'
REGISTER=OUT/'freeze.json'
CHIPS=[(ca.DT,'dt'),(ca.PR,'pr')]


def save(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    width,={(r['shape'][1],r['shape'][2]) for r in ca.read(ROOT/'results/v0.28/matmul/overview-prices.json')['rows']
            if r['model']==ca.FLASH and r['node']=='indexer.wk'}
    known={tuple(r['shape']) for r in ca.read(ROOT/'results/v0.29/matmul/score.json')['rows']}
    for rel in ('v0.28/matmul/overview-prices.json','v0.29/matmul/timing-freeze.json','v0.29/matmul/full-load/freeze.json',
                'v0.29/matmul/decode/freeze.json'):
        reg=ca.read(ROOT/'results'/rel)
        known.update(tuple(r['shape']) for r in reg.get('rows',[]) if 'shape' in r)
    scenarios={}
    for batch in (1,9,11,16):
        for q in (1,9):
            scenarios.setdefault(batch*q,[]).append({'batch':batch,'tokens_per_request':q,'phase':'decode'})
    scenarios[524288]=[{'batch':8,'tokens_per_request':65536,'phase':'prefill'}]
    jobs=[]
    for m in (9,1,11,16,81,99,144,524288):
        shape=[m,*width]
        if tuple(shape) in known:
            raise ValueError(f'Already registered or measured {shape}')
        prices={}
        for sku in (ca.DT,ca.PR,ca.FUT):
            base=estimate('matmul',shape,chip=sku)
            supply=estimate_matmul_supply(shape,chip=sku)
            place=matmul_placement(shape,chip=sku)
            prices[sku]=dict(us=base.latency_us,public_active_aic=base.active_aic,public_active_aiv=base.active_aiv,
                             placement_us=place['latency_us'],placement_route=place['route'],
                             supply_us=supply['latency_us'],supply_waves=supply.get('waves'))
        large=m==524288
        jobs.append(dict(shape=shape,scenarios=scenarios[m],predictions=prices,wall_s=21600 if large else 900,
                         raw_mib=24576 if large else 512))
    paths=[Path(__file__),ca.BIN,Path(ca.__file__),ROOT/'development/validation/matmul28/drain.py',
           ROOT/'operator_model/matmul_supply30.py',ROOT/'results/v0.30/matmul/prices.json',ROOT/'operator_model/matmul_placement.py']
    with REGISTER.open('x') as f:
        json.dump(dict(utc=ca.now(),model=ca.FLASH,node='indexer.wk',rows=jobs,
                       hashes={str(p.relative_to(ROOT)):ca.digest(p) for p in paths},
                       admission='One endpoint at a time after host-control pilot, positive niceness. Storage reserve 2x per-run raw cap +16 GiB. Preserve interrupted raw and duplicate protection. Distinct M=9 aliases charged once.',
                       qualification='BF16 ND aclnnMatmul, all outputs and all participating cores through END plus asynchronous retire and official time. Missing endpoint stays unmeasured. 970DT extrapolation only.'),f,indent=2)
        f.write('\n')


def run():
    reg=ca.read(REGISTER)
    if not all(ca.digest(ROOT/p)==h for p,h in reg['hashes'].items()):
        raise RuntimeError('Frozen collector or price artifact changed')
    RAW.mkdir(parents=True,exist_ok=True)
    os.nice(12)
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    def state(phase,**kw):
        save(OUT/'state.json',dict(utc=ca.now(),pid=os.getpid(),phase=phase,register_sha256=ca.digest(REGISTER),**kw))
    with (RAW/'supervisor.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state('awaiting host pilot lane')
        with (ROOT/'runs/host30/lane.lock').open('a') as slot:
            fcntl.flock(slot,fcntl.LOCK_EX)
            for job in reg['rows']:
                for sku,tag in CHIPS:
                    label=f"{job['shape'][0]}-{tag}"
                    folder=RAW/label
                    ticket=OUT/(label+'.json')
                    if ticket.exists():continue
                    if not (folder/'cost.json').exists():
                        if folder.exists():
                            state('interrupted raw retained',case=label);return
                        while shutil.disk_usage(ROOT).free < 2*job['raw_mib']*2**20+16*2**30:
                            state('awaiting disk',case=label);time.sleep(30)
                        state('acquiring',case=label)
                        folder.mkdir()
                        ca.measure(job,sku,folder)
                    cost=ca.read(folder/'cost.json')
                    try:
                        raw=ca.qualify(job,sku,folder)
                    except (OSError,ValueError,IndexError,AssertionError,KeyError) as exc:
                        raw=dict(qualified=False,reason=repr(exc))
                    initial=OUT/(label+'-initial.json')
                    raw.update(chip=sku,shape=job['shape'],cost=cost)
                    save(initial,raw)
                    try:
                        final=raw if raw['qualified'] else recover(job,raw,folder,acquisition=initial) or raw
                    except (OSError,ValueError,IndexError,AssertionError,KeyError) as exc:
                        final=dict(qualified=False,reason=repr(exc))
                    save(ticket,dict(final,chip=sku,shape=job['shape'],cost=cost,registration_sha256=ca.digest(REGISTER)))
        state('finished')


if __name__=='__main__':
    {'freeze':freeze,'run':run}[sys.argv[1]]()
