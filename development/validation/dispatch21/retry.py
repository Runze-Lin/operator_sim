"""Recover the two initial opposite-SKU timeouts once, after the main dispatch queue."""
from pathlib import Path
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import read, write, digest, CHIPS, now
from development.validation.sku19.collect import capture, qualify
from development.validation.large20.recover_cost import recover_cost

OUT=ROOT/'results/v0.21/dispatch'
RAW=ROOT/'runs/dispatch21-retry'
REG=OUT/'retry-registration.json'


def register():
    frozen=read(OUT/'freeze.json')
    if REG.exists():
        raise FileExistsError(REG)
    attempts=[]
    for jobid,side in (('addmm_17_80_96','dt'),('mhc_post_5_4_288','pr')):
        job=next(j for j in frozen['jobs'] if j['id']==jobid)
        chip=CHIPS[side]
        path=OUT/'receipts'/(jobid+'_'+chip+'.json')
        failed=read(path)
        assert not failed['qualified'] and failed['cost']['stop_reason']=='wall cap'
        attempts.append(dict(id=jobid,chip=chip,original_sha=digest(path),wall_s=job['wall_s']))
    write(REG,dict(utc=now(),original_freeze_sha=digest(OUT/'freeze.json'),attempts=attempts,
          concurrency=1,trigger='Both initial attempts lack a complete kernel; wait for the main forty dispatch receipts; same binary, shape, SKU, host worker count and timeout. No replacement of original run or prediction.'))


def run():
    plan=read(REG)
    frozen=read(OUT/'freeze.json')
    assert digest(OUT/'freeze.json')==plan['original_freeze_sha']
    while len(list((OUT/'receipts').glob('*.json')))<2*len(frozen['jobs']):
        time.sleep(60)
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    for a in plan['attempts']:
        chip=a['chip']
        job=next(j for j in frozen['jobs'] if j['id']==a['id'])
        filename=job['id']+'_'+chip+'.json'
        source=OUT/'receipts'/filename
        assert digest(source)==a['original_sha'] and digest(ROOT/job['runner'])==job['runner_sha256']
        dest=OUT/'retries'/filename
        dest.parent.mkdir(exist_ok=True)
        if dest.is_file():
            continue
        folder=RAW/dest.stem
        if (folder/'cost.json').is_file():
            cost=read(folder/'cost.json')
        elif (folder/'run.log').is_file():
            cost=recover_cost(folder,job['wall_s'],job['raw_mib']*2**20)
            write(folder/'cost.json',cost)
        else:
            cost=capture(job,chip,folder)
        assert cost['started_utc']>plan['utc']
        try:
            obs,=qualify(job,chip,folder)
            outcome=dict(qualified=True,observation=obs)
        except (ValueError,KeyError,IndexError,OSError) as exc:
            outcome=dict(qualified=False,error=str(exc),cost=cost)
        write(dest,dict(id=job['id'],op=job['op'],shape=job['shape'],chip=chip,**outcome))


if __name__=='__main__':
    {'register':register,'run':run}[sys.argv[1]]()
