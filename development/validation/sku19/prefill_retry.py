"""One same-contract PR chunk recapture with passive waits and full instruction output."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import OUT, PR, now, read, write, digest
from development.validation.sku19.recover_windows import parse
from campaign_collect import capture

TARGET=ROOT/'results/v0.19/prefill'
REG=TARGET/'chunk-retry-registration.json'
FOLDER=ROOT/'runs/sku19-retry'/('rmsnorm_2048_5120_'+PR)


def register():
    assert not REG.exists() and not FOLDER.exists()
    j=next(x for x in read(OUT/'freeze.json')['jobs'] if x['id']=='rmsnorm_2048_5120')
    original=ROOT/'runs/sku19'/FOLDER.name/'cost.json'
    write(REG,dict(utc=now(),job=dict(j,raw_mib=16384,wall_s=2400),chip=PR,
                   main_freeze_sha256=digest(OUT/'freeze.json'),original_cost=read(original),dump=True,
                   max_launches=1,parallel=1,omp_wait_policy='PASSIVE',gomp_spincount='0',
                   reason='Original complete host golden but 4 of 56 AIVs omitted the final END_LABEL in CSV, and core12.veccore0 omitted the last two BF16 output rows from its trace. Reuse fixed host/shape; request vendor instruction dumps and allow final draining.'))


def run():
    r=read(REG);j=r['job']
    assert not FOLDER.exists() and digest(OUT/'freeze.json')==r['main_freeze_sha256']
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    cost=capture(j,PR,FOLDER,r)
    outcome=dict(id=j['id'],op=j['op'],shape=j['shape'],chip=PR,**cost)
    try:
        rec=parse(j,PR,FOLDER)
        rec.update(backend='official all-core PR chunk retry; explicit full instruction export',
                   role='supplemental complete prefill chunk')
        write(FOLDER/'qualified.json',rec)
        outcome.update(eligible=True,reason=None,ca_us=rec['ca_us'],trace_sha256=rec['trace_sha256'])
    except (ValueError,KeyError,IndexError,AssertionError,OSError) as exc:
        outcome.update(eligible=False,reason=repr(exc))
    write(FOLDER/'outcome.json',outcome)
    write(TARGET/'chunk-retry-outcome.json',outcome)
    print(outcome['eligible'],outcome['reason'],flush=True)


if __name__=='__main__':
    {'register':register,'run':run}[sys.argv[1]]()
