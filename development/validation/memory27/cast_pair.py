"""Independent, serial replacement geometry for a missing PR capacity-boundary window."""
from argparse import ArgumentParser
from hashlib import sha256
from math import prod
from pathlib import Path
import fcntl
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT/'results/v0.25/cache'
OUT = ROOT/'results/v0.27/memory/cast-pair'
sys.path[:0] = [str(ROOT/'development/validation/cache25'),str(ROOT)]
from development.validation.cache25 import campaign
from shadow import predict

SHAPE = [5632,5120]  # Eight 704-token chunks in the real Flash full hidden width.


def register():
    OUT.mkdir(exist_ok=True)
    existing = campaign.read(ROOT/'results/v0.27/replay.json')['rows']
    if any(r['op']=='cast_bf16_fp32' and prod(r['shape'])==prod(SHAPE) for r in existing):
        raise ValueError('Flattened geometry already replayed')
    original = campaign.read(OLD/'freeze.json')
    for prefix in ('cast_bf16_fp32','sigmoid_fp32'):
        assert all(j['shape']!=SHAPE for j in original['jobs'] if j['op']==prefix)
    cases = [j for j in original['jobs'] if j['op']=='cast_bf16_fp32']
    forecasts = {sku:predict('cast_bf16_fp32',SHAPE,sku) for sku in (campaign.PR,campaign.DT,campaign.FUTURE)}
    for j in cases:
        j.update(shape=SHAPE,args=list(map(str,SHAPE)),expected_elements=prod(SHAPE),
                 predictions=forecasts,working_set_MiB=prod(SHAPE)*6/2**20,wall_s=2700,raw_mib=12288)
    original.update(jobs=cases,utc=campaign.utc(),sha256={**original['sha256'],
                    str(Path(__file__).resolve().relative_to(ROOT)):campaign.digest(Path(__file__))},
                    role='New holdout with real Flash hidden width. Each SKU once after original single slot. Does not replace the failed original window in any metric.',
                    bounds='Two serial CA jobs, 45 min and 12 GiB each. Same ELF and qualification. No 970 CA.',
                    assessment='Score all frozen policies on both endpoints; check whether pressure lowers PR absolute latency error and whether DT stays within 25% of CA. Preserve all failures.')
    with (OUT/'freeze.json').open('x') as f:
        json.dump(original,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')


def run():
    os.nice(10)
    with (ROOT/'runs/memory27-followup/remaining.lock').open('a') as slot:
        fcntl.flock(slot,fcntl.LOCK_EX)
        reg = OUT/'freeze.json'
        original = campaign.read(reg)
        campaign.OUT, campaign.RAW = OUT,ROOT/'runs/memory27-cast-pair'
        for index, job in enumerate(original['jobs']):
            single = OUT/(job['id']+'-registration.json')
            if not single.exists():
                campaign.save(single,dict(original,jobs=[job],parent_sha256=campaign.digest(reg)))
            verified = campaign.read(single)
            if verified['jobs']!=[job] or verified['sha256']!=original['sha256'] or verified['parent_sha256']!=campaign.digest(reg):
                raise ValueError('single-case registration differs from frozen pair')
            campaign.REG=single
            campaign.run()
        results = {j['sku']:campaign.read(OUT/(j['id']+'-result.json')) for j in original['jobs']}
        campaign.save(OUT/'score.json',{k:{f:r.get(f) for f in ('shape','qualified','ca_us','error_pct','reason','cost')}
                                        for k,r in results.items()})


if __name__=='__main__':
    args=ArgumentParser(description=__doc__)
    args.add_argument('action',choices=['register','run'])
    {'register':register,'run':run}[args.parse_args().action]()
