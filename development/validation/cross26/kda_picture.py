"""Render the four-block FwdH from the frozen model and the actual final AIC/AIV group."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'development/validation/kda_prefetch24'))
os.environ.setdefault('MPLCONFIGDIR','/tmp/operator-sim-fig26')
from figure import materialize,draw
from evidence import sha

out=ROOT/'results/v0.26/kda'
original=ROOT/'results/v0.24/kda-prefetch/supplement'
for sku,slug in [('Ascend950DT_9582','dt'),('Ascend950PR_957b','pr')]:
    src=original/f'supplement-{sku}-193-2.json'
    receipt=json.loads(src.read_text())
    assert receipt['qualified'] and receipt['registration_sha256']==sha(original/'register.json')
    d=materialize(src)
    d['evidence_role']='Prospectively frozen second cohort, T=193/H=2; complete official FLA 26.9.1 V2 kernel with all four numerical references accepted.'
    d['prediction_register_sha256']=receipt['registration_sha256']
    (out/(slug+'-four-chunks.json')).write_text(json.dumps(d,indent=2)+'\n')
    draw(d,out/(slug+'-four-chunks.png'))
