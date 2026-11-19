"""Draw four FwdH blocks from the committed intervals, or re-extract from raw CA."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'development/validation/kda_prefetch24'))
os.environ.setdefault('MPLCONFIGDIR','/tmp/operator-sim-fig26')
from figure import materialize,draw
from evidence import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract',action='store_true',help='Read the original qualified per-core shards again.')
    args=parser.parse_args()
    out=ROOT/'results/v0.26/kda'
    original=ROOT/'results/v0.24/kda-prefetch/supplement'
    for sku,slug in [('Ascend950DT_9582','dt'),('Ascend950PR_957b','pr')]:
        path=out/(slug+'-four-chunks.json')
        if args.extract:
            src=original/f'supplement-{sku}-193-2.json'
            receipt=json.loads(src.read_text())
            assert receipt['qualified'] and receipt['registration_sha256']==sha(original/'register.json')
            d=materialize(src)
            d['evidence_role']='Prospectively frozen second cohort, T=193/H=2; complete official FLA 26.9.1 V2 kernel with all four numerical references accepted.'
            d['prediction_register_sha256']=receipt['registration_sha256']
            path.write_text(json.dumps(d,indent=2)+'\n')
        else:
            d=json.loads(path.read_text())
        draw(d,out/(slug+'-four-chunks.png'))


if __name__=='__main__':
    main()
