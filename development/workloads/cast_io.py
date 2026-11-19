"""Audit DMA instruction coverage against logical Cast bytes in qualified windows."""
from collections import Counter
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'development/validation/sku17'))
from campaign_collect import events


def inspect(rec):
    stage=rec['kernel']
    seen=Counter()
    cores=set()
    for x in events(ROOT/stage['trace']):
        if x['ph']!='X' or x['name']!='MOV_SRC_TO_DST_ALIGNv2':
            continue
        q=x['tid']
        if q not in ('MTE2','MTE3'):continue
        info=json.loads(x['args']['detail'])
        if info.get('is_duration_excluded'):continue
        seen[q]+=info['processed_bytes']
        if q=='MTE3':
            cores.add(x['pid'])
    n=rec['shape'][0]*rec['shape'][1]
    return dict(shape=rec['shape'],sku=rec['sku'],read_bytes_observed=seen['MTE2'],write_bytes_observed=seen['MTE3'],
                logical_read_bytes=2*n,logical_write_bytes=4*n,writers_observed=len(cores),
                active_aiv=stage['active_aiv'],trace_sha256=stage['trace_sha256'])


if __name__=='__main__':
    out=ROOT/'results/v0.19/workloads'
    rows=[inspect(r) for r in json.loads((out/'cast-qualified.json').read_text()) if r['qualified']]
    (out/'cast-dma-coverage.json').write_text(json.dumps(rows,indent=2)+'\n')
    for r in rows:
        if r['write_bytes_observed']!=r['logical_write_bytes']:print(r)
