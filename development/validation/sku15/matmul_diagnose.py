"""Replay old paired MMAD descriptors on both CA backends. No v0.15 CA input."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model.chips import load_chip
from operator_model.matmul_v15 import matmul_plan
from operator_model.target_tiling import matmul_plan as old_plan


def hist(record):
    path=ROOT/record['trace']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['trace_sha256']
    if path.suffix=='.db':
        conn=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)
        try:
            rows=[e[0] for e in conn.execute("select ExtendParams from ExecutedInstructions where ExecInstrName='MMAD'")]
        finally:conn.close()
    else:
        rows=[e['args']['detail'] for e in json.loads(path.read_bytes())['traceEvents'] if e['name']=='MMAD' and e['ph']=='X']
    return Counter(tuple(j.get(k) for k in ('m','n','k')) for j in map(json.loads,rows))


def as_hist(shape,tile):
    def partition(n,t):
        full,tail=divmod(n,t)
        return [(k,c) for k,c in ((t,full),(tail,1)) if k and c]
    from itertools import product
    return {(m,n,k):a*b*c for (m,a),(n,b),(k,c) in product(*(partition(n,t) for n,t in zip(shape,tile)))}


def main():
    out=[]
    for f in ['results/v0.13/sku/pairs.json','results/v0.13/sku/holdout.json','results/v0.14/sku/holdout.json']:
        for r in json.loads((ROOT/f).read_text()):
            if r['op']!='matmul':continue
            for sku,name in [('dt','Ascend950DT_9582'),('pr','Ascend950PR_957b')]:
                rec=r[sku];shape=r['shape'];chip=load_chip(name);p=matmul_plan(shape,chip)
                old=old_plan(shape,chip.cube_cores);observed=hist(rec)
                out.append(dict(shape=shape,sku=sku,archive=f,trace_sha256=rec['trace_sha256'],
                                observed_aic=rec['active_aic'],predicted_aic=p['active_aic'],old_aic=old['active_aic'],
                                observed_mmad_count=sum(observed.values()),new_tile=p['tile'],old_tile=old['tile'],
                                read_page_k=p['read_page_k'],old_histogram_match=observed==as_hist(shape,old['tile']),
                                new_histogram_match=observed==as_hist(shape,p['tile']),
                                observed_mmad_mnk={','.join(map(str,k)):v for k,v in sorted(observed.items())}))
    (ROOT/'results/v0.15/matmul-development.json').write_text(json.dumps(out,indent=2)+'\n')
    print(len(out),'pairs of SKU and shape',sum(r['new_histogram_match'] for r in out),'exact histograms')
if __name__=='__main__': main()
