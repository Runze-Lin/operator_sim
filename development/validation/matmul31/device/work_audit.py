"""Recover useful and padded matrix work independently from all retired MMAD descriptors."""
from collections import Counter
from math import ceil
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from development.validation.matmul31.device.campaign import ca, OUT, REG, CHIPS


def expected(shape,model):
    m,n,k=shape; bm,bn,bk=(model[key] for key in ('base_m','base_n','base_k'));s=model['k_splits'];depth=model['max_segment_k']
    kinds=Counter()
    for mi in range(0,m,bm):
        for ni in range(0,n,bn):
            for i in range(s):
                segment=(model['tail_segment_k'] if i==s-1 else depth)
                for ki in range(0,segment,bk):kinds[min(m-mi,bm),min(n-ni,bn),min(segment-ki,bk)]+=1
    return kinds


def describe(hist):
    rows=sorted(hist.items());logical=sum(2*m*n*k*count for (m,n,k),count in rows)
    padded=sum(2*16*ceil(m/16)*16*ceil(n/16)*16*ceil(k/16)*count for (m,n,k),count in rows)
    return dict(mmads=sum(hist.values()),useful_ops=logical,hypothetical_mma16_rounded_ops=padded,
                mma16_lane_fraction=logical/padded,histogram=[dict(m=m,n=n,k=k,count=count) for (m,n,k),count in rows])


def build():
    reg=ca.read(REG);rows=[];received=0
    for job in reg['rows']:
        m,n,k=job['shape']
        for sku,tag in CHIPS:
            slug=f'{m}-{n}-{k}-{tag}';path=OUT/(slug+'.json');p=job['predictions'][sku]['geometry']
            theoretical=expected(job['shape'],p); predicted=describe(theoretical); assert predicted['mmads']==p['mmads_total'] and predicted['useful_ops']==2*m*n*k
            row=dict(shape=job['shape'],sku=sku,expected=predicted,actual=None,matches=None)
            if path.exists():
                received+=1;d=ca.read(path);g=d.get('actual_geometry')
                if g:
                    observed=Counter({(v['m'],v['n'],v['k']):v['count'] for v in g['mmads']})
                    row.update(actual=describe(observed),matches=theoretical==observed,trace_sha256=d['stages'][0]['trace_sha256'])
            rows.append(row)
    ca.save(OUT/'work.json',dict(freeze_sha256=ca.digest(REG),rows=rows,
        scope='Useful ops from actual MMAD m,n,k, FMA=2. Additional ops from padding each dimension to 16 are theoretical tile occupancy, not measured hardware allocation or issue speed. Comparison checks every descriptor kind and total, not just their count.'))
    return received==len(rows)

if __name__=='__main__':
    if '--watch' in sys.argv:
        while not build():time.sleep(30)
    else:build()
