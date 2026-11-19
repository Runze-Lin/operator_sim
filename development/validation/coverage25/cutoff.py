"""Seal which finished KDA and Pro windows belong to this iteration's audit."""
from datetime import datetime,timezone
import json
from audit import ROOT,OUT,read,sources

def snapshot():
    p=OUT/'continuing-runs-cutoff.json'
    if p.exists():raise FileExistsError(p)
    selected={}
    for f in ['results/v0.24/kda-prefetch/comparison.json','results/v0.24/kda-prefetch/supplement/comparison.json']:
        for r in read(f)['rows']:
            path=r.get('file') or r['source'];rc=read(path)
            if rc['qualified']:selected[path]=dict(receipt=path,sha256=sources[path],role=r.get('role') or 'v0.24 prospective supplement')
    pro=[]
    for f in sorted((ROOT/'results/v0.23/pro-prefill').glob('pro-b8-*.json')):
        rc=read(f);ref=str(f.relative_to(ROOT))
        if rc.get('qualified') and rc.get('stage'):pro.append(dict(receipt=ref,sha256=sources[ref]))
    with p.open('x') as f:
        json.dump(dict(utc=datetime.now(timezone.utc).isoformat(),kda=list(selected.values()),pro=pro),f,indent=2);f.write('\n')

if __name__=='__main__':snapshot()
