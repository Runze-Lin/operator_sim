"""Audit prospective input tuples against historical result manifests."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import read,write
OUT=ROOT/'results/v0.25/post'
keys={tuple(r['shape']) for r in read(OUT/'freeze.json')['rows']}; hits=[]; files=0
def visit(o,ctx,path):
    if isinstance(o,list):
        for el in o:visit(el,ctx,path)
    elif isinstance(o,dict):
        ctx=o.get('op',o.get('operator',ctx))
        for k,v in o.items():
            if ('shape' in k or k=='dims') and isinstance(v,list) and len(v)==3 and all(type(i)==int for i in v) and tuple(v) in keys:
                hits.append(dict(file=str(path.relative_to(ROOT)),shape=v,key=k,operator=ctx))
            elif isinstance(v,(dict,list)):visit(v,ctx,path)
for p in (ROOT/'results').rglob('*.json'):
    if 'v0.25' in p.parts or p.name=='trace.json':continue
    visit(read(p),None,p);files+=1
write(OUT/'novelty.json',dict(manifests=files,matches=hits,shapes=sorted(keys)))
print(files,len(hits))
