"""Check every historical JSON result manifest for this round's built-in Post shapes."""
from pathlib import Path
import json,re
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.24/post'
reg=json.loads((OUT/'freeze.json').read_text())
targets={tuple(r['shape']) for r in reg['rows']}
hits=defaultdict(list);seen=0
def visit(obj,path,where=(),op=None,implementation=None):
    if isinstance(obj,dict):
        name=obj.get('op',obj.get('operator',obj.get('op_name',op)))
        attrs=obj.get('attrs',{})
        impl=obj.get('implementation',attrs.get('implementation',implementation) if isinstance(attrs,dict) else implementation)
        for k,v in obj.items():
            if isinstance(v,list) and len(v)==3 and all(type(q) is int for q in v) and tuple(v) in targets and ('shape' in k or k=='dims'):
                hits[str(tuple(v))].append(dict(file=str(path.relative_to(ROOT)),key='.'.join((where+(k,))[-5:]),operator=name,implementation=impl))
            else:visit(v,path,where+(k,),name,impl)
    elif isinstance(obj,list):
        for v in obj:visit(v,path,where,op,implementation)
for p in (ROOT/'results').rglob('*.json'):
    if 'v0.24' in p.parts or p.name=='trace.json':continue
    v=json.loads(p.read_text());seen+=1;visit(v,p)
reserve_manifest=json.loads((ROOT/'results/v0.8/fusion/reserve.json').read_text())
reserve=next(r for r in reserve_manifest['cases'] if r['shape']==[1,4,192])
candidate_folder=ROOT/'runs/v8cv'/reserve['id']
published=(ROOT/'results/v0.8/fusion/rows.json').read_text() + (ROOT/'results/v0.8/fusion/costs.json').read_text()
reserve_check=dict(id=reserve['id'],op=reserve['op'],registered_only=(not candidate_folder.exists() and reserve['id'] not in published),raw_folder=str(candidate_folder.relative_to(ROOT)),rows_costs_reference_to_id=reserve['id'] in published)
(OUT/'novelty.json').write_text(json.dumps(dict(historical_manifests=seen,shape_fields_with_equal_dimensions=hits,older_chain_reserve=reserve_check,
  scope='Historical result manifests scanned recursively; event-only trace.json excluded. A matching dimension tuple is insufficient for operator identity. Compare shape-field owner and effective installed implementation.'),indent=2)+'\n')
print(seen,[(key,len(val)) for key,val in hits.items()])
