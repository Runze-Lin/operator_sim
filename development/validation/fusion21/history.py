"""Check frozen shapes against historical measured, registered and planned JSON contracts."""
from pathlib import Path
import json
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.21/fusion'
OPS=('attn_res_fwd','mhc_sinkhorn','mhc_pre_sinkhorn')


def walk(node, op, path, hits):
    if isinstance(node, list):
        for item in node:
            if isinstance(item,list) and op in OPS and item and all(type(v) is int for v in item):
                hits[(op,tuple(item))].add(str(path))
            else:walk(item,op,path,hits)
        return
    if not isinstance(node,dict):return
    name=node.get('op',node.get('operator',op))
    if not isinstance(name,str):name=op
    if name in OPS:
        s=node.get('shape')
        if s and isinstance(s,list) and all(type(v) is int for v in s):hits[(name,tuple(s))].add(str(path))
        if name=='attn_res_fwd' and all(k in node for k in ('tokens','hidden','valid')):
            hits[(name,(node['tokens'],node['hidden'],node.get('slots',8),node['valid']))].add(str(path))
    for k,item in node.items():walk(item,k if k in OPS else name,path,hits)


def audit():
    hits=defaultdict(set)
    paths=sorted(p for p in (ROOT/'results').rglob('*.json') if p.relative_to(ROOT/'results').parts[0]!='v0.21')
    for p in paths:
        name='attn_res_fwd' if 'attn-res-' in p.name else None
        walk(json.loads(p.read_text()),name,p.relative_to(ROOT),hits)
    # Some preregistrations and old files use positional shape fields; also test
    # kernel-specific host logs for invocations outside the versioned registries.
    raw=[p for p in (ROOT/'runs').iterdir() if p.is_dir() and (p.name.startswith(('v8','v9','v10','v11','v12','sku','workload','model-v','extensions')))]
    log_count=0
    import os,re
    for base in raw:
        for d,dirs,files in os.walk(base):
            dirs[:]=[v for v in dirs if v not in {'sources','profile','log','log_ca','dump','simulator','npusim','device0','record','ca_workspace'} and not v.startswith(('OPPROF_','npusim_','PROF_'))]
            if 'run.log' not in files:continue
            p=Path(d)/'run.log'; text=p.read_text(errors='replace')
            log_count+=1
            for m in re.finditer(r'correctness shape=(\d+)x(\d+) S=(\d+) N=(\d+) groups=output,prefix', text):
                hits[('attn_res_fwd',tuple(map(int,m.groups())))].add(str(p.relative_to(ROOT)))
            for m in re.finditer(r'api=aclnnMhcPreSinkhorn T=(\d+) N=(\d+) D=(\d+)',text):
                hits[('mhc_pre_sinkhorn',tuple(map(int,m.groups())))].add(str(p.relative_to(ROOT)))
            for m in re.finditer(r'api=aclnnMhcSinkhorn T=(\d+) N=(\d+)',text):
                hits[('mhc_sinkhorn',tuple(map(int,m.groups())))].add(str(p.relative_to(ROOT)))
    rows=[];seen=set()
    for c in json.loads((OUT/'freeze.json').read_text())['cases']:
        key=(c['op'],tuple(c['shape']))
        if key in seen:continue
        seen.add(key)
        rows.append(dict(op=c['op'],shape=c['shape'],attrs=c['attrs'],prior_shape_matches=sorted(hits.get(key,[]))))
    result=dict(scope='Shape-only match is stricter than shape + attrs. Includes model plans as potential prior exposure; only versioned public evidence and matching archived host logs have been inspected.',json_files=len(paths),historic_host_logs=log_count,contracts=rows)
    (OUT/'history-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('files',len(paths),'host logs',log_count)
    for r in rows:
        if r['prior_shape_matches']:print(r['op'],r['shape'],r['prior_shape_matches'][:8])
if __name__=='__main__':audit()
