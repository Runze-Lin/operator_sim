"""Register previously unseen shape identities and the collection budget without opening new labels."""
import hashlib
import json
import os
from pathlib import Path
import re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/v0.18/sku'
RAW = ROOT/'runs/sku18'
DT, PR, FUTURE = 'Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'
OPS = ('rmsnorm', 'quant')

# Some input pairs intentionally straddle DT's 64 and PR's 56 advertised vector cores.
DESIGN = {
    'rmsnorm': [
        (1,5504,'small-wide'), (3,7552,'small-wide'), (8,6016,'small-wide'),
        (18,6912,'small-wide'), (55,1728,'core-boundary'), (57,1536,'core-boundary'),
        (65,2816,'core-boundary'), (113,2112,'ordinary'),
        (393,3840,'large'), (511,3968,'large'),
    ],
    'quant': [
        (1,704,'tiny'), (3,1920,'small-wide'), (7,4480,'small-wide'),
        (27,576,'ordinary'), (55,1152,'core-boundary'), (57,3200,'core-boundary'),
        (65,2624,'core-boundary'), (255,5440,'large'),
        (448,6592,'large'), (513,5504,'large'),
    ],
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buffer in iter(lambda: f.read(1024*1024), b''):
            h.update(buffer)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def records(obj):
    if isinstance(obj,list):
        for item in obj:yield from records(item)
    elif isinstance(obj,dict):
        op = obj.get('op', obj.get('operator'))
        shape = obj.get('shape')
        if op in OPS and isinstance(shape,list) and len(shape)==2 and all(type(v) is int for v in shape):
            yield op, tuple(shape)
        for item in obj.values():yield from records(item)


def history():
    """Union of released/archive identities, actual host logs and batch list inputs, including unsuccessful attempts."""
    known = {op:set() for op in OPS}
    sources = []
    for folder in [ROOT/'results', ROOT/'docs/extensions']:
        for path in sorted(folder.rglob('*.json')):
            if 'v0.18' in path.parts: continue
            before = sum(map(len,known.values()))
            for op,shape in records(read(path)):known[op].add(shape)
            if sum(map(len,known.values()))!=before:sources.append(str(path.relative_to(ROOT)))
    for tree, dirs, files in os.walk(ROOT/'runs'):
        dirs[:] = [d for d in dirs if d not in {'profile','record','data','dump','sku18'}]
        for fn in set(files)&{'run.log','inputs.txt','batch.txt','input.json','result.json'}:
            path=Path(tree)/fn
            if fn.endswith('.json'):
                for op,shape in records(read(path)):known[op].add(shape)
                continue
            txt=path.read_text(errors='replace')
            for m in re.finditer(r'case=dynamic_mx_quant rows=(\d+) cols=(\d+)',txt):known['quant'].add(tuple(map(int,m.groups())))
            for m in re.finditer(r'case=rmsnorm .*?R=(\d+) C=(\d+)',txt):known['rmsnorm'].add(tuple(map(int,m.groups())))
            for m in re.finditer(r'^rmsnorm\s+\d+\s+(\d+)\s+(\d+)\s+\d+\s*$',txt,re.M):known['rmsnorm'].add(tuple(map(int,m.groups())))
    return {op:[list(s)for s in sorted(allshapes)]for op,allshapes in known.items()},sources


def main():
    manifest=OUT/'registration.json'
    if manifest.exists():raise SystemExit('registration already exists')
    previous,sources=history()
    hosts={'quant':'build/smoke/three_ops','rmsnorm':'build/smoke/extension_norm_addmm'}
    shapes=[]
    for op in OPS:
        for rows,cols,regime in DESIGN[op]:
            shape=[rows,cols]
            if shape in previous[op]:raise ValueError((op,shape,'previously used'))
            shapes.append(dict(op=op,shape=shape,id=f'{op}_{rows}_{cols}',regime=regime,
                attrs={},dtype='bf16',runner=hosts[op],runner_sha256=digest(ROOT/hosts[op]),args=[op,str(rows),str(cols)],batch=None,nchecks=1 if op=='quant' else 2,kernels=1,
                wall_s=360 if rows*cols>=500000 else 200,raw_mib=3200 if rows*cols>=500000 else 950))
    write(OUT/'excluded-history.json',dict(shapes=previous,release_files_contributing_new_identities=sources,inventory_scope='All pre-v0.18 result JSON, extension documents, raw host logs, run input/results and batch lists; qualification irrelevant.'))
    write(manifest,dict(utc=utc(),chips=[DT,PR],future_projection=FUTURE,workers=4,total_wall_s=3600,total_raw_gib=24,max_physical_attempts=40,jobs=shapes,history_sha256=digest(OUT/'excluded-history.json'),
        policy='First isolated host call for every op/chip/shape. No cache/resident hint in prediction. Both chips use exact runtime SKU and full official msopprof. No replacement or error-dependent selection.'))
    print({k:len(v)for k,v in previous.items()})

if __name__=='__main__':main()
