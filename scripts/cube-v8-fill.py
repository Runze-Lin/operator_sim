"""Run prespecified replacement shapes until both measured Cube families reach ten."""
import json,importlib.util
from pathlib import Path
root=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('isolated',root/'scripts/cube-v8-isolated.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
f=json.loads((c.out/'replacement-freeze.json').read_text());assert f['model_freeze_sha256']==c.sha256(c.out/'freeze.json')
for op in ('matmul','addmm'):
    refs=[json.loads(p.read_text()) for folder in ('v8cube','v8cube-isolated') for p in (root/'runs'/folder).glob(op+'*/result.json')];ok=len({r['id'] for r in refs if r['status']=='ok'})
    for case in f['cases']:
        if case['op']==op and ok<10 and not (c.runs/case['id']).exists():ok+=int(c.run(case)['status']=='ok')
