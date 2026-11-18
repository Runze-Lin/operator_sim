import runpy,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from operator_model.npusim_v7 import extract_db
m=runpy.run_path('scripts/vector-v8-freeze.py')
p=Path('results/v0.8/vector')
f=json.load(open(p/'profile.json'))
rows=json.load(open(p/'rows.json'))
for r in rows:
 op=r['op']
 if op=='kda_recurrent':continue
 if r['role']=='validation':
  ph=m['phases'](extract_db(Path(r['source']).resolve()))
 else:ph=next(x['phase_us'] for x in f[op]['source'] if x['id']==r['id'])
 r['phase_us']=ph
 print(r['id'], [round(ph.get(s,0),2) for s in ('arrival','MTE2','MTE3','SIMD','SCALAR','other')])
Path('results/v0.9/vector/development-phases.json').write_text(json.dumps([r for r in rows if r['op']!='kda_recurrent']))
