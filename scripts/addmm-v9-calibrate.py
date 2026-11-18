"""Measure visible control and the unhidden portion of non-bottleneck queues on the critical AIC."""
import json
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from operator_model.npusim_v7 import extract_db
from operator_model.cube_v8 import predict as previous

OUT = ROOT/'results/v0.9/addmm'
rows = []
resources = ('CUBE','MTE1','MTE2','MTE3','FIXP')
for record in json.loads((ROOT/'results/v0.8/cube/rows.json').read_text()):
    if record['op'] != 'addmm' or record['source'].startswith('docs/'):
        continue
    db = next((ROOT/record['source']).parent.rglob('*.db'))
    trace = extract_db(db)
    core = max(trace['cores'], key=lambda c: c['end_ticks'])
    with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as cx:
        spans = cx.execute("""SELECT ExecInstrTickStart,ExecInstrTickEnd FROM ExecutedInstructions
            WHERE CoreTypeId=? AND ExecInstrCoreId=? AND ExecInstrIsDurationExcluded=0
            AND json_extract(ExtendParams,'$.queue_type') IN ('CUBE','MTE1','MTE2','FIXP','MTE3')
            ORDER BY ExecInstrTickStart""",(core['core_type_id'],core['core_id']))
        end = union = 0
        for a,b in spans:
            union += max(0,b-max(a,end))
            end = max(b,end)
    union /= 1650
    queues = [core['pipeline_union_us'].get(q,0) for q in resources]
    bottleneck = max(queues)
    exposed = (union-bottleneck)/(sum(queues)-bottleneck)
    arrival = trace['reference_us']-core['span_us']
    rows.append(dict(shape=record['shape'], active=trace['active_aic'], sha256=trace['db_sha256'],
                     queues=dict(zip(resources,queues)), union=union, control=core['span_us']-union,
                     arrival=arrival, exposed=exposed))
def med(key):
    return statistics.median(r[key] for r in rows)
profile=dict(control_us=med('control'), exposed_fraction=med('exposed'), observations=rows)
(OUT/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
print([(r['shape'],round(r['control'],2),round(r['exposed'],2),round(r['arrival'],2)) for r in rows])
from math import log2
from operator_model.model import fit_nonnegative_line

for row in rows:
    plan = previous('addmm',row['shape'])['plan']
    row['units']=plan['units']
profile['queue_costs']={}
for queue,unit in [('CUBE','cube'),('MTE2','read'),('MTE1','read'),('FIXP','fix')]:
    profile['queue_costs'][queue]=fit_nonnegative_line([r['units'][unit] for r in rows],
                                                      [r['queues'][queue] for r in rows])
profile['arrival_us_per_log_core']=statistics.median(r['arrival']/log2(r['active']) for r in rows if r['active']>1)
(OUT/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
