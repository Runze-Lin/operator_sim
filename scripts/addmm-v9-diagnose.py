"""Post-freeze decomposition of the worst qualified holdout by critical AIC."""
import json
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from operator_model.npusim_v7 import extract_db
from operator_model.addmm_v9 import predict
OUT=ROOT/'results/v0.9/addmm'
rows=json.loads((OUT/'rows.json').read_text())
hold=[r for r in rows if r['role']=='validation']
worst=max(hold,key=lambda r:abs(r['predicted_us']/r['reference_us']-1))
db=next((ROOT/worst['source']).parent.rglob('*.db'))
trace=extract_db(db)
core=max(trace['cores'],key=lambda x:x['end_ticks'])
union=end=0
with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as cx:
    for a,b in cx.execute("""SELECT ExecInstrTickStart,ExecInstrTickEnd FROM ExecutedInstructions
        WHERE CoreTypeId=? AND ExecInstrCoreId=? AND ExecInstrIsDurationExcluded=0
        AND json_extract(ExtendParams,'$.queue_type') IN ('CUBE','MTE1','MTE2','MTE3','FIXP')
        ORDER BY ExecInstrTickStart""",(core['core_type_id'],core['core_id'])):
        union+=max(0,b-max(end,a))
        end=max(end,b)
union/=1650
p=predict('addmm',worst['shape'],worst['attrs'])
observed=dict(service_union_us=union,control_us=core['span_us']-union,arrival_us=trace['reference_us']-core['span_us'],
              queues=core['pipeline_union_us'],active_aic=trace['active_aic'])
estimated=dict(service_union_us=p['predicted_us']-p['control_us']-p['arrival_us'],control_us=p['control_us'],
               arrival_us=p['arrival_us'],queues=p['stage_services_us'],active_aic=p['plan']['active_aic'])
out=dict(shape=worst['shape'],kernel=worst['kernel'],critical_core=core['core_id'],predicted_us=p['predicted_us'],
         reference_us=worst['reference_us'],db_sha256=trace['db_sha256'],observed=observed,estimated=estimated)
(OUT/'worst.json').write_text(json.dumps(out,indent=2)+'\n')
print(out)
