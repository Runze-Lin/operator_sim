"""Freeze the first two resident-branch held-out comparisons."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from operator_model.chips import load_chip
from operator_model.attention_residual_timing import resource_attention

out=ROOT/'results/v0.19/workloads/attn-res-holdout-freeze.json'
if out.exists():
    raise FileExistsError(out)
paths=['operator_model/attention_residual_timing.py','operator_model/attention_residual.py',
       'results/v0.19/workloads/attn-res-prices.json','build/smoke/attn_res_fwd19','build/smoke/attn_res_fwd19_h512']
rows=[]
for t,h in [(1,7168),(2,512)]:
    runner=paths[3] if h==7168 else paths[4]
    for chip in ('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'):
        duration,_,d=resource_attention((t,h,8,1),load_chip(chip),dict(addend=True,write_idx=1,materialize=True))
        rows.append(dict(chip=chip,tokens=t,hidden=h,slots=8,valid=1,runner=runner,latency_us=duration,
                         stages_us=d['stages_us'],tiling_key=d['target_plan']['tiling_key'],active_aiv=d['payload_aiv']))
out.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),name='attnres_hold19',output='attn-res-holdout.json',
     scope='Fresh 950 pairs, previously unseen token/hidden/valid combination. Both complete CPU oracle and identical kernel required. 970 extrapolated.',
     wall_s_per_attempt=240,max_raw_mib_per_attempt=420,max_workers=2,
     hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scenarios=rows),indent=2)+'\n')
