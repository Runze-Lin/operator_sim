"""Freeze the conservative structural cost floors before attempting exact CA."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from operator_model.attention_residual import attention_residual_work

if __name__=='__main__':
    out=ROOT/'results/v0.19/workloads/attn-res-registration.json'
    if out.exists():
        raise FileExistsError(out)
    rows=[]
    for tokens,valid in [(1,2),(9,6)]:
        for chip in ('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'):
            pred=attention_residual_work(tokens,7168,8,valid,chip=chip,addend=True,write_idx=valid,materialize=True)
            rows.append(dict(chip=chip,tokens=tokens,hidden=7168,slots=8,valid=valid,source_estimate=pred))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [ROOT/'operator_model/attention_residual.py', ROOT/'development/workloads/attention_residual_smoke.cpp',ROOT/'build/smoke/attn_res_fwd19']}
    out.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),role='source cost floors and exact-entry feasibility, no total device-time model',
        wall_s_per_attempt=180,max_raw_mib_per_attempt=512,max_workers=2, hashes=hashes,scenarios=rows),indent=2)+'\n')
