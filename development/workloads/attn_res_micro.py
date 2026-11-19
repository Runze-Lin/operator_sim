"""Build hidden-width variants of the same CPU oracle and register micro CA."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from operator_model.attention_residual import attention_residual_work

out=ROOT/'results/v0.19/workloads'
raw=ROOT/'runs/workloads19/sources'
op=raw/'full/csrc'
base=ROOT/'development/workloads/attention_residual_smoke.cpp'
manifest=out/'attn-res-micro-registration.json'
if manifest.exists():
    raise FileExistsError(manifest)

hashes={str(base.relative_to(ROOT)):hashlib.sha256(base.read_bytes()).hexdigest()}
jobs=[]
for h,n,sku in [(256,1,'Ascend950DT_9582'),(512,2,'Ascend950PR_957b'),(512,2,'Ascend950DT_9582')]:
    binary=ROOT/f'build/smoke/attn_res_fwd19_h{h}'
    if not any(j['hidden']==h for j in jobs):
        src=raw/f'smoke_h{h}.cpp'
        template=base.read_text()
        assert template.count('const int H=7168, S=8;')==1
        src.write_text(template.replace('const int H=7168, S=8;',f'const int H={h}, S=8;'))
        cmd='g++ -std=c++17 -O2 "$1" -I"$2" -I"$3/attention/attn_res_fwd/op_host/op_api" -I"$3/common/include/external" -I"$ASCEND_HOME_PATH/include" -L"$3/build" -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath-link,"$ASCEND_HOME_PATH/lib64" -lcust_opapi -lopapi -lnnopbase -lascendcl -o "$4"'
        subprocess.run(['bash',str(ROOT/'scripts/cann-run.sh'),'bash','-c',cmd,'compile',str(src),str(base.parent),str(op),str(binary)],check=True)
        hashes[str(binary.relative_to(ROOT))]=hashlib.sha256(binary.read_bytes()).hexdigest()
    pred=attention_residual_work(1,h,8,n,chip=sku,addend=True,write_idx=n,materialize=True)
    jobs.append(dict(chip=sku,tokens=1,hidden=h,slots=8,valid=n,source_estimate=pred,runner=str(binary.relative_to(ROOT))))
manifest.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),name='attnres_micro19',output='attn-res-micro-ca.json',role='diagnostic reduced hidden size; Kimi K3 has H=7168',
    wall_s_per_attempt=150, max_raw_mib_per_attempt=256,max_workers=2,hashes=hashes,scenarios=jobs),indent=2)+'\n')
