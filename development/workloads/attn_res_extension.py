"""Register eight additional input contracts against the original frozen resident model."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from operator_model.chips import load_chip
from operator_model.attention_residual_timing import resource_attention

OUT=ROOT/'results/v0.19/workloads'
SMOKE=ROOT/'development/workloads/attention_residual_smoke.cpp'
# Run the largest input first. The first two examples checked in v0.19 are
# independent of the model but did inform selection of these additional cases.
CASES=[(1,7168,3),(2,4096,2),(5,2048,1),(2,1024,4),
       (8,512,2),(3,512,1),(4,256,3),(1,256,2)]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def runner(h):
    if h==7168:
        return ROOT/'build/smoke/attn_res_fwd19'
    binary=ROOT/f'build/smoke/attn_res_fwd19_h{h}'
    if h in (256,512):
        return binary
    raw=ROOT/'runs/workloads19/sources'
    source=raw/f'smoke_h{h}.cpp'; template=SMOKE.read_text()
    assert template.count('const int H=7168, S=8;')==1
    source.write_text(template.replace('const int H=7168, S=8;',f'const int H={h}, S=8;'))
    op=raw/'full/csrc'
    cmd='g++ -std=c++17 -O2 "$1" -I"$2" -I"$3/attention/attn_res_fwd/op_host/op_api" -I"$3/common/include/external" -I"$ASCEND_HOME_PATH/include" -L"$3/build" -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath-link,"$ASCEND_HOME_PATH/lib64" -lcust_opapi -lopapi -lnnopbase -lascendcl -o "$4"'
    subprocess.run(['bash',str(ROOT/'scripts/cann-run.sh'),'bash','-c',cmd,'compile',
                    str(source),str(SMOKE.parent),str(op),str(binary)],check=True)
    return binary


if __name__=='__main__':
    target=OUT/'attn-res-extension-freeze.json'
    if target.exists():raise FileExistsError(target)
    early=json.loads((OUT/'attn-res-holdout-freeze.json').read_text())
    for name in ['operator_model/attention_residual_timing.py','results/v0.19/workloads/attn-res-prices.json']:
        assert early['hashes'][name]==sha(ROOT/name)
    prior=set()
    for name in ['attn-res-holdout-freeze.json','attn-res-registration.json','attn-res-micro-registration.json']:
        for s in json.loads((OUT/name).read_text())['scenarios']:
            prior.add((s['tokens'],s['hidden'],s['valid']))
    assert len(set(CASES))==8 and not set(CASES)&prior
    paths=[ROOT/n for n in ['operator_model/attention_residual_timing.py','operator_model/attention_residual.py',
            'results/v0.19/workloads/attn-res-prices.json','results/v0.19/workloads/attn-res-holdout-freeze.json']]
    paths.append(SMOKE)
    binaries={h:runner(h) for h in sorted({s[1] for s in CASES})}
    paths.extend(binaries.values())
    rows=[]
    for t,h,n in CASES:
        for chip in ('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'):
            pred,_,d=resource_attention((t,h,8,n),load_chip(chip),dict(addend=True,write_idx=n,materialize=True))
            assert pred is not None and d['target_plan']['iterations']==1 and d['tiling_key']==40020
            rows.append(dict(chip=chip,tokens=t,hidden=h,slots=8,valid=n,runner=str(binaries[h].relative_to(ROOT)),latency_us=pred,
                             stages_us=d['stages_us'],tiling_key=d['tiling_key'],active_aiv=d['payload_aiv']))
    target.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),name='attnres_ext19',output='attn-res-extension-holdout.json',
        scope='Supplemental prospective holdout selected after inspecting the first two; all stage prices and schedule unchanged; distinct even from failed or registered cases. Limit two simultaneous CA processes.',
        wall_s_per_attempt=300,max_raw_mib_per_attempt=640,max_workers=2,
        hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},scenarios=rows),indent=2)+'\n')
