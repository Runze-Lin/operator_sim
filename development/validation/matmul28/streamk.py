"""Check the split-K hypothesis against frozen ordinary-tile predictions."""
from hashlib import sha256
from math import ceil, log2
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import load_chip
from operator_model.matmul_resource import _costs,resource_matmul

OUT=ROOT/'results/v0.28/matmul'
DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'


def scenario(shape,chip):
    m,n,k=shape
    ordinary,aic,aiv,d=resource_matmul(shape,chip)
    plan=d['target_plan']; tiles=plan['output_tiles']; cores=chip.cube_cores
    threshold=max(8192,256*cores)/2
    # Eligible model for a single unsplit output rectangle and one K partition per AIC.
    if tiles!=1 or ceil(k/256)*256<threshold:
        return dict(us=ordinary,aic=aic,aiv=aiv,route='ordinary',threshold_k=threshold)
    splits=min(cores,ceil(k/64))
    local=64*ceil(ceil(k/splits)/64)
    # Use the same MTE2/L0/Cube prices on a shorter K segment. Reprice the shared
    # reads across all AICs. The partial output is FP32, the final output BF16.
    _,_,_,part=resource_matmul([m,n,local],chip)
    f=chip.frequency_ghz; p=_costs()
    stages=part['stages_us']
    launch=(p['launch_cycles']+p['arrival_log2_cycles']*log2(splits))/(f*1000)
    body=sum(stages[x] for x in ('read_fill','l0_lead','page_intervals','last_mmad','write_tail'))
    input_floor=2*k*(m+n)/(chip.hbm_bandwidth_gbs*1000)
    aiv=min(chip.vector_cores,ceil(splits*chip.vector_cores/cores))
    fixup_bytes=2*m*n*(4*splits-1)
    # The DAV_3510 host model's vector dispatch and per-K-split issue cycles.
    # Scale the per-split work at 256 FP32 ops/AIV/cycle (DT dual issue).
    vec_per_cycle=chip.vector_tops['32']*1000/(chip.vector_cores*f)
    issue=(1000+60*splits*256/vec_per_cycle)/(f*1000)
    l2=chip.details.get('l2_bandwidth_gbs') or chip.hbm_bandwidth_gbs
    merge=issue+fixup_bytes/(1000*l2)
    us=launch+max(body,input_floor)+merge
    if us>=ordinary:
        return dict(us=ordinary,aic=aic,aiv=aiv,route='ordinary',threshold_k=threshold)
    return dict(us=us,aic=splits,aiv=aiv,route='one_output_split_k',threshold_k=threshold,
                k_splits=splits,max_aligned_depth=local,partial_fixup_bytes=fixup_bytes,
                launch_us=launch,partial_pipeline_us=body,cold_floor_us=input_floor,merge_us=merge)


def freeze():
    path=OUT/'streamk-freeze.json'
    # Registration is immutable and excludes all already-running endpoints from validation.
    if path.exists():raise FileExistsError(path)
    rows=[]
    for r in json.loads((OUT/'freeze.json').read_bytes())['rows']:
        vals={}
        for chip,tag in ((DT,'dt'),(PR,'pr'),(FUT,None)):
            slug='-'.join(map(str,r['shape']))
            observed=bool(tag and (ROOT/'runs/matmul28'/(slug+'-'+tag)).exists())
            vals[chip]=dict(**scenario(r['shape'],load_chip(chip)),already_started=observed)
        rows.append(dict(shape=r['shape'],predictions=vals))
    pkgs=[Path(__file__),ROOT/'operator_model/matmul_resource.py',ROOT/'results/v0.16/matmul.json',
          ROOT/'third_party/ops-nn/matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_streamk_tiling.cpp']
    with path.open('x') as out:
        json.dump(dict(utc=datetime.now(timezone.utc).isoformat(),rows=rows,
            role='Diagnostic candidate. Single-output rectangle, long K; unchanged prices for ordinary tiles. Only endpoints not started at registration are prospective. Route confirmation requires installed key; 970 extrapolated.',
            source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in pkgs}),out,indent=2)
        out.write('\n')


if __name__=='__main__':freeze()
