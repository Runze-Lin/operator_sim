"""Freeze a cross-L2 pair, and distinguish prior full instructions from native envelopes."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import load_chip, estimate
from operator_model.quant_stream_memory import scenario
from development.validation.rms_ordered_verify.verify import digest

OUT=ROOT/'results/v0.29/quant'
OLD=ROOT/'results/v0.27/quant-transfer'
FROZEN=OUT/'freeze.json'
DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'
PAIR=(16384,7168)
RUNTIME=ROOT/'build/smoke/large_vector'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_bytes())


def store(path,obj,*,exclusive=False):
    content=json.dumps(obj,indent=2,allow_nan=False)+'\n'
    if exclusive:
        with path.open('x') as f:f.write(content)
    else:
        tmp=path.with_name(path.name+'.tmp')
        tmp.write_text(content);tmp.replace(path)


def forecasts(shape,sku):
    c=load_chip(sku)
    base=estimate('quant',shape,chip=c).latency_us
    eta=read(OLD/'freeze.json')['efficiency']['mid']
    actual_pin_ratio=4096/4000 if sku==DT else 1.
    # There is no verified PR or 970 controller calibration at registration.
    # Price their *configured* data link directly. Transferring eta is conditional.
    q=scenario(shape,c,base,mixed_bus_efficiency=eta,wire_to_configured_bandwidth=actual_pin_ratio)
    ideal=scenario(shape,c,base,mixed_bus_efficiency=1.,wire_to_configured_bandwidth=actual_pin_ratio)
    q['ideal_mixed_service_us']=ideal['latency_us']
    q['bandwidth_gbs']=c.hbm_bandwidth_gbs
    return q


def register():
    if FROZEN.exists() or (ROOT/'runs/quant29').exists():raise FileExistsError('existing freeze or acquisition')
    old=read(ROOT/'results/v0.28/replay.json')
    assert all(tuple(r['shape'])!=PAIR for r in old['rows'] if r['op']=='quant')
    import os,re
    seen=set()
    for base in (ROOT/'runs',ROOT/'results'):
        for folder,dirs,fs in os.walk(base):
            dirs[:]=[d for d in dirs if d not in ('dump','profile','simulator','log','log_ca','record','ca_workspace') and not d.startswith(('OPPROF_','PROF_'))]
            if 'run.log' not in fs:continue
            p=Path(folder)/'run.log'
            if p.stat().st_size>32*1024**2:continue
            if OLD/'raw' in p.parents:continue  # B4 target still running; no observations from its job used.
            txt=p.read_text(errors='replace')
            for a,b in re.findall(r'case=dynamic_mx_quant rows=(\d+) cols=(\d+)',txt):seen.add((int(a),int(b)))
    assert PAIR not in seen
    native=OLD/'native.json'
    if native.exists():raise ValueError('B4 observation already available')
    shapes={'pair':[ *PAIR ],'existing_b4':[131072,7168],'historical_b8':[262144,7168]}
    pred={name:{sku:forecasts(s,sku) for sku in (DT,PR,FUT)} for name,s in shapes.items()}
    arch=ROOT/'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_nn/dynamic_mx_quant/DynamicMxQuant_bf16toe4m3_high_performance.o'
    src=[Path(__file__),Path(__file__).with_name('direction.py'),Path(__file__).with_name('collect.py'),RUNTIME,arch,
         ROOT/'smoke/large_vector.cpp',ROOT/'operator_model/quant_resource.py', ROOT/'operator_model/quant_stream_memory.py',
         ROOT/'operator_model/quant_fragments.py',ROOT/'operator_model/target_tiling.py',OLD/'freeze.json',OUT/'direction-prices.json',
         ROOT/'development/validation/sku19/collect.py',ROOT/'development/validation/rms_ordered_verify/verify.py',
         ROOT/'scripts/cann-run.sh', *[ROOT/f'configs/chips/{s}.json' for s in (DT,PR,FUT)]]
    reg=dict(utc=now(),role='One new full-instruction paired geometry. B4 new-candidate predictions sealed during previously registered independent native collection, before a final receipt or derived B4 DMC data; B8 is development replay.',
      shapes=shapes,predictions=pred,history=dict(checked_host_shapes=sorted(seen),largest_replay_logical_bytes=max(r['shape'][0]*(3*r['shape'][1]+2*((r['shape'][1]+63)//64)) for r in old['rows'] if r['op']=='quant')),
      cross_check='Candidate uses one isotropic mixed service efficiency, frozen previously from Cast. Leave-one-Cast-out two-direction prices did not consistently improve, so are diagnostic only. No quant whole duration fits.',
      official=dict(skus=[DT,PR],order=[PR,DT],shape=list(PAIR),scenario='B8, 2048 newly scheduled tokens/request in a prefill chunk; Kimi hidden_size=7168. Equivalent per-dispatch token count may arise from other scheduler contracts.',output_counts=[PAIR[0]*PAIR[1],PAIR[0]*PAIR[1]//32],kernel='DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv',vendor=str(arch.relative_to(ROOT)),utc_role='both endpoint simulations must start after freeze',
        budget=dict(wall_s=3*3600,raw_mib=64*1024),budget_basis='[2048,5120] official same family 7.4/8.3 min, 4.29/3.86 GiB peak; new input 11.2x; one serial pair, no automatic retry, >140 GiB free before launch.',
        qualification='same runtime SKU & controller mode, installed ELF text, 1 launch, exact shape, all numerics, expected AIVs with one END each, whole physical tag windows agree with each duration table row, exact logical output bytes. Controllers are independent process-scope optional evidence; no incomplete timing.',
        checks='On the new pair compare old and candidate signed errors per SKU and DT/PR speedup, 25%/50% target; additionally test whether 950DT measured DDR writes are nearer corrected retained suffix than old clean-first. Persist 950PR reports if available; if no independently verified PR link rate use no aggregate utilization claim. A single pair cannot certify whole large family.',
        priority='Wait for v0.27 B4 native to reach a terminal receipt. Reuse that queue lock, one simulator at a time, nice and CPU headroom gate, no duplicate interrupted/failed endpoint. Assess B4 separately even if official pair delayed.'),
      hashes={str(p.relative_to(ROOT)):digest(p) for p in src})
    store(FROZEN,reg,exclusive=True)
    print(reg['utc'],flush=True)

if __name__=='__main__':register()
