"""Seal the physical schedule and all remaining endpoint forecasts before opening CA receipts."""
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model.matmul_supply30 import estimate_matmul_supply

OUT=ROOT/'results/v0.30/matmul';OLD=ROOT/'results/v0.29/matmul/full-load'
def read(p):return json.loads(p.read_bytes())
def digest(p):return sha256(p.read_bytes()).hexdigest()


def freeze():
    rows=[]
    for orig in read(OLD/'freeze.json')['rows']:
        chip=orig['chip'];s=orig['shape'];name=orig['acquisition']
        if not name:role='extrapolation only'
        elif s[0]<1048576 or (s[0]==1048576 and 'DT' in chip):role='development'
        else:
            p=ROOT/name;raw=ROOT/'runs/matmul28/long-prefill'/p.stem
            role='receipt pre-existed unread' if p.exists() else 'running unseen' if raw.exists() else 'strictly before acquisition'
        start=time.perf_counter_ns();cand=estimate_matmul_supply(s,chip=chip);query_ms=(time.perf_counter_ns()-start)/1e6
        variants={}
        if cand['latency_us'] is not None:
            for idx,eta in zip(('low_eta','high_eta'),cand['pricing']['external_efficiency_range']):
                variants[idx]=estimate_matmul_supply(s,chip=chip,mixed_efficiency=eta)['latency_us']
            variants['input_cold']=estimate_matmul_supply(s,chip=chip,input_mode='cold')['latency_us']
            variants['unity_mixed_efficiency']=estimate_matmul_supply(s,chip=chip,mixed_efficiency=1.)['latency_us']
        rows.append(dict(shape=s,chip=chip,role=role,acquisition=name,older_public_us=orig['legacy_us'],
                         v029_us=orig['candidate']['latency_us'],candidate=cand,ablation_us=variants,query_ms=query_ms))
    files=[Path(__file__),OUT/'external-service.json',OUT/'prices.json',OLD/'freeze.json',OLD/'131072-dt-intervals.json',OLD/'pr32-geometry.json',
           ROOT/'results/v0.27/memory/read-windows.json',ROOT/'results/v0.27/quant-channel-service.json',
           ROOT/'results/v0.16/matmul.json',ROOT/'operator_model/chips.py',ROOT/'operator_model/matmul_v15.py',
           ROOT/'operator_model/matmul_resource.py',ROOT/'operator_model/matmul_full_load.py',ROOT/'operator_model/matmul_supply30.py',
           Path(__file__).with_name('controller.py'),Path(__file__).with_name('inputs.py')]
    files+=list((ROOT/'configs/chips').glob('Ascend*.json'))
    with (OUT/'freeze.json').open('x') as f:
        json.dump(dict(utc=datetime.now(timezone.utc).isoformat(),rows=rows,hashes={str(p.relative_to(ROOT)):digest(p) for p in files},
             registered='Three new pairs of shape/SKU endpoints: existing PR 128Ki in flight and unseen, DT/PR 256Ki each awaiting acquisition. Only eligible original complete windows with numerical gates and END/drain. Model, price source, primary, sensitivity, v0.29 and original public forecast all frozen before scoring. No new CA. 970 target-only.'),f,indent=2);f.write('\n')
if __name__=='__main__':freeze()
