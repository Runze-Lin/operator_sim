"""Extract direction-matched controller efficiency and independent logical-ingress constraints."""
from pathlib import Path
import json
from hashlib import sha256

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.30/matmul'

def extract():
    service=OUT/'external-service.json';source=ROOT/'results/v0.27/memory/read-windows.json'
    rows=json.loads(source.read_bytes());ports={}
    for r in rows:
        if r['op'] not in ('cast_bf16_fp32','sigmoid_fp32'):continue
        s=ports.setdefault(r['chip'],dict(qualified_observations=[],minimum_logical_gbs=0,reference_nominal_gbs=r['nominal_GBps']))
        value=r['peak_rate_interval']['rate_GBps']
        s['minimum_logical_gbs']=max(s['minimum_logical_gbs'],value)
        s['qualified_observations'].append(dict(geometry=r['shape'],kernel=r['op'],demand_bound_gbs=value,origin=r['source']))
    result=dict(saturated_write_fraction_deciles=json.loads(service.read_bytes())['consensus'],logical_ingress_by_calibrated_sku=ports,
                provenance={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in (service,source)})
    (OUT/'prices.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':extract()
