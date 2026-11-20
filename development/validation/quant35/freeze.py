"""Register the remaining dual-SKU case before either original capture can start."""
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.35/quant'
if len(sys.argv)>1:
    sys.path.insert(0,str(Path(sys.argv[1]).resolve()))
else:
    sys.path.insert(0,str(ROOT))
from operator_model import estimate, load_chip
from operator_model.quant_stream_memory import scenario

OLD=ROOT/'results/v0.29/quant/freeze.json'
DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'


def forecasts(shape, eff):
    result={}
    for sku in (DT,PR,FUT):
        chip=load_chip(sku)
        base=estimate('quant',shape,chip=chip)
        price=scenario(shape,chip,base.latency_us,mixed_bus_efficiency=eff)
        result[sku]={'previous_us':base.latency_us,**price}
    return result


def main():
    original=json.loads(OLD.read_bytes())
    shape=original['official']['shape']
    for sku in (DT,PR):
        assert not (ROOT/'runs/quant29'/sku).exists()
        assert not (OLD.parent/(sku+'.json')).exists()
    pricefile=OUT/'service.json'
    eff=json.loads(pricefile.read_bytes())['quant_mixed_bandwidth_efficiency']
    result=dict(utc=datetime.now(timezone.utc).isoformat(),source_freeze_sha256=sha256(OLD.read_bytes()).hexdigest(),
                service_sha256=sha256(pricefile.read_bytes()).hexdigest(),shape=shape,
                predictions=forecasts(shape,eff),status='prospective; uses the previously registered capture, no new geometry',
                implementation_sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
                                        (Path(__file__),ROOT/'operator_model/quant_stream_memory.py',ROOT/'operator_model/quant_resource.py')},
                previous_native_dataset={'completed_b8':forecasts([262144,7168],eff),'partial_b4':forecasts([131072,7168],eff)})
    with (OUT/'freeze.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')

if __name__=='__main__':main()
