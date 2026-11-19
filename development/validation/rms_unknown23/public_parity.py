"""Time the public no-fit RMS history-band service separately from research decision rules."""
import json
from pathlib import Path
import sys
from time import perf_counter_ns
from hashlib import sha256
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model.rms_history import estimate_rms_history_band
from development.validation.sku19.campaign import write
from development.validation.rms_unknown23.candidate import forecast
from operator_model import load_chip
OUT = ROOT/'results/rms-unknown23'


def main():
    frozen = json.loads((OUT/'freeze.json').read_text())['rows']
    records = []
    n = 0
    first_time = None
    for row in frozen:
        start = perf_counter_ns()
        band = estimate_rms_history_band(row['shape'],chip=row['chip'])
        elapsed = (perf_counter_ns()-start)/1e6
        if first_time is None:
            first_time=elapsed
        assert len(band.by_resident_aiv_us)==len(row['conditional_us_by_H'])
        error = max(abs(a-b) for a,b in zip(band.by_resident_aiv_us,row['conditional_us_by_H']))
        assert error<1e-11 and abs(band.lower_us-row['lower_us'])<1e-11 and abs(band.upper_us-row['upper_us'])<1e-11
        n+=len(band.by_resident_aiv_us);records.append(dict(chip=row['chip'],shape=row['shape'],elapsed_ms=elapsed,max_abs_us=error))
    hist = json.loads((ROOT/'results/v0.22/rms-unknown/audit.json').read_text())['rows']
    shapes=sorted({(r['chip'],tuple(r['shape'])) for r in hist});max_hist=0;hn=0
    for name,shape in shapes:
        result=estimate_rms_history_band(shape,chip=name)
        old=forecast(shape,load_chip(name))
        assert len(result.by_resident_aiv_us)==len(old['conditional_us_by_H'])
        d=max(abs(a-b) for a,b in zip(result.by_resident_aiv_us,old['conditional_us_by_H']));assert d<1e-11
        max_hist=max(max_hist,d);hn+=len(result.by_resident_aiv_us)
    write(OUT/'public-parity.json',dict(source_sha256=sha256((ROOT/'operator_model/rms_history.py').read_bytes()).hexdigest(),
        frozen=sha256((OUT/'freeze.json').read_bytes()).hexdigest(),rows=records,total_frozen_H=n,history_shapes=len(shapes),history_H=hn,
        largest_frozen_difference_us=max(r['max_abs_us'] for r in records),largest_history_difference_us=max_hist,
        first_call_ms=first_time,max_ms=max(r['elapsed_ms'] for r in records),max_ms_per_chip={c:max(r['elapsed_ms'] for r in records if r['chip']==c) for c in sorted({r['chip'] for r in records})},
        timing='Direct public call using chip name, no shape-result cache, includes local setup and all predecessor states. Python import outside measurement. First call starts with empty chip and calibration parameter caches.'))
if __name__=='__main__':main()
