"""Verify the deployed wheel from outside the checkout, using a released replay."""
import json
import math
import sys
from pathlib import Path
from time import perf_counter_ns


def main(path):
    frozen = json.loads(path.read_text())
    start = perf_counter_ns()
    import operator_model
    from operator_model import estimate, load_chip
    import_ms = (perf_counter_ns() - start) / 1e6
    if 'site-packages' not in Path(operator_model.__file__).parts or operator_model.__version__ != '0.13.0':
        raise RuntimeError('Expected an isolated v0.13.0 wheel')
    results = {}
    for chip_name, field in [('Ascend950DT_9582','dt_prediction_us'),
                             ('Ascend950PR_957b','pr_projection_us')]:
        start = perf_counter_ns()
        chip = load_chip(chip_name)
        load_ms = (perf_counter_ns() - start) / 1e6
        times = []
        for r in frozen['rows']:
            start = perf_counter_ns()
            ans = estimate(r['op'], r['shape'], r['attrs'], chip=chip)
            times.append((perf_counter_ns()-start)/1e6)
            if ans.latency_us is None or not math.isclose(r[field], ans.latency_us, rel_tol=1e-9):
                raise ValueError((chip_name,r['op'],r['shape'],ans))
            if r['logical_read_bytes'] != ans.work['logical_read_bytes']:
                raise ValueError('Logical accounting mismatch')
        results[chip_name] = dict(load_ms=load_ms,calls=len(times),first_ms=times[0],max_ms=max(times))
        if max(times) >= 100:
            raise ValueError('Single-call wall time exceeded 100 ms')
    print(json.dumps(dict(version=operator_model.__version__,import_ms=import_ms,checks=results),indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
