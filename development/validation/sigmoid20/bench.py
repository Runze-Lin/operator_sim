"""Time the first public API call in a fresh process, separate from module import."""

import json
from pathlib import Path
from subprocess import check_output
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from development.validation.sigmoid20.campaign import DEST, DT, PR, FUTURE, ROOT, write

SCRIPT = """from time import perf_counter_ns
import json, sys
start = perf_counter_ns()
from operator_model import estimate
ready = perf_counter_ns()
r = estimate('sigmoid_fp32', (int(sys.argv[1]), 12), chip=sys.argv[2], dtype='fp32')
complete = perf_counter_ns()
assert r.latency_us is not None
print(json.dumps({'import_ms': (ready-start)/1e6, 'first_call_ms': (complete-ready)/1e6,
        'model_us': r.latency_us, 'shape': [int(sys.argv[1]),12], 'chip': sys.argv[2]}))"""

if __name__ == "__main__":
    rows = []
    for size in (72, 262144, 1048576, 2097152):
        for chip in (DT, PR, FUTURE):
            rows.append(
                json.loads(
                    check_output(
                        [sys.executable, "-c", SCRIPT, str(size), chip],
                        cwd=ROOT,
                        text=True,
                    )
                )
            )
    write(
        DEST / "first-query.json",
        dict(
            scope="Fresh interpreter per call, import and first uncached public estimate separate, includes chip config read and all resource module imports",
            n=len(rows),
            max_first_call_ms=max(r["first_call_ms"] for r in rows),
            rows=rows,
        ),
    )
