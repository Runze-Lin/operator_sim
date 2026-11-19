"""Recompute the DT-only exposed tail prices; never uses PR or total latency."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[3]
source = json.loads((root/'results/v0.19/workloads/attn-res-extension-evaluation.json').read_text())
samples = [r for r in source['rows'] if r['chip'].endswith('9582') and r['ca_phases_us']]
x, y = [], []
for r in samples:
    _, h, _, n = r['shape']
    lanes = (h+63)//64
    x.append([1, lanes, lanes*(n+1)])
    y.append(1650*r['ca_phases_us'][-1] - 2*h/48.3629559884)
coeff, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
assert rank == 3 and min(coeff) > 0
print('fixed cycles, cycles / norm chunk, cycles / bank chunk', *coeff)
