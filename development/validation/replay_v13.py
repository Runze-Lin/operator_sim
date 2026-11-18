"""Replay the DT reference and exercise the PR projection on the same contracts."""
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from operator_model import estimate, load_chip


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / 'results/v0.12/evaluation.json'
    frozen = json.loads(source.read_text())
    target = load_chip('Ascend950PR_957b')
    timings = defaultdict(lambda: defaultdict(list))
    rows = []
    for r in frozen['rows']:
        op, shape, attrs = r['op'], r['shape'], r.get('attrs', {})
        outputs = {}
        for chip in (load_chip(), target):
            for _ in range(3):
                start = perf_counter_ns()
                result = estimate(op, shape, attrs, chip=chip)
                timings[chip.name][op].append((perf_counter_ns()-start)/1e6)
            if not result.work or result.latency_us is None or not math.isfinite(result.latency_us):
                raise ValueError((op, shape, chip.name, result))
            outputs[chip.name] = result
        ref, projected = outputs.values()
        if not math.isclose(ref.latency_us, r['predicted_us'], rel_tol=1e-8):
            raise ValueError(('DT latency differs from freeze', r['id'], ref, r['predicted_us']))
        if ref.work != projected.work or projected.status != 'extrapolated':
            raise ValueError(('inconsistent work or projection label', r['id']))
        rows.append(dict(id=r.get('id', f'{op}-{len(rows)}'), op=op, shape=shape, attrs=attrs, role=r['role'],
                         dt_ca_us=r['reference_us'], dt_prediction_us=ref.latency_us,
                         dt_query_max_ms=max(timings[ref.chip_name][op][-3:]),
                         pr_query_max_ms=max(timings[projected.chip_name][op][-3:]),
                         pr_projection_us=projected.latency_us, cube_ops=ref.work['cube_ops'],
                         vec_ops=ref.work['vec_ops'], logical_read_bytes=ref.work['logical_read_bytes'],
                         logical_write_bytes=ref.work['logical_write_bytes']))
    chunk = estimate('kda_chunk', (1, 64, 1, 1, 128, 128), chip=target)
    if chunk.status != 'unavailable' or chunk.latency_us is not None or not chunk.work:
        raise ValueError('Missing structure-only KDA contract')
    perf = {chip: {op: {'calls': len(times), 'max_ms': max(times), 'median_ms': median(times)}
                   for op, times in operators.items()} for chip, operators in timings.items()}
    versions = [*sorted((ROOT/'configs/chips').glob('*.json')),
                *(ROOT/'operator_model'/name for name in
                  ('backend.py', 'chips.py', 'workload.py', 'transfer.py', 'predictor.py',
                   'quant_repeat.py', 'target_tiling.py'))]
    output = dict(version='v0.13', reference_file=str(source.relative_to(ROOT)),
                  reference_sha256=fingerprint(source), model_sha256={str(p.relative_to(ROOT)):fingerprint(p)
                  for p in versions}, timed_rows=len(rows), dt_summary=frozen['operator_summary'],
                  timing_method='3 calls per shape per configured SKU in one Python process, chip already loaded',
                  query_wall=perf, rows=rows)
    for summary in output['dt_summary']:
        group = perf[load_chip().name].get(summary['op'], {})
        summary['query_max_ms'] = group.get('max_ms')
    out = ROOT/'results/v0.13/evaluation.json'
    out.write_text(json.dumps(output, indent=2)+'\n')
    for chip, ops in perf.items():
        worst = max(group['max_ms'] for group in ops.values())
        print(f'{chip}: {len(rows)} shapes, slowest {worst:.3f} ms')
        if worst >= 100:
            raise ValueError('In-process prediction exceeded 100 ms')


if __name__ == '__main__':
    main()
