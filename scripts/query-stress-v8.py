"""Time uncached queries on boundary shapes without running the CA reference."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from operator_model.predictor import predict

CASES = [
    ('quant', (1048576, 65536), {}),
    ('fa', (9, 524288, 64, 1, 1), {}),
    ('moe_ffn', (1152, 4096, 4096, 144), {'expert_counts': [8] * 144}),
    ('matmul', (4096, 4096, 4096), {}),
    ('kda_chunk', (1, 64, 1, 2, 128, 128), {}),
]


def main():
    measurements = []
    for op, shape, attrs in CASES:
        start = time.perf_counter_ns()
        result = predict(op, shape, attrs)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        measurements.append(dict(op=op, shape=shape, attrs=attrs, query_ms=elapsed_ms,
                                 returned_time=result['predicted_us'] is not None, status=result['status']))
    path = ROOT / 'results/v0.8/query-stress.json'
    path.write_text(json.dumps(dict(scope='uncached query wall milliseconds; no CA validation on stress shapes',
                                    cases=measurements), indent=2) + '\n')
    if any(row['query_ms'] >= 1000 for row in measurements):
        raise RuntimeError('Query exceeded one second')


if __name__ == '__main__':
    main()
