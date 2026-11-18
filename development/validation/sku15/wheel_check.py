"""Measure the installed wheel and check its estimates against two immutable releases."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from time import perf_counter_ns


def read(path):
    return json.loads(path.read_text())


def run(root):
    start = perf_counter_ns()
    import operator_model
    from operator_model import estimate, load_chip

    import_ms = (perf_counter_ns() - start) / 1e6
    if operator_model.__version__ != '0.15.0' or 'site-packages' not in Path(operator_model.__file__).parts:
        raise RuntimeError('Use the installed v0.15.0 wheel')
    archive = read(root/'results/v0.13/evaluation.json')['rows']
    freeze = read(root/'results/v0.15/sku/freeze.json')['predictions']
    prior_quant = read(root/'results/v0.14/sku/freeze.json')['predictions']
    result = {}
    for sku, key in [('Ascend950DT_9582', 'dt_us'), ('Ascend950PR_957b', 'pr_us')]:
        chip = load_chip(sku)
        durations = defaultdict(list)
        rows = {}

        def call(op, shape, attrs=None):
            before = perf_counter_ns()
            answer = estimate(op, shape, attrs, chip=chip)
            durations[op].append((perf_counter_ns() - before) / 1e6)
            if answer.latency_us is None or not math.isfinite(answer.latency_us) or answer.latency_us <= 0:
                raise AssertionError((op, shape, answer.reason))
            return answer

        for r in archive:
            answer = call(r['op'], r['shape'], r['attrs'])
            rows[r['id']] = durations[r['op']][-1]
            if answer.work['logical_read_bytes'] != r['logical_read_bytes']:
                raise AssertionError(('work mismatch', r['id']))
            expected = r['dt_prediction_us'] if key == 'dt_us' else r['pr_projection_us']
            if (key == 'dt_us' or r['op'] not in ('quant','fa','matmul')) and not math.isclose(answer.latency_us,expected,rel_tol=1e-10):
                raise AssertionError(('archived prediction mismatch',r['id'],sku))
        for r in prior_quant:
            if r['op'] == 'quant' and not math.isclose(call('quant',r['shape']).latency_us,r[key],rel_tol=1e-10):
                raise AssertionError(('v0.14 quant drift',r['id']))
        for r in freeze:
            for _ in range(5):
                if not math.isclose(call(r['op'],r['shape']).latency_us,r[key],rel_tol=1e-10):
                    raise AssertionError(('frozen prediction mismatch',r['id']))
        all_calls = [value for values in durations.values() for value in values]
        if max(all_calls) >= 100:
            raise AssertionError((sku,'prediction exceeded 100 ms',max(all_calls)))
        result[sku] = dict(calls=len(all_calls), median_ms=median(all_calls), max_ms=max(all_calls),
                           per_op={k:dict(calls=len(v),median_ms=median(v),max_ms=max(v)) for k,v in durations.items()},
                           archived_row_ms=rows)
    return dict(version=operator_model.__version__, import_ms=import_ms, checks=result,
                archived_rows=len(archive), frozen_cases_including_reserves=len(freeze))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(json.dumps(run(parser.parse_args().root.resolve()),indent=2))
