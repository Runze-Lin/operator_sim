"""Run with the isolated release wheel. No raw traces or CANN runtime are used."""
import argparse
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter_ns


def check(root):
    start = perf_counter_ns()
    import operator_model
    from operator_model import load_chip, estimate

    import_ms = (perf_counter_ns() - start) / 1e6
    if ('site-packages' not in Path(operator_model.__file__).parts or
            operator_model.__version__ != '0.14.0'):
        raise ValueError('Run with the installed v0.14.0 wheel')
    source = root/'results/v0.13/evaluation.json'
    archive = json.loads(source.read_text())
    frozen = json.loads((root/'results/v0.14/sku/freeze.json').read_text())
    results = {}
    for name, field, holdout_key in [('Ascend950DT_9582','dt_prediction_us','dt_us'),
                                    ('Ascend950PR_957b','pr_projection_us','pr_us')]:
        chip = load_chip(name)
        elapsed = []
        per_op = {}
        samples = {}
        archived_row_ms = {}
        deltas = []
        for r in archive['rows']:
            start = perf_counter_ns()
            answer = estimate(r['op'], r['shape'], r['attrs'], chip=chip)
            ms = (perf_counter_ns()-start)/1e6
            elapsed.append(ms)
            per_op[r['op']] = max(ms, per_op.get(r['op'], 0))
            samples.setdefault(r['op'], []).append(ms)
            archived_row_ms[r['id']] = ms
            if answer.latency_us is None or answer.work['logical_read_bytes'] != r['logical_read_bytes']:
                raise ValueError(('missing estimation/work mismatch',name,r['id']))
            delta = abs(answer.latency_us-r[field])
            deltas.append(delta)
            if (holdout_key == 'dt_us' or r['op'] not in ('matmul','quant')) and delta>1e-8:
                raise ValueError(('unexpected reference drift',name,r['id'],delta))
        for f in frozen['predictions']:
            for _ in range(5):
                start = perf_counter_ns()
                answer = estimate(f['op'], f['shape'], chip=chip)
                ms = (perf_counter_ns()-start)/1e6
                elapsed.append(ms)
                per_op[f['op']] = max(ms, per_op[f['op']])
                samples[f['op']].append(ms)
                if not math.isclose(answer.latency_us, f[holdout_key],rel_tol=1e-9):
                    raise ValueError(('frozen vs installed disagreement',f['shape'],answer.latency_us))
        if max(elapsed) >= 100:
            raise ValueError(('prediction exceeded 100 ms', name, max(elapsed)))
        results[name] = dict(calls=len(elapsed), max_ms=max(elapsed),median_ms=median(elapsed),
                             per_op_max_ms=per_op,archive_max_delta_us=max(deltas),archived_row_ms=archived_row_ms,
                             per_op={op: dict(calls=len(values), max_ms=max(values), median_ms=median(values))
                                     for op, values in samples.items()})
    return dict(version=operator_model.__version__, import_ms=import_ms, checks=results,
                archived_rows=len(archive['rows']),frozen_cases_including_reserves=len(frozen['predictions']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.root.resolve()),indent=2))
