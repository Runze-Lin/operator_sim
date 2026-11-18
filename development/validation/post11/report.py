"""Join immutable per-shape predictions to completed records and benchmark the online path."""
import json
import math
from pathlib import Path
import runpy
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model.ca import sha256
from operator_model.post_v11 import predict
from operator_model.wide_v10 import predict as previous

OUT = ROOT / 'results/v0.11/post'
OLD = ROOT / 'results/v0.10/wide/rows.json'
inspect = runpy.run_path(str(ROOT / 'scripts/wide-v10-trace.py'))['inspect']
stats = runpy.run_path(str(ROOT / 'scripts/wide-v10-report.py'))['stats']


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2) + '\n')


def main():
    frozen = json.loads((OUT / 'freeze.json').read_text())
    for path, digest in frozen['sha256'].items():
        assert sha256(ROOT / path) == digest
    dev = [r for r in json.loads(OLD.read_text()) if r['op'] == 'mhc_post']
    fixed = statistics.median(r['reference_us'] for r in dev)
    for r in dev:
        shape = r['shape']
        p = predict('mhc_post', shape, r['attrs'])
        contextual = predict('mhc_post', shape, {**r['attrs'], 'resident_aiv':r['prior_aiv']})
        r.update(role='development', prediction=p, previous_us=previous('mhc_post',shape,r['attrs'])['predicted_us'],
                 predicted_us=p['predicted_us'], sequence_prediction=contextual,
                 sequence_us=contextual['predicted_us'], fixed_us=fixed,
                 group='few_wide' if shape[0] <= 4 and shape[-1] >= 2048 else 'controls')
    hold = json.loads((OUT / 'mhc_post-0-10.json').read_text())
    assert len(hold) == 10 and len(set(tuple(r['shape']) for r in hold + dev)) == len(hold + dev)
    for frozen_case, r in zip(frozen['cases'], hold):
        assert all(r[k] == v for k, v in frozen_case.items())
        assert sha256(ROOT / r['source']) == r['db_sha256'] and r['output_audit']['qualified']
        r.update(inspect(ROOT / r['source']))
        assert math.isclose(sum(r['phase_us'].values()), r['reference_us'], abs_tol=1e-7)
        r['sequence_us'] = r['sequence_prediction']['predicted_us']
        r['expected_kernel'] = 'entry_kernel_6' if r['prediction']['route'] == 'post_token' else 'entry_kernel_5'
        r['kernel_match'] = r['expected_kernel'] == r['kernel']
        r['active_match'] = r['prediction']['predicted_active_aiv'] == r['active_aiv']
    for r in dev + hold:
        samples = []
        for _ in range(100):
            start = time.perf_counter_ns()
            p = predict('mhc_post', r['shape'], r['attrs'])
            samples.append((time.perf_counter_ns() - start) / 1000.)
        assert math.isclose(p['predicted_us'], r['predicted_us'], abs_tol=1e-9)
        r['query_median_us'] = statistics.median(samples)
        r['query_max_us'] = max(samples)
    summary = {}
    for role, rows in [('development', dev), ('validation', hold)]:
        parts = dict(all=rows, few_wide=[r for r in rows if r['group'] == 'few_wide'],
                     controls=[r for r in rows if r['group'] == 'controls'])
        if role == 'validation':
            parts['fixed_width_ladder'] = [r for r in rows if r['shape'][0] >= 64]
        summary[role] = {group: {k: stats(part, k)
                                for k in ('predicted_us','previous_us','fixed_us','sequence_us')}
                         for group, part in parts.items()}
    original = json.loads((OUT / 'original-attempt.json').read_text())
    folders = [(ROOT / 'runs/v10wide/mhc_post-0-10', original['wall_s'], 10)]
    for path in sorted(OUT.glob('retry-v11_mhc_post_*.json')):
        result = json.loads(path.read_text())
        folders.append((ROOT / 'runs/v11post' / path.stem, result['wall_s'], 2))
    batches = [dict(name=folder.name, wall_s=wall,
                    db_bytes=sum(p.stat().st_size for p in folder.rglob('*.db')),
                    raw_bytes=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()),
                    launches=count) for folder,wall,count in folders]
    costs = dict(batch_wall_s=sum(b['wall_s'] for b in batches),
                 db_bytes=sum(b['db_bytes'] for b in batches),raw_bytes=sum(b['raw_bytes'] for b in batches),
                 launches=sum(b['launches'] for b in batches),distinct_holdouts=len(hold),
                 prior_traces_reused=len(dev),batches=batches,
                 failed_timing_attempts=len(original['timing_failed']) + 1,
                 context_warmups=len(batches)-1,
                 timing_exclusions='Two empty original DBs and one mixed-output DB; all golden checks passed.',
                 benchmark='100 uncached calls per shape (profile loaded). CA wall sums workers, including retries and warmups.',
                 query_median_us=statistics.median(r['query_median_us'] for r in hold),
                 query_max_us=max(r['query_max_us'] for r in hold))
    save('rows.json', dev + hold)
    save('summary.json', {'mhc_post': summary})
    save('costs.json', {'mhc_post': costs})
    print(json.dumps(summary['validation'], indent=2), costs)


if __name__ == '__main__':
    main()
