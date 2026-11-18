"""Freeze context cohorts, phase service costs and two generations of predictions."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model.ca import sha256
from operator_model.post_v11 import predict, cohort
from operator_model.vector_v9 import schedule
from operator_model.wide_v10 import occupancy, predict as previous

OUT = ROOT / 'results/v0.11/post'
OLD = ROOT / 'results/v0.10/wide/rows.json'
STATES = ('first', 'expand', 'reuse')
ATTRS = dict(dtype='bf16', has_h_res=True)


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')


def main():
    if (OUT / 'freeze.json').exists():
        raise RuntimeError('Already frozen')
    historical = [row for row in json.loads(OLD.read_text()) if row['op'] == 'mhc_post']
    groups = defaultdict(Counter)
    by_route = defaultdict(list)
    for row in historical:
        route, active, rows, units = schedule('mhc_post', row['shape'], ATTRS)
        state = occupancy(active, row['prior_aiv'])
        groups[cohort(route, row['shape'][0], active)][state] += 1
        by_route[route].append((row['phase_us'], state, rows, units))
    counts = {k: {s: v[s] for s in STATES} for k, v in groups.items()}
    phases = defaultdict(dict)
    for route, samples in by_route.items():
        for queue in ('arrival', 'SCALAR', 'MTE2', 'SIMD', 'MTE3', 'other'):
            state_terms = queue in ('arrival', 'SCALAR', 'MTE2', 'SIMD')
            service = queue == 'SIMD' or (queue == 'MTE2' and route == 'post_token')
            row_term = route == 'post_token' and queue in ('SCALAR', 'MTE3')
            labels = list(STATES) if state_terms else ['all']
            design, target = [], []
            for measured, state, rows, units in samples:
                x = [float(label in (state, 'all')) for label in labels]
                if service:
                    x.append(units[queue])
                if row_term:
                    x.append(rows - 1)
                design.append(x)
                target.append(measured.get(queue, 0.))
            x = np.asarray(design)
            fit = least_squares(lambda b: x @ b - target, np.full(x.shape[1], .01),
                                bounds=(0, np.inf), loss='soft_l1', f_scale=.1)
            b = dict(zip(labels + (['service'] if service else [])
                         + (['per_extra_row'] if row_term else []), map(float, fit.x)))
            phases[route][queue] = {s: b[s] if state_terms else b['all'] for s in STATES}
            phases[route][queue].update(service=b.get('service', 0.), per_extra_row=b.get('per_extra_row', 0.))
    save('profile.json', dict(contract=ATTRS, context_counts=counts, phases=phases))
    shapes = [(3,4,4352), (8,4,640), (2,4,3328), (4,4,6144), (1,4,2304),
              (1,4,7680), (64,4,1024), (128,4,1024), (192,4,1024), (256,4,1024)]
    high = 0
    cases = []
    baseline = statistics.median(row['reference_us'] for row in historical)
    for i, shape in enumerate(shapes):
        assert all(list(shape) != row['shape'] for row in historical)
        p = predict('mhc_post', shape, ATTRS)
        group = 'few_wide' if shape[0] <= 4 and shape[2] >= 2048 else 'controls'
        ctx = predict('mhc_post', shape, {**ATTRS, 'resident_aiv': high})
        cases.append(dict(id=f'v11_mhc_post_{i}', op='mhc_post', shape=shape, attrs=ATTRS, kind='MHC',
                          groups=1, args=shape, role='validation', group=group, prediction=p,
                          previous_us=previous('mhc_post', shape, ATTRS)['predicted_us'], fixed_us=baseline,
                          prior_aiv=high, sequence_prediction=ctx))
        high = max(high, p['predicted_active_aiv'])
    paths = ['operator_model/post_v11.py', 'operator_model/wide_v10.py', 'operator_model/vector_v9.py',
             'operator_model/vector_v8.py', 'results/v0.11/post/profile.json', 'results/v0.10/wide/profile.json',
             'results/v0.10/wide/rows.json', 'scripts/wide-v10-collect.py', 'scripts/wide-v10-trace.py',
             'development/validation/post11/collect.py', 'development/validation/post11/freeze.py',
             'build/vector-v8/MHC', 'build/capture/libsim_soc_adapter.so', 'scripts/cann-run.sh',
             'operator_model/npusim_v7.py', 'operator_model/ca.py']
    save('freeze.json', dict(utc=datetime.now(timezone.utc).isoformat(), cases=cases,
                             sha256={p: sha256(ROOT / p) for p in paths}, per_batch_seconds=900,
                             disk_budget_gib=3, prior_source='33 unique historical shapes; one stream per batch; no duplicate control',
                             plan='5 few/wide and 5 controls; final four fixed width, 64/128/192/256 tokens. No timing replay after capture.'))


if __name__ == '__main__':
    main()
