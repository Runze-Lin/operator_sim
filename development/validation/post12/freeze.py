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
from operator_model.post_v12 import predict
from operator_model.post_v11 import cohort, predict as previous
from operator_model.vector_v9 import schedule
from operator_model.wide_v10 import occupancy

OUT = ROOT / 'results/v0.12/post'
OLD = ROOT / 'results/v0.11/post/rows.json'
STATES = ('first', 'expand', 'reuse')
ATTRS = dict(dtype='bf16', has_h_res=True)


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')


def fit(historical):
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
            labels = [s for s in STATES if any(entry[1] == s for entry in samples)] if state_terms else ['all']
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
            phases[route][queue] = {s: b.get(s, statistics.median(b[l] for l in labels)) if state_terms else b['all'] for s in STATES}
            phases[route][queue].update(service=b.get('service', 0.), per_extra_row=b.get('per_extra_row', 0.))
    return dict(contract=ATTRS, context_counts=counts, phases=phases)


def main():
    if (OUT / 'freeze.json').exists():
        raise RuntimeError('Already frozen')
    dev = json.loads(OLD.read_text())
    assert len({tuple(r['shape']) for r in dev}) == len(dev) == 43
    audit = json.loads((ROOT / 'results/v0.11/post/development-output-audit.json').read_text())
    eligible = {r['db_sha256'] for r in audit if r['qualified']}
    for r in dev:
        if 'output_audit' in r:
            assert r['output_audit']['qualified'] and r['db_sha256'] == r['output_audit']['db_sha256']
        else:
            assert sha256(ROOT/r['db_source']) in eligible
    save('profile.json', fit(dev))
    batches = [[(1,3456),(2,4480),(3,3840)], [(1,2816),(4,5888),(12,1024),(31,256),(64,1536)],
               [(4,6144),(128,1536)], [(4,6144),(256,1536)]]
    fixed = statistics.median(r['reference_us'] for r in dev)
    cases=[]; batches_out=[]
    for j,batch in enumerate(batches):
        resident=0; plan=[]
        for i,(t,d) in enumerate(batch):
            shape = [t,4,d]; warm = j>=2 and i==0
            pred = predict('mhc_post',shape,ATTRS)
            c = dict(id=f'v12_post_{j}_{i}',op='mhc_post',shape=shape,attrs=ATTRS,
                     prior_aiv=resident, prediction=pred, role='context_warmup' if warm else 'validation',
                     previous_us=previous('mhc_post',shape,ATTRS)['predicted_us'],fixed_us=fixed,
                     sequence_prediction=predict('mhc_post',shape,dict(ATTRS,resident_aiv=resident)),
                     group='few_wide' if t<=4 and d>=2048 else 'controls')
            resident=max(resident,pred['predicted_active_aiv']); plan.append(c)
            if not warm: cases.append(c)
        batches_out.append(plan)
    assert len(cases)==len({tuple(c['shape']) for c in cases})==10
    assert not {tuple(c['shape']) for c in cases}&{tuple(r['shape']) for r in dev}
    paths=['operator_model/post_v12.py','operator_model/post_v11.py','operator_model/vector_v9.py','operator_model/vector_v8.py','operator_model/wide_v10.py',
           'results/v0.12/post/profile.json','results/v0.11/post/profile.json','results/v0.11/post/rows.json',
           'development/validation/post12/freeze.py','development/validation/post12/collect.py','development/validation/post11/recover.py',
           'development/validation/post11/output_audit.py','scripts/wide-v10-trace.py','build/vector-v8/MHC',
           'build/capture/libsim_soc_adapter.so','scripts/cann-run.sh','operator_model/npusim_v7.py']
    save('freeze.json',dict(utc=datetime.now(timezone.utc).isoformat(),cases=cases,batches=batches_out,
                            sha256={p:sha256(ROOT/p) for p in paths},per_batch_seconds=240,per_batch_gib=.7,
                            workers=2, recovery_budget_launches=0, budgets='4 processes, 12 launches including 2 warmups, <=960 process-s and 2.8 GiB DB; no retry by default.',
                            policy='Unknown channel-split history: minimize worst relative error over first/expand/reuse conditional estimates. Other routes use cohort median.',
                            prior_source='43 qualified historical shapes; original empty and mixed-output DBs excluded'))


if __name__=='__main__': main()
