"""Audit logical chunk work by enumeration and a sparse CPU forward identity."""
import json
import math
from collections import Counter
from itertools import accumulate
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from operator_model.cube_v8_chunk import predict

BASE = ROOT / 'results/v0.8/cube'
NAMES = {'qk': 'aqk_useful_dot_flops', 'kk': 'akk_useful_dot_flops',
         'wu': 'post_wu_dense_equivalent_flops', 'state_update': 'fwd_h_matmul_flops',
         'state_decay': 'fwd_h_state_decay_multiplies', 'finalize': 'finalize_dense_equivalent_flops'}


def enumerate_work(shape, width):
    b, seq, _, heads, k, v = shape
    counts = Counter()
    other = Counter()
    tiles = [range(begin, min(begin + width, seq)) for begin in range(0, seq, width)]
    for _ in range(b):
        for _ in range(heads):
            other['state_chains'] += 1
            other['state_elements'] += k * v
            for chunk in tiles:
                other['parallel_chunk_head_tasks'] += 1
                counts['state_decay'] += k * v
                for i in chunk:
                    other['gate_elements'] += k
                    other['beta_elements'] += 1
                    counts['state_update'] += 2*k*v + 2*v*k
                    counts['finalize'] += 2*k*v
                    for j in chunk:
                        if j <= i:
                            counts['qk'] += 2*k
                        if j < i:
                            counts['kk'] += 2*k
                        counts['wu'] += 2*k + 2*v
                        counts['finalize'] += 2*v
    other['chunks_per_sequence'] = len(tiles)
    other['full_chunks'] = sum(len(t) == width for t in tiles)
    other['tail_tokens'] = sum(len(t) for t in tiles if len(t) < width)
    other['state_tensor_bytes'] = other['state_elements'] * 4
    return counts, other


def compare_cpu(shape, attrs):
    """One live K lane and one live V lane per head. All other outputs/states vanish."""
    b, seq, _, heads, k_dim, _ = shape
    width = attrs['chunk_size']
    scale = k_dim ** -0.5
    max_error, n_outputs, n_boundaries = 0., 0, 0
    for chain in range(b*heads):
        # Dense reference semantics on this sparse subspace reduce exactly to one lane.
        blocked = direct = (chain + 1)/32
        for start in range(0, seq, width):
            size = min(width, seq - start)
            q, k, v, beta, gates, ref = [], [], [], [], [], []
            for t in range(start, start + size):
                phase = t + chain
                q.append((3 + phase % 5)/16)
                k.append((6 - phase % 7)/16)
                v.append((phase % 11 - 5)/8)
                beta.append((phase % 3 + 1)/4)
                if attrs.get('safe_gate'):
                    # A_log=0, dtBias=0.
                    gates.append(attrs['lower_bound'] / (1 + math.exp(-(phase % 9 - 4)/8)))
                else:
                    gates.append(-(phase % 3 + 1)/128)
                direct *= math.exp(gates[-1])
                direct += k[-1] * beta[-1] * (v[-1] - k[-1]*direct)
                ref.append(scale*q[-1]*direct)
            g = list(accumulate(gates))
            w, u, delta = [], [], []
            for i in range(size):
                # Solve two lower-triangular right-hand sides for the chunk.
                coeff = [beta[i]*k[i]*k[j]*math.exp(g[i]-g[j]) for j in range(i)]
                w.append(beta[i]*k[i]*math.exp(g[i]) - sum(c*x for c,x in zip(coeff,w)))
                u.append(beta[i]*v[i] - sum(c*x for c,x in zip(coeff,u)))
                delta.append(u[i]-w[i]*blocked)
                out = q[i]*math.exp(g[i])*blocked
                out += sum(q[i]*k[j]*math.exp(g[i]-g[j])*delta[j] for j in range(i+1))
                max_error = max(max_error, abs(scale*out-ref[i]))
            blocked = math.exp(g[-1])*blocked + sum(
                k[j]*math.exp(g[-1]-g[j])*delta[j] for j in range(size))
            max_error = max(max_error, abs(blocked-direct))
            n_outputs += size
            n_boundaries += 1
    return dict(live_outputs=n_outputs, live_chunk_end_states=n_boundaries,
                max_abs_error=max_error, tolerance=1e-12, passed=max_error < 1e-12)


def audit(case, source):
    shape, attrs = case['shape'], case['attrs']
    prediction = predict('kda_chunk', shape, attrs)
    counts, other = enumerate_work(shape, attrs['chunk_size'])
    plan = prediction['structural_plan']
    checks = {}
    for stage in prediction['stages']:
        key = stage['id']
        serial = key in ('state_decay','state_update')
        expected_tasks = other['state_chains' if serial else 'parallel_chunk_head_tasks']
        expected_steps = other['chunks_per_sequence'] if serial else 1
        checks[key] = ([counts[key],expected_tasks,expected_steps,serial] ==
                       [stage['work'],stage['parallel_tasks'],stage['serial_steps'],stage['state_serial']]
                       and counts[key] == plan['work'][NAMES[key]])
    for key, value in other.items():
        checks[key] = value == {**plan['tiling'], **plan['work']}[key]
    positive = [key for key in NAMES if counts[key]]
    checks['nonzero_stages_and_dependency_order'] = [
        (stage['id'],stage['dependencies']) for stage in prediction['stages']
    ] == [(key,positive[i-1:i] if i else []) for i,key in enumerate(positive)]
    checks['all_stage_work_including_zeros'] = all(
        counts[key] == plan['work'][work_key] for key,work_key in NAMES.items())
    checks['unknown_work_remains_unqualified'] = (plan['work']['triangular_inverse_flops'] is None and
        plan['work']['total_executed_flops'] is None and prediction['predicted_us'] is None)
    cpu = compare_cpu(shape, attrs)
    return dict(id=case['id'], shape=shape, attrs=attrs, freeze_source=source,
                enumerated_stage_work={s:counts[s] for s in NAMES}, enumerated_partition=dict(other),
                checks=checks, sparse_cpu_forward=cpu, passed=all(checks.values()) and cpu['passed'],
                ca_reference_us=None)


def main():
    selections = [('freeze.json', [
        'kda_chunk_1_1_1_1_16_16','kda_chunk_1_8_1_1_16_16','kda_chunk_1_63_1_1_16_16',
        'kda_chunk_1_65_1_1_16_16','kda_chunk_1_96_1_1_128_128','kda_chunk_1_128_1_1_128_128']),
        ('safe-chunk-freeze.json', ['kda_chunk_safe_64_2','kda_chunk_safe_128_4',
                                    'kda_chunk_safe_64_8','kda_chunk_safe_256_4'])]
    rows = []
    for source, ids in selections:
        cases = {c['id']:c for c in json.loads((BASE/source).read_text())['cases']}
        rows += [audit(cases[name],source) for name in ids]
    report = dict(protocol='Previously frozen shapes. Explicit enumeration of each chunk, token and '
                  'ordered pair. CPU float64 single-live-lane chunk triangular solve versus token '
                  'recurrence; compares outputs and boundary states, before BF16 rounding. '
                  'Does not exercise the installed kernel, raw intermediate exports or physical scheduling.',
                  reference='third_party/ops-transformer/attention/chunk_kda_fwd/docs/aclnnChunkKdaFwd.md',
                  verified=sum(r['passed'] for r in rows), ca_timing_qualified=0, cases=rows)
    (BASE/'chunk-structure-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    if not all(r['passed'] for r in rows):
        raise SystemExit('structure or sparse forward check failed')
    print(report['verified'], max(r['sparse_cpu_forward']['max_abs_error'] for r in rows))


if __name__ == '__main__':
    main()
