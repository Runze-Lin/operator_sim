"""Archive-only Post first-row MTE2 response/issue decomposition and grouped replay."""
from collections import defaultdict
from math import ceil
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.post_candidate.evaluate import DT, PR, OUT, STATES, PHASES, read, write, train, phases, metric
from operator_model import load_chip
from operator_model.rms_resource import READ_PER_KIB


def payloads(row):
    width = row['fragment_width']
    if row['route'] == 'post_channel':
        # Each of the four residual lanes has a 32-byte BF16 DMA stride.
        lane_bytes = 32 * ceil(width * 2 / 32)
        return [lane_bytes, 4 * lane_bytes]
    return [64, 16, 2 * width, 8 * width]


def price(chip):
    c = load_chip(chip)
    gbps = c.details.get('mte2_bandwidth_gbs')
    return 1024 * c.frequency_ghz / gbps if gbps else READ_PER_KIB


def measured(row):
    sizes = payloads(row)
    dma = row['dma_read'][1:1 + len(sizes)]
    assert row['dma_read'][0]['bytes'] == 96
    assert sizes == [d['bytes'] for d in dma], (row['shape'], sizes, dma)
    start = dma[0]['start_us']
    issued = [d['start_us'] - start for d in dma]
    spans = [d['end_us'] - d['start_us'] for d in dma]
    wait = row['first_vector_operand_us'] - max(d['end_us'] for d in dma)
    assert wait >= 0 and all(a <= b for a, b in zip(issued, issued[1:]))
    ready = max(t + s for t, s in zip(issued, spans)) + wait
    assert abs(ready - row['phase_us']['read']) < 1e-6
    return dict(issue_us=issued, response_us=spans, handoff_us=wait)


def fit_ports(rows):
    signals = defaultdict(list)
    for r in rows:
        s = measured(r)
        frequency = 1000 * load_chip(r['chip']).frequency_ghz
        starts = s['issue_us']
        gaps = [b - a for a, b in zip(starts, starts[1:])]
        costs = [duration * frequency - size * price(r['chip']) / 1024
                 for duration, size in zip(s['response_us'], payloads(r))]
        signals[r['route'], r['state']].append(
            [v * frequency for v in gaps] + costs + [s['handoff_us'] * frequency])
    calibration = {}
    for (route, state), observations in signals.items():
        n = 2 if route == 'post_channel' else 4
        typical = np.maximum(0, np.median(observations, axis=0)).tolist()
        calibration.setdefault(route, {})[state] = dict(n=len(observations), gap_cycles=typical[:n-1],
                  response_fixed_cycles=typical[n-1:-1], handoff_cycles=typical[-1])
    return calibration


def predicted_read(port, row, state):
    by_state = port[row['route']]
    if state not in by_state:
        fallback = {'expand':'first', 'after_pre_sinkhorn':'reuse', 'first':'expand', 'reuse':'after_pre_sinkhorn'}
        state = fallback[state]
        if state not in by_state:
            state = max(by_state, key=lambda s: by_state[s]['n'])
    coefs = by_state[state]
    frequency = 1000 * load_chip(row['chip']).frequency_ghz
    starts = (np.cumsum([0] + coefs['gap_cycles']) / frequency).tolist()
    latency = [(cycles + size * price(row['chip']) / 1024) / frequency
               for cycles, size in zip(coefs['response_fixed_cycles'], payloads(row))]
    handoff = coefs['handoff_cycles'] / frequency
    ready = max(t + l for t, l in zip(starts, latency)) + handoff
    return ready, dict(issue_us=starts, response_us=latency, handoff_us=handoff)


def public_state(row):
    # These archived labels are derived from the actual launch order, prior
    # active counts and producer order in extract.py, never from timings.
    previous = row['prior_aiv']
    if previous is None:
        assert row['state'] == 'after_pre_sinkhorn'
        return 'after_pre_sinkhorn'
    state = 'first' if previous == 0 else 'expand' if previous < row['active_aiv'] else 'reuse'
    assert state == row['state']
    return state


def estimate(model, port, row):
    totals = {}
    intervals = {}
    stages = {}
    for state in STATES:
        if state == 'expand' and row['active_aiv'] == 1:
            continue
        segments = phases(model, row, state)
        segments['read'], intervals[state] = predicted_read(port, row, state)
        stages[state] = segments
        totals[state] = sum(segments.values())
    low = min(totals.values())
    high = max(totals.values())
    s = public_state(row)
    a, b = totals.get('expand', totals['reuse']), totals['reuse']
    partial = totals[s] if s in ('first', 'after_pre_sinkhorn') else 2*a*b/(a+b)
    return dict(conditioned_us=totals[s], unknown_us=2*low*high/(low+high),
                coarse_context_us=partial, state_envelope_us=[low, high], predicted_state_us=totals,
                predicted_milestones_us=stages[s], predicted_read_window=intervals[s])


def clean_sample(row):
    # The online descriptor calculation cannot see profile IDs, process names
    # or recorded device durations; state is read only for conditional outputs.
    return {k: row[k] for k in ('chip','route','fragment_width','rows_per_core','active_aiv','state','prior_aiv')}


def run():
    rows = read(OUT/'phase-evidence.json')
    old = {r['trace']:r for r in read(OUT/'cross-validation.json')['rows']}
    assert len(old) == len(rows) == 90
    by_process = defaultdict(list)
    for row in rows:
        by_process[row['process']].append(row)
    results = []
    folds = []
    for name, held in by_process.items():
        shapes = {tuple(r['shape']) for r in held}
        training = [r for r in rows if r['process'] != name and tuple(r['shape']) not in shapes]
        folds.append(dict(held_process=name, held_shapes=sorted(shapes), train_n=len(training),
                          train_processes=sorted({r['process'] for r in training})))
        base, price_model = train(training), fit_ports(training)
        for row in held:
            previous = old[row['trace']]
            assert len(training) == previous['train_n']
            pred = estimate(base, price_model, clean_sample(row))
            lo, hi = pred['state_envelope_us']
            results.append(dict(**{k:row[k] for k in ('trace','sha256','shape','chip','state','process','group','us','active_aiv','route')},
                         **pred,extrapolated_state_envelope_covers=lo <= row['us'] <= hi,
                         prior_stage_candidate_us=previous['given_history_us'],
                         original_unknown_us=previous['old_unknown_us'],
                         prior_stage_unknown_us=previous['no_history_us'],
                         prior_first_read_us=previous['given_history_stages_us']['read'],
                         ca_phases_us=row['phase_us'],operand_bytes=payloads(row),ca_read_window=measured(row)))
    summary = []
    for chip in (DT, PR):
        pool = [r for r in results if r['chip'] == chip]
        for group in ('all', *STATES, 'few token/wide', 'one row/core','multiple rows/core'):
            g = [r for r in pool if group == 'all' or group in (r['state'],r['group'])]
            if g:
                err = {key:metric(g,key) for key in ('original_unknown_us','prior_stage_unknown_us','unknown_us','coarse_context_us','prior_stage_candidate_us','conditioned_us')}
                stages = {q:float(np.median([abs(r['predicted_milestones_us'][q]-r['ca_phases_us'][q]) for r in g])) for q in PHASES}
                summary.append(dict(chip=chip,group=group,models=err,stage_median_absolute_error_us=stages,
                     previous_read_median_abs_error_us=statistics.median(abs(r['prior_first_read_us']-r['ca_phases_us']['read']) for r in g),
                     empirical_state_envelope_coverage=sum(r['extrapolated_state_envelope_covers'] for r in g)/len(g)))
    pairs = read(OUT/'protocol-feasibility.json')['exact_same_elf_pairs']
    # A per-process fold can train different coefficients for the two endpoints
    # of a matched shape. Fit ONE model after withholding all matched processes
    # and every shape that occurred in them (both SKUs) before comparing protocols.
    target_hashes = {g[k] for limits in pairs.values() for g in limits['exact_pair_set']['groups']
                     for k in ('high_trace_sha256', 'low_trace_sha256')}
    matched_processes = {r['process'] for r in rows if r['sha256'] in target_hashes}
    matched_shapes = {tuple(r['shape']) for r in rows if r['process'] in matched_processes}
    train_pairs = [r for r in rows if r['process'] not in matched_processes and tuple(r['shape']) not in matched_shapes]
    paired_model, paired_ports = train(train_pairs), fit_ports(train_pairs)
    match_predictions = []
    for r in rows:
        if r['sha256'] not in target_hashes:
            continue
        pred = estimate(paired_model, paired_ports, clean_sample(r))
        match_predictions.append(dict(shape=r['shape'], chip=r['chip'], state=r['state'], sha256=r['sha256'],
                                 us=r['us'], unknown_us=pred['unknown_us'], conditioned_us=pred['conditioned_us']))
    paired = []
    for chip, limits in pairs.items():
        groups = limits['exact_pair_set']['groups']
        for g in groups:
            two = [r for r in match_predictions if r['chip'] == chip and r['shape'] == g['shape']]
            assert len(two) == 2 and two[0]['unknown_us'] == two[1]['unknown_us']
        for state in ('first','after_pre_sinkhorn'):
            pool = [r for r in match_predictions if r['chip'] == chip and r['state'] == state]
            assert len(pool) == len(groups)
            paired.append(dict(chip=chip,state=state,n=len(pool),unknown_us=metric(pool,'unknown_us'),
                               conditioned_us=metric(pool,'conditioned_us')))
    write(OUT/'issue-response.json',dict(scope='Only archived v0.12/v0.17 and earlier. Shape- and process-purged grouped replay, no new CA. Fractional endpoints in protocol-feasibility.json.',
            detail='First-row MTE2 queue gaps and per-request response intercepts are route/state medians measured independently of end-to-end time. Fixed intercept excludes independent local-byte floor. Predictions from recorded public workload shape/tiling, chip resource configs and optional caller history only.',
            chips={chip:dict(cycles_per_us=load_chip(chip).frequency_ghz*1000,mte2_cycles_per_KiB=price(chip)) for chip in (DT, PR)},
            archive_ports=fit_ports(rows),paired_per_protocol=paired,matched_predictions=match_predictions,
            matched_split=dict(held_processes=sorted(matched_processes),held_shapes=sorted(matched_shapes),n_train=len(train_pairs)),
            summary=summary,folds=folds,rows=results))
    for r in summary:
        if r['group']=='all':
            print(r['chip'],{k:(v['wape'],v['max_err']) for k,v in r['models'].items()}, r['stage_median_absolute_error_us'],r['empirical_state_envelope_coverage'])


if __name__ == '__main__':
    run()
