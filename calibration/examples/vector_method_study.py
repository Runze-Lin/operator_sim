"""Compare the public methods with the same shapes, paired chips and query boundaries."""
import argparse
import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import platform
import random
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

from operator_model import estimate, load_chip

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/v2.2'
DT = 'Ascend950DT_9582'
PR = 'Ascend950PR_957b'
FUTURE = 'Ascend970DT'
MODES = {'original': {}, 'classes': {'vector_model': 'instruction_classes'}}
DTYPES = {'fa': 'bf16', 'cast_bf16_fp32': 'bf16', 'sigmoid_fp32': 'fp32', 'quant_token_fp8': 'bf16'}
SCENARIOS = [(9, 32768, 64, 1, 1), (64, 768, 64, 1, 1), (63, 640, 128, 1, 1)]


def evaluate(op, shape, chip, mode):
    result = estimate(op, shape, MODES[mode], chip=chip, dtype=DTYPES[op])
    if result.latency_us is None:
        raise ValueError((op, chip.name, shape, mode, result.reason))
    return result


def archive():
    rows = defaultdict(list)
    for line in (ROOT / 'calibration/reference/windows.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['op'] in DTYPES and not r['attrs']:
            rows[r['op']].append(r)
    return rows


def distribution(times):
    return {'p50_us': median(times), 'p95_us': sorted(times)[math.ceil(.95 * len(times)) - 1],
            'n': len(times)}


def latency(op, shape, chip, mode):
    start = perf_counter_ns()
    r = evaluate(op, shape, chip, mode)
    return (perf_counter_ns() - start) / 1000, r.latency_us


def timed_study():
    rng = random.Random(950)
    observations = []
    cohorts = archive()
    for op, rows in cohorts.items():
        # Give every distinct observed chip/shape equal weight at query time.
        cases = sorted({(r['chip'], tuple(r['shape'])) for r in rows})
        tasks = [(load_chip(chip), shape) for chip, shape in cases]
        for mode in MODES:
            latency(op, tasks[0][1], tasks[0][0], mode)
        samples = defaultdict(list)
        for rep in range(15):
            rng.shuffle(tasks)
            for i, (chip, shape) in enumerate(tasks):
                for mode in list(MODES)[::1 if (i + rep) % 2 else -1]:
                    samples[mode].append(latency(op, shape, chip, mode)[0])
        exemplar = cases[len(cases)//2][1]
        for mode in MODES:
            # Each child imports the package and resolves its chip before its timer starts.
            # First-demand parameter loads and lazy imports remain inside the timed call.
            args = [sys.executable, str(Path(__file__).resolve()), '--first', op, mode,
                    ','.join(map(str, exemplar))]
            cold = [float(subprocess.check_output(args, text=True)) for _ in range(9)]
            observations.append(dict(op=op, mode=mode, cases=len(cases),
                                     steady=distribution(samples[mode]), first_shape=exemplar,
                                     first_dt=distribution(cold)))
    return {'timer': 'perf_counter_ns, public estimate incl work and diagnostic assembly; CPU wall time',
            'steady': '15 rounds of shuffled distinct archive chip+shape, alternating order, preloaded chips',
            'first': '9 fresh interpreters, DT config and module import before timing, one representative shape',
            'python': platform.python_version(), 'cpu': platform.machine(), 'results': observations}


def wape(actual, forecasts):
    return 100 * sum(abs(v - m) for v, m in zip(actual, forecasts)) / sum(actual)


def replay():
    rows = []
    for op, cases in archive().items():
        for r in cases:
            chip = load_chip(r['chip'])
            time = {m: evaluate(op, r['shape'], chip, m).latency_us for m in MODES}
            if not math.isclose(time['original'], r['predicted_us'], rel_tol=1e-8):
                raise ValueError(f'Default drift: {op} {r["shape"]} {chip.name}')
            future = {m: evaluate(op, r['shape'], load_chip(FUTURE), m).latency_us for m in MODES}
            rows.append(dict(op=op, shape=tuple(r['shape']), chip=chip.name, future_970dt_us=future,
                             ca=r['ca_us'], **time))
    pairs = []
    for op in DTYPES:
        by_key = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if op == r['op']:
                by_key[r['shape']][r['chip']].append(r)
        for shape, chips in by_key.items():
            if not ({DT, PR} <= chips.keys()):
                continue
            def ratio(field):
                return median(r[field] for r in chips[DT]) / median(r[field] for r in chips[PR])
            pairs.append(dict(op=op, shape=shape, **{f: ratio(f) for f in ('ca', *MODES)}))
    summary = {}
    for op in DTYPES:
        sub = [r for r in pairs if r['op'] == op]
        metrics = {}
        for m in MODES:
            # Relative error on each speed ratio weights each geometry once.
            errors = [100*abs(r[m]/r['ca'] - 1) for r in sub]
            non_ties = [r for r in sub if abs(r['ca']-1) > .01]
            sign = lambda x: 1 if x>1.01 else -1 if x<.99 else 0
            metrics[m] = dict(mean_abs_speedup_error_pct=sum(errors)/len(errors) if errors else None,
                              clear_observed_directions=len(non_ties),
                              correct_direction=sum(sign(r[m]) == sign(r['ca']) for r in non_ties))
        per_chip = {}
        for chip in (DT, PR):
            chosen = [r for r in rows if (r['op'],r['chip']) == (op,chip)]
            per_chip[chip] = {m: wape([r['ca'] for r in chosen], [r[m] for r in chosen]) for m in MODES}
        summary[op] = {'paired_shapes': len(sub), 'speed_ratio': metrics, 'whole_window_wape_pct': per_chip}
    return {'role':'archive replay; duplicate CA windows of each SKU/shape collapsed by median for speed ratios',
            'summary':summary, 'pairs':pairs, 'windows':rows}


def sensitivity():
    future = load_chip(FUTURE)
    profiles = {'nominal': future}
    for name, options in {'longer_exp_II': {'fa_vf_exp_ii_cycles':8},
                          'narrow_issue': {'fa_vf_issue_width':1},
                          'half_read_port': {'fa_mte_bytes_per_cycle':40},
                          'two_vector_workers': {'fa_aiv_per_aic':2}}.items():
        profiles[name] = replace(future, details={**future.details, **options})
    obs = []
    for shape in SCENARIOS:
        for scenario, chip in profiles.items():
            for mode in MODES:
                r = evaluate('fa', shape, chip, mode)
                w = r.projection['timing_diagnostics']['wave_types'][0]
                repeat = w['repeat'] or w['first']
                obs.append(dict(shape=shape, scenario=scenario, mode=mode, predicted_us=r.latency_us,
                                workers=r.active_aiv, repeat_vector_cycles=repeat['vec1'] + repeat['vec2'],
                                repeat_bound=repeat['steady_bottleneck']))
    vector_trials = []
    for op, shape, field, ii in [('cast_bf16_fp32',(1024,4096),'vf_convert_ii_cycles',2),
                                 ('sigmoid_fp32',(16384,48),'vf_exp_ii_cycles',8),
                                 ('quant_token_fp8',(144,5120),'vf_divide_ii_cycles',8)]:
        slow_unit = replace(future, details={**future.details,field:ii})
        for mode in MODES:
            nominal = evaluate(op,shape,future,mode).latency_us
            changed = evaluate(op,shape,slow_unit,mode).latency_us
            vector_trials.append(dict(op=op, shape=shape, mode=mode, field=field, ii=ii,
                                      predicted_us=nominal, changed_us=changed,
                                      change_pct=100*(changed/nominal-1)))
    return {'status':'970DT scenario only; one input change per run; no target CA',
            'observations':obs, 'non_fa':vector_trials}


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    transfer = json.loads((OUT/'vector-transfer.json').read_text())['summary']
    costs = json.loads((OUT/'vector-cost.json').read_text())['results']
    ticks = list(range(len(DTYPES)))
    labels = ['FA', 'Cast', 'Sigmoid', 'Token quant']
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
    for i, (mode, color, label) in enumerate((('original','#247878','Original'),
                                              ('classes','#b35d32','Instruction classes'))):
        xs = [j + (i-.5)*.36 for j in ticks]
        for ax, sku, title in zip(axes[:2], (DT, PR), ('950DT', '950PR')):
            ax.bar(xs, [transfer[op]['whole_window_wape_pct'][sku][mode] for op in DTYPES],
                   .35, color=color, label=label)
            ax.set_title(title + ' · archive replay')
            ax.set_ylabel('Whole-window WAPE (%)')
        cost = {r['op']: r['steady']['p95_us'] for r in costs if r['mode'] == mode}
        axes[2].bar(xs, [cost[op] for op in DTYPES], .35, color=color)
    axes[2].set_title('Public estimate · warm process')
    axes[2].set_ylabel('CPU wall time p95 (µs)')
    for ax in axes:
        ax.set_xticks(ticks, labels, rotation=16)
        ax.grid(axis='y',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.savefig(ROOT/'docs/figures/vector-method-study.png',dpi=160)
    plt.close(fig)

    folder = OUT/'fa-validation'
    holdouts = [row for group in (folder, folder/'long') for row in
                json.loads((group/'comparison.json').read_text())['rows']]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.1), constrained_layout=True)
    for col, sku in enumerate((DT, PR)):
        rows = [row for row in holdouts if row['chip'] == sku]
        for ax, metric, title in zip(axes[:, col], ('vf_difference_pct', 'device_error_pct'),
                                    ('Busy-AIV RV union', 'Whole operation')):
            for key, label, color in [('calibrated','Original','#277c8b'),('v21_classes','Classes v2.1','#766c94'),
                                      ('instruction_classes','Classes v2.2','#bc6229')]:
                ax.plot(range(len(rows)), [r[key][metric] for r in rows], marker='o',label=label,color=color)
            ax.axhline(0,color='black',linewidth=.65)
            ax.set_xticks(range(len(rows)), [str(r['shape'][1]) for r in rows])
            ax.set_title(sku.split('_')[0] + ' · ' + title)
            ax.grid(alpha=.2)
            ax.set_xlabel('KV length (last point selected separately)')
    axes[0,0].legend(fontsize=7)
    for ax in axes[:,0]:
        ax.set_ylabel('Signed deviation (%)')
    fig.savefig(ROOT/'docs/figures/fa-v22-holdout.png',dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--first', nargs=3, metavar=('OP','MODE','SHAPE'))
    cli.add_argument('--study', choices=('cost','transfer','sensitivity','plot'), default='transfer')
    args = cli.parse_args()
    if args.first:
        op, mode, shape = args.first
        print(latency(op, tuple(map(int,shape.split(','))),load_chip(DT),mode)[0])
    elif args.study == 'plot':
        plot()
    else:
        fn = {'cost':timed_study,'transfer':replay,'sensitivity':sensitivity}[args.study]
        OUT.mkdir(exist_ok=True, parents=True)
        (OUT / ('vector-' + args.study + '.json')).write_text(json.dumps(fn(),indent=2)+'\n')
