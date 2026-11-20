"""Compare a qualified B-resident kernel's critical AIC issue intervals with the frozen cycle schedule."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import accumulate
import json
from pathlib import Path
from statistics import mean, median
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/v0.29/matmul/full-load'
sys.path[:0] = [str(ROOT),str(ROOT/'development/validation/sku17')]
from campaign_collect import events


def read(path):
    return json.loads(path.read_bytes())


def inspect(receipt, registration):
    measured = read(receipt)
    acquisition = ROOT / registration['acquisition']
    initial = read(acquisition)
    shape, chip = registration['shape'], registration['chip']
    assert initial['shape'] == shape and initial['chip'] == chip
    if receipt != acquisition:
        assert measured['acquisition_sha256'] == sha256(acquisition.read_bytes()).hexdigest()
    assert measured['qualified']
    stage, = measured['stages']
    assert stage['kernel'].endswith('_131074') and stage['active_aiv'] == 0
    assert measured['correctness'] == [f"correctness count={shape[0]*shape[1]} bad=0"]
    critical = max(stage['windows'],key=lambda key:stage['windows'][key][1])
    path = (ROOT / stage['trace']).parent / critical / 'trace.json'
    candidate = registration['candidate']
    tick_rate = 1000 * read(ROOT/'configs/chips'/(chip+'.json'))['frequency_ghz']
    windows = defaultdict(list)
    geometry = Counter()
    for e in events(path):
        name = e['name']
        if e['ph'] != 'X' or name not in ('MMAD','FIX_L0C_TO_DST','MOV_OUT_TO_L1_MULTI_ND2NZ'):
            continue
        detail = json.loads(e['args']['detail'])
        if any(detail.get(k) for k in ('is_wait','is_sync','is_duration_excluded')):
            continue
        if name == 'MOV_OUT_TO_L1_MULTI_ND2NZ':
            name = {256:'A page',512:'B matrix'}[detail['n']]
        if name == 'MMAD':
            geometry[detail['m'],detail['n'],detail['k']] += 1
        windows[name].append((round(e['ts']*tick_rate),round((e['ts']+e['dur'])*tick_rate)))
    windows = {k:sorted(v) for k,v in windows.items()}
    per_tile = candidate['mmads_per_tile']
    tile_count = len(windows['FIX_L0C_TO_DST'])
    assert len(windows['B matrix'])==1 and len(windows['A page'])==4*tile_count
    assert geometry == {tuple(candidate['tile']):per_tile*tile_count}
    firsts = [start for start,end in windows['MMAD'][::per_tile]]
    steps = [b-a for a,b in zip(firsts,firsts[1:])]
    within = [r[0]-l[0] for i,(l,r) in enumerate(zip(windows['MMAD'],windows['MMAD'][1:])) if (i+1)%per_tile]
    boundary = [windows['MMAD'][i][0]-windows['MMAD'][i-1][0]
                for i in range(per_tile,len(windows['MMAD']),per_tile)]
    nominal = candidate['cycles']['cube_issue']
    # Actual start-to-start delays inside and between output tiles can be larger
    # than the MMAD initiation interval even when the accumulated FLOPs agree.
    groups = []
    for i in range(4):
        part=steps[i*len(steps)//4:(i+1)*len(steps)//4]
        if part:
            groups.append(dict(tiles=len(part),mean_cycles=mean(part),median_cycles=median(part)))
    with path.open('rb') as f:
        h=sha256()
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return dict(shape=shape,chip=chip,critical_aic=critical,critical_tile_count=tile_count,
                core_roster=stage['active_aic'],mmads_whole=stage['mmad'],ca_us=stage['ca_us'],
                legacy_us=registration['legacy_us'],frozen_candidate_us=candidate['latency_us'],
                all_stores_cold_sensitivity_us=candidate['all_output_writeback_us'],
                predicted_tile_interval_cycles=candidate['cycles']['per_aic_step'],
                observed_tile_intervals_cycles=steps,interval_quarters=groups,
                observed_interval_mean_cycles=mean(steps),observed_interval_median_cycles=median(steps),
                minimum_mmad_interval_cycles=min(within),median_mmad_interval_cycles=median(within),
                within_tile_mmad_longer_than_nominal=sum(v>nominal for v in within),
                within_tile_extra_issue_cycles=sum(v-nominal for v in within),
                between_tile_extra_issue_cycles=sum(v-nominal for v in boundary),
                time_from_first_to_last_tile_issue_us=(firsts[-1]-firsts[0])/tick_rate,
                predicted_first_to_last_tile_issue_us=(tile_count-1)*candidate['cycles']['per_aic_step']/tick_rate,
                observed_weights_loaded_by_last_aic=len(windows['B matrix']),cycles_per_us=tick_rate,
                source_receipt=str(receipt.relative_to(ROOT)),receipt_sha256=sha256(receipt.read_bytes()).hexdigest(),
                aggregate_trace_sha256=stage['trace_sha256'],critical_trace_sha256=h.hexdigest(),
                freeze_sha256=sha256((OUT/'freeze.json').read_bytes()).hexdigest())


def comparison(current, previous):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    profiles = [read(previous),current]
    fig, axes = plt.subplots(len(profiles), 1, figsize=(10, 5.8), layout='constrained')
    for ax, profile in zip(axes,profiles):
        steps=profile['observed_tile_intervals_cycles']
        ax.plot(range(len(steps)+1),[x/profile['cycles_per_us'] for x in accumulate(steps,initial=0)],
                marker='.',markersize=2.7,color='#116e79',label='CA critical AIC')
        ax.plot(range(len(steps)+1),[i*profile['predicted_tile_interval_cycles']/profile['cycles_per_us']
                for i in range(len(steps)+1)],color='#71539b',linestyle='--',label='Frozen resource model')
        ax.set_title(f"{profile['chip']}   M = {profile['shape'][0]:,}   ·   {profile['critical_aic']}",loc='left',fontsize=11)
        ax.set_ylabel('Elapsed time (µs)')
        ax.set_xlabel('Output tile index on this AIC')
        ax.legend(frameon=False,loc='upper left')
        ax.grid(axis='y',alpha=.25)
        ax.spines[['right','top']].set_visible(False)
    fig.suptitle('Time from the first MMAD to the start of each output tile',fontsize=15)
    fig.savefig(OUT/'interval-scaling.png',dpi=165)
    plt.close(fig)


if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument('receipt',type=Path,help='Qualified acquisition receipt, relative to repository root')
    parser.add_argument('--compare',type=Path,help='Previous interval JSON to show in the example figure')
    args=parser.parse_args()
    row,=[r for r in read(OUT/'freeze.json')['rows'] if r['acquisition']==str(args.receipt)]
    receipt=ROOT/args.receipt
    supplemental=receipt.with_name(receipt.stem+'-drain.json')
    info=inspect(supplemental if supplemental.exists() else receipt,row)
    output=OUT/(args.receipt.stem+'-intervals.json')
    output.write_text(json.dumps(info,indent=2)+'\n')
    if args.compare:
        comparison(info,args.compare)
    print({k:v for k,v in info.items() if k not in ('observed_tile_intervals_cycles','interval_quarters')})
