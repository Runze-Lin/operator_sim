"""Rebuild the worked-example timeline from the published phase receipts and estimate."""
import argparse
from pathlib import Path

from calibration.quant_update.bundle import DEFAULT, load
from operator_model import estimate


def draw(data, output):
    import matplotlib.pyplot as plt

    e = estimate('quant_mxfp4', (9, 288), chip='Ascend950DT_9582')
    clock = 1650
    wave = e.projection['critical_wave']
    tile = wave['tile']
    ready = clock*e.projection['stages_us']['first_issue'] + wave['arrival_cycles']
    read = ready + tile['ub_clear_cycles']
    vector = ready + tile['read_cycles']
    stores = vector + tile['vector_cycles']
    scale = clock*(e.latency_us - e.projection['stages_us']['drain'])
    model = [((read, vector),), ((ready, read), (vector, stores)),
             ((stores, stores+tile['data_write_cycles']),), ((stores, scale),)]
    last = max(data['development_core_phases']['9x288-dt'], key=lambda c: c['end'])
    measured = [((last['first_read'], last['read_retire']),),
                ((last['first_vector'], last['last_vector']),),
                ((last['write_issue'], last['data_retire']),), ((last['scale_issue'], last['scale_retire']),)]
    labels = ['Read · 64 B', 'Vector', 'Write packed · 16 B', 'Write scale · 2 B']
    colors = ['#007d83','#7757a6','#bf673a','#a14124']
    plt.rcParams['svg.fonttype']='none'
    plt.rcParams['svg.hashsalt']='quant-mxfp4-950dt'
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 4.8), sharex=True)
    views = [('Model · logical owner '+str(wave['owner']), e.latency_us, model),
             ('CA · '+last['core'], last['end']/clock, measured)]
    for index, (ax, (name, end_us, rows)) in enumerate(zip(axes, views)):
        ax.set_title(name, loc='left', fontsize=12)
        ax.set_title(f'{end_us:.3f} µs', loc='right', fontsize=12)
        for y, times in enumerate(rows):
            for start, end in times:
                # CA only identifies the first and last RVECEX in this compressed record.
                ax.barh(y, (end-start)/clock, left=start/clock, color=colors[y], height=.37,
                        alpha=.25 if y==1 and index else .95)
        ax.axvline(end_us, color='#304960', linestyle='dashed', linewidth=1)
        ax.set_yticks(range(4), labels=labels)
        ax.invert_yaxis()
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color='#d5e0e7', linewidth=.7)
        ax.set_axisbelow(True)
        ax.set_xlim(0,5.25)
    fig.suptitle('950DT · BF16 → MXFP4 · 9 × 288', x=.18, y=.985, ha='left', fontsize=16, fontweight='bold')
    axes[-1].set_xlabel('Device time from earliest instruction on any AIV (µs)')
    fig.subplots_adjust(top=.83,bottom=.18,hspace=.53,left=.22,right=.96)
    fig.text(.22,.035,'Pale bar spans first to last RVECEX and includes idle time. Scale descriptors include waiting.',fontsize=8)
    fig.savefig(output, metadata={'Date':None})
    plt.close(fig)
    if output.suffix == '.svg':
        output.write_text('\n'.join(line.rstrip() for line in output.read_text().splitlines())+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=DEFAULT)
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('tail-timeline.svg'))
    args=p.parse_args()
    draw(load(args.bundle),args.output)
