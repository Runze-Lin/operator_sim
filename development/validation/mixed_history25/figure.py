"""Plot three launches of the same RMS tile using frozen model stages and real descriptors."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.mixed_history25.campaign import OUT, write
from development.validation.rms_ordered_verify.verify import CLOCK, digest, events
from operator_model import load_chip
from operator_model.rms_dispatch import price_rms

PIPES = ('Front end', 'MTE2', 'Vector', 'MTE3')
COLORS = ('#8296a3', '#127b87', '#7052a6', '#b64d28')
OPS = {0x3e9c: ('tiling', 1), 0x4878: ('gamma', 1), 0x4a48: ('x', 1),
       0x4d1c: ('sum', 2), 0x4e64: ('rstd', 2), 0x5008: ('normalize', 2),
       0x4f38: ('rstd write', 3), 0x5128: ('y write', 3)}
LABELS = {1: 'First RMS\nafter Addmm', 2: 'Immediate\nRMS repeat', 4: 'RMS repeat\nafter Addmm'}


def collect(chip):
    receipt = json.loads((OUT / f'development-paired-{chip}-windows.json').read_text())
    assert all(r['qualified'] for r in receipt['rows'])
    rows = []
    for r in receipt['rows']:
        if r['i'] not in LABELS:
            continue
        assert r['shape'] == [2, 2944]
        core = max(r['core_windows'], key=lambda c: r['core_windows'][c][1])
        shard = ROOT / r['trace']
        base = int((shard.parent.parent / 'dump/pc_start_addr.txt').read_text(), 16)
        shard = shard.parent / core / 'trace.json'
        assert digest(shard) == r['shards'][core]
        actual = []
        for e in events(shard):
            if e['name'] != 'VF' and not e['name'].startswith('MOV_SRC_TO_DST'):
                continue
            pc = int(e['args']['pc_addr'], 16) - base
            name, lane = OPS[pc]
            start = e['ts'] - r['first_tick'] / CLOCK
            actual.append(dict(name=name, pc=hex(pc), pipe=PIPES[lane], start=start,
                               end=start+e['dur'], bytes=json.loads(e['args']['detail']).get('processed_bytes')))
        assert len(actual) == len(OPS) == len({a['pc'] for a in actual})
        pred, _, _, diag = price_rms(r['shape'], load_chip(chip), {'resident_aiv': r['previous_aiv']})
        assert abs(pred-r['preserve_us']) < 1e-10
        start = 0
        model = []
        for name, duration in diag['stages_us'].items():
            pipe = next((k for k in PIPES if name.startswith(k)), None)
            pipe = pipe or ('Vector' if name.startswith('SIMD') else 'Front end')
            if duration:
                model.append(dict(name=name, pipe=pipe, start=start, end=start+duration))
            start += duration
        rows.append(dict(i=r['i'], previous_aiv=r['previous_aiv'], core=core, model_us=pred,
                         ca_us=r['ca_us'], unannotated_us=r['public_us'], model=model, ca=actual,
                         core_start_us=(r['core_windows'][core][0]-r['first_tick'])/CLOCK,
                         trace=r['trace'], trace_sha256=r['trace_sha256'], shard_sha256=r['shards'][core]))
    return rows


def draw(chip, rows, output):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/mixed25-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.9), sharex=True)
    fig.subplots_adjust(left=.18, right=.875, bottom=.145, top=.80, hspace=.43)
    ink = '#19303c'
    fig.text(.18,.95,f'{chip[6:11]} RMSNorm  |  2 × 2944',size=21,weight='bold',color=ink)
    fig.text(.18,.90,'2 AIVs  ·  1 BF16 row per core  ·  3 launches on the same stream',size=12,color='#415a67')
    fig.legend(handles=[Patch(facecolor=c,label=p) for p,c in zip(PIPES,COLORS)],
               loc='upper left',bbox_to_anchor=(.172,.887),ncol=4,frameon=False)
    for ax, predicted, title in zip(axes, (True,False), ('Resource model · exposed critical path','CA trace · last AIV to retire')):
        ax.set_title(title,loc='left',pad=7,weight='bold',color=ink,fontsize=13)
        ax.set(xlim=(0,4.5),ylim=(2.55,-.55),yticks=range(3),yticklabels=[LABELS[r['i']] for r in rows])
        ax.spines[['top','left','right']].set_visible(False)
        ax.grid(axis='x',color='#dde6eb')
        ax.set_axisbelow(True)
        ax.tick_params(axis='y',length=0,pad=12)
        for y,r in enumerate(rows):
            ax.axhline(y+.49,color='#e4ecf0',lw=.8)
            span=r['model_us' if predicted else 'ca_us']
            ax.vlines(span,y-.38,y+.38,color=ink,linestyles='--',lw=1)
            ax.text(4.56,y,f'{span:.3f} µs',va='center',color=ink,weight='bold',fontsize=10)
            if not predicted:
                ax.plot(r['core_start_us'],y-.28,marker='|',color=COLORS[0],ms=10)
                ax.text(4.56,y+.27,r['core'],fontsize=7,color='#51646b')
            for bar in r['model' if predicted else 'ca']:
                lane=PIPES.index(bar['pipe'])
                offset=(lane-1.5)*.155
                if not predicted:
                    # Same-pipe descriptors have individual visible origins even when overlapping.
                    offset += {'gamma':-.04,'x':.04,'rstd write':-.04,'y write':.04}.get(bar['name'],0.)
                a,b=bar['start'],bar['end']
                ax.barh(y+offset,b-a,left=a,height=.10,color=COLORS[lane])
                if not predicted:
                    ax.plot(a,y+offset,marker='|',ms=4.4,color=COLORS[lane])
    axes[-1].set_xlabel('Device time from the first instruction of each launch (µs)')
    fig.text(.18,.04,'Upper bars sum to the frozen device prediction. Lower bars are issue-to-retirement windows and can overlap.\n'
             'Gray ticks mark the first instruction of the last AIV. Dashed segments mark the end across all cores.',fontsize=9,color='#425969')
    fig.savefig(output,dpi=155)
    plt.close(fig)


if __name__=='__main__':
    for chip,tag in [('Ascend950DT_9582','dt'),('Ascend950PR_957b','pr')]:
        rows=collect(chip)
        write(OUT/f'{tag}-rms-tile.json',rows)
        draw(chip,rows,OUT/f'{tag}-rms-tile.png')
