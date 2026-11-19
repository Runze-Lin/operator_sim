"""Use actual paired terminal-core GM and UB addresses; retain the common time origin."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.cast22.slot_figure import measured
from operator_model import estimate

OUT = ROOT / 'results/v0.25/cache'
SOURCE = ROOT / 'results/v0.23/prefill/flash-128ki-pr.json'
STAGES = ('Read', 'Vector', 'Write')


def load(path):
    return json.loads(path.read_bytes())


def evidence():
    d = load(SOURCE)
    job = load(ROOT / d['receipt'])
    job['api'] = 'aclnnCast'
    m = measured(ROOT, job)
    estimate_us = estimate(d['op'], d['shape'], chip=d['chip'], dtype='bf16').latency_us
    assert math.isclose(estimate_us, d['model_us'])
    assert m['core'] == d['ca_core'] and m['all_core_trace_sha256'] == d['trace_sha256']
    rows = []
    for i, (raw, archival, model) in enumerate(zip(m['tiles'], d['ca_tiles'], d['model_tiles'], strict=True)):
        actual = dict(Read=raw['MTE2']['window'], Vector=raw['vcvt'], Write=raw['MTE3']['window'])
        for stage in STAGES:
            assert max(abs(a-b) for a,b in zip(archival[stage],actual[stage])) < 0.001
        deltas = {s:dict(start=model[s][0]-actual[s][0], retirement=model[s][1]-actual[s][1],
                         duration=(model[s][1]-model[s][0])-(actual[s][1]-actual[s][0])) for s in STAGES}
        rows.append(dict(tile=i+1,slot=raw['slot'],elements=raw['n'],model_elements=d['model_tile_elements'][i],
                         raw=raw,ca_us=actual,rate_equivalent_us=model,model_minus_ca_us=deltas))
    assert len(rows)==29 and m['active_aiv']==56
    return dict(model=d['model_name'],chip=d['chip'],shape=d['shape'],dtype='bf16 → fp32',signed_pct=100*(estimate_us/d['ca_us']-1),
                model_us=estimate_us,ca_us=d['ca_us'],projection_970dt_us=d['projection_970dt_us'],op=d['op'],
                first_mte2_issue_gap_us=rows[0]['model_minus_ca_us']['Read']['start'],
                tile_15_mte2_issue_gap_us=rows[14]['model_minus_ca_us']['Read']['start'],
                actual_last_core=m['core'],active_aiv=m['active_aiv'],kernel=m['kernel'],correctness=m['correctness'],
                trace_sha256=m['all_core_trace_sha256'],binary_sha256=m['binary_sha256'],shard=m['shard'],shard_sha256=m['shard_sha256'],
                receipt=d['receipt'],intervals=str(SOURCE.relative_to(ROOT)),intervals_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                ca_wall_seconds=m['wall_s'],cost_new_ca_seconds=0,rows=rows[-3:],
                origin='Earliest X instruction across the entire launch; shared for all device times, no realignment.',
                model_semantics='Equivalent service schedule: the exposed steady time is allocated by tile payload; issue offsets are hypotheses, not hardware timestamps.',
                measured_semantics='Positive-duration MTE2/MTE3 descriptor issue→retire. Vector is the first-to-last positive VCVT in the matching address group. Aggregate timing is max(all retire) – min(all issue).')


def draw(d):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/operator-sim-cache25')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator
    ink='#192f3e'
    colors={'Read':'#137d85','Vector':'#7652a3','Write':'#b74c29'}
    fig,axs=plt.subplots(2,1,figsize=(11.4,6.9),sharex=True)
    left=.20
    fig.subplots_adjust(left=left,right=.963,top=.751,bottom=.178,hspace=.62)
    fig.text(left,.952,'950PR Cast  |  Flash B8 × 128k prefill',fontsize=21,color=ink,weight='bold')
    fig.text(left,.90,'BF16 → FP32   ·   1,048,576 × 32   ·   56 AIVs   ·   last three of 29 UB blocks',fontsize=12,color='#476170')
    fig.legend(handles=[Patch(color=colors[s],label=l) for s,l in zip(STAGES,('MTE2 / GM → UB','Vector / VCVT','MTE3 / UB → GM'))],
               loc='upper left',bbox_to_anchor=(left-.009,.870),frameon=False,ncol=3,fontsize=10)
    for ax,key,figtitle,t in zip(axs,('rate_equivalent_us','ca_us'),('Resource model  ·  rate-equivalent block service',
                'CA trace  ·  '+d['actual_last_core']+' (last to finish)'),(d['model_us'],d['ca_us'])):
        for i,row in enumerate(d['rows']):
            for j,s in enumerate(STAGES):
                a,b=row[key][s];ax.barh(i+(j-1)*.20,b-a,left=a,height=.15,color=colors[s],zorder=3)
            if i<2:ax.axhline(i+.5,color='#dfe6e9',lw=.75)
        ax.axvline(t,color=ink,ls=(0,(3,3)),lw=1)
        ax.set_ylim(2.5,-.5)
        n='model_elements' if key=='rate_equivalent_us' else 'elements'
        ax.set_yticks(range(3),[f"UB {r['tile']}  ·  slot {r['slot']}\n{r[n]:,} elements" for r in d['rows']],fontsize=9.1)
        ax.grid(axis='x',color='#dbe4e9',lw=.8);ax.set_axisbelow(True);ax.spines[:].set_visible(False);ax.tick_params(axis='y',length=0,pad=10)
        ax.set_title(figtitle,loc='left',fontsize=13,fontweight='bold',color=ink);ax.set_title(f'{t:.3f} µs',loc='right',fontsize=14,fontweight='bold',color=ink)
    minx=min(r[k]['Read'][0] for r in d['rows'] for k in ('rate_equivalent_us','ca_us'))
    axs[-1].set_xlim(math.floor(minx)-.4,math.ceil(max(d['model_us'],d['ca_us']))+.65);axs[-1].xaxis.set_major_locator(MultipleLocator(2))
    axs[-1].set_xlabel('Device time from the first instruction on any AIV (µs)',labelpad=8)
    fig.text(left,.079,'Model bars allocate shared service by payload. CA matches GM byte ranges, two UB slots and every positive-duration VCVT.',fontsize=9,color='#476170')
    fig.text(left,.047,'The final MTE2 starts 19.870 µs late in the model. Its predicted response is shorter than the observed descriptor lifetime.',fontsize=9,color='#476170')
    fig.savefig(OUT/'flash-128ki-pr-tail.png',dpi=160,facecolor='white');plt.close(fig)


if __name__=='__main__':
    d=evidence();(OUT/'flash-128ki-pr-tail.json').write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');draw(d)
