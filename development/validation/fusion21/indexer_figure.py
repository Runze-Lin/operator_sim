"""Reconstruct the frozen overlap template in original event order; compare to 950PR CA.

Replot from committed arrays: python indexer_figure.py --replot
Re-extract with the archived trace.json/core shards and the calibration databases present.
"""
from collections import Counter, defaultdict
from hashlib import sha256
from math import isclose
from pathlib import Path
import argparse
import json
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'development/validation/sku20')]
from timelines import trace_lanes, union
from operator_model import estimate, load_chip
from operator_model.fusion_v8 import work
from operator_model.predictor import _profile
from operator_model.transfer import _queue_factors

OP = 'lightning_indexer'
SHAPE = [1,2,3072,8,128]
PR = 'Ascend950PR_957b'
STEM = 'lightning-indexer-3072-pr'
DATA = ROOT/'results/v0.21/fusion'/f'{STEM}.json'
FIG = ROOT/'docs/examples/figures'/f'{STEM}.png'
QUEUES = {'AIC':('MTE2','MTE1','CUBE','FIXP','SCALAR'), 'AIV':('MTE2','SIMD','MTE3','SCALAR')}
COLORS = {'MTE2':'#15798a','MTE1':'#5173ac','CUBE':'#7150a0','FIXP':'#b34d2a',
          'SCALAR':'#718392','SIMD':'#6553a1','MTE3':'#cd6a28'}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def projected_core(db, ticks, lane, rate):
    """Apply the frozen max-over-coactive rule to each original chronological interval."""
    changes = defaultdict(Counter)
    end_tick = ticks + lane['end_ticks']
    changes[end_tick]
    sql = """SELECT ExecInstrTickStart, ExecInstrTickEnd, json_extract(ExtendParams,'$.queue_type')
             FROM ExecutedInstructions WHERE CoreTypeId=? AND ExecInstrCoreId=?
             AND ExecInstrIsDurationExcluded=0"""
    for a,b,q in db.execute(sql,(lane['core_type_id'],lane['core_id'])):
        changes[a][q] += 1
        changes[b][q] -= 1
    active = Counter()
    spans = defaultdict(list)
    prev, elapsed = ticks, 0.
    for tick, change in sorted(changes.items()):
        mask = [q for q,n in active.items() if n] or ['IDLE']
        end = elapsed + (tick-prev)/1650 * max(rate[q] for q in mask)
        if end>elapsed:
            for q in mask:
                spans[q].append([elapsed,end])
        active.update(change)
        prev, elapsed = tick, end
    return dict(end_us=elapsed, lanes={q:union(intervals) for q,intervals in spans.items()})


def extract():
    pair = next(p for p in read(ROOT/'results/v0.17/sku/pairs.json') if p['op']==OP and p['shape']==SHAPE)
    attrs = pair['attrs']
    actual = pair['pr']['stages'][0]
    public = estimate(OP,SHAPE,attrs,chip=PR)
    assert isclose(public.latency_us,pair['pr_us'],abs_tol=1e-8)
    assert actual['ca_us']==pair['pr_ca_us']
    # Exact call contract of transfer.project's single operator _template branch.
    factors,_ = _queue_factors(OP,SHAPE,attrs,{'predicted_us':public.projection['reference_latency_us']},
                               public.work,load_chip(),load_chip(PR),[])
    demand = work(OP,SHAPE,attrs)
    candidates = []
    kernels = {}
    for p in (ROOT/'runs/model-v7').glob('lightning_indexer*/result.json'):
        r = read(p)
        if r['status']=='ok':
            k, = r['measurement']['kernels']
            kernels[k['db_sha256']] = k
    for template in _profile('results/v0.8/fusion/model.json')['templates'][OP]:
        k = kernels[template['db_sha256']]
        path = Path(k['db_path'])
        assert digest(path)==template['db_sha256']
        rate = {q:factors.get(q,factors['SCALAR'])*v/template['work'][q] for q,v in demand.items()}
        offline = max(sum(us*max(rate[q] for q in mask.split('+')) for mask,us in phase.items())
                      for phase in template['cores'])
        lanes = []
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
            earliest = db.execute('SELECT MIN(ExecInstrTickStart) FROM ExecutedInstructions').fetchone()[0]
            for kind in (1,2,3):
                last = max((c for c in k['cores'] if c['core_type_id']==kind),key=lambda c:c['end_ticks'])
                lanes.append(dict(kind=kind,ref_core_id=last['core_id'],**projected_core(db,earliest,last,rate)))
        terminal = max(c['end_us'] for c in lanes)
        assert isclose(offline,terminal,abs_tol=1e-7)
        candidates.append(dict(shape=template['shape'],template_us=terminal,db=str(path.relative_to(ROOT)),
                               sha256=template['db_sha256'],rates=rate,cores=lanes))
    selected=sorted(candidates,key=lambda r:r['template_us'])[1]
    assert isclose(selected['template_us'],public.latency_us,abs_tol=1e-7)
    predicted={f:max([c for c in selected['cores'] if (c['kind']==1)==(f=='AIC')],key=lambda c:c['end_us'])
               for f in QUEUES}
    file=ROOT/actual['trace']
    assert digest(file)==actual['trace_sha256']
    cores=trace_lanes(actual,file,0,actual['ca_us'])
    observed={('AIC' if 'cubecore' in c['core'] else 'AIV'):c for c in cores}
    # Preserve full windows and DMA/MMAD issue times; reduce interval precision
    # only after looking up phases and verifying the double-precision result.
    def compact(lanes):
        return {q:[[round(a,5),round(b,5)] for a,b in intervals] for q,intervals in lanes.items()
                if q in COLORS}
    out=dict(shape=SHAPE,attrs=attrs,chip=PR,api=pair['pr']['api'],installed_kernel=actual['kernel'],
             ca_trace=actual['trace'],ca_sha256=actual['trace_sha256'],ca_us=actual['ca_us'],pred_us=public.latency_us,
             target_core_jobs=[public.active_aic,public.active_aiv],ca_launched=[actual['active_aic'],actual['active_aiv']],
             work=demand,physical_whole_chip_mmad=actual['mmad'],
             checks=pair['pr']['correctness'],template_candidates=[{k:r[k] for k in ('shape','db','sha256','template_us')} for r in candidates],
             selected_template=selected['sha256'],selected_queue_multipliers=selected['rates'],model={},ca={})
    for family in QUEUES:
        c=predicted[family];out['model'][family]=dict(source_core=c['ref_core_id'],source_core_type=c['kind'],
                    end_us=c['end_us'],lanes=compact(c['lanes']))
        c=observed[family];out['ca'][family]=dict(core=c['core'],shard=c['file'],sha256=c['sha256'],window=c['window'],
                    lanes=compact(c['lanes']),mmad_issues_us=c['issues'].get('CUBE',[]))
    DATA.write_text(json.dumps(out,indent=2)+'\n')
    return out


def draw(d):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    fig,axes=plt.subplots(2,1,figsize=(12.4,8.7),sharex=True)
    fig.subplots_adjust(left=.19,right=.965,top=.815,bottom=.12,hspace=.38)
    dark='#18313e'
    fig.text(.19,.956,'950PR Lightning Indexer  |  Q = 2, KV = 3072',fontsize=20,color=dark,weight='bold')
    fig.text(.19,.917,'8 heads  ·  head dim 128  ·  causal Top-512  ·  BF16',fontsize=12,color='#465e6f')
    fig.legend(handles=[Patch(color=COLORS[q],label=q) for q in COLORS],loc='upper left',
               bbox_to_anchor=(.18,.899),ncol=7,frameon=False,fontsize=9)
    rows=[(family,q) for family,queues in QUEUES.items() for q in queues]
    for ax,label,group,duration in zip(axes,['Model · scheduled queues of median DT template',
                                            'CA trace · terminal AIC and terminal AIV'],
                                       [d['model'],d['ca']],[d['pred_us'],d['ca_us']]):
        ca=group is d['ca']
        ax.set_title(label,loc='left',pad=12,fontsize=12,weight='bold',color=dark)
        ax.set_title(f'{duration:.3f} µs',loc='right',pad=12,fontsize=14,weight='bold',color=dark)
        for j,(family,q) in enumerate(rows):
            for a,b in group[family]['lanes'].get(q,[]):
                ax.broken_barh([(a,b-a)],(j-.17,.34),facecolors=COLORS[q],linewidths=0)
            if q=='CUBE' and ca:
                ax.vlines(group[family]['mmad_issues_us'],j-.25,j+.25,color='#39255a',lw=.8)
        ax.axhline(4.5,lw=.8,color='#d2dce3')
        ax.set_yticks(range(len(rows)),[f'{f} / {q}' for f,q in rows],fontsize=9)
        ax.set_ylim(len(rows)-.5,-.58);ax.grid(axis='x',lw=.65,color='#d2dce3');ax.set_axisbelow(True)
        ax.axvline(duration,color='#415a69',ls=(0,(3,3)),lw=.85)
        ax.set_xlim(0,1.035*max(d['ca_us'],d['pred_us']))
        ax.tick_params(axis='y',length=0,pad=8)
        for spine in ax.spines.values():spine.set_visible(False)
    axes[1].set_xlabel('Device time since first instruction on any core (µs)',fontsize=11)
    fig.text(.19,.049,'CA: '+d['ca']['AIC']['core']+' / '+d['ca']['AIV']['core']+
             '. Dark ticks on the Cube row mark measured MMAD issues.',fontsize=9,color='#41596a')
    fig.text(.19,.025,'Model intervals preserve DT calibration order. A stretched interval represents queue occupancy; it does not predict individual MMAD issues.',
             fontsize=8.6,color='#41596a')
    fig.savefig(FIG,dpi=148,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--replot',action='store_true')
    args=parser.parse_args()
    draw(read(DATA) if args.replot else extract())
