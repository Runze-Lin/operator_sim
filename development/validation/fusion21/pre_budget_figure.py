"""Compare mHC Pre's exclusive resource budgets to the terminal PR core, without invented issue times."""
from collections import Counter,defaultdict
import argparse
from math import isclose
import json

from indexer_figure import ROOT, PR, digest, read
from operator_model import estimate,load_chip
from operator_model.transfer import _queue_factors
from operator_model import mhc_v9

DATA=ROOT/'results/v0.21/fusion/mhc-pre-exclusive-pr.json'
FIG=ROOT/'docs/examples/figures/mhc-pre-exclusive-pr.png'
CLASSES=('compute','vector','traffic','control')
PALETTE={'compute':'#7552a5','vector':'#597daa','traffic':'#167b86','control':'#788c99'}


def exclusive(qs):
    if 'CUBE' in qs:return 'compute'
    if 'SIMD' in qs:return 'vector'
    if any(q.startswith(('MTE','FIX')) for q in qs):return 'traffic'
    return 'control'


def extract():
    op='mhc_pre';shape=[1,4,128]
    pair=next(r for r in read(ROOT/'results/v0.17/sku/pairs.json')if r['op']==op and r['shape']==shape)
    attrs=pair['attrs'];k=pair['pr']['stages'][0]
    r=estimate(op,shape,attrs,chip=PR)
    factors,_=_queue_factors(op,shape,attrs,{'predicted_us':r.projection['reference_latency_us']},r.work,
                              load_chip(),load_chip(PR),[])
    branch,work=mhc_v9.partition('pre',shape)
    prices=mhc_v9.profile()['stages']['pre'][branch]
    qs={'compute':'CUBE','vector':'SIMD','traffic':'MTE2','control':'SCALAR'}
    budget={q:c['startup_us']*factors['SCALAR']+c['unit_us']*work[q]*factors[qs[q]] for q,c in prices.items()}
    assert isclose(sum(budget.values()),r.latency_us,abs_tol=1e-8)
    file=ROOT/k['trace'];assert digest(file)==k['trace_sha256']
    core=max(k['windows'],key=lambda v:k['windows'][v][1]);shard=file.parent/core/'trace.json'
    es=[e for e in read(shard)['traceEvents'] if e['ph']=='X']
    origin=min(e['ts'] for e in es)-k['windows'][core][0]
    changes=defaultdict(Counter);changes[0.];changes[k['ca_us']]
    unknown=[]
    for e in es:
        raw=e['args'].get('detail')
        if not raw:
            # Raw PR trace contains a few intrinsics without queue metadata.
            # Leave these with residual control/wait if no tagged queue covers them.
            unknown.append((e['name'],e['dur']))
            continue
        detail=json.loads(raw)
        if detail.get('is_duration_excluded') or e['dur']<=0:continue
        q=detail['queue_type']
        a=max(0.,e['ts']-origin);b=min(k['ca_us'],e['ts']+e['dur']-origin)
        if b>a:
            changes[a][q]+=1;changes[b][q]-=1
    prev=0.;active=Counter();ca={q:0. for q in CLASSES};segments=[]
    for time,delta in sorted(changes.items()):
        q=exclusive([q for q,n in active.items() if n])
        if time>prev:
            ca[q]+=time-prev
            if segments and segments[-1][0]==q:
                segments[-1][-1]=time
            else:segments.append([q,prev,time])
        active.update(delta);prev=time
    assert isclose(sum(ca.values()),k['ca_us'],abs_tol=1e-8)
    return dict(shape=shape,attrs=attrs,api=pair['pr']['api'],installed_kernel=k['kernel'],chip=PR,
        cashape_role='first Pre launch within serialized 3-kernel parent; only the first kernel window is plotted',
        model_us=r.latency_us,ca_us=k['ca_us'],core=core,ca_trace=k['trace'],sha256=k['trace_sha256'],
        shard=str(shard.relative_to(ROOT)),shard_sha256=digest(shard),
        priority='CUBE > SIMD > MTE*/FIX* > remaining, including idle. queue_type; remove duration_excluded.',
        model=budget,ca=ca,ordered_ca=[[q,round(a,6),round(b,6)] for q,a,b in segments],
        work=work,rate_parameters=prices,untyped_events=len(unknown),untyped_duration_upper_bound_us=sum(v for _,v in unknown),ca_physical_mmad=k['mmad'],correctness=pair['pr']['correctness'])


def draw(d):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    fig,(a,b)=plt.subplots(2,1,figsize=(11.5,5.6),gridspec_kw={'height_ratios':[1.35,1]})
    fig.subplots_adjust(left=.25,right=.95,top=.74,bottom=.15,hspace=.8)
    dark='#19343f'
    fig.text(.25,.93,'950PR mHC Pre   |   1 × 4 × 128',fontsize=21,weight='bold',color=dark)
    fig.text(.25,.87,'One Pre launch   ·   critical AIV   ·   BF16',fontsize=13,color='#486373')
    fig.legend(handles=[Patch(color=PALETTE[q],label={'control':'control / waits','compute':'Cube','traffic':'data movement','vector':'Vector'}[q])
                        for q in CLASSES],loc='upper left',bbox_to_anchor=(.24,.845),ncol=4,frameon=False)
    a.set_title('Exclusive service budgets · each row sums to its full window',loc='left',weight='bold',fontsize=12)
    for j,key in enumerate(('model','ca')):
        cur=0.;values=d[key]
        for q in CLASSES:
            v=values[q]
            a.barh(j,v,left=cur,height=.46,color=PALETTE[q],linewidth=0)
            if v>.7:a.text(cur+v/2,j,f'{v:.2f}',ha='center',va='center',fontsize=9,color='white',weight='bold')
            cur+=v
        a.text(cur+.10,j,f'{cur:.3f} µs',va='center',weight='bold',fontsize=10)
    a.set_xlim(0,max(d['model_us'],d['ca_us'])*1.16)
    a.set_yticks([0,1],['Model','CA · same accounting']);a.set_ylim(1.6,-.6)
    a.set_xlabel('Accumulated exclusive service (µs); order in stack is accounting order',fontsize=9)
    b.set_title('CA · observed sequence on '+d['core'],loc='left',weight='bold',fontsize=12)
    for q,l,r in d['ordered_ca']:b.broken_barh([(l,r-l)],(-.19,.38),facecolors=PALETTE[q],linewidths=0)
    b.set_yticks([0],['Terminal core']);b.set_ylim(-.48,.48);b.set_xlim(a.get_xlim());b.axvline(d['ca_us'],ls='--',lw=1,color=dark)
    b.set_xlabel('Device time from the first instruction on any core (µs)',fontsize=9)
    for ax in (a,b):
        ax.grid(axis='x',lw=.6,color='#d5e0e5');ax.set_axisbelow(True);ax.tick_params(axis='y',length=0,pad=8)
        for s in ax.spines.values():s.set_visible(False)
    fig.savefig(FIG,dpi=148,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replot',action='store_true');args=parser.parse_args()
    if not args.replot:DATA.write_text(json.dumps(extract(),indent=2)+'\n')
    draw(read(DATA))
