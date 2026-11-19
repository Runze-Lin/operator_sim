"""Recover and render the two middle model schedules for the same two-expert GMM chain."""
import argparse
from math import isclose
from pathlib import Path
from statistics import median
import sqlite3
import json

from indexer_figure import ROOT, PR, COLORS, QUEUES, read, digest, projected_core
from timelines import trace_lanes
from operator_model import estimate, load_chip
from operator_model.fusion_v8 import work
from operator_model.predictor import _profile
from operator_model.transfer import _queue_factors

SHAPE=[8,256,128,2]
OPS={'grouped_matmul_swiglu_quant_v2':'GMM1', 'gmm2':'GMM2'}
OUT=ROOT/'results/v0.21/fusion'


def cases():
    archive=read(ROOT/'results/v0.17/sku/pairs.json')
    return [next(p for p in archive if p['op']==op and p['shape']==SHAPE) for op in OPS]


def exact_mx(trace,stage,group):
    counts={}
    for c in stage['windows']:
        if 'cubecore' not in c:continue
        shard=trace.parent/c/'trace.json'
        es=[e for e in read(shard)['traceEvents'] if e['ph']=='X']
        ticks=[e['ts'] for e in es if e['name']=='MMAD_MX']
        if ticks:counts[c]=len(ticks)
        if c==group['core']:
            origin=min(e['ts'] for e in es)-stage['windows'][c][0]
            group['mx_issues_us']=[round(t-origin,6) for t in sorted(ticks)]
    return counts


def extract(case,kernels):
    op=case['op'];attrs=case['attrs'];pub=estimate(op,SHAPE,attrs,chip=PR)
    assert isclose(pub.latency_us,case['pr_us'],abs_tol=1e-7)
    stage=case['pr']['stages'][0];trace=ROOT/stage['trace']
    assert digest(trace)==stage['trace_sha256'] and isclose(stage['ca_us'],case['pr_ca_us'])
    factors,_=_queue_factors(op,SHAPE,attrs,{'predicted_us':pub.projection['reference_latency_us']},
                              pub.work,load_chip(),load_chip(PR),[])
    demand=work(op,SHAPE,attrs)
    candidates=[]
    def compact(lanes):
        return {q:[[round(a,6),round(b,6)] for a,b in v] for q,v in lanes.items() if q in COLORS}
    for t in _profile('results/v0.8/fusion/model.json')['templates'][op]:
        k=kernels[t['db_sha256']];path=Path(k['db_path'])
        assert digest(path)==t['db_sha256']
        rates={q:factors.get(q,factors['SCALAR'])*v/t['work'][q] for q,v in demand.items()}
        expected=max(sum(us*max(rates[q] for q in mask.split('+')) for mask,us in c.items()) for c in t['cores'])
        family={}
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
            zero=db.execute('SELECT min(ExecInstrTickStart) FROM ExecutedInstructions').fetchone()[0]
            for kind in {c['core_type_id'] for c in k['cores']}:
                last=max([c for c in k['cores'] if c['core_type_id']==kind],key=lambda c:c['end_ticks'])
                c=projected_core(db,zero,last,rates)
                f='AIC' if kind==1 else 'AIV'
                if f not in family or family[f]['end_us']<c['end_us']:
                    family[f]=dict(end_us=c['end_us'],lanes=compact(c['lanes']),source_type=kind,source_core=last['core_id'])
        assert isclose(max(c['end_us'] for c in family.values()),expected,abs_tol=1e-7)
        candidates.append(dict(shape=t['shape'],sha256=t['db_sha256'],file=str(path.relative_to(ROOT)),
                               end_us=expected,family=family,rates=rates))
    candidates.sort(key=lambda c:c['end_us'])
    assert len(candidates)==4 and isclose(median(c['end_us'] for c in candidates),pub.latency_us,abs_tol=1e-7)
    selected=candidates[1:3];ca={}
    for c in trace_lanes(stage,trace,0,stage['ca_us']):
        f='AIC' if 'cubecore' in c['core'] else 'AIV'
        ca[f]=dict(window=c['window'],lanes=compact(c['lanes']),core=c['core'],sha256=c['sha256'],file=c['file'])
    counts=exact_mx(trace,stage,ca['AIC'])
    return dict(op=op,shape=SHAPE,attrs=attrs,api=case['pr']['api'],installed_kernel=stage['kernel'],
        work_units=demand,template_totals=[{q:c[q] for q in ('shape','sha256','file','end_us')}for c in candidates],
        middle_templates=selected,pred_us=pub.latency_us,ca_us=stage['ca_us'],ca=ca,trace=stage['trace'],
        trace_sha256=stage['trace_sha256'],mx_issues_by_core=counts,plain_mmad=stage['mmad'],checks=case['pr']['correctness'])


def draw(d,stem):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    op=d['op'];fams=d['ca'].keys()
    # Model rows deliberately retain BOTH independently reconstructed middle templates.
    # Averaging event positions would imply issue predictions the backend never made.
    rows=[(f,q)for f in ('AIC','AIV') if f in fams for q in QUEUES[f]]
    fig,axes=plt.subplots(2,1,figsize=(12.6,9 if len(rows)>6 else 7.4),sharex=True)
    fig.subplots_adjust(left=.17,right=.96,bottom=.135,top=.795,hspace=.45)
    dark='#193541'
    fig.text(.17,.955,f'950PR  {OPS[op]}   |   8 routed tokens  ·  2 experts',color=dark,fontsize=21,weight='bold')
    fig.text(.17,.91,'Expert rows [1, 7]   ·   hidden 256   ·   intermediate 128   ·   '+('fused SwiGLU + MX quant' if 'swiglu' in op else 'MX down projection'),fontsize=13,color='#415c6d')
    used={q for f,q in rows}
    fig.legend(handles=[Patch(color=color,label=q) for q,color in COLORS.items() if q in used],loc='upper left',
        bbox_to_anchor=(.165,.893),ncol=len(used),frameon=False,fontsize=9)
    t0,t1=d['middle_templates']
    for ax,title,duration in zip(axes,('Model · two middle templates on each row', 'CA trace · '+', '.join(c['core'] for c in d['ca'].values())),(d['pred_us'],d['ca_us'])):
        ca=ax is axes[1]
        ax.set_title(title,loc='left',color=dark,pad=9,fontsize=12,weight='bold')
        ax.set_title(f'{duration:.3f} µs',loc='right',pad=9,fontsize=14,weight='bold',color=dark)
        for i,(family,q) in enumerate(rows):
            groups=[(d['ca'][family],0,.35,1)] if ca else [(t0['family'][family],-.19,.27,1),(t1['family'][family],.19,.27,.52)]
            for c,offset,h,alpha in groups:
                for a,b in c['lanes'].get(q,[]):
                    ax.broken_barh([(a,b-a)],(i+offset-h/2,h),linewidths=0,facecolors=COLORS[q],alpha=alpha)
            if ca and q=='CUBE':ax.vlines(d['ca']['AIC']['mx_issues_us'],i-.28,i+.28,color='#30204b',lw=1)
        if 'AIV' in fams:ax.axhline(4.5,color='#d7e2e7',lw=.8)
        ax.axvline(duration,ls=(0,(3,3)),lw=.9,color='#465f70')
        ax.set_yticks(range(len(rows)),[f'{f} / {q}' for f,q in rows],fontsize=9)
        ax.set_ylim(len(rows)-.5,-.55);ax.set_xlim(0,1.045*max(t1['end_us'],d['ca_us']))
        ax.grid(axis='x',color='#d8e2e7',lw=.7);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0,pad=8)
        for spine in ax.spines.values():spine.set_visible(False)
    axes[1].set_xlabel('Device time since the first instruction of this kernel (µs)')
    fig.text(.17,.058,f"Top of each model row: darker template {t0['end_us']:.3f} µs.  Bottom: lighter template {t1['end_us']:.3f} µs.  Dashed line: mean of their end times.",
             color='#425e6d',fontsize=9)
    fig.text(.17,.030,f"CA ticks: MMAD_MX issues on the terminal AIC.  All payload AICs issued {sum(d['mx_issues_by_core'].values())} MMAD_MX instructions in total.",color='#425e6d',fontsize=9)
    fig.savefig(ROOT/'docs/examples/figures'/f'{stem}.png',dpi=148,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replot',action='store_true');args=parser.parse_args()
    kernels={}
    if not args.replot:
        for p in (ROOT/'runs/model-v7').glob('moe*/result.json'):
            r=read(p)
            for k in (r.get('measurement') or {}).get('kernels',[]):kernels[k['db_sha256']]=k
    for op in OPS:
        stem=OPS[op].lower()+'-two-experts-pr';file=OUT/(stem+'.json')
        if not args.replot:
            d=extract(next(p for p in cases() if p['op']==op),kernels);file.write_text(json.dumps(d,indent=2)+'\n')
        draw(read(file),stem)
