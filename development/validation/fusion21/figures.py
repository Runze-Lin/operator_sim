"""Render common-origin model/CA panels from archived critical-core trace shards."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import sqlite3
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'development/validation/sku20')]
from timelines import trace_lanes, union
from operator_model import estimate, load_chip
from operator_model.mhc_sink_resource import partition, compute_scale
from operator_model.predictor import _profile

OUT=ROOT/'results/v0.21/fusion'
PR='Ascend950PR_957b'
COLORS={'MTE2':'#117f88','MTE3':'#c15b26','MTE1':'#5078af','CUBE':'#6851a5',
        'FIXP':'#a74725','SIMD':'#8059aa','SCALAR':'#4b687d','IDLE':'#d2dce1',
        'other':'#a7b4bb','arrival':'#8a9fa9','phase':'#407a86'}
PHASE_NAMES=['Dispatch','Read score weights','Read / add prefix','Read bank + score','Weight / normalize / write']
def read(path):return json.loads(path.read_text())
def cumulative(stages):
    start=0.;spans=[]
    for q,d in stages.items():
        if d>0:
            spans.append([q,start,start+d]);start+=d
    return spans


def calibrated_pre(shape):
    # Recover the exact order of overlapping queue masks underlying median DT template.
    kernels={}
    for p in (ROOT/'runs/model-v7').glob('mhc_pre_sinkhorn*/result.json'):
        result=read(p)
        if result['status']=='ok':
            k=result['measurement']['kernels'][0];kernels[k['db_sha256']]=k
    ref=load_chip();target=load_chip(PR); plan=partition(shape,target)
    results=[]; m,k=plan['tile'];clk=ref.frequency_ghz/target.frequency_ghz
    for t in _profile('results/v0.8/fusion/model.json')['templates']['mhc_pre_sinkhorn']:
        kk=kernels[t['db_sha256']];base=partition(t['shape'],ref);rm,rk=base['tile'];rows=plan['rows_per_aiv']/base['rows_per_aiv']
        cube=m*k/(rm*rk)
        factors=dict(CUBE=cube*compute_scale(ref,target,'cube','16'),MTE1=cube*clk,FIXP=m/rm*clk,SIMD=rows*compute_scale(ref,target,'vector','32'),MTE2=rows*clk,MTE3=rows*clk)
        bycore=[]
        with sqlite3.connect(Path(kk['db_path']).resolve().as_uri()+'?mode=ro',uri=True) as db:
            zero=db.execute('select min(ExecInstrTickStart) from ExecutedInstructions').fetchone()[0]
            for typ in {c['core_type_id'] for c in kk['cores']}:
                core=max([c for c in kk['cores'] if c['core_type_id']==typ],key=lambda c:c['end_ticks'])
                delta=defaultdict(Counter)
                for a,b,q in db.execute("""select ExecInstrTickStart,ExecInstrTickEnd,json_extract(ExtendParams,'$.queue_type')
                    from ExecutedInstructions where CoreTypeId=? and ExecInstrCoreId=? and ExecInstrIsDurationExcluded=0""",(typ,core['core_id'])):
                    delta[a][q]+=1;delta[b][q]-=1
                delta[zero+core['end_ticks']]
                active=Counter(); prev=zero; elapsed=0.; lanes=defaultdict(list)
                for tick,change in sorted(delta.items()):
                    qs=[q for q,n in active.items() if n] or ['IDLE']
                    end=elapsed+(tick-prev)*max(factors.get(q,clk) for q in qs)/1650
                    if end>elapsed:
                        for q in qs:lanes[q].append([elapsed,end])
                    elapsed=end;prev=tick;active.update(change)
                bycore.append((elapsed,typ,{q:union(ranges) for q,ranges in lanes.items()}))
        results.append((max(c[0] for c in bycore),bycore,t['db_sha256']))
    dur,cores,sha=sorted(results,key=lambda z:z[0])[len(results)//2]
    families=[('AIC',max((c for c in cores if c[1]==1),key=lambda c:c[0])),
              ('AIV',max((c for c in cores if c[1]!=1),key=lambda c:c[0]))]
    lanes={(family,q):v for family,(_end,_kind,queues) in families for q,v in queues.items()}
    return lanes,{'reference_db_sha256':sha,'family_terminal_us':{family:c[0] for family,c in families},'latency_us':dur}


def example(op,shape,independent=False):
    if op=='attn_res_fwd':
        if independent:
            rec=read(OUT/'receipts'/('_'.join((op,*map(str,shape),PR))+'.json'))
            assert rec['qualified']
        else:
            ms=read(ROOT/'results/v0.19/workloads/attn-res-extension-holdout.json')
            rec=next(z for z in ms if z['chip']==PR and z['shape']==shape)
        k=rec['kernel'];attrs=dict(addend=True,materialize=True,write_idx=shape[-1])
    else:
        pair=next(r for r in read(ROOT/'results/v0.17/sku/pairs.json') if r['op']==op and r['shape']==shape)
        k=pair['pr']['stages'][0];attrs=pair['attrs']
    r=estimate(op,shape,attrs,chip=PR);tl=r.projection['stages_us']
    groups=trace_lanes(k,ROOT/k['trace'],0,k['ca_us'])
    core=max(groups,key=lambda g:g['window'][1])
    if op=='mhc_pre_sinkhorn':
        model,extra=calibrated_pre(shape)
        assert abs(extra['latency_us']-r.latency_us)<1e-6
        stages={'AIC':('MTE2','MTE1','CUBE','FIXP','SCALAR'), 'AIV':('MTE2','SIMD','MTE3','SCALAR','IDLE')}
        lanes=[family+' / '+q for family,qs in stages.items() for q in qs]
        top=[(family+' / '+q,a,b,q) for (family,q),v in model.items() for a,b in v if q in stages[family]]
        bottom=[]
        extra['ca_core_shards']=[]
        for g in groups:
            family='AIC' if 'cubecore' in g['core'] else 'AIV'
            extra['ca_core_shards'].append(dict(core=g['core'],file=g['file'],sha256=g['sha256']))
            for q,v in g['lanes'].items():
                if q in stages[family]:bottom.extend((family+' / '+q,a,b,q) for a,b in v)
        title='Target resource schedule · terminal AIC and AIV in median template'
        title2='CA · terminal AIC and AIV ('+', '.join(g['core'] for g in groups)+')'
        note='Overlapping model queues share each reference interval. IDLE includes exposed vector waits in the reference.'
    elif op=='mhc_sinkhorn':
        extra={};lanes=['arrival','SCALAR','SIMD','MTE2','MTE3','other']
        top=[(q,a,b,q)for q,a,b in cumulative(tl)]
        bottom=[(q,a,b,q) for q,v in core['lanes'].items() for a,b in v if q in lanes]
        title='Model · exclusive contributions (cumulative)'
        title2=f'CA instruction timeline · {core["core"]} (last to finish)'
        note='Top bar ordering is accounting order. The fitted exclusive services do not identify instruction issue times.'
    else:
        if independent:
            from development.workloads.attn_res_extension_report import phases
            observed_core,ca=phases(rec)
            assert observed_core==core['core']
        else:
            rows=read(ROOT/'results/v0.19/workloads/attn-res-extension-evaluation.json')['rows']
            ca=next(z['ca_phases_us'] for z in rows if z['shape']==shape and z['chip']==PR)
        lanes=PHASE_NAMES
        top=[(lanes[i],a,b,'phase') for i,(_,a,b) in enumerate(cumulative(tl))]
        extra={'ca_issue_to_issue_us':ca}
        bottom=[]
        for phase,a,b in cumulative(dict(zip(lanes,ca))):
            for q in ('MTE2','SIMD','MTE3','SCALAR'):
                for l,h in core['lanes'].get(q,[]):
                    if max(a,l)<min(b,h):bottom.append((phase,max(a,l),min(b,h),q))
        title='Model · predicted DMA issue-to-issue windows'
        title2=f'CA · same issue landmarks, {core["core"]}'
        note='DMA / SIMD / control tracks may overlap within a phase. Phase boundaries are payload read issues.'
    fig,axes=plt.subplots(2,1,figsize=(12,7.8),sharex=True)
    fig.subplots_adjust(left=.24,right=.96,bottom=.13,top=.81,hspace=.38)
    fig.text(.24,.944,f'950PR   {op}   |   '+ ' × '.join(map(str,shape)),fontsize=20,weight='bold',color='#18303b')
    used=[q for q in COLORS if any(z[-1]==q for z in top+bottom)]
    fig.legend(handles=[Patch(color=COLORS[q],label=q) for q in used],loc='upper left',bbox_to_anchor=(.23,.918),frameon=False,ncol=8,fontsize=9)
    for ax,title,items,dur in zip(axes,(title,title2),(top,bottom),(r.latency_us,k['ca_us'])):
        for lane,a,b,q in items:
            j=lanes.index(lane)
            mult=op=='attn_res_fwd' and ax is axes[1]
            dy={'SCALAR':-.27,'MTE2':-.09,'SIMD':.09,'MTE3':.27}.get(q,0) if mult else 0
            ax.broken_barh([(a,b-a)],(j+dy-(.075 if mult else .18),.15 if mult else .36),facecolors=COLORS[q],linewidths=0)
        ax.set_title(title,fontsize=11,loc='left',weight='bold',pad=9)
        ax.set_title(f'{dur:.3f} µs',fontsize=13,loc='right',weight='bold',pad=9)
        ax.set_yticks(range(len(lanes)),lanes);ax.set_ylim(len(lanes)-.5,-.5)
        ax.set_xlim(0,1.055*max(k['ca_us'],r.latency_us));ax.axvline(dur,ls='--',lw=.9,color='#3c5562')
        ax.grid(axis='x',lw=.6,color='#d4dfe5');ax.set_axisbelow(True)
        ax.tick_params(axis='y',length=0)
        for spine in ax.spines.values():spine.set_visible(False)
    axes[1].set_xlabel('Device time from first instruction on any core (µs)')
    fig.text(.06,.035,note,fontsize=9,color='#405767')
    stem=op+'-independent' if independent else op
    fig.savefig(ROOT/'docs/examples/figures'/f'{stem}-v21-pr.png',dpi=145,bbox_inches='tight');plt.close(fig)
    (OUT/f'{stem}-figure.json').write_text(json.dumps(dict(shape=shape,chip=PR,model_us=r.latency_us,ca_us=k['ca_us'],
        ca_core=core['core'],ca_trace=k['trace'],ca_sha256=k['trace_sha256'],shard_sha256=core['sha256'],
        top=top,bottom=bottom,note=note,**extra),indent=2)+'\n')
if __name__=='__main__':
    for op,shape in [('attn_res_fwd',[1,7168,8,3]),('mhc_sinkhorn',[64,4]),('mhc_pre_sinkhorn',[1,4,4096])]:
        example(op,shape)

    example('attn_res_fwd',[2,4096,8,1],independent=True)
