"""Rebuild the page-aligned PR Addmm worked example from a qualified archived trace."""
from pathlib import Path
from math import isclose
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import write,read,digest,CHIPS
from operator_model import estimate
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

OUT = ROOT/'results/v0.21/dispatch/addmm-pr-page.json'
FIG = ROOT/'docs/examples/figures/addmm-v21-pr-page.png'


def extract():
    shape=[512,512,256]
    row=next(r for r in read(ROOT/'results/v0.17/evaluation.json')['rows'] if r['op']=='addmm' and r['shape']==shape)
    obs=row['pr_observation']
    path=ROOT/obs['trace'][0]
    assert digest(path)==obs['trace_sha256'][0]
    ev=[e for e in read(path)['traceEvents'] if e['ph']=='X']
    first=min(e['ts'] for e in ev)
    terminal=max(ev,key=lambda e:e['ts']+e['dur'])
    end=terminal['ts']+terminal['dur']-first
    assert isclose(end,row['pr_ca_us'],abs_tol=1e-5)
    lane={'MOV_OUT_TO_L1_MULTI_ND2NZ':'MTE2','MOV_SRC_TO_DST_ALIGNv2':'bias GM → L1',
          'LOAD_2Dv2':'MTE1','MOV_L1_TO_BT':'bias L1 → BT','MMAD':'MMAD','FIX_L0C_TO_DST':'FIXP'}
    actual=[]
    for e in ev:
        if e['pid']!=terminal['pid'] or e['name'] not in lane:continue
        arg=json.loads(e['args']['detail'])
        if arg.get('is_duration_excluded'):continue
        actual.append(dict(lane=lane[e['name']],start=e['ts']-first,end=e['ts']+e['dur']-first,
                           opcode=e['name']))
    pred=estimate('addmm',shape,chip=CHIPS['pr'])
    cost=pred.projection['stages_us'];clock=cost['launch']; stages=[]
    # Port durations include the small fused bias transfer. Its independent
    # descriptor issue time does not exist in the analytic model.
    for stage,label in [('read_fill','MTE2'),('l0_lead','MTE1'),('last_mmad','MMAD'),('write_tail','FIXP')]:
        stages.append(dict(lane=label,start=clock,end=clock+cost[stage]))
        clock+=cost[stage]
    assert isclose(clock,pred.latency_us)
    write(OUT,dict(chip=CHIPS['pr'],shape=shape,kernel=obs['kernel'],trace=str(path.relative_to(ROOT)),
        trace_sha256=digest(path),core=terminal['pid'],start='earliest instruction over all cores',
        model_us=pred.latency_us,old_us=row['pr_us'],ca_us=end,model=stages,ca=actual,
        launch_us=cost['launch'],target_plan=pred.projection['target_plan'],ca_aic=obs['active_aic'],
        bias_l1_to_bt_bytes=pred.projection['timing_diagnostics']['l1_to_bt_bytes']))


def render():
    d=read(OUT)
    colors={'MTE2':'#147786','bias GM → L1':'#147786','MTE1':'#5177b9','bias L1 → BT':'#5177b9',
            'MMAD':'#7653a8','FIXP':'#b54a25'}
    names=list(colors)
    ys={name:len(names)-i for i,name in enumerate(names)}
    fig,axes=plt.subplots(2,1,figsize=(11,7.6),sharex=True)
    fig.subplots_adjust(left=.17,right=.96,top=.81,bottom=.14,hspace=.57)
    fig.text(.17,.957,'950PR Addmm  |  512 × 512 × 256',fontsize=21,weight='bold',color='#19303b')
    fig.text(.17,.907,'28 AICs  ·  one output wave  ·  BF16 fused column bias  ·  56 MMADs',fontsize=12,color='#486171')
    for ax,desc,entries,total in zip(axes,['Resource model · critical output tile',f"CA trace · {d['core']} (last to finish)"],
            [d['model'],d['ca']],[d['model_us'],d['ca_us']]):
        ax.set_title(desc,fontsize=13,loc='left',weight='bold')
        ax.set_title(f'{total:.3f} µs',fontsize=13,loc='right',weight='bold')
        ax.set_yticks(list(ys.values()),['GM → L1','bias → L1','L1 → L0','bias → BT','Cube / MMAD','FIXP write'])
        ax.set_ylim(.4,6.7);ax.grid(axis='x',color='#d9e3e8',linewidth=.7);ax.set_axisbelow(True)
        ax.spines[:].set_visible(False);ax.tick_params(axis='y',length=0)
        ax.axvline(total,color='#586f7a',linestyle=(0,(3,3)),lw=1)
        for e in entries:
            y=ys[e['lane']]; alpha=.38 if e['lane']=='FIXP' and ax is axes[1] else .92
            ax.barh(y,e['end']-e['start'],left=e['start'],height=.37,color=colors[e['lane']],alpha=alpha)
            if ax is axes[1]:
                ax.vlines(e['start'],y-.25,y+.25,color=colors[e['lane']],linewidth=1.1)
    axes[0].barh(ys['MTE2'],d['launch_us'],left=0,height=.37,color='#e6edef')
    axes[0].text(d['launch_us']/2,ys['MTE2'],'Launch',ha='center',va='center',fontsize=9,color='#496270')
    for name in ('bias GM → L1','bias L1 → BT'):
        axes[0].text(d['launch_us']+.05,ys[name],'included in port service',va='center',fontsize=9,color='#6b7d83')
    axes[1].set_xlim(0,max(d['model_us'],d['ca_us'])*1.06)
    axes[1].set_xlabel('Device time since first instruction (µs)')
    fig.text(.17,.045,'Lower ticks mark actual issue. The FIXP descriptor can wait in flight before the result is ready.',fontsize=10,color='#486171')
    fig.savefig(FIG,dpi=155,facecolor='white');plt.close(fig)


if __name__=='__main__':
    if '--extract' in sys.argv:extract()
    render()
