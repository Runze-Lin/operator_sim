"""Redraw the 1x128x512 model/CA illustration with a shared whole-device zero."""
import argparse
import gzip
import json
from pathlib import Path

from operator_model import estimate


def case_panels(bundle):
    cases=bundle['freeze']['cases']
    cases={c['id']:c for c in cases if c['shape']==[1,128,512]}
    panels=[]
    for item in bundle['observations']:
        c=cases.get(item['measurement']['case_id'])
        if c is None:
            continue
        e=estimate(c['op'],c['shape'],c['attrs'],dtype=c['dtype'],chip=c['chip'])
        if abs(e.latency_us-c['predicted_us'])>1e-8:
            raise ValueError('Current schedule differs from the frozen model plotted here')
        plan=e.projection['target_plan']
        if plan['ub_tiles']!=1 or plan['shared_floor_cycles']>plan['local_pipeline_cycles']:
            raise ValueError('This illustration represents one local UB tile on a critical worker')
        pieces=[plan[k] for k in ('front_cycles','read_service_cycles','vector_service_cycles',
                                    'write_service_cycles')]
        pieces.append(plan['exposed_tail_wait_cycles']+plan['tail_cycles'])
        if abs(sum(pieces)/1650-c['predicted_us'])>1e-7:
            raise ValueError('Plotted stage budget differs from the frozen window')
        ph=item['terminal_phases']
        panels.append((c['chip'],e.active_aiv,c['predicted_us'],item['measurement']['ca_us'],pieces,ph))
    if len(panels)!=2:
        raise ValueError('Both complete reference traces are required')
    return sorted(panels)


def plot(panels,output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    colors=['#798b9b','#147e83','#8456a5','#bb542a','#bead7b']
    labels=['Front end','Packed / scale read','Vector','BF16 write','Other waits']
    fig,axs=plt.subplots(2,2,figsize=(12.5,6.9),sharex=True,sharey=True)
    fig.subplots_adjust(left=.153,right=.966,bottom=.196,top=.811,hspace=.66,wspace=.13)
    fig.text(.153,.954,'CacheLoad  |  E2M1 + E8M0 to BF16',size=22,weight='bold',color='#203749')
    fig.text(.153,.902,'1 × 128 × 512     ·     contiguous block32, one materialized output',size=12,color='#4c6273')
    for col,(sku,aiv,model,ca,stages,ph) in enumerate(panels):
        upper,lower=axs[:,col]
        name=sku.removeprefix('Ascend').split('_')[0]
        upper.set_title(f'{name}  ·  model budget  ·  {aiv} AIVs',loc='left',fontsize=12,weight='bold')
        lower.set_title(f'CA terminal worker  ·  {ph["core"]}',loc='left',fontsize=11,weight='bold')
        def block(ax,row,lo,hi,shift=0,height=.4):
            ax.barh(row+shift,(hi-lo)/1650,left=lo/1650,height=height,color=colors[row],zorder=3)
        pos=0
        for i,duration in enumerate(stages):
            block(upper,i,pos,pos+duration)
            pos+=duration
        # Copy-in order from this one-tile binary: tiling packet, packed values, then scales.
        dma=ph['dma']
        if [d['kind'] for d in dma]!=['MTE2','MTE2','MTE2','MTE3']:
            raise ValueError('Inspect terminal DMA ordering before plotting a new implementation')
        for d,row,offset in zip(dma,[0,1,1,3],[0,-.17,.17,0]):
            block(lower,row,d['issue'],d['retire'],offset,.27 if row==1 else .4)
        block(lower,2,ph['first_rvecex'],ph['last_rvecex_retire'])
        for ax,end in [(upper,model),(lower,ca)]:
            ax.axvline(end,ls='--',lw=1.3,color='#283e4c')
            ax.text(end-.04,4.7,f'{end:.3f} µs',ha='right',va='center',fontsize=10,weight='bold')
            ax.set_yticks(range(5),labels,fontsize=9)
            ax.set_xlim(0,5.1);ax.set_ylim(5.0,-.65)
            ax.grid(axis='x',color='#dce4e8',lw=.9,zorder=0)
            ax.xaxis.set_major_locator(MultipleLocator(1))
            for s in ('top','right','left'):
                ax.spines[s].set_visible(False)
            ax.spines['bottom'].set_color('#c7d2d8')
            ax.tick_params(axis='y',length=0)
        lower.set_xlabel('Time from first instruction on any core (µs)',fontsize=10)
    fig.text(.153,.091,'Model: “Other waits” is an exposed timing budget; its position in the trace can vary.',size=10,color='#4c6273')
    fig.text(.153,.052,'CA: bars enclose actual descriptors or Vector issue/retirements. Blank regions include unshown scalar and queue events.',size=10,color='#4c6273')
    fig.savefig(output,dpi=150,facecolor='white')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    plot(case_panels(json.loads(gzip.decompress(a.bundle.read_bytes()))),a.output)


if __name__=='__main__':
    main()
