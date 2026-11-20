"""Draw only model milestones and measured DMA for matching logical output fragments."""
from pathlib import Path
import sys
from study import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def build(c,tag):
    previous=read(OLD/('wide-'+tag+'.json'))
    args=dict(chip=c,resident_aiv=load_chip(c).vector_cores,previous_shape=previous['previous_shape'],inspect_rows=(0,1))
    pred=estimate_post_fragments(previous['shape'],**args)
    rows=[]
    for j,v in enumerate(pred['timeline']):
        rr=previous['rows'][j//2];f=rr['fragments'][j%2]
        assert f['columns']==v['columns'] and f['token']==rr['token']
        rows.append(dict(token=rr['token'],offset=f['first_column'],width=f['columns'],model=v,ca={k:f[k]for k in ('h','x','y')}))
    dt=dict(shape=previous['shape'],previous_shape=previous['previous_shape'],chip=c,source=previous['input_trace'],
        source_sha256=previous['input_trace_sha256'],active_aiv=previous['active'],core=previous['core'],
        original_ordered_us=previous['public_us'],aggregate_us=previous['experimental_aggregate_us'],
        conditional_us=pred['us'],ca_us=previous['ca_us'],records=rows)
    save(OUT/f'fragments-{tag}.json',dt)
    fig,axes=plt.subplots(2,1,figsize=(11.2,6.6),sharex=True)
    fig.subplots_adjust(left=.245,right=.972,top=.745,bottom=.14,hspace=.42)
    col=('#086c82','#43a1a4','#ba562d')
    fig.text(.245,.978,f"{c.split('_')[0]} mHC Post  |  65 × 4 × 5,120",va='top',fontsize=21,weight='bold',color='#183442')
    fig.text(.245,.904,'33 AIVs   ·   two tokens on a critical lane   ·   two alternating channel fragments',fontsize=11,color='#48606d')
    fig.legend(handles=[Patch(color=col[0],label='MTE2   ·   hOut'),Patch(color=col[1],label='MTE2   ·   residual X'),
        Patch(color=col[2],label='MTE3   ·   output'),Line2D([0],[0],linestyle='none',marker='|',markersize=10,color=col[2],label='write issue')],
        loc='upper left',bbox_to_anchor=(.236,.88),frameon=False,ncol=4,fontsize=9)
    labels=[f"token {r['token']}  ·  D {r['offset']}–{r['offset']+r['width']-1}" for r in rows]
    for ax,total,title,key in zip(axes,(pred['us'],previous['ca_us']),('Conditional two-slot schedule','CA trace  ·  '+previous['core']+' (last AIV)'),('model','ca')):
        ax.set_title(title,loc='left',fontsize=13,color='#183442',weight='bold');ax.set_title(f'{total:.3f} µs',loc='right',fontsize=12)
        for i,r in enumerate(rows):
            y=3-i
            g=r[key]
            pairs=([g['h_read_us'],g['x_read_us']] if key=='model' else [[g[k]['start_us'],g[k]['end_us']]for k in ('h','x')])
            for z,((a,b),color) in enumerate(zip(pairs,col)):
                ax.broken_barh([(a,b-a)],(y+.16-z*.17,.115),facecolor=color,zorder=3)
            a=g['write_issue_us'] if key=='model' else g['y']['start_us']
            b=g['final_write_ack_us'] if key=='model' else g['y']['end_us']
            ax.plot(a,y-.17,'|',markersize=8,color=col[2],zorder=4)
            if b is not None:ax.broken_barh([(a,b-a)],(y-.23,.115),facecolor=col[2],zorder=3)
        ax.set_yticks([3,2,1,0],labels,fontsize=9);ax.tick_params(length=0,pad=6,colors='#45606d');ax.set_ylim(-.43,3.48)
        ax.grid(axis='x',color='#d4e0e5',lw=.8);ax.axvline(total,color='#50636f',linestyle='--',lw=1)
        for sp in ax.spines.values():sp.set_visible(False)
    axes[-1].set_xlim(0,1.04*max(pred['us'],previous['ca_us']));axes[-1].set_xlabel('Device time since earliest AIV instruction (µs)')
    fig.text(.245,.040,'Each input row maps to GM and an alternating UB slot. The model prices the terminal write acknowledgement only.',fontsize=9,color='#48606d')
    fig.savefig(OUT/f'fragments-{tag}.png',dpi=150,facecolor='white');plt.close(fig)
if __name__=='__main__':
    for c,tag in [(DT,'dt'),(PR,'pr')]:build(c,tag)
