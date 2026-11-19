"""Match physical DMA payloads and VF launches to the five real UB tiles."""
from pathlib import Path
import json
from math import ceil
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from timelines import OUT,ROOT,COLORS,read

def extract(data):
    g,=data['ca'][0]['groups']
    records=read(ROOT/g['file'])['traceEvents']
    origin=min(e['ts'] for e in records if e['ph']=='X')-g['window'][0]
    lanes={'MTE2':[],'MTE3':[],'VF':[]}
    for e in records:
        if e['ph']!='X':continue
        info=json.loads(e['args'].get('detail') or '{}');q=info.get('instr_type')
        if e['name']=='VF' and q=='PUSHQ':q='VF'
        elif e['name']!='MOV_SRC_TO_DST_ALIGNv2' or q not in ('MTE2','MTE3'):continue
        a=e['ts']-origin
        lanes[q].append(dict(a=a,b=a+e['dur'],size=info.get('processed_bytes'),pc=e['args']['pc_addr']))
    for events in lanes.values():events.sort(key=lambda e:e['a'])
    header=lanes['MTE2'].pop(0)
    assert header['size']==160
    plan=data['model']['plan']
    assert plan['iterations']==len(lanes['MTE2'])==len(lanes['VF'])//3==len(lanes['MTE3'])//2
    tiles=[]
    for i,rd in enumerate(lanes['MTE2']):
        w=plan['cols_per_core'];h=min(plan['tile'][0],plan['rows_per_core']-i*plan['tile'][0])
        wr=lanes['MTE3'][2*i:2*i+2];vf=lanes['VF'][3*i:3*i+3]
        assert rd['size']==2*h*w and [v['size'] for v in wr]==[h*w,2*h*ceil(w/64)]
        assert len(set(e['pc'] for e in vf))==3
        if 0<i<len(lanes['MTE2'])-1:assert [e['pc'] for e in vf]==[e['pc'] for e in tiles[0]['VF']]
        tiles.append(dict(label=f'UB tile {i+1} · {h}×{w}',MTE2=[rd],VF=vf,MTE3=wr))
    return dict(core=g['core'],source_sha256=g['sha256'],metadata_read=header,tiles=tiles)

def draw(data,raw,name):
    fig,axes=plt.subplots(2,1,figsize=(10.4,6.5),sharex=True)
    fig.subplots_adjust(left=.218,right=.96,bottom=.14,top=.82,hspace=.47)
    short='950DT' if '950DT' in data['chip'] else '950PR'
    fig.text(.218,.948,short+'  DynamicMxQuant  |  2048 × 5120',size=18,weight='bold',color='#1d3440')
    fig.text(.218,.905,'BF16 → FP8 + scale  ·  five ping-pong UB tiles on the terminal AIV',color='#415a68')
    fig.legend([Patch(color=COLORS[k]) for k in ('MTE2','SIMD','MTE3','fabric')],
        ['MTE2 read','SIMD service / VF descriptor','MTE3 data + scale','Exposed shared link'],
        loc='upper left',bbox_to_anchor=(.21,.893),ncol=4,frameon=False,fontsize=9)
    ts=raw['tiles'];rows=['Dispatch / metadata']+[t['label'] for t in ts]+['Shared link / drain']
    for ax,dur,title in zip(axes,[data['model']['latency_us'],data['ca_us']],
        ['Resource model · two-slot critical path','CA trace · '+raw['core']+' (last to finish)']):
        ax.set_title(title,loc='left',fontsize=11.5,weight='bold')
        ax.text(1,1.035,f'{dur:.3f} µs',transform=ax.transAxes,ha='right',weight='bold')
        ax.set_yticks(range(len(rows)),rows,fontsize=9);ax.set_ylim(len(rows)-.5,-.5)
        ax.grid(axis='x',c='#dce4e8',lw=.7);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
        ax.axvline(dur,ls=(0,(3,3)),lw=1,c='#435964')
        for sp in ax.spines.values():sp.set_visible(False)
    offsets={'MTE2':-.23,'SIMD':0,'VF':0,'MTE3':.23}
    for label,a,b,q in data['model']['segments']:
        row=rows.index(label) if label.startswith('UB tile') else 0 if label=='Dispatch' else len(rows)-1
        axes[0].barh(row+offsets.get(q,0),b-a,left=a,height=.18 if row not in (0,len(rows)-1) else .35,color=COLORS[q])
    def event(row,q,e):
        color=COLORS['SIMD' if q=='VF' else q]
        y=row+offsets[q]
        axes[1].barh(y,e['b']-e['a'],left=e['a'],height=.16,color=color)
        axes[1].vlines(e['a'],y-.13,y+.13,lw=.7,color='#263946')
    event(0,'MTE2',raw['metadata_read'])
    for i,t in enumerate(ts,1):
        for q in offsets:
            for e in t.get(q,[]):event(i,q,e)
    axes[1].set_xlabel('Device time from first instruction (µs)',labelpad=7);axes[1].set_xlim(left=0)
    fig.text(.218,.038,'Ticks mark each descriptor issue. Each CA tile has one input read, three VF launches and two output stores.',fontsize=8.6)
    fig.text(.218,.012,'The upper purple band prices SIMD work; a lower VF window can include waiting. Shared-link exposure has no ISA lane.',fontsize=8.6)
    fig.savefig(ROOT/'docs/examples/figures'/(name+'-tiles.png'),dpi=160);plt.close(fig)

if __name__=='__main__':
    import sys
    for side in ('dt','pr'):
        name='quant-chunk-'+side
        data=read(OUT/(name+'.json'))
        archive=OUT/(name+'-tiles.json')
        if '--extract' in sys.argv:
            archive.write_text(json.dumps(extract(data),indent=2)+'\n')
        draw(data,read(archive),name)
