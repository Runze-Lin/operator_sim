"""Zoom three consecutive, address-verified UB blocks of the Pro PR prefill trace."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'results/v0.23/pro-prefill/pro-b8-128ki-pr-blocks.json'
OUT=ROOT/'results/v0.25'
os.environ.setdefault('MPLCONFIGDIR','/tmp/pro25-mpl')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


def main():
    r=json.loads(SOURCE.read_text())
    assert r['shape']==[1048576,64] and r['active_aiv']==56
    idx=[27,28,29]
    assert all(r['model_tile_elements'][i]==r['ca_tile_elements'][i]==21120 for i in idx)
    figures,axes=plt.subplots(2,1,figsize=(11.2,6.6),sharex=True)
    figures.subplots_adjust(left=.20,right=.965,top=.77,bottom=.17,hspace=.53)
    ink='#182e3b'
    figures.text(.20,.935,'950PR Cast BF16 → FP32  |  1,048,576 × 64',color=ink,fontsize=20,weight='bold')
    figures.text(.20,.879,'DeepSeek V4 Pro   ·   B8 × 128k prefill   ·   UB blocks 28–30 of 57',color='#426174',fontsize=12.5)
    colors={'Read':'#197c88','Vector':'#7552a4','Write':'#b44e29'}
    figures.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors.values(),('MTE2  /  GM → UB','Vector  /  convert','MTE3  /  UB → GM'))],
                   loc='upper left',bbox_to_anchor=(.193,.853),frameon=False,ncol=3)
    for ax,key,name,device in [(axes[0],'model_tiles','Resource model · shared bandwidth per block',r['model_us']),
                              (axes[1],'ca_tiles',f"CA trace · {r['ca_core']} (last to finish)",r['ca_us'])]:
        ax.set_title(name,loc='left',color=ink,fontsize=12,weight='bold')
        ax.set_title(f'{device:.3f} µs  full window',loc='right',color=ink,fontsize=10)
        ax.set(ylim=(2.55,-.55),yticks=range(3),yticklabels=[f'UB {i+1}\n21,120 elements' for i in idx])
        ax.grid(axis='x',color='#d9e3ea')
        ax.set_axisbelow(True)
        ax.spines[['top','left','right']].set_visible(False)
        ax.tick_params(axis='y',length=0,pad=8)
        for y,i in enumerate(idx):
            for k,(pipe,color) in enumerate(colors.items()):
                start,end=r[key][i][pipe]
                pos=y+.18*(k-1)
                ax.barh(pos,end-start,left=start,height=.13,color=color)
                ax.plot(start,pos,marker='|',color=color,ms=6)
            if y<2:
                ax.axhline(y+.50,color='#dee8ed',lw=.8)
    axes[1].set_xlim(78,133)
    axes[1].set_xlabel('Device time from the first instruction on any AIV (µs)')
    figures.text(.20,.048,'Both panels show the same absolute time range. Marks indicate issue. Upper bars allocate exposed service;\n'
                 'lower bars span issue to retirement. A pending DMA can overlap transfers on the next block.',size=9,color='#3d5969')
    figures.savefig(OUT/'pro-128ki-pr-middle.png',dpi=155,facecolor='white')
    plt.close(figures)
    values=[]
    for i in idx:
        model,actual=r['model_tiles'][i],r['ca_tiles'][i]
        values.append(dict(block=i+1,elements=21120,model=model,ca=actual,
                           read_issue_gap_us=model['Read'][0]-actual['Read'][0],
                           write_retire_gap_us=model['Write'][1]-actual['Write'][1],
                           addresses=r['actual_tiles_with_addresses'][i]))
    (OUT/'pro-128ki-pr-middle.json').write_text(json.dumps(dict(source=str(SOURCE.relative_to(ROOT)),
               source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),trace_sha256=r['trace_sha256'],
               profile_core=r['ca_core'],model_us=r['model_us'],ca_us=r['ca_us'],blocks=values),indent=2)+'\n')


if __name__=='__main__':
    main()
