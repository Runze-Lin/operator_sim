"""Zoom the same four Pro Cast UB tiles above/below, preserving device-time zero."""
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from common import *


def build():
    source=ROOT/'results/v0.23/pro-prefill/pro-b8-256ki-dt-blocks.json'
    row=read(source);op=row['op'];p=forecast(op,row['shape'],DT)
    if abs(p['public_us']-row['model_us'])>1e-6:
        raise ValueError("archived stage schedule no longer matches current total")
    witness=[]
    for i,raw in enumerate(row['actual_tiles_with_addresses'][:4]):
        n=raw['n'];m=row['model_tiles'][i];ca=row['ca_tiles'][i]
        assert n==row['model_tile_elements'][i] and raw['MTE2']['bytes']==2*n and raw['MTE3']['bytes']==4*n
        assert ca['Read']==raw['MTE2']['window'] and ca['Write']==raw['MTE3']['window'] and ca['Vector']==raw['vcvt']
        offset=(raw['MTE2']['gm']-row['actual_tiles_with_addresses'][0]['MTE2']['gm'])//2
        out=(raw['MTE3']['gm']-row['actual_tiles_with_addresses'][0]['MTE3']['gm'])//4
        assert offset==out==sum(row['model_tile_elements'][:i])
        witness.append(dict(tile=i+1,slot=raw['slot'],local_offset=offset,n=n,model=m,ca=ca))
    src=dict(source=rel(source),trace_sha256=row['trace_sha256'],kernel=row['kernel'],core=row['ca_core'],
             model_us=row['model_us'],ca_us=row['ca_us'],tile_count=len(row['ca_tiles']),shape=row['shape'],tiles=witness)
    save(OUT/'pro-cast-startup.json',src)
    color={'Read':'#166e82','Vector':'#794b9a','Write':'#c45e34'}
    fig,axes=plt.subplots(2,1,figsize=(11.2,6.5),sharex=True)
    fig.subplots_adjust(left=.19,bottom=.16,right=.98,top=.73,hspace=.42)
    fig.text(.19,.98,'950DT Cast  |  2,097,152 × 64',fontsize=22,weight='bold',color='#173443',va='top')
    fig.text(.19,.895,'DeepSeek V4 Pro   ·   B8 × 256 Ki prefill   ·   64 AIVs   ·   startup, tiles 1–4 of 100',fontsize=11.5,color='#435963')
    handles=[Patch(color=c,label=name) for c,name in zip(color.values(),['MTE2  ·  GM → UB','SIMD  ·  VCVT','MTE3  ·  UB → GM'])]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.18,.88),ncol=3,frameon=False,fontsize=10)
    for ax,key,title,total in [(axes[0],'model','Resource model · critical AIV',row['model_us']),
                               (axes[1],'ca','CA trace · '+row['ca_core'],row['ca_us'])]:
        ax.set_title(title,loc='left',weight='bold',color='#173443',fontsize=13)
        ax.set_title(f'Full device window   {total:.3f} µs',loc='right',color='#334d58',fontsize=10)
        for i,tile in enumerate(witness):
            y=3-i
            for field,shift in [('Read',.18),('Vector',0),('Write',-.18)]:
                start,end=tile[key][field]
                ax.broken_barh([(start,end-start)],(y+shift-.08,.16),facecolors=color[field],zorder=3)
        ax.set_yticks([3,2,1,0],[f"tile {t['tile']}   ·   UB {t['slot']}" for t in witness],fontsize=10)
        ax.set_ylim(-.46,3.46);ax.set_xlim(0,12)
        ax.grid(axis='x',color='#d9e3e8',lw=.8);ax.tick_params(length=0,pad=5,colors='#435963')
        for spine in ax.spines.values():spine.set_visible(False)
    axes[-1].set_xlabel('Device time since first instruction on any core (µs)',fontsize=10)
    fig.text(.19,.048,'Each row contains 21,120 elements. CA pairs each read and write by GM byte offset and ping-pong UB slot.',fontsize=9,color='#435963')
    fig.savefig(OUT/'pro-cast-startup.png',dpi=150,facecolor='white');plt.close(fig)

if __name__=='__main__':build()
