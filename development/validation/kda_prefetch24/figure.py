"""One time origin, one row layout, two independent schedules for the exact FwdH chunk."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from evidence import ROOT,OUT,read,sha,extract,compact
from frozen import predict,engine


STYLE={'W':('#087f82',-.27),'H':('#346b9c',-.09),'K':('#58aaa9',.09),'right':('#8397c0',.27),
       'mm0':('#654aa3',-.12),'mm2':('#8566ac',.12),'U':('#d29a43',-.22),
       'corrected':('#a14d79',0),'state':('#cc839f',.22),'write_right':('#ad5237',-.11),'write_state':('#db825a',.11),
       'initial_read':('#d29a43',-.2),'initial_vf':('#a14d79',0),'initial_write':('#db825a',.2)}
SECTIONS=(('W','H','K','right'),('mm0','mm2'),('U','corrected','state'),('write_right','write_state'))
LABELS=('AIC reads · W/H/K/right','Cube · MMAD 0/2','AIV · U / VF1 / VF3','MTE3 · right / state')


def materialize(src):
    frozen=engine()[0];Tile=frozen['Tile'];FAMILIES=frozen['FAMILIES']
    r=read(src);shape=r['case']['shape'];chip=frozen['load_chip'](r['case']['sku'])
    ca=extract(src);st=r['stages'][1]
    vec=compact(st,ca['cluster']+'.veccore0')
    pred=predict(shape,chip)
    if len(pred['fwd_h_trace_model_ticks'])!=len(ca['tiles']):raise ValueError('Figure must show every chunk explicitly')
    def ca_window(x):return [x['begin'],x['end']]
    def tls(events):
        return [{key:ca_window(t[key]) for group in SECTIONS for key in group} for t in events]
    a=pred['active_aic'][1];tmp=Tile(chip,FAMILIES[1],a);n=min(shape[1],64)
    first=pred['fwd_h_trace_model_ticks'][0];base=first['W'][0]-334.5
    ri=base+tmp.dma('OUT->UB',65536);vc=ri+tmp.vec(0,n)
    model0={'initial_read':[base,ri],'initial_vf':[ri,vc],'initial_write':[vc,first['H'][0]]}
    actual0={'initial_read':ca_window(ca['initial_state_read']),'initial_vf':ca_window(vec['VF'][0]),
             'initial_write':ca_window(ca['initial_H_write'])}
    reg=read(OUT/'register.json');assert sha(Path(__file__).with_name('model.py'))==reg['source_hashes']['development/validation/kda_prefetch24/model.py']
    return dict(shape=shape,sku=chip.name,core=ca['cluster'],last_core=ca['last_core'],device_text_sha256=st['device_text_sha256'],
                aggregate_sha256=ca['aggregate_sha256'],shard_sha256=[ca['cube_sha256'],ca['vec_sha256']],
                case_receipt=str(src.relative_to(ROOT)),model_register_sha256=sha(OUT/'register.json'),
                us_per_tick=1/(chip.frequency_ghz*1000),zero='earliest core instruction in the complete FwdH kernel, both panels',
                evidence_role='independent' if src.parent==OUT else 'v0.23 release replay, used for mechanism development',
                chunks=[x['rows'] for x in ca['tiles']],model={'init':model0,'tiles':pred['fwd_h_trace_model_ticks'],'full_us':pred['stages_us'][1]},
                ca={'init':actual0,'tiles':tls(ca['tiles']),'full_us':st['ca_us']})


def draw(j,path):
    chunks=j['chunks'];n=len(chunks)
    fig,axs=plt.subplots(2,1,sharex=True,figsize=(14,4.75+1.8*n))
    fig.subplots_adjust(left=.24,right=.975,top=.815,bottom=.105,hspace=.26)
    _,t,h,*_=j['shape'];sku=j['sku'].split('_')[0].replace('Ascend','')
    fig.text(.24,.967,f'{sku} KDA FwdH  |  {t} tokens × {h} heads',size=22,weight='bold',color='#183542')
    fig.text(.24,.927,f'{n} recurrent chunks  ·  head dimension 128  ·  one AIC + paired AIV group per chain  ·  archived FLA 26.9.1 V2',size=11,color='#456575')
    labels=[('W','W'),('H','H'),('K','Kg'),('right','right'),('mm0','MMAD'),('U','U'),
            ('corrected','corrected VF'),('state','state VF'),('write_right','right write'),('write_state','H / VK write')]
    fig.legend([Patch(color=STYLE[k][0]) for k,l in labels],[l for k,l in labels],loc='upper left',
               bbox_to_anchor=(.234,.909),frameon=False,ncol=5,fontsize=9.6,handlelength=1.8,columnspacing=1.4)
    tlabels=['Initial FP32 → BF16 H']+[f'C{i+1} · {count} {"row" if count==1 else "rows"}  |  {s}' for i,count in enumerate(chunks) for s in LABELS]
    stop=max(j['ca']['full_us'],j['model']['full_us'])
    unit=j['us_per_tick']
    for ax,side,title in zip(axs,['model','ca'],['Resource model · recurrent critical chain',f'CA trace · {j["core"]}, last to finish: {j["last_core"]}']):
        ev=j[side]
        for k,(start,end) in ev['init'].items():
            color,offset=STYLE[k]
            ax.barh(offset,(end-start)*unit,left=start*unit,height=.14,color=color,linewidth=0,zorder=3)
        for i,chunk in enumerate(ev['tiles']):
            for s,group in enumerate(SECTIONS):
                for k in group:
                    start,end=chunk[k];color,offset=STYLE[k];y=1+i*4+s+offset
                    ax.barh(y,(end-start)*unit,left=start*unit,height=.14,color=color,linewidth=0,zorder=3)
            ax.axhline(.5+i*4,color='#b8cbd3',lw=.8,zorder=1)
        ax.set_ylim(len(tlabels)-.5,-.5);ax.set_xlim(0,stop*1.025);ax.set_yticks(range(len(tlabels)),tlabels)
        ax.tick_params(length=0,labelsize=8,pad=4);ax.spines[:].set_visible(False)
        ax.grid(axis='x',color='#d2dce3',lw=.8);ax.set_axisbelow(True)
        ax.axvline(ev['full_us'],lw=1.0,ls=(0,(3,3)),color='#3b5463')
        ax.set_title(title,loc='left',weight='bold',color='#193746',pad=7,fontsize=13)
        ax.set_title(f'{ev["full_us"]:.3f} µs',loc='right',weight='bold',color='#193746',fontsize=15,pad=7)
    axs[1].set_xlabel('Time since the earliest FwdH device instruction (µs)',fontsize=11,labelpad=7)
    fig.text(.24,.045,'Each bar is one named resource window. W and U for the next chunk may overlap this chunk; H waits for the preceding state write.',size=9.6,color='#405d6b')
    fig.text(.24,.022,'CA shows descriptor issue to retirement. Unshown scalar, flags, L1 → L0 and FIX operations still contribute to the marked full kernel window.',size=9.3,color='#405d6b')
    fig.savefig(path,dpi=140,bbox_inches='tight',facecolor='white');plt.close(fig)


if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('receipt');cli.add_argument('name');cli.add_argument('--snapshot',action='store_true')
    args=cli.parse_args();p=Path(args.receipt).resolve()
    if args.snapshot:j=read(p)
    else:
        j=materialize(p);(OUT/(args.name+'.json')).write_text(json.dumps(j,indent=2)+'\n')
    draw(j,OUT/(args.name+'.png'))
