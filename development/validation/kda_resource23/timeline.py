"""Two panels per kernel, same local origin and scale. Qualified DT S2 only."""
from collections import defaultdict
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from model import DT, FAMILIES, OUT, ROOT, predict, read
from extract import extract
from pathlib import Path
import argparse

NAMES=[['Front end','V0 gates','V1 pack','C2 scores','V3 inverse','C4/C5 + V6','C7 W/U'],
       ['Front end','Initial state','S0 W @ H','S1 values','S2 update','S3 state'],
       ['Front end','Gather inputs','QH + AqkV','Output']]
from plot_schedule import COLORS,model_bars


def ca(st, index):
    lanes=defaultdict(list)
    def take(e,row,label):
        lanes[row].append((e['begin']/1650,e['end']/1650,label))
    def split(items,counts,rows,label):
        assert len(items)==sum(counts)
        pos=0
        for count,row in zip(counts,rows):
            for e in items[pos:pos+count]:take(e,row,label)
            pos+=count
    def route(key,counts,rows,label):
        split([e for e in st['transfers'] if e['route']==key],counts,rows,label)
    if index==0:
        route('OUT->UB',[2,6,4],[0,1,5],'MTE2')
        route('OUT->L1',[6,1,2],[3,5,6],'MTE2')
        route('UB->OUT',[3,2,6,4],[1,2,4,5],'MTE3 / FIXP')
        route('L1->L0A',[2,1],[3,6],'MTE1')
        route('L1->L0B',[1,2],[3,6],'MTE1')
        route('L0C->UB',[1],[3],'MTE3 / FIXP')
        route('L0C->OUT',[2],[6],'MTE3 / FIXP')
        split(st['vfs'],[1]*4,[1,2,4,5],'Vector VF')
        split(st['mmads'],[1,2],[3,6],'Cube')
    elif index==1:
        route('OUT->UB',[1,1,2],[1,3,4],'MTE2')
        route('OUT->L1',[2,2],[2,4],'MTE2')
        route('UB->OUT',[1,2,1],[1,3,5],'MTE3 / FIXP')
        for key in ('L1->L0A','L1->L0B'):
            route(key,[1,1],[2,4],'MTE1')
        route('L0C->UB',[1,1],[2,4],'MTE3 / FIXP')
        split(st['vfs'],[1]*3,[1,3,5],'Vector VF')
        split(st['mmads'],[1,1],[2,4],'Cube')
    else:
        route('OUT->L1',[4],[1],'MTE2')
        for key in ('L1->L0A','L1->L0B'):
            route(key,[2],[2],'MTE1')
        route('L0C->OUT',[1],[3],'MTE3 / FIXP')
        split(st['mmads'],[2],[2],'Cube')
    return lanes


def render(receipt=None):
    if receipt is None:
        n=2;h=1;sku=DT;ref=read(OUT/'dt-primitive-services.json')['2']['stages']
        pred=predict([1,2,1,1,128,128],DT)
        label='development';out=OUT/'dt-s2-stage-comparison.png'
    else:
        data=read(receipt)
        assert (data['qualified'] and data['case']['id'].startswith('holdout-')
                and data['case']['shape'] in ([1,3,1,1,128,128],[1,5,2,2,128,128]))
        shape=data['case']['shape'];n=shape[1];h=shape[2];sku=data['case']['sku']
        ref=extract(receipt)['stages']
        for actual in data['stages']:
            st=ref[actual['family']]
            terminal=max(actual['cores'],key=lambda k:actual['cores'][k]['last_tick'])
            group=terminal.split('.')[0]
            st['terminal_workgroup']=group
            for key in ('mmads','transfers','vfs'):
                st[key]=[v for v in st[key] if v['core'].split('.')[0]==group]
        pred=next(x for x in read(OUT/'freeze.json')['predictions'] if x['shape']==shape and x['chip']==sku)
        label='frozen holdout';stem=sku.removeprefix('Ascend')[:5].lower()
        out=OUT/f'{stem}-t{n}-holdout-comparison.png'
        anchor=dict(view='registered holdout, stage-local cycles',receipt=str(receipt.relative_to(ROOT)),
                    sku=sku,shape=shape,model_freeze_sha256='7e278842c78dca05192219b2e8a36512b9e3b4180d277dc2cd1200081b0b61f1',stages=ref)
        out.with_suffix('.json').write_text(json.dumps(anchor,indent=2)+'\n')
    fig=plt.figure(figsize=(16,7.4))
    grid=fig.add_gridspec(2,3,width_ratios=[1.35,1,0.87],left=.09,right=.99,bottom=.18,top=.77,hspace=.34,wspace=.45)
    fig.suptitle(f'{sku.removeprefix("Ascend").split("_")[0]}   |   Chunk KDA V2   |   {n} tokens × {h} local head{"s" if h!=1 else ""}, K = V = 128',
                 x=.055,y=.976,ha='left',fontsize=21,weight='bold',color='#182f3b')
    fig.text(.055,.912,f'Official FLA 26.9.1, direct BF16 V2 · {label}. Each column starts at its own first device instruction.',
             fontsize=11,color='#3c5665')
    fig.legend(handles=[Patch(facecolor=v,label=k) for k,v in COLORS.items() if k!="Shared phase"],
               loc='upper left',bbox_to_anchor=(.05,.886),ncol=7,frameon=False,fontsize=10)
    labels=['Prepare','FwdH','Finalize']
    for i,st in enumerate(pred['stages']):
        obs=ref[FAMILIES[i]]
        window=(obs['ca_window_ticks'][1]-obs['ca_window_ticks'][0])/1650
        phase=list(next(iter(st['representative_tile_phase_cycles'].values())).values())
        values=[st['first_payload_cycles']]+phase
        names=NAMES[i];assert len(values)==len(names)
        model=fig.add_subplot(grid[0,i]);meas=fig.add_subplot(grid[1,i],sharex=model)
        for ax,lanes in ((model,model_bars(i,st,n,sku)),(meas,ca(obs,i))):
            for r,evs in lanes.items():
                for a,b,k in evs:
                    off={'MTE2':-.21,'MTE1':-.10,'Cube':0,'Vector VF':.10,'MTE3 / FIXP':.21,'Front end':0}[k]
                    ax.plot([a,b],[r+off]*2,lw=3.6,solid_capstyle='butt',color=COLORS[k])
                    if k=='Cube':ax.plot([a,a],[r-.14,r+.14],lw=.7,color=COLORS[k])
        for ax,title,t in ((model,labels[i]+' · model',st['device_us']), (meas,labels[i]+' · CA'+(' / '+obs['terminal_workgroup'] if h>1 else ''),window)):
            ax.set_yticks(range(len(names)),names,fontsize=9)
            ax.set_ylim(len(names)-.4,-.6);ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
            ax.set_xlim(0,1.05*max(st['device_us'],window));ax.axvline(t,color='#253e50',ls='--',lw=1)
            for sp in ax.spines.values():sp.set_visible(False)
            ax.tick_params(axis='both',length=0,pad=3)
            ax.set_title(title,loc='left',fontsize=12,fontweight='bold')
            ax.set_title(f'{t:.3f} µs',loc='right',fontsize=11)
        meas.set_xlabel('Device time (µs)')
    fig.text(.055,.095,'Top: resources scheduled from descriptor bytes, VF and MMAD work. Bottom: measured transfers, vector functions and MMADs of the last finishing workgroup, grouped by the same source phase.',fontsize=10)
    fig.text(.055,.059,'Dashes mark all-core windows. Front end is a modeled control budget. CA scalar/control and waits, and kernel-to-kernel dispatch gaps, are omitted.',fontsize=10)
    fig.savefig(out,dpi=150)
    plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--holdout-receipt',type=Path)
    args=parser.parse_args();render(args.holdout_receipt.resolve() if args.holdout_receipt else None)
