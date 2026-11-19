"""Draw the same work units on two axes with a common zero and microsecond scale."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from extract import OUT, ROOT, read, sha, extract
from model import predict

COLORS={'post':'#377c90','comb':'#699ba6','x':'#127887','res':'#57aeb5','arith':'#654399','cast':'#bc8431','write':'#bb4d29'}
OFF={'post':-.14,'comb':.14,'x':-.25,'res':-.12,'arith':.01,'cast':.14,'write':.27}


def draw(file, prefix):
    rc=read(file)
    compact=OUT/'primitives'/(file.stem+'.json')
    if compact.exists():
        data=read(compact)
        if data['trace_sha256']!=rc['stage']['trace_sha256']:raise ValueError('Trace receipt mismatch')
    else:
        data=extract(file)
    chip=data['sku']; _,t,_,d=data['shape']
    model=predict(t,d,chip,timeline=True)
    name,critical=max(data['cores'].items(),key=lambda item:item[1]['end'])
    actual=[]
    for row_id,row in enumerate(critical['rows']):
        for key in ['post','comb']:
            e=row[key];actual.append([row_id,0,0,key,e['begin'],e['end']])
        for idx,tile in enumerate(row['tiles']):
            for key in ['x','res','arith','cast','write']:
                e=tile[key];actual.append([row_id,idx,tile['width'],key,e['begin'],e['end']])
    def lane(entries):
        return [(r,i,w) for r,i,w,k,a,b in entries if k in ('post','x')]
    # The selected panels have the same local row geometry.
    lanes=lane(actual)
    if lanes!=lane(model['bars']):raise ValueError('Different model/CA critical-core workload; choose a different source or explicitly map both profiles')
    factor=1650
    assert abs(rc['ca_us']-data['window_ticks']/factor)<0.001
    record=dict(shape=data['shape'],sku=chip,role=rc['role'],source=str(file.relative_to(ROOT)),
                trace_sha256=data['trace_sha256'],device_text_sha256=data['device_text_sha256'],model_sha256=sha(Path(__file__).with_name('model.py')),
                lane_order=['local_row','fragment','useful_width','stage','issue_tick','end_tick'],cycles_per_us=factor,
                zero='earliest instruction on any active core',model=model,ca={'core':name,'first_rvec_tick':critical['first_rvec'],
                'full_us':rc['ca_us'],'bars':actual,'start_tick':critical['start'],'end_tick':critical['end'],'meta':critical['meta']})
    (OUT/(prefix+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    labels=['Row coefficients' if w==0 else f'Fragment {i+1}  ·  {w:,} D' for r,i,w in lanes]
    fig,axs=plt.subplots(2,1,figsize=(13.5,8.3),sharex=True)
    fig.subplots_adjust(left=.18,right=.96,top=.795,bottom=.16,hspace=.48)
    fig.text(.18,.965,f'{chip.split("_")[0].replace("Ascend","")} HcPost  |  [1, {t}, 4, {d:,}]',size=22,weight='bold',color='#193342')
    fig.text(.18,.92,f'{model["active_aiv"]} AIVs    ·    BF16 activations / FP32 coefficients    ·    {len(lanes)-1} feature fragments per row',size=12,color='#496273')
    tags={'post':'post / comb (MTE2)','x':'x (MTE2)','res':'residual (MTE2)','arith':'MulAdd VF','cast':'CAST VF','write':'output (MTE3)'}
    fig.legend([Patch(color=COLORS[k]) for k in tags],tags.values(),loc='upper left',bbox_to_anchor=(.175,.894),frameon=False,ncol=6,fontsize=10,columnspacing=1.5,handlelength=1.7)
    end=max(model['device_us'],rc['ca_us'])
    xmax=math.ceil(end*1.025/(1 if end<14 else 2))*(1 if end<14 else 2)
    for ax,bars,title,stop in zip(axs,[model['bars'],actual],['Resource model · critical cohort',f'CA trace · {name} (last to finish)'],[model['device_us'],rc['ca_us']]):
        for row,idx,w,k,begin,finish in bars:
            y=lanes.index((row,idx,w))+OFF[k]
            ax.barh(y,(finish-begin)/factor,left=begin/factor,height=.105 if idx or w else .18,color=COLORS[k],linewidth=0,zorder=3)
        if ax==axs[0]:
            ax.barh(-.27,model['bars'][0][4]/factor,left=0,height=.11,color='#cedde2',linewidth=0)
            ax.text(.04,-.29,'Launch / coefficient issue',va='center',size=8,color='#496273')
        else:
            m=critical['meta']
            ax.barh(-.28,(m['end']-m['begin'])/factor,left=m['begin']/factor,height=.10,color='#899eaa',linewidth=0)
            ax.axvline(critical['start']/factor,color='#899eaa',lw=.7,ls=':')
            ax.plot(critical['first_rvec']/factor,1+OFF['arith'],'|',color='black',ms=7,mew=1.25,zorder=4)
        for y in range(1,len(lanes)):ax.axhline(y-.5,color='#e5ebef',lw=.65,zorder=0)
        ax.axvline(stop,color='#334e60',dashes=[3,3],lw=1.1,zorder=4)
        ax.set_title(title,loc='left',fontsize=13,weight='bold',pad=9,color='#193342')
        ax.set_title(f'{stop:.3f} µs',loc='right',fontsize=15,weight='bold',pad=8,color='#193342')
        ax.set(yticks=range(len(lanes)),yticklabels=labels,xlim=(0,xmax),ylim=(len(lanes)-.52,-.48))
        ax.spines[:].set_visible(False);ax.tick_params(length=0,pad=6,labelsize=10)
        ax.grid(axis='x',color='#d5e0e6',lw=.8);ax.set_axisbelow(True)
    axs[-1].set_xlabel('Time since earliest device instruction (µs)',size=12,labelpad=8)
    delta=(model['device_us']/rc['ca_us']-1)*100
    fig.text(.18,.07,f'Full device window   {delta:+.2f}% prediction error.  Colors track issue-to-retirement intervals; overlapping VF descriptors include dispatch.',fontsize=10,color='#405b6a')
    fig.text(.18,.039,'Grey CA segment = 96-byte tiling read.  Black tick = first RVEC child.  Every output CAST converts 16,384 entries, including the short final fragment.',fontsize=9.4,color='#405b6a')
    fig.savefig(OUT/(prefix+'.png'),dpi=140,bbox_inches='tight',facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('receipt');parser.add_argument('prefix')
    args=parser.parse_args();draw(Path(args.receipt).resolve(),args.prefix)
