"""Plot full-Kimi resource phases and archived raw DDR service with explicit clock uncertainty."""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import re
import sys
from study import ROOT,OUT,FROZEN,DT,read,store

sys.path.append(str(ROOT/'development/validation/kda27'))
from quant_service import HEADER,PANEL,NAMES,FIELDS


def extract():
    folder=ROOT/'runs/kimi-full-quant20/log'
    files=sorted(folder.glob('lpddr_state*.log'))
    assert len(files)==128
    bins=defaultdict(lambda:[0,0,0]);fingerprints={};horizons=set()
    for path in files:
        data=path.read_bytes();fingerprints[path.name]=sha256(data).hexdigest()
        prev=0
        for p in data.split(PANEL)[1:]:
            end,dmc,ddr=map(int,HEADER.search(p,0,400).groups())
            at=p.index(b' Request Statistics ');stop=p.index(b' Latency Statistics ',at)
            a={k:int(v) for k,v,_ in FIELDS.findall(p,at,stop)}
            r=sum(a[k] for k in NAMES[:2]);w=sum(a[k] for k in NAMES[2:6])
            assert 32*(r+w)==ddr and 0<end-prev<=1000
            # End=1000 is the period [0,1000); final period may be partial.
            v=bins[(end-1)//8000];v[0]+=r;v[1]+=w;v[2]+=end-prev;prev=end
        horizons.add(end)
    assert len(horizons)==1
    horizon=horizons.pop()
    archive=read(ROOT/'results/v0.27/quant-channel-service.json')['rows'][0]
    full=archive['full']['summed_channel_counts']
    assert sum(q[0] for q in bins.values())==full['read'] and sum(q[1] for q in bins.values())==full['write']
    periods=[dict(start_us=i*8,end_us=min((i+1)*8,horizon/1000),read_beats=v[0],write_beats=v[1],dfi_channel_cycles=v[2])
             for i,v in sorted(bins.items())]
    frozen=read(FROZEN);forecast=frozen['predictions']['historical_b8'][DT]
    native=read(ROOT/'results/v0.20/full-quant/native.json')
    starts=native['core_ticks'];origin=min(x for x,y in starts.values())
    kernel={c:[(a-origin)/1650,(b-origin)/1650] for c,(a,b) in starts.items()}
    gross=native['core_envelope_us']
    out=dict(shape=native['shape'],sku=DT,prediction=forecast,native_us=gross,
             dfi_us=horizon/1000,wrapper_start_in_dfi_us=[0,horizon/1000-gross],
             clock_note='All-AIV wrapper and controller DFI use independent origins. Given containment, only the origin interval is identified.',
             native_source='results/v0.20/full-quant/native.json',model_source='results/v0.29/quant/freeze.json',controller_source=str(folder.relative_to(ROOT)),
             raw_sha256=fingerprints,native_windows=kernel,controller_periods=periods)
    store(OUT/'b8-service.json',out)
    return out


def draw():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    data=read(OUT/'b8-service.json') if (OUT/'b8-service.json').exists() else extract()
    p=data['prediction'];ca=data['native_us'];last=data['dfi_us']
    fig,ax=plt.subplots(2,1,figsize=(11.7,7.2),sharex=True,gridspec_kw={'height_ratios':[1,1.2]})
    fig.subplots_adjust(left=.19,right=.96,top=.78,bottom=.18,hspace=.43)
    fig.text(.19,.935,'950DT  |  DynamicMXQuant  ·  262,144 × 7,168',fontsize=21,weight='bold',color='#193744')
    fig.text(.19,.88,'Kimi prefill B8 × 32 Ki   ·   64 AIVs   ·   820 critical UB tiles   ·   128 memory channels',fontsize=11,color='#496574')
    fig.text(.19,.835,'BF16 input in GM     →     local 32-element scale groups     →     E4M3FN values + E8M0 scales',fontsize=11,color='#496574')
    a,b=ax
    a.set_title('Resource model · sequential memory service',loc='left',weight='bold',fontsize=13)
    a.set_title(f"{p['latency_us']:,.1f} µs",loc='right',weight='bold',fontsize=13)
    a.barh(0,p['prefix_us'],color='#197a7e',height=.39);a.barh(1,p['mixed_us'],left=p['prefix_us'],color='#6478ad',height=.39)
    for val,color,label in [(p['base_us'],'#85644a','Public'),(p['old_resource_us'],'#52717e','Old resource')]:
        a.plot(val,2,marker='v',color=color);a.text(val,2.34,label,ha='center',fontsize=9,color=color)
    a.set_yticks([0,1,2],['Read prefix','Mixed DDR service','Earlier point estimates']);a.set_ylim(2.68,-.53)
    a.axvline(p['latency_us'],ls='--',lw=1.2,color='#2e4650')
    b.set_title('CA · 128-channel DDR data and all-AIV wrapper',loc='left',weight='bold',fontsize=13)
    b.set_title(f'{ca:,.1f} µs',loc='right',weight='bold',fontsize=13)
    bins=data['controller_periods'];xx=[u['start_us'] for u in bins]+[last]
    for i,key in enumerate(('read_beats','write_beats')):
        y=np.array([u[key]/u['dfi_channel_cycles'] for u in bins]);y=np.append(y,y[-1])
        b.fill_between(xx, i+.34-.70*y,i+.34,step='post',color=('#197a7e','#bc602e')[i],lw=0)
    wins=sorted(data['native_windows'].values(),key=lambda r:r[1])
    b.fill_between([0,ca],[1.82,1.82],[2.12,2.12],color='#b5c5cb')
    for a0,z in wins:
        b.vlines(z,1.79,2.15,color='#405d6a',lw=.58)
    b.axvline(ca,ls='--',lw=1.2,color='#2e4650');b.set_yticks([0,1,2],['DDR read occupancy','DDR write occupancy','AIV wrapper ends']);b.set_ylim(2.48,-.48)
    for panel in ax:
        panel.tick_params(axis='y',length=0);panel.grid(axis='x',color='#dce6eb',lw=.7);panel.spines[['top','right','left']].set_visible(False);panel.spines['bottom'].set_color('#adc0c8')
    b.set_xlim(0,1950);b.set_xlabel('Elapsed device time (µs) from each source’s own start',labelpad=9)
    fig.text(.19,.074,'DDR bands show measured bus duty in 8 µs bins. Ticks mark physical AIV wrapper ends; per-instruction timing is unavailable.',fontsize=9,color='#45606e')
    fig.text(.19,.044,f'Clock alignment: wrapper start is contained within the first {last-ca:.3f} µs of DDR reporting. No bus beat is assigned to a UB tile.',fontsize=9,color='#45606e')
    fig.savefig(OUT/'b8-service.png',dpi=160);plt.close(fig)

if __name__=='__main__':draw()
