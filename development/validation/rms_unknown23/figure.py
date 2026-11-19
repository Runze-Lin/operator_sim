"""Compare the registered H=1 and H=2 resource paths with measured last-to-finish AIV descriptors."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from operator_model import load_chip
from operator_model.rms_dispatch import price_rms
from development.validation.rms_ordered_verify.verify import digest,events
from development.validation.sku19.campaign import write
OUT=ROOT/'results/rms-unknown23'

def build():
    facts=[]
    for p in sorted((OUT/'verified').glob('wide-*.json')):
        data=json.loads(p.read_text());chip=load_chip(data['chip'])
        for r in data['rows']:
            if r['shape']!=[2,2944] or not r['timing_qualified']:continue
            s=r['stage'];first=s['first_tick']/1650;core=max(s['cores'],key=lambda c:s['cores'][c]['last_tick']);trace=ROOT/s['trace'];shard=trace.parent/core/'trace.json';assert digest(shard)==s['cores'][core]['shard_sha256'];base=int((trace.parent.parent/'dump/pc_start_addr.txt').read_text(),16)
            price,_,_,diag=price_rms(r['shape'],chip,{'resident_aiv':r['prior_aiv']});assert abs(price-r['known_us'])<1e-12
            t=0.;stage=[]
            for name,val in diag['stages_us'].items():stage.append([name,t,t+val]);t+=val
            descriptors=[]
            for e in events(shard):
                if e['name']!='VF' and not e['name'].startswith('MOV_SRC_TO_DST'):continue
                pc=int(e['args']['pc_addr'],16)-base;desc=json.loads(e['args']['detail']);name={0x3e9c:'tiling input',0x4878:'gamma',0x4a48:'x',0x4d1c:'reduce',0x4e64:'rstd VF',0x5008:'normalize',0x4f38:'rstd write',0x5128:'y write'}[pc]
                descriptors.append(dict(name=name,pc=hex(pc),bytes=desc.get('processed_bytes'),start=e['ts']-first,finish=e['ts']+e['dur']-first))
            assert len(descriptors)==8
            facts.append(dict(chip=chip.name,H=r['prior_aiv'],known_us=price,unknown_us=r['candidate_us'],old_unknown_us=r['previous_us'],ca_us=s['ca_us'],critical_core=core,critical_lane=diag['lane_schedule']['lane_class'],actual_arrival_us=(s['cores'][core]['first_tick']-s['first_tick'])/1650,stages=stage,ca=descriptors,trace=s['trace'],trace_sha256=s['trace_sha256'],shard_sha256=digest(shard)))
    write(OUT/'paired-2944.json',facts)
    return facts


def draw(data):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/rms23-plot-mpl')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    colors=['#61788a','#167e88','#7354a4','#b35b39'];fig,axes=plt.subplots(4,2,figsize=(12.5,10.0),sharex=True)
    for row in data:
        chip=row['chip'];H=row['H'];col=0 if '950DT' in chip else 1;row_i=0 if H==1 else 2;top=axes[row_i,col];bottom=axes[row_i+1,col]
        for panel in [top,bottom]:
            panel.set_xlim(0,4.85);panel.set_ylim(3.65,-.65);panel.set_yticks(range(4),['Dispatch','MTE2 read','Vector','MTE3 write'] if col==0 else ['','','','']);panel.grid(axis='x',color='#dce3e9');panel.set_axisbelow(True);panel.spines[['top','left','right']].set_visible(False);panel.tick_params(axis='y',length=0,pad=4)
            panel.axvline(row['unknown_us'],color='#384b5b',ls=':',lw=1.2)
        for name,a,b in row['stages']:
            lane=0 if name in ('arrival','frontend_to_rstd_issue','drain') else 1 if name=='MTE2' else 2 if name.startswith('SIMD') else 3
            top.barh(lane,b-a,left=a,color=colors[lane],height=.40,alpha=.55 if name=='arrival' else 1)
        c=row['ca'];arrival=row['actual_arrival_us'];bottom.plot(arrival,0,marker='|',ms=13,color=colors[0],mew=2)
        for inst in c:
            name=inst['name'];lane=1 if name in ('tiling input','gamma','x') else 2 if name in ('reduce','rstd VF','normalize') else 3
            idx={'tiling input':-.20,'gamma':0,'x':.20,'reduce':-.20,'rstd VF':0,'normalize':.20,'rstd write':-.13,'y write':.13}[name]
            bottom.barh(lane+idx,inst['finish']-inst['start'],left=inst['start'],height=.16,color=colors[lane]);
        for ax,val in [(top,row['known_us']),(bottom,row['ca_us'])]:ax.axvline(val,color='#1c3240',ls='--',lw=1)
        cls='H=1 · opens second partition' if H==1 else 'H=2 · immediate full reuse'
        top.set_title(f'{chip[6:11]}   {cls}\nModel · {row["critical_lane"]} · {row["known_us"]:.3f} µs',loc='left',fontsize=11,fontweight='bold')
        bottom.set_title(f'CA · {row["critical_core"]} · {row["ca_us"]:.3f} µs',loc='left',fontsize=10)
        if row_i==2:bottom.set_xlabel('Time from earliest core instruction (µs)')
    fig.suptitle('RMSNorm  |  [2, 2944]  |  same inputs on successive launches',x=.11,ha='left',y=.985,fontsize=17,fontweight='bold');fig.legend(handles=[Patch(color=c,label=t) for c,t in zip(colors,['Dispatch / scalar','MTE2','Vector','MTE3'])]+[Line2D([0],[0],color='#384b5b',ls=':',label='Unknown-H point'),Line2D([0],[0],color='#1c3240',ls='--',label='Window end')],loc='upper left',bbox_to_anchor=(.105,.957),ncol=6,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.11,right=.985,top=.86,bottom=.105,hspace=.63,wspace=.09);fig.text(.11,.028,'Model bars give additive critical-path stages. CA shows the last physical AIV’s input DMA, VF issue-to-completion and output DMA windows.\nThe short tick on the CA dispatch lane marks the core’s first instruction. The dotted line is identical for both histories on each chip.',fontsize=9,color='#425866');fig.savefig(OUT/'paired-2944.png',dpi=150);plt.close(fig)

if __name__=='__main__':draw(build())
