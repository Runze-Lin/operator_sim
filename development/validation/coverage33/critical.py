"""Compare model endpoints against archived critical-AIV physical descriptor windows."""
from pathlib import Path
import json
import statistics
import sys
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
sys.path.append(str(Path(__file__).resolve().parent))
from rms_archive import moves, OUT, ROOT, CLOCK, save, read, DT, PR

def core_data(r):
    core=max(r['cores'],key=lambda c:r['cores'][c]['merged_window_us'][1])
    source=ROOT/r['trace'];pc=int((source.parent.parent/'dump/pc_start_addr.txt').read_text(),16)
    observed=r['cores'][core]
    origin=observed['farthest_retire']-round(CLOCK*observed['merged_window_us'][1])
    lanes={k:[] for k in ['MTE2 · X','MTE3 · rstd','MTE3 · Y']}
    for event in moves(source.parent/core/'trace.json'):
        detail=json.loads(event['args']['detail'])
        if any(detail.get(k) for k in ('is_wait','is_duration_excluded','is_sync')):continue
        key={0x4a48:'MTE2 · X',0x4f38:'MTE3 · rstd',0x5128:'MTE3 · Y'}.get(int(event['args']['pc_addr'],16)-pc)
        if key:
            begin=round(event['ts']*CLOCK)-origin
            retire=round((event['ts']+event['dur'])*CLOCK)-origin
            lanes[key].append(dict(start_tick=begin,retire_tick=retire,bytes=detail['processed_bytes'],burst_count=detail['burst_number']))
    numbers={}
    for key,vals in lanes.items():
        vals.sort(key=lambda v:v['start_tick'])
        full=[x for x in vals if x['bytes']==max(v['bytes'] for v in vals)]
        issue=[b['start_tick']-a['start_tick'] for a,b in zip(full,full[1:])]
        service=[v['retire_tick']-v['start_tick'] for v in full]
        numbers[key]=dict(full_descriptor_count=len(full),median_issue_ticks=statistics.median(issue),median_descriptor_ticks=statistics.median(service))
    return dict(chip=r['chip'],shape=r['shape'],core=core,origin_shard_tick=origin,trace_sha256=r['trace_sha256'],ca_us=r['ca_us'],lanes=lanes,summary=numbers)

def main():
    coverage=read(OUT/'coverage.json')
    pair=next(pair for pair in coverage['rms_pairs'] if pair[0]['shape']==[8192,5120])
    data=[]
    for row in pair:
        x=core_data(read(ROOT/row['receipt']));x['price_us']=row['price_us'];data.append(x)
    save(OUT/'rms-critical.json',dict(clock_ticks_per_us=CLOCK,model_version=coverage['version'],rows=data))
    fig,axs=plt.subplots(2,1,figsize=(10.2,5.8),sharex=True)
    colors=['#157888','#996418','#ab4f32']
    for ax,r in zip(axs,data):
        for k,(label,values) in enumerate(r['lanes'].items()):
            for v in values:
                x=v['start_tick']/CLOCK;y=2-k;span=(v['retire_tick']-v['start_tick'])/CLOCK
                ax.broken_barh([(x,span)],(y-.21,.42),facecolors=colors[k],alpha=.68)
                ax.plot(x,y,marker='|',markersize=6,color=colors[k])
        ax.hlines(3,0,r['price_us'],color='#615189',lw=3)
        ax.plot(r['price_us'],3,'|',color='#615189',ms=13)
        ax.axvline(r['ca_us'],lw=1.1,ls=':',color='#24394b')
        ax.set_yticks(range(4),['MTE3 · Y','MTE3 · rstd','MTE2 · X','Model window'])
        ax.set_title(f"{r['chip']}   |   {r['core']}   |   model {r['price_us']:.2f} µs, CA {r['ca_us']:.2f} µs",loc='left',size=11)
        ax.grid(axis='x',lw=.5,alpha=.5);ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
        ax.set_ylim(-.5,3.55)
    axs[1].set_xlabel('Device time since earliest instruction on any core (µs)')
    xmax=max(max(r['ca_us'],r['price_us']) for r in data)*1.035
    axs[0].set_xlim(0,xmax);axs[0].xaxis.set_major_locator(MaxNLocator(8))
    fig.suptitle('RMSNorm   |   8192 tokens × 5120 hidden channels',size=16,x=.16,ha='left',fontweight='bold')
    fig.text(.16,.035,'Top line: complete model window. Bars: physical DMA issue through descriptor retirement; waiting is included.\nOverlapping intervals share the same lane. The dotted line marks full-device CA retirement.',size=9,color='#3d5263')
    fig.subplots_adjust(left=.16,right=.97,hspace=.48,top=.85,bottom=.22)
    fig.savefig(OUT/'rms-critical.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
