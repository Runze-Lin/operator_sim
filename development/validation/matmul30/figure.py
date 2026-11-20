"""Render frozen wave budgets vs last-core physical descriptors on the same all-core time origin."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from freeze import OUT,read

r=read(OUT/'critical-1048576-dt.json')
spec,=[x for x in read(OUT/'freeze.json')['rows'] if x['shape']==r['shape'] and x['chip']==r['chip']]
p=spec['candidate'];scale=1650;tiles=r['tiles'];ca=r['qualified_ca_us']
colors={'core':'#65519b','memory_supply':'#187b83','l2':'#4778b0'}
ink='#18303d';grey='#d9e1e4';rust='#b86536'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':ink,'axes.labelcolor':ink,'xtick.color':ink,'ytick.color':ink})
fig,(a,b,z)=plt.subplots(3,1,figsize=(13,8.1),gridspec_kw=dict(height_ratios=[1.1,1.5,1.8],hspace=.72))
fig.subplots_adjust(left=.155,right=.975,top=.80,bottom=.145)
fig.text(.055,.942,'950DT  MatMulV3  |  1,048,576 × 128 × 512',fontsize=23,weight='bold')
fig.text(.055,.895,'DeepSeek V4.1 Flash   ·   B8 × 128 Ki prefill   ·   32 AICs   ·   128 complete outputs per AIC',fontsize=12)
for ax in (a,b,z):
    ax.grid(axis='x',color=grey,lw=.7)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.tick_params(axis='y',length=0,pad=6)
for ax in (a,b):
    ax.set_xlim(0,420);ax.set_xticks(range(0,401,50));ax.set_yticks([])
a.set_title(f'Capacity and resource schedule   ·   {p["latency_us"]:.3f} µs',loc='left',fontweight='bold',pad=10)
a.set_ylim(-.6,.7);a.set_yticks([0],['Output wave'])
start=p['first_cube_cycles']/scale;durations=[]
wavedata=[]
for group in p['segments']:
    for wi in range(group['first_wave'],group['exclusive_end_wave']):
        duration=group['cycles_per_wave']/scale
        c=colors[group['limiter']]
        a.broken_barh([(start,duration)],(-.20,.40),facecolors=c,zorder=3)
        a.vlines(start,-.20,.2,color='white',lw=.25,zorder=4)
        wavedata.append(dict(index=wi,start_us=start,end_us=start+duration,limiter=group['limiter']))
        durations.append(group['cycles_per_wave']);start+=duration
transition=next(d['start_us'] for d in wavedata if d['limiter']=='memory_supply')
a.axvline(transition,color=ink,ls=':',lw=1);a.annotate('Predicted output spill',xy=(transition,.22),xytext=(transition+4,.43),ha='left',fontsize=9)
a.axvline(p['latency_us'],ls='--',color=ink,lw=1)
a.text(34,.43,'Cube issue limits wave rate',color=colors['core'],fontsize=9)
a.text(308,.43,'Mixed read + write service',color=colors['memory_supply'],fontsize=9)
b.set_title(f'CA trace   ·   {r["actual_core"]}   ·   {ca:.3f} µs',loc='left',fontweight='bold',pad=8)
b.set_ylim(2.55,-.55);b.set_yticks([0,1,2],['MTE2 · A pages','Cube · MMAD','FIXP · BF16 store'])
for i,(key,color) in enumerate((('a_pages',colors['memory_supply']),('mmads',colors['core']),('write',rust))):
    for t in tiles:
        spans=[t[key]] if key=='write' else t[key]
        b.broken_barh([(lo/scale,(hi-lo)/scale) for lo,hi in spans],(i-.14,.28),facecolors=color,alpha=.8 if i==2 else 1,zorder=3)
b.axvline(ca,ls='--',color=ink,lw=1);b.set_xlabel('Device time since earliest participating instruction (µs)')
observed=[tiles[i+1]['mmads'][0][0]-t['mmads'][0][0] for i,t in enumerate(tiles[:-1])]
z.set_title('Interval between the first MMAD of consecutive output tiles',loc='left',fontweight='bold',pad=9)
z.plot(range(1,128),observed,lw=.9,color='#637780',alpha=.9,label='CA on the last AIC')
z.step(range(1,129),durations,where='post',lw=2,color=colors['memory_supply'],label='Capacity and supply model')
z.axhline(4112,ls='--',lw=1.25,color=colors['core'],label='v0.29 per-core service')
z.legend(loc='upper left',frameon=False,ncol=3,fontsize=9)
z.set_xlim(0,129);z.set_ylim(3000,max(observed)*1.13);z.set_ylabel('Device cycles');z.set_xlabel('Output tile on the critical AIC')
fig.text(.055,.060,'The purple and teal wave costs are modeled bottlenecks. CA shows recorded descriptors; queue intervals can overlap and include dependency waits.',fontsize=9)
fig.text(.055,.032,'Output residency and writeback are model conditions. The CA stage records GM → L1 transfers, not direct L2 hit or DRAM eviction times.',fontsize=9)
fig.savefig(OUT/'flash-b8-128ki.png',dpi=154);plt.close(fig)
(OUT/'figure-data.json').write_text(json.dumps(dict(shape=r['shape'],source=r['critical_shard'],sha256=r['shard_sha256'],qualified_ca_us=ca,
    predicted_us=p['latency_us'],predicted_wave_intervals=wavedata,measured_cycles_between_tile_starts=observed),indent=1)+'\n')
