"""Explain the first Pre M-K candidate on a legacy PR window; no new sampling."""
from pathlib import Path
import json
import math
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import stage_candidate as source
D=source.OUT
p=source.read(D/'stage-freeze.json')['prices_ticks']
entry=next(r for r in source.read(source.ROOT/'results/v0.17/sku/pairs.json') if r['op']=='mhc_pre' and r['shape']==[9,4,1024])
s=entry['pr']['stages'][0];parent=(source.ROOT/s['trace']).parent
def extract(core):
    path=parent/core/'trace.json'
    events=[v for v in source.read(path)['traceEvents'] if v.get('ph')=='X']
    origin=min(e['ts'] for e in events)-s['windows'][core][0]
    return [dict(op=e['name'],pc=e['args'].get('pc_addr'),us=e['ts']-origin,dur=e['dur'],detail=json.loads(e['args'].get('detail')or '{}'))for e in events if e['name'] in ('VF','MMAD')],dict(path=source.rel(path),sha256=source.sha(path),origin_us=origin)
vec,v=extract('core0.veccore0');cube,c=extract('core0.cubecore0')
assert len(vec)==7 and len(cube)==8 and all(e['detail']['k']==32 and e['detail']['m']==9 for e in cube)
ratio=(27/64)/(23/56);clock=1650
front=p['first_cast_issue_ticks']/clock
cast=(p['cast_fixed_ticks']+p['cast_ticks_per_element']*5*256*ratio)/clock
period=(p['mmad_fixed_ticks']+p['mmad_m16_ticks']*(432/32)/(378/28))/clock
mmad_start=front+cast;mmad_end=mmad_start+8*period
handoff=p['from_last_mmad_to_first_stage2_ticks']/clock
coeff=p['stage2_coeff_group_ticks']*ratio/clock
hin=2*1024*p['hin_ticks_per_output']*ratio/clock
localwait=p['residual_drain_ticks']/clock
stage2=mmad_end+handoff;hissue=stage2+coeff+localwait;finish=hissue+hin
bands=['Input conversion','Projection · 8 MMAD','Reduce + coefficients','Mix 4 streams → HIn']
model=[[(front,cast)],[(mmad_start+k*period,period)for k in range(8)],[(stage2,coeff)],[(hissue,hin)]]
ca=[[(vec[0]['us'],vec[0]['dur'])],[(e['us'],e['dur']) for e in cube],
    [(e['us'],e['dur'])for e in vec[1:-1]],[(vec[-1]['us'],vec[-1]['dur'])]]
data=dict(shape=[9,4,1024],sku=entry['pr']['chip'],old_development_reference=True,secs_unit='device microseconds',
  freeze_sha256=source.sha(D/'stage-freeze.json'),source_figure_sha256=source.sha(__file__),protocol='single AIC/AIV pair core0 only; zero is earliest instruction of any core; endpoints use complete official launch and model budget; blank intervals are unassigned; prices use DT development windows',
  core_sources=[c,v],aggregate=s['trace'],aggregate_sha256=s['trace_sha256'],all_core_us=s['ca_us'],
  last_core=max(s['windows'],key=lambda name:s['windows'][name][1]),model_us=finish,model=model,ca=ca,cube_events=cube,vec_events=vec)
(D/'pre-mk9-pr.json').write_text(json.dumps(data,indent=2)+'\n')
fig,axes=plt.subplots(2,1,figsize=(11.5,6.75),sharex=True)
colors=['#147d8b','#744b9d','#4b73a7','#b34c2e']
for ax,tracks,end,label in zip(axes,[model,ca],[finish,s['ca_us']],['Resource schedule · local M and K tile','CA trace · core0.cubecore0 + core0.veccore0']):
    for idx,row in enumerate(tracks):
        y=3-idx
        for start,duration in row:
            ax.barh(y,duration,left=start,height=.32,color=colors[idx])
            if idx==1:ax.vlines(start,y-.24,y+.24,color='#402960',linewidth=.8)
    ax.set_yticks([3,2,1,0],bands);ax.axvline(end,color='#263f4c',ls='--',lw=1)
    ax.text(1,1.045,f'{end:.3f} µs',transform=ax.transAxes,ha='right',weight='bold',fontsize=13)
    ax.set_title(label,loc='left',fontsize=13,weight='bold',pad=9)
    ax.xaxis.set_major_locator(MultipleLocator(1));ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    ax.spines[['left','right','top']].set_visible(False);ax.set_ylim(-.48,3.54);ax.tick_params(axis='y',length=0)
fig.suptitle('950PR  ·  mHC Pre  |  9 × 4 × 1024',x=.23,y=.98,ha='left',fontsize=21,weight='bold')
fig.text(.23,.912,'One M tile, local K = 256, outputs = 24. Each issue advances K by 32.',fontsize=11,color='#405868')
axes[-1].set_xlim(0,math.ceil(max(finish,s['ca_us']))+.18)
axes[-1].set_xlabel('Device time from earliest core instruction (µs)')
fig.text(.23,.033,'Top uses 950DT instruction/VF prices. The bottom endpoint marks all-core retirement; only one pair is drawn.\nBlank regions include dispatch, handoff and waits. The upper model ends after its estimated HIn service.',fontsize=9,color='#405868')
fig.subplots_adjust(left=.23,bottom=.16,right=.96,top=.80,hspace=.63);fig.savefig(D/'pre-mk9-pr.png',dpi=165);plt.close(fig)
