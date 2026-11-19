"""Rebuild the worked MatMul illustration from the archived per-page intervals."""
from pathlib import Path
import json
from math import isclose
import sys
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

root = Path(__file__).resolve().parents[3]
out = root / 'docs/examples/figures'
data = json.loads((root/'results/v0.20/examples/matmul-tile-pages.json').read_text())
sys.path.insert(0, str(root))
from operator_model import estimate
current = estimate('matmul', data['shape'], chip='Ascend950DT_9582', timing_mode='resource')
assert isclose(current.latency_us, data['model_us'], rel_tol=1e-9)
assert current.projection['timing_diagnostics']['mmads_total'] == data['counts']['CUBE']
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10.4, 7.4))
fig.subplots_adjust(left=.195, right=.95, top=.806, bottom=.13, hspace=.38)
ink='#192e3b'
subtitle='#425866'
colors = {'MTE2':'#197786', 'MTE1':'#5275ae', 'MMAD':'#7254a6','FIXP':'#af4a29'}
fig.text(.195,.955,'950DT MatMulV3 | 1280 × 2048 × 512',size=19,weight='bold',color=ink)
fig.text(.195,.92,'32 AICs · one full + one split-tail tile per core · two L1 K pages per tile',size=11,color=subtitle)
fig.legend([Patch(color=v) for v in colors.values()],['MTE2 / GM → L1','MTE1 / L1 → L0','Cube / MMAD','FIXP'],loc='upper left',
           bbox_to_anchor=(.185,.905),ncol=4,frameon=False,labelcolor=ink,fontsize=10)

def draw_segments(ax, y, segments, color, alpha=1, linestyle='-', height=.15):
    for start,stop in segments:
        ax.barh(y, stop-start, left=start, height=height, color=color, alpha=alpha,
                linestyle=linestyle, linewidth=0, zorder=3)

labels=['256×256 · K page 1','256×256 · K page 2','256×256 · write',
        '128×128 · K page 1','128×128 · K page 2','128×128 · write']
for idx,(ax,cpt,tiles,finish) in enumerate(zip(axes,
    ['Resource model · wave critical path',f"CA trace · {data['core'].split('.')[0]} (last to finish)"],
    [data['model_tiles'],data['ca_tiles']],[data['model_us'],data['ca_us']])):
    ax.set_ylim(-.55,5.55)
    ax.set_xlim(0, max(data['model_us'],data['ca_us']) + .6)
    ax.set_yticks(range(5,-1,-1),labels,fontsize=10)
    ax.tick_params(axis='y',length=0,pad=9,labelcolor=ink)
    ax.tick_params(axis='x',colors=subtitle,labelsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_axisbelow(True)
    ax.grid(axis='x',color='#dbe3e8',linewidth=.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhline(2.5,color='#dbe3e8',linewidth=.9)
    ax.set_title(cpt,loc='left',pad=9,fontsize=12,weight='bold',color=ink)
    ax.text(1.0,1.06,f'{finish:.3f} µs',transform=ax.transAxes,ha='right',va='bottom',fontsize=12,weight='bold',color=ink)
    ax.axvline(finish,linestyle=(0,(3,3)),color='#536776',linewidth=1)
    if idx==0:
        launch=data['launch_us']
        ax.barh(5,launch,height=.40,color='#e5ecef')
        ax.text(launch/2,5,'Launch',va='center',ha='center',color=subtitle,fontsize=9)
    for i,tile in enumerate(tiles):
        y=5-3*i
        for j,page in enumerate(tile['pages']):
            yp=y-j
            for lane,unit in enumerate(('MTE2','MTE1','MMAD')):
                segs=page[unit]
                if idx==0:
                    segs=[segs]
                draw_segments(ax,yp+.23-lane*.22,segs,colors[unit],
                              alpha=.57 if idx==0 and (j==1 or i==1) and unit in ('MTE2','MTE1') else 1)
                if idx==1 and unit=='MMAD':
                    ax.vlines([s for s,_ in segs],yp-.35,yp-.095,
                              color='#492a77',linewidth=.75,zorder=4)
        w=tile['write']
        if idx==0:
            draw_segments(ax,y-2,[w[:2]],colors['FIXP'],height=.21)
            draw_segments(ax,y-2,[w[1:]],colors['FIXP'],alpha=.38,height=.21)
        else:
            draw_segments(ax,y-2,[w],colors['FIXP'],alpha=.38,height=.21)
axes[-1].set_xlabel('Device time from first instruction (µs)',fontsize=11,color=ink,labelpad=10)
fig.text(.195,.041,'Model: faded read/load bars show timing implied by the page interval.  CA: purple ticks mark each MMAD issue.',fontsize=9.2,color=subtitle)
fig.text(.195,.016,'Pale orange: modeled write drain above, full FIXP descriptor window including dependency waits below.',fontsize=9.2,color=subtitle)
p=out/'matmul-tile-pages.png'
fig.savefig(p,dpi=160,facecolor='white')
print(p)
