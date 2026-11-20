"""Compare first/last output tiles on the actual last AIC; no stage fit to total."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.29/matmul/full-load'
TEAL='#177783';BLUE='#4678ab';PURPLE='#7553a1';ORANGE='#b65b2a';INK='#213540';PAPER='#fbfcfd'
CA='core0.cubecore0'
train=next(r for r in json.loads((OUT/'development-traces.json').read_bytes()) if r['case']=='262144-128-512-dt')
assert train['core']==CA
reg=json.loads((OUT/'freeze.json').read_bytes())
row=next(r for r in reg['rows'] if r['shape']==train['shape'] and r['chip']==train['chip'])
pred=row['candidate'];cs=pred['cycles']
scale=1650
selected=((0,1),(30,31))
data=dict(reference='ticks since globally earliest participating instruction',cycles_per_us=scale,core=CA,
          heldout=False,receipt=train['drain_receipt'],source_sha256=train['core_sha256'],model=[],actual=[])


def add(level, column, y, name, lo, hi, resource, **extra):
    data[level].append(dict(column=column,lane=y,label=name,begin_ticks=lo,end_ticks=hi,resource=resource,**extra))


for column,ids in enumerate(selected):
    if column==0:
        add('model',column,0,'one B + first A page (combined service and response)',cs['launch'],cs['first_cube']-cs['l0_first'],'MTE2')
        add('model',column,1,'first operand transfer',cs['first_cube']-cs['l0_first'],cs['first_cube'],'MTE1')
        b=train['b'];add('actual',column,0,'resident B',b['begin'],b['end'],'MTE2',offset=-.18,pc=b['pc'])
    for local,i in enumerate(ids):
        off=cs['first_cube']+i*cs['per_aic_step']
        tile=train['tiles'][i]
        for u in range(pred['mmads_per_tile']):
            lo=off+u*cs['cube_issue'];hi=lo+cs['cube_issue']
            add('model',column,2+2*local,'MMAD issue interval',lo,hi,'CUBE',mark_issue=True,tile=i)
        if i==pred['waves']-1:
            add('model',column,2+2*local,'final response tail',hi,hi+cs['response_after_last_issue_interval'],'CUBE',tile=i)
            add('model',column,3+2*local,'exposed BF16 write and drain',cs['core_finish'],cs['total'],'FIXP',tile=i)
        for page,interval in enumerate(tile['a_pages']):
            add('actual',column,0,'A page '+str(page),interval['begin'],interval['end'],'MTE2',tile=i,offset=(-.015+.12*local+.038*(page%2)),pc=interval['pc'])
        for k,interval in enumerate(tile['l0']):
            add('actual',column,1,'operand L1→L0',interval['begin'],interval['end'],'MTE1',tile=i,offset=(-.10+.07*(k%3)),pc=interval['pc'])
        for interval in tile['mmad']:
            add('actual',column,2+2*local,'MMAD',interval['begin'],interval['end'],'CUBE',tile=i,mark_issue=True,pc=interval['pc'])
        f=tile['fix']
        add('actual',column,3+2*local,'FIXP descriptor, dependency wait included',f['begin'],f['end'],'FIXP',tile=i,pale=True,pc=f['pc'])
        last=tile['mmad'][-1]['end']
        if f['end']>last:
            add('actual',column,3+2*local,'descriptor after last MMAD retirement',max(f['begin'],last),f['end'],'FIXP',tile=i,pc=f['pc'])

data['global_model_end_us']=pred['latency_us'];data['global_ca_end_us']=train['ca_us']
(OUT/'figure-data.json').write_text(json.dumps(data,indent=1)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
fig,axes=plt.subplots(2,2,figsize=(13.6,7.7),gridspec_kw={'hspace':.44,'wspace':.11,'width_ratios':[1.3,1]},facecolor='white')
fig.subplots_adjust(left=.17,right=.965,top=.755,bottom=.169)
fig.text(.06,.94,'950DT MatMulV3  |  262,144 × 128 × 512',size=23,weight='bold')
fig.text(.06,.892,'DeepSeek V4.1 Flash · indexer.wk · 8 requests × 32 Ki tokens · 32 AICs · 32 output tiles per AIC',size=12)
fig.legend(handles=[Patch(color=c,label=s) for s,c in [('MTE2  GM → L1',TEAL),('MTE1  L1 → L0',BLUE),('Cube  MMAD',PURPLE),('FIXP  L0C → GM',ORANGE)]],loc='upper left',bbox_to_anchor=(.055,.87),ncol=4,frameon=False,columnspacing=2.3,handlelength=1.9)
fig.text(.17,.79,f'Resource schedule   {pred["latency_us"]:.3f} µs',weight='bold',size=15)
fig.text(.17,.445,f'CA trace   {CA}   ·   {train["ca_us"]:.3f} µs',weight='bold',size=15)
for level,axrow in zip(('model','actual'),axes):
    for column,ax in enumerate(axrow):
        ids=selected[column]
        ax.set_xlim(*((0,12) if column==0 else (78,87)))
        ax.set_ylim(5.5,-.52);ax.set_facecolor(PAPER)
        labels=['B / A reads','L1 → L0',f'Tile {ids[0]+1} · MMAD',f'Tile {ids[0]+1} · FIXP',f'Tile {ids[1]+1} · MMAD',f'Tile {ids[1]+1} · FIXP']
        ax.set_yticks(range(6));ax.set_yticklabels(labels if column==0 else ['','','Tile 31','', 'Tile 32',''])
        ax.tick_params(axis='y',length=0,pad=8)
        ax.xaxis.set_major_locator(MultipleLocator(2 if column==0 else 1))
        ax.grid(axis='x',color='#d9e3ea',linewidth=.7)
        ax.axhline(3.5,color='#cbd8dd',lw=.7)
        for s in ax.spines.values():s.set_visible(False)
        colors={'MTE1':BLUE,'MTE2':TEAL,'CUBE':PURPLE,'FIXP':ORANGE}
        for bar in data[level]:
            if bar['column']!=column:continue
            y=bar['lane']+bar.get('offset',0);lo=bar['begin_ticks']/scale;hi=bar['end_ticks']/scale
            h=.13 if level=='actual' and bar['resource'] in ('MTE1','MTE2') else .25
            c=colors[bar['resource']]
            ax.broken_barh([(lo,hi-lo)],(y-h/2,h),facecolors=c,alpha=.30 if bar.get('pale') else 1,zorder=3)
            if bar.get('mark_issue'):ax.vlines(lo,y-.21,y+.20,color='#4c3479',lw=.65,zorder=4)
        if level=='model' and column==0:
            ax.text(cs['launch']/scale/2,0,'Launch',ha='center',va='center',fontsize=8)
        if level=='model' and column==1:
            ax.text(78.3,.5,'Steady transfers constrained by port capacity;\nper-page start times unresolved.',fontsize=8,va='center',color='#62737b')
        if column==1:
            end=pred['latency_us'] if level=='model' else train['ca_us']
            ax.axvline(end,color=INK,ls='--',lw=.9)
        if level=='actual':
            ax.set_xlabel('Device time from first instruction (µs)')
fig.text(.06,.08,'Left: first two outputs. Right: last two outputs. Both panels use the same origin; the middle 66 µs is omitted.',size=10)
fig.text(.06,.048,'Ticks mark individual MMAD issues. Pale FIXP spans the descriptor including dependency waits. Dark FIXP continues after the final MMAD response.',size=9)
fig.savefig(OUT/'flash-b8-32ki-matmul.png',dpi=165,facecolor='white');plt.close(fig)
