"""Render four model write issues and all genuine CA writes on a shared source clock."""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prices import OUT,read,write,sha
from candidate import predict
from pathlib import Path
DT='Ascend950DT_9582'
shape=[1,4,192]
r=next(r for r in read('results/v0.32/post/wide-repeats.json') if r['chip']==DT and r['shape']==shape)
qualified=next(x for x in read('results/v0.33/post/replay.json')['rows'] if x['sha256']==r['ca_sha256'])
z=predict(shape,DT,read(OUT/'prices.json'));info=z['details']['hot_cycles']
first=(info['arrival']+info['first_operand_and_first_write']+info['reads'])/1650
ii=info['remaining_write_issues']/3/1650
modeled=[first+i*ii for i in range(4)]
from json import loads
e=[x for x in read(r['shard'])['traceEvents'] if x.get('ph')=='X'];zero=min(x['ts'] for x in e)
writes=[]
for x in e:
    if x['name']!='MOV_SRC_TO_DST_ALIGNv2':continue
    d=loads(x['args']['detail'])
    if d.get('queue_type')!='MTE3' or d.get('dst_mem')!='OUT' or d.get('processed_bytes')!=384:continue
    writes.append([x['ts']-zero,x['ts']+x['dur']-zero])
writes.sort();assert len(writes)==4
data=dict(shape=shape,sku=DT,shape_only_point_us=z['us'],state='immediate repeat',modeled_write_issues_us=modeled,
    modeled_last_write_response_us=info['write_response']/1650,modeled_hot_window_us=z['conditional_us']['repeat'],
    measured_writes_us=writes,ca_all_core_window_us=qualified['ca_us'],trace=r['shard'],shard_sha256=sha(r['shard']))
write('single-aiv-example.json',data)
fig,axes=plt.subplots(2,1,figsize=(8,4.6),sharex=True); fig.subplots_adjust(left=.19,top=.80,bottom=.16,hspace=.51)
fig.suptitle('950DT  ·  mHC Post  [1, 4, 192]',x=.19,y=.975,ha='left',fontsize=16,weight='bold')
fig.text(.19,.878,'1 AIV   ·   four 384 B output writes   ·   immediately repeated launch',fontsize=10)
color='#a34522'
for ax,model in zip(axes,(True,False)):
    us=data['modeled_hot_window_us'] if model else data['ca_all_core_window_us']
    ax.set_title(('Conditional resource model' if model else 'CA  ·  core0.veccore0')+f'                 {us:.3f} µs',loc='left',fontsize=10,weight='bold')
    for i in range(4):
        y=3-i
        if model:
            ax.scatter([modeled[i]],[y],marker='|',s=115,c=color,linewidths=1.8,zorder=3)
            if i==3:ax.barh(y,info['write_response']/1650,left=modeled[i],height=.24,color=color)
        else:
            lo,hi=writes[i];ax.barh(y,hi-lo,left=lo,height=.25,color=color);ax.vlines(lo,y-.16,y+.16,color='#4e2a19',lw=1)
    ax.axvline(us,color='#17394c',ls='--',lw=1.3);ax.set(xlim=(0,1.82),ylim=(-.55,3.55),yticks=(0,1,2,3),yticklabels=('output 4','output 3','output 2','output 1'))
    ax.grid(axis='x',color='#cddde6',lw=.6);ax.set_axisbelow(True)
    for side in ('left','top','right'):ax.spines[side].set_visible(False)
    ax.tick_params(axis='y',length=0)
axes[-1].set_xlabel('Device time from first instruction (µs)');fig.savefig(OUT/'single-aiv-example.png',dpi=165,bbox_inches='tight');plt.close(fig)
