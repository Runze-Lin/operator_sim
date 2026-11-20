"""Compare frozen issue times with the actual critical AIV descriptor records."""
import argparse
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from landmarks import read,save,OUT,ROOT
from schedule28 import estimate,load_chip
DT='Ascend950DT_9582';PR='Ascend950PR_957b'
LABELS=['Tiling  ·  MTE2','Input  ·  MTE2','Setup  ·  VF','Softmax  ·  VF','First column  ·  VF','Row  ·  VF','Column  ·  VF','Scatter  ·  VF','Output  ·  MTE3']
COLORS=['#167B80','#167B80','#566d77','#5966a3','#765aae','#4a6faa','#795998','#5e7183','#b05830']

def render(sku,t,forward,group="sequence"):
    if forward:
        sys.path.insert(0,str(ROOT/'development/validation/sink26'))
        from evidence import shard
        reg=read(OUT/'register.json');j=next(x for x in reg['jobs'] if x['chip']==sku and x['shape'][0]==t and x['group']==group)
        receipt=OUT/'receipts'/(group+'-'+sku+'.json')
        d=read(receipt)['rows'][j['index']];assert d['qualified']
        s=d['stage'];path,digest,core,origin,_,_,ev=shard(s)
        for e in ev:e['begin']=round(e['begin']-origin);e['end']=round(e['end']-origin)
        ev.sort(key=lambda e:e['begin']);full=s['ca_us'];it=j['iters'];k=j['conditional'];r=j['range_us'];point=j['point_us'];h=j['prior_h'];source=str(receipt.relative_to(ROOT));aggregate=s['trace_sha256']
        name='independent'
    else:
        d=next(x for x in read(OUT/'landmarks.json') if x['sku']==sku and x['shape'][0]==t and x['role'].startswith('v0.26'))
        ev=d['events'];it=d['iters'];a=estimate([t,4],it,load_chip(sku));k=a['states'][d['state']];r=a['range_us'];point=a['point_us'];h=d['history']
        full=d['full_cycles']/1650;path=d['trace'];digest=d['trace_sha256'];core=path.split('/')[-2];source=d['receipt'];aggregate=d['aggregate_sha256'];name='replay'
    e=k['epochs_us'];model=[]
    def add(lane,start,end=None):model.append(dict(lane=lane,begin_us=start,end_us=end))
    add(0,e['tiling_begin'],e['tiling_end']);add(1,e['read_begin'],e['read_end'])
    add(2,e['setup_vf']);add(3,e['softmax_vf']);add(4,e['first_col_vf'])
    for n in range(it-1):
        row=e['first_row_vf']+(0 if n==0 else e['first_pair_period']+(n-1)*e['steady_pair_period'])
        add(5,row);add(6,row+(e['first_row_to_col'] if n==0 else e['steady_row_to_col']))
    add(7,e['scatter_vf']);add(8,e['write_begin'],e['write_end'])
    dma=[x for x in ev if x['name'].startswith('MOV_SRC_TO_DST')];vf=[x for x in ev if x['name']=='VF']
    actual=[]
    def ca(lane,x):actual.append(dict(lane=lane,begin_us=x['begin']/1650,end_us=x['end']/1650,pc=x['pc'],descriptor_bytes=x.get('size')))
    for l,x in zip((0,1,8),dma):ca(l,x)
    for l,x in zip([2,3,4]+[5,6]*(it-1)+[7],vf):ca(l,x)
    assert len(dma)==3 and len(vf)==2*it+2
    aiv=k['active'];g=k['groups'];short=sku[6:11]
    stem=OUT/f'{name}-{short.lower()}-t{t}-i{it}'
    save(stem.with_suffix('.json'),dict(phase='conditional model vs measured actual state',model=model,measured=actual,
        timing_origin='earliest instruction on any core for this complete launch, microseconds',sku=sku,t=t,iters=it,h=h,active=aiv,groups=g,critical_core=core,
        missing_history_point_us=point,modeled_range_us=r,conditional_model_us=k['us'],ca_device_us=full,trace=path,trace_sha256=digest,aggregate_sha256=aggregate,receipt=source))
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.labelcolor':'#203541','xtick.color':'#40596a'})
    fig,axes=plt.subplots(2,1,figsize=(12,7.5),sharex=True)
    fig.subplots_adjust(left=.19,right=.96,bottom=.17,top=.79,hspace=.36)
    fig.text(.19,.94,f'{short}  |  MhcSinkhorn  ·  {t:,} × 4 × 4',size=24,color='#1C3542',weight='bold')
    fig.text(.19,.887,f'{it} iterations   ·   {aiv} AIVs   ·   up to {k["rows"]} matrices per core   ·   {g} register groups   ·   H = {h} ({k["state"]})',size=12,color='#42606e')
    fig.text(.19,.842,f'Unknown history  {point:.3f} µs      Reachable model endpoints  {r[0]:.3f}–{r[1]:.3f} µs',size=10,color='#435b64')
    for ax,spans,end,title in zip(axes,(model,actual),(k['us'],full),('Resource model · conditioned on dispatch history','CA trace · '+core+' (last to finish)')):
        measured=ax is axes[1]
        ax.set_title(title,loc='left',weight='bold',color='#203541',fontsize=13)
        ax.set_title(f'{end:.3f} µs',loc='right',weight='bold',color='#203541',fontsize=13)
        for v in spans:
            l,x,y=v['lane'],v['begin_us'],v['end_us']
            if y is not None:ax.barh(l,width=y-x,left=x,height=.32,color=COLORS[l],alpha=.37 if measured and 2<=l<=7 else .91,zorder=2)
            if 2<=l<=7:ax.vlines(x,l-.23,l+.23,lw=1.9,color=COLORS[l],zorder=3)
        ax.axvline(end,ls=(0,(3,3)),color='#304955',lw=1.2)
        ax.set_yticks(range(9),LABELS,fontsize=9);ax.set_ylim(8.6,-.6);ax.tick_params(axis='y',length=0)
        ax.grid(axis='x',color='#d5e0e6',lw=.7);ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color('#afbec5')
    axes[1].set_xlim(0,max(k['us'],full)*1.035);axes[1].set_xlabel('Device time from first instruction on any core (µs)',labelpad=8,fontsize=12)
    fig.text(.19,.072,'Ticks mark VF issue. Pale CA bars show descriptor lifetimes including in-flight work. Solid DMA bars show transfer windows.',size=9,color='#45606e')
    fig.text(.19,.045,'Dashed line marks the whole-kernel end. The model prices issue intervals; it exposes no separate VF retirement timestamp.',size=9,color='#45606e')
    fig.savefig(stem.with_suffix('.png'),dpi=165);plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--t',type=int,default=4097);p.add_argument('--pr',action='store_true');p.add_argument('--forward',action='store_true');p.add_argument('--group',choices=('first','sequence'),default='sequence');a=p.parse_args()
    render(PR if a.pr else DT,a.t,a.forward,a.group)
