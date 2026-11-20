"""Render model and ELF-identified CA DMA/VF descriptors for *the same* output rows."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.rms28.campaign import OUT,verify,read,write
from operator_model.rmsnorm_pipeline import rmsnorm_pipeline

COLORS={'gamma':'#1b7c85','x':'#1b7c85','reduce':'#734fa6','rstd_vf':'#9664a1','normalize':'#734fa6','rstd':'#b45232','y':'#b45232'}
NAMES={'gamma':'Read γ','x':'Read x','reduce':'Sum of squares','rstd_vf':'Inverse RMS','normalize':'Normalize y','rstd':'Write rstd','y':'Write y'}
def project(record):
    path=ROOT/record['trace']
    verify.checked(verify.digest(path)==record['sha256'],'published trace changed')
    folder=path.parent
    pc0=int((folder.parent/'dump/pc_start_addr.txt').read_text(),16)
    all_lanes=[]
    for shard in sorted(folder.glob('core*/trace.json')):
        ops={key:[] for key in COLORS}
        last=0
        for ev in verify.events(shard):
            if ev['name']=='END_LABEL':continue
            end=round((ev['ts']+ev['dur'])*verify.CLOCK)
            last=max(last,end-record['first_tick'])
            if ev['name']!='VF' and not ev['name'].startswith('MOV_SRC_TO_DST'):continue
            pc=int(ev['args']['pc_addr'],16)-pc0
            key={0x4878:'gamma',0x4a48:'x',0x4e64:'rstd_vf',0x5008:'normalize',0x4f38:'rstd',0x5128:'y'}.get(pc)
            if key is None:
                if ev['name']!='VF':continue
                key='reduce'
            info=json.loads(ev['args']['detail'])
            ops[key].append(dict(b=round(ev['ts']*verify.CLOCK)-record['first_tick'],e=end-record['first_tick'],
                pc=hex(pc),bytes=info.get('processed_bytes'),src=info.get('src_addr'),dst=info.get('dst_addr')))
        for v in ops.values():v.sort(key=lambda x:x['b'])
        all_lanes.append(dict(lane=shard.parent.name,ops=ops,last_cycles=last,shard_sha256=verify.digest(shard)))
    all_lanes.sort(key=lambda z:z['ops']['rstd'][0]['dst'])
    statistic0=all_lanes[0]['ops']['rstd'][0]['dst']
    y0=all_lanes[0]['ops']['y'][0]['dst']
    row=0
    for block,lane in enumerate(all_lanes):
        assert lane['ops']['rstd'][0]['dst']==statistic0+row*4
        assert lane['ops']['y'][0]['dst']==y0+row*record['shape'][1]*2
        lane['row0']=row;lane['block']=block
        row+=sum(s['bytes']//4 for s in lane['ops']['rstd'])
    assert row==record['shape'][0] and len(all_lanes)==record['active_aiv']
    return all_lanes

def build(group,shape,slug,chip="Ascend950PR_957b"):
    if (OUT/(slug+'.json')).exists():
        cached=read(OUT/(slug+'.json'))['entries']
        draw(cached,OUT/(slug+'.png'))
        return
    receipt=OUT/'receipts'/(group+'-'+chip+'.json')
    if not receipt.exists():return
    pair=[r for r in read(receipt)['rows'] if r['shape']==shape and r['timing_qualified']]
    if len(pair)!=2:return
    traces=[project(r) for r in pair]
    block=max(range(len(traces[0])),key=lambda i:traces[0][i]['last_cycles'])
    data=[]
    for rec,lanes in zip(pair,traces):
        lane=lanes[block]
        m=rmsnorm_pipeline(shape,chip=rec['chip'],resident_aiv=rec['prior_aiv'],
            immediate_repeat=rec['role']=='repeat',diagnostics=True)
        tiles=[s for s in m['stages'] if s['block']==block]
        assert tiles[0]['row0']==lane['row0'] and len(tiles)==len(lane['ops']['x'])
        data.append(dict(shape=shape,chip=rec['chip'],role=rec['role'],H=rec['prior_aiv'],active=rec['active_aiv'],
            ca_us=rec['ca_us'],model_us=m['us'],unknown_us=rmsnorm_pipeline(shape,chip=rec['chip'])['us'],
            lane=lane,model_tiles=tiles,trace=rec['trace'],sha256=rec['sha256'],bounds_cycles=m['bounds_cycles']))
    for k,f in [('gamma','src'),('x','src'),('rstd','dst'),('y','dst')]:
        assert [n[f] for n in data[0]['lane']['ops'][k]]==[n[f] for n in data[1]['lane']['ops'][k]]
    report=dict(time_origin='first physical instruction over the complete active AIV roster; axes in target microseconds',
        scope='Same output row range recovered from GM addresses and byte counts on both calls. All-device terminal line; relative to first AIV. Estimated reduction bar is issue-to-next-issue, hatched. No fabricated reduce retirement time.',
        entries=data)
    write(OUT/(slug+'.json'),report)
    draw(data,OUT/(slug+'.png'))

def draw(data,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    l=data[0]['lane']
    keys=['gamma']+[k for _ in data[0]['model_tiles'] for k in ('x','reduce','rstd_vf','normalize','rstd','y')]
    def row_label(tile):
        start = tile['row0']
        return f"row {start}" if tile['rows']==1 else f"rows {start}–{start+tile['rows']-1}"
    top=['Read γ']+[f"{NAMES[k]} · {row_label(t)}" for t in data[0]['model_tiles'] for k in ('x','reduce','rstd_vf','normalize','rstd','y')]
    fig,axes=plt.subplots(2,2,figsize=(14.5,5.8+len(keys)*.25),sharex=True,sharey=True)
    fig.subplots_adjust(left=.165,right=.975,top=.76,bottom=.13,wspace=.12,hspace=.4)
    a=data[0];xmax=max(max(r['ca_us'],r['model_us'],r['unknown_us']) for r in data)*1.09
    fig.suptitle(f"{a['chip'].split('_')[0].removeprefix('Ascend')} RMSNorm  |  {a['shape'][0]} × {a['shape'][1]}",x=.165,y=.985,ha='left',fontsize=24,color='#182e3b',weight='bold')
    fig.text(.165,.912,f"aclnnRmsNorm / BF16 key 5000   ·   {a['active']} AIVs   ·   block {l['block']}   ·   matched x / γ / y / rstd GM addresses across both calls",fontsize=12,color='#344e60')
    fig.legend(handles=[Patch(color=COLORS['x'],label='MTE2 / GM → UB'),Patch(color=COLORS['reduce'],label='Vector / VF'),Patch(color=COLORS['y'],label='MTE3 / UB → GM'),
        Patch(facecolor='white',edgecolor=COLORS['reduce'],hatch='///',label='Issue to next issue'),Line2D([],[],ls=':',color='#637989',label='Unknown-history point')],ncol=5,loc='upper left',bbox_to_anchor=(.16,.882),frameon=False,fontsize=10)
    for col,d in enumerate(data):
        ta,ba=axes[:,col];lane=d['lane'];h=d['H']
        ta.set_title(f"Model · H={h} · "+('exact repeat' if col else 'expansion'),loc='left',weight='bold',fontsize=12)
        ba.set_title(f"CA · {lane['lane']} · matched block",loc='left',weight='bold',fontsize=12)
        for ax,wall in ((ta,d['model_us']),(ba,d['ca_us'])):
            ax.set_title(f"{wall:.3f} µs",loc='right',weight='bold',fontsize=12)
            ax.axvline(wall,ls='--',color='#344d5d',lw=1.4)
            ax.axvline(d['unknown_us'],ls=':',color='#637989',lw=1.1)
            ax.set_xlim(0,xmax);ax.set_ylim(len(keys)-.5,-.5);ax.set_yticks(range(len(keys)),top,fontsize=9)
            ax.grid(axis='x',color='#dfe7eb');ax.set_axisbelow(True);ax.tick_params(axis='y',length=0,pad=7)
            ax.spines[['top','left','right']].set_visible(False)
        idx={k:0 for k in COLORS}
        for line,key in enumerate(keys):
            jt=idx[key];idx[key]+=1
            t=d['model_tiles'][jt]
            vv=lane['ops'][key][jt]
            if key=='reduce':window=[t['reduce_issue'],t['rstd_vf'][0]]
            elif key=='rstd':window=[t['rstd_issue'],t['rstd_retire']]
            else:window=t[key]
            for ax,(b,e) in [(ta,window),(ba,[vv['b'],vv['e']])]:
                color=COLORS[key];hatched=key=='reduce' and ax==ta
                ax.barh(line,(e-b)/verify.CLOCK,left=b/verify.CLOCK,height=.43,
                    facecolor='white' if hatched else color,edgecolor=color,hatch='////' if hatched else None,linewidth=.8)
                ax.plot(b/verify.CLOCK,line,'|',ms=8,color=color)
    fig.supxlabel('Device time from first physical instruction on any active AIV (µs)',y=.075,fontsize=12)
    fig.text(.165,.019,'Dashed line = full device window. Bars include descriptor wait. Only the bottom reduction bars show full VF response. Each store row has its own destination.',fontsize=9,color='#466070')
    fig.savefig(dest,dpi=135,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':
    build('partition',[4,784],'partition-repeat')
    build('multi',[539,4000],'multi-ub-repeat-dt','Ascend950DT_9582')
    build('multi',[539,4000],'multi-ub-repeat')
