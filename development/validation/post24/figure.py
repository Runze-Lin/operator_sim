"""Map real GM output addresses to one identical logical slice across first/repeat calls."""
from collections import defaultdict
import json,os,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.rms_ordered_verify.verify import events
from development.validation.sku19.campaign import read,write
OUT=ROOT/'results/v0.24/post'
COL={'control':'#78909f','raw':'#7f929e','MTE2':'#147887','SIMD':'#7556a4','MTE3':'#b85332','body':'#6e5a9d'}
def union(intervals):
    merged=[]
    for a,b in sorted(intervals):
        if b<=a:continue
        if merged and a<=merged[-1][1]+1e-6:merged[-1][1]=max(merged[-1][1],b)
        else:merged.append([a,b])
    return merged
def mapped(r):
    tr=ROOT/r['stage']['trace'];shape=r['shape'];tokens,n,d=shape;stores=[]
    for ev in events(tr):
        if not ev['name'].startswith('MOV_SRC_TO_DST'):continue
        det=json.loads(ev['args']['detail'])
        if det['queue_type']=='MTE3' and not any(det.get(i) for i in ('is_sync','is_wait','is_duration_excluded')):
            assert det['burst_number']==1 and det['dst_mem']=='OUT'
            stores.append((det['dst_addr'],det['processed_bytes'],ev['pid']))
    base=min(x[0] for x in stores);prev=base
    for adr,sz,_ in sorted(stores):assert adr==prev;prev+=sz
    assert prev-base==2*tokens*n*d
    selected,=[item for item in stores if item[0]==base];core=selected[2];extent=selected[1]//2
    start=r['stage']['first_tick']/1650; lanes=defaultdict(list);anchors=[]
    for ev in events(tr.parent/core/'trace.json'):
        det=json.loads(ev['args']['detail']);q=det['queue_type'];nm=ev['name'];t=ev['ts']-start;end=t+ev['dur']
        if ev['dur']<=0 or any(det.get(i) for i in ('is_sync','is_wait','is_duration_excluded')):continue
        if nm.startswith('MOV_SRC_TO_DST') and q=='MTE2':anchors.append((t,end,det['processed_bytes']))
        if nm.startswith('MOV_SRC_TO_DST') and q=='MTE3':
            off=(det['dst_addr']-base)//2;token,row_and_d=divmod(off,n*d);row,x=divmod(row_and_d,d)
            assert token==0 and x==0 and det['processed_bytes']==2*extent
            lanes['out'+str(row)].append([t,end])
        if det['instr_type'].startswith(('RVEC','SIMD')):lanes['simd'].append([t,end])
    assert len(anchors)==3 and [x[2] for x in sorted(anchors)]==[96,2*extent,8*extent] and all(len(lanes['out'+str(i)])==1 for i in range(4))
    for j,key in enumerate(('meta','h','x')):lanes[key]=[sorted(anchors)[j][:2]]
    lanes['simd']=union(lanes['simd']); return dict(core=core,extent=extent,base_address=hex(base),source=str(tr.relative_to(ROOT)),ca_us=r['stage']['ca_us'],lanes=dict(lanes))
def main(small=False):
    # All columns use the same logical tile addressed by the first global output byte.
    path=OUT/f"receipts/{'channel' if small else 'wide'}-Ascend950DT_9582.json"
    if not path.exists():raise FileNotFoundError(path)
    rs=read(path)['rows'][:2];assert all(r['timing_qualified'] and r['shape']==([1,4,192] if small else [3,4,7168]) for r in rs)
    mapped_rows=[mapped(r) for r in rs];assert len({r['extent'] for r in mapped_rows})==1
    write(OUT/('small-figure-data.json' if small else 'figure-data.json'),dict(shape=rs[0]['shape'],scope='Select exact GM output token 0, d offset 0 via contiguous coverage, match all four writes by address. Global summaries above, lane intervals below.',rows=[dict(role=r['role'],shape=r['shape'],known_us=r['us'],unknown_us=r['candidate_unknown_us'],stages_us=r['stages'],port=r['port'],**m) for r,m in zip(rs,mapped_rows)]))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'text.color':'#203541','axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
    fig,ax=plt.subplots(2,2,figsize=(12.5,7.1),sharex=True,sharey='row',gridspec_kw={'height_ratios':[1,2.3]})
    fig.subplots_adjust(left=.17,right=.96,top=.8,bottom=.15,hspace=.52,wspace=.1)
    fig.suptitle(f"950DT mHC Post  |  {rs[0]['shape']}",x=.17,y=.98,ha='left',weight='bold',fontsize=19)
    fig.text(.17,.906,f"Token 0, channels 0–{mapped_rows[0]['extent']-1}  ·  same four output rows and deterministic input  ·  one stream",fontsize=12)
    mx=max(max(r['us'],r['stage']['ca_us'],r['candidate_unknown_us']) for r in rs)*1.06
    for j,(r,m,condition) in enumerate(zip(rs,mapped_rows,['First dispatch','Immediate repeat'])):
        top,lower=ax[0,j],ax[1,j];top.set_title(f"{condition}   |   model {r['us']:.3f} µs",loc='left',fontsize=11,weight='bold')
        st=r['stages'];t=0
        for k,v in st.items():
            color=COL['MTE2'] if k in ('MTE2','read') else COL['MTE3'] if k in ('MTE3','retire') else COL['SIMD'] if k in ('SIMD','body') else COL['control']
            top.barh(0,v,left=t,height=.42,color=color);t+=v
        top.set_yticks([0,1,2],['Critical-path budget','hOut tile read','X tile read']);top.set_ylim(2.65,-.65)
        if r['port']:
            anchor=st['arrival']+st['control'];port=r['port']
            for i,(s,length) in enumerate(zip(port['issue_us'],port['response_us'])):top.barh(i+1,length,left=anchor+s,height=.35,color=COL['MTE2'])
        else:top.text(.06,.28,'Per-read assignment unavailable',transform=top.transAxes,color='#687d88',fontsize=8)
        top.axvline(r['us'],color='#213846',lw=1.2);top.axvline(r['candidate_unknown_us'],color='#828c92',ls=':',lw=1)
        lower.set_title(f"CA  |  {m['core']}   ·   full window {m['ca_us']:.3f} µs",loc='left',fontsize=10.5,weight='bold')
        keys=['meta','h','x','simd',*[f'out{i}' for i in range(4)]]
        for idx,k in enumerate(keys):
            color=COL['MTE2'] if idx<3 else COL['SIMD'] if idx==3 else COL['MTE3']
            itv=m['lanes'][k];lower.broken_barh([(a,b-a) for a,b in itv],(idx-.18,.36),facecolor=color)
        lower.set_yticks(range(8),['96 B metadata','hOut tile read','X tile read','Vector on the tile',*[f'y[0, {i}, tile] write' for i in range(4)]])
        lower.set_ylim(7.8,-.7);lower.axvline(m['ca_us'],color='#213846',lw=1.2);lower.set_xlabel('Device time (µs)')
        for a in (top,lower):a.set_xlim(0,mx);a.set_axisbelow(True);a.grid(axis='x',color='#d8e1e8',lw=.7);a.tick_params(axis='y',length=0)
    fig.legend(handles=[Patch(color=COL['control'],label='Arrival / control'),Patch(color=COL['MTE2'],label='Read'),Patch(color=COL['SIMD'],label='Vector / body'),Patch(color=COL['MTE3'],label='Write / retire'),Line2D([],[],color='#828c92',ls=':',label='Shape-only estimate')],loc='upper left',bbox_to_anchor=(.161,.89),ncol=5,frameon=False)
    fig.text(.17,.065,'Model bars in the budget row are additive stage contributions. Only the repeat model resolves individual DMA responses.',fontsize=9)
    fig.text(.17,.043,'CA bars use actual descriptor windows and the union of non-wait vector instructions. Global time starts at the earliest AIV instruction.',fontsize=9)
    fig.savefig(OUT/('post-small-pipeline.png' if small else 'post-ordered-pipeline.png'),dpi=160);plt.close(fig)
if __name__=='__main__':main('small' in sys.argv)
