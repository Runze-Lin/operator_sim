"""Plot true matching rows using contiguous output GM and stride-checked operand descriptors."""
import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.rms_ordered_verify.verify import events
from development.validation.post24.figure import union
from development.validation.post25.calibrate import predict, OUT
from development.validation.sku19.campaign import read,write
from operator_model import load_chip

def map_rows(r):
    st=r['stage'];t,n,d=r['shape'];tr=ROOT/st['trace'];origin=st['first_tick'];by=[]
    for e in events(tr):
        if e['name'].startswith('MOV_SRC_TO_DST'):
            a=json.loads(e['args']['detail'])
            if a['queue_type']=='MTE3' and not any(a.get(f) for f in ('is_sync','is_wait','is_duration_excluded')):
                assert a['dst_mem']=='OUT' and a['burst_number']*a['burst_len']==a['processed_bytes']
                assert a['burst_number']==1 or a['dst_stride']==a['burst_len']
                by.append((a['dst_addr'],a['processed_bytes'],e['pid']))
    by.sort();start=by[0][0];stride=2*n*d
    assert len(by)==t and all(addr==start+i*stride and size==stride for i,(addr,size,c) in enumerate(by))
    core=max(st['windows'],key=lambda k:st['windows'][k][1])
    rr=[];ww=[];vec=[]
    for e in events(tr.parent/core/'trace.json'):
        a=json.loads(e['args']['detail']);s=(round(e['ts']*1650)-origin)/1650.;b=(round((e['ts']+e['dur'])*1650)-origin)/1650.
        if b<=s or any(a.get(f) for f in ('is_sync','is_wait','is_duration_excluded')):continue
        if a['instr_type'].startswith(('RVEC','SIMD')):vec.append([s,b])
        if e['name'].startswith('MOV_SRC_TO_DST'):
            q=(s,b,a['src_addr'],a['dst_addr'],a['processed_bytes'])
            if a['queue_type']=='MTE2':rr.append(q)
            elif a['queue_type']=='MTE3':ww.append(q)
    rr.sort();ww.sort();assert rr[0][-1]==96 and len(rr)==1+4*len(ww)
    assert len(ww)==r['plan']['rows_per_core'],'terminal core has fewer than predicted work assignments'
    items=[];last=None
    for i,w in enumerate(ww):
        token=(w[3]-start)//stride;inputs=rr[1+4*i:5+4*i]
        assert by[token]==(w[3],stride,core)
        steps=[64,16,2*d,8*d];assert [z[-1] for z in inputs]==steps
        if last is not None:
            assert token==items[-1]['token']+1 and [z[2]-o[2] for z,o in zip(inputs,last)]==steps
        items.append(dict(token=token,reads_us=[z[:2] for z in inputs],write_us=w[:2]));last=inputs
    measured=dict(core=core,metadata_us=rr[0][:2],aiv_vector_us=union(vec),ca_us=st['ca_us'],items=items,trace=st['trace'],trace_sha256=st['trace_sha256'])
    pred=predict(r['shape'],load_chip(r['chip']),r['prior_aiv'],read(OUT/'prices.json'),range(len(items)),r['previous_shape'])
    assert abs(pred['us']-r['us'])<1e-12 and all(j['row']==i for i,j in enumerate(pred['pipeline']['timeline']))
    return dict(shape=r['shape'],role=r['role'],sku=r['chip'],active=r['active'],prior=r['prior_aiv'],model=pred['pipeline'],measured=measured)

def draw(data,slug):
    m=data['model'];ca=data['measured'];t,n,d=data['shape'];items=ca['items'];rows=len(items)
    colors={'read':'#177c87','write':'#b5522e','body':'#b39bc8','launch':'#95a6af','vector':'#7360a8'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'#223946','axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
    fig,axes=plt.subplots(2,1,figsize=(12,6.5+.49*rows),sharex=True,sharey=False)
    fig.subplots_adjust(left=.205,right=.964,top=.79,bottom=.125,hspace=.35)
    sku='950DT' if '950DT' in data['sku'] else '950PR'
    fig.suptitle(f"{sku} mHC Post  |  {t} × {n} × {d}",x=.205,y=.965,ha='left',weight='bold',fontsize=21)
    fig.text(.205,.90,f"{data['active']} AIVs  ·  {rows} token items on the last AIV  ·  {m['state']}  ·  BF16",fontsize=12)
    fig.legend(handles=[Patch(color=colors[k],label=s) for k,s in [('read','MTE2 read'),('body','Operand ready to write issue'),('write','MTE3 write'),('vector','AIV vector instructions')]],loc='upper left',bbox_to_anchor=(.199,.887),ncol=4,frameon=False,fontsize=9)
    titles=['Frozen row schedule · predicted longest AIV',f"CA trace · {ca['core']} finishes last"]
    for ix,(a,src,us) in enumerate(zip(axes,[m['timeline'],items],[m['us'],ca['ca_us']])):
        a.set_title(titles[ix],loc='left',weight='bold',fontsize=13);a.set_title(f'{us:.3f} µs',loc='right',weight='bold',fontsize=13)
        def bar(y,h,lo,hi,color):
            a.broken_barh([(lo,hi-lo)],(y-h/2,h),facecolor=color)
        if ix==0:
            ar=m['stages_us']['arrival'];ctl=m['stages_us']['control_to_first_read']
            bar(-1,.22,0,ar,'#596f7c');bar(-1,.22,ar,ar+ctl,'#b7c8cd')
        else:
            bar(-1,.16,*ca['metadata_us'],colors['read'])
        for i,row in enumerate(src):
            for j,(s,e) in enumerate(row['reads_us']):bar(i+(j-1.9)*.11,.085,s,e,colors['read'])
            ws,we=row['write_us'];bar(i+.24,.11,ws,we,colors['write'])
            if ix==0:bar(i+.12,.08,max(z[1] for z in row['reads_us']),ws,colors['body'])
        if ix==1:
            for s,e in ca['aiv_vector_us']:bar(rows,.25,s,e,colors['vector'])
        else:
            a.text(.012,rows,'Per-instruction issue unavailable',ha='left',va='center',color='#637681',fontsize=8,transform=a.get_yaxis_transform())
        for y in range(-1,rows):a.axhline(y+.5,color='#d5e0e4',lw=.7,zorder=0)
        a.axvline(us,color='#37505c',lw=1,ls='--');a.set_axisbelow(True);a.grid(axis='x',color='#d5e0e4');a.tick_params(axis='y',length=0,pad=7)
        a.set_yticks(list(range(-1,rows+1)),['Dispatch to first read' if ix==0 else 'Metadata · 96 B']+[f"Token {v['token']} · 4 streams" for v in items]+['AIV vector activity']);a.set_ylim(rows+.5,-1.55);a.set_xlim(0,1.05*max(m['us'],ca['ca_us']))
    axes[-1].set_xlabel('Device time from first instruction on any AIV (µs)')
    fig.text(.205,.052,'Token IDs come from contiguous output GM addresses. The upper panel uses ordinal positions on a longest AIV.',fontsize=9)
    fig.text(.205,.030,'Four read bars per token, top to bottom: mix weights, post weights, hOut, residual. Purple on each modeled row is a handoff span.',fontsize=9)
    fig.savefig(OUT/(slug+'.png'),dpi=160);plt.close(fig);write(OUT/(slug+'.json'),data)

if __name__=='__main__':
    group,token,width,slug=sys.argv[1:];sku='Ascend950DT_9582'
    rows=read(OUT/'receipts'/(group+'-'+sku+'.json'))['rows']
    r,=[r for r in rows if r['shape']==[int(token),4,int(width)] and r['role']=='repeat' and r['timing_qualified']]
    draw(map_rows(r),slug)
