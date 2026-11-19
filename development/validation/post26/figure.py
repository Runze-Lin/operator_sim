"""Plot the frozen public aggregate row against each measured strided GM fragment."""
from collections import defaultdict
from pathlib import Path
import json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post24.collect import events
from development.validation.post24.figure import union
from development.validation.post26.freeze import OUT,OLD
from development.validation.sku19.campaign import read,write
from operator_model import OperatorStream
from operator_model.post_token_pipeline import token_pipeline
from operator_model.post_repeat import channel_tiles

def measure():
    path=OLD/'receipts/width-Ascend950DT_9582.json';rows=read(path)['rows']
    r,=[x for x in rows if x['shape']==[65,4,5120] and x['role']=='repeat']
    assert r['timing_qualified'];t,n,d=r['shape'];st=r['stage'];origin=st['first_tick']
    p=ROOT/st['trace'];last=max(st['windows'],key=lambda k:st['windows'][k][1]);extents=[]
    for e in events(p):
        if not e['name'].startswith('MOV_SRC_TO_DST'):continue
        v=json.loads(e['args']['detail'])
        if v['queue_type']!='MTE3' or any(v.get(k) for k in ('is_sync','is_wait','is_duration_excluded')):continue
        assert v['dst_mem']=='OUT' and v['burst_number']==n and v['burst_len']*n==v['processed_bytes'] and v['dst_stride']==2*d
        for j in range(n):
            s=v['dst_addr']+j*v['dst_stride'];extents.append((s,s+v['burst_len']))
    extents.sort();base=extents[0][0];prev=base
    for lo,hi in extents:assert lo==prev;prev=hi
    assert prev-base==t*n*d*2
    rr=[];ww=[];simd=[]
    for e in events(p.parent/last/'trace.json'):
        a=json.loads(e['args']['detail']);u=(round(e['ts']*1650)-origin)/1650.;v=(round((e['ts']+e['dur'])*1650)-origin)/1650.
        if v<=u or any(a.get(k) for k in ('is_sync','is_wait','is_duration_excluded')):continue
        if a['instr_type'].startswith(('RVEC','SIMD')):simd.append((u,v))
        if not e['name'].startswith('MOV_SRC_TO_DST'):continue
        row=dict(start=u,end=v,bytes=a['processed_bytes'],src=a['src_addr'],dst=a['dst_addr'])
        if a['queue_type']=='MTE2':rr.append(row)
        if a['queue_type']=='MTE3':ww.append(row)
    rr.sort(key=lambda q:q['start']);ww.sort(key=lambda q:q['start'])
    stream=OperatorStream(r['chip'],experimental_post_repeats=True);res=None
    for s in rows:
        res=stream.advance('mhc_post',s['shape'])
        if s['i']==r['i']:break
    diag=res.projection['timing_diagnostics'];plan=diag['target_plan'];tiles=diag['channel_tiles'];count=tiles['count'];assert count==2 and tiles['inner']==tiles['tail']==2560
    assert rr[0]['bytes']==96 and len(ww)==count*plan['rows_per_core']
    assert len(rr)==1+(2+2*count)*plan['rows_per_core']
    model=token_pipeline(r['shape'],stream.chip,plan,'repeat',read(OLD/'prices.json'),diag['shared_floor_us'],range(plan['rows_per_core']))
    assert abs(model['us']-res.latency_us)<1e-12
    items=[]
    for i in range(plan['rows_per_core']):
        reads=rr[1+i*(2+2*count):1+(i+1)*(2+2*count)]
        assert [v['bytes'] for v in reads[:2]]==[64,16];parts=[]
        for j in range(count):
            h,x=reads[2+2*j:4+2*j];w=ww[i*count+j];offset=(w['dst']-base)//2;token,k=divmod(offset,n*d);stream_idx,col=divmod(k,d)
            assert stream_idx==0 and col==j*tiles['inner'] and x['bytes']==w['bytes']==4*h['bytes']==2*n*tiles['inner']
            if parts:assert h['src']-parts[-1]['h']['src']==x['src']-parts[-1]['x']['src']==2*tiles['inner']
            parts.append(dict(token=token,col=col,h=h,x=x,w=w))
        assert len({v['token'] for v in parts})==1
        items.append(dict(token=parts[0]['token'],weights=reads[:2],parts=parts))
    assert all(b['token']==a['token']+1 for a,b in zip(items,items[1:]))
    return dict(shape=r['shape'],core=last,active=r['active'],model=model,CA_us=st['ca_us'],items=items,metadata=rr[0],vector=union(simd),
                tiles=tiles,full_GM_coverage_bytes=prev-base,fragments_checked=len(extents),trace=st['trace'],sha256=st['trace_sha256'],
                formula='N=4, two 32B-aligned FP32 coefficient queues plus double-buffered operands/work: 320 + 52 * aligned_inner < target UB; round each width up to 16, prefer smallest tile count.',
                source=dict(host_blob='082a3a52c5fb5bec7c1ccb9434178ad6334b4967',installed_header_sha256='26709934b798b877a0d8c48214d1183713c7cce58ede9ada55b30508ef424226',reference_header_git_blob='54c4a55d0a35a1924c280d09f4312e191d683c08',installed_header_git_blob='d6bb3ae4f80e6905c4e76f61ffcbb019150a9795',host_git='14d022b1a4c3df263bf697534e3c25c7c992ff5d:mhc/mhc_post/op_host/op_tiling/arch35/mhc_post_tiling_regbase_arch35.cpp',installed_kernel='cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/mhc_post/op_kernel/arch35/mhc_post_regbase.h'))

def render(x):
    col=dict(read='#157987',write='#af4d2a',handoff='#ad99bf',vector='#7756a3',launch='#b6c6cb')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'#203745','axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
    fig,(a,b)=plt.subplots(2,1,figsize=(12.2,8.9),sharex=True,gridspec_kw={'height_ratios':[1,1.6]})
    fig.subplots_adjust(left=.215,right=.96,hspace=.38,bottom=.135,top=.80)
    fig.suptitle('950DT mHC Post  |  65 × 4 × 5120',x=.215,y=.97,ha='left',fontweight='bold',fontsize=21)
    fig.text(.215,.915,'33 AIVs  ·  exact repeat  ·  2 token rows on the last core  ·  2 channel fragments per row in CA',fontsize=11.6)
    fig.legend(handles=[Patch(color=col[k],label=t) for k,t in [('read','MTE2 reads'),('handoff','Ready to write issue'),('write','MTE3 writes'),('vector','AIV vector activity')]],loc='upper left',bbox_to_anchor=(.208,.903),ncol=4,frameon=False)
    def bar(ax,y,lo,hi,color,h=.12):
        ax.broken_barh([(lo,hi-lo)],(y-h/2,h),facecolor=col[color])
    def desc(ax,y,v,key,h=.12):bar(ax,y,v['start'],v['end'],key,h)
    m=x['model'];a.set_title('Model · one aggregate response per operand and token',loc='left',fontsize=13,weight='bold');a.set_title(f"{m['us']:.3f} µs",loc='right',fontsize=13,weight='bold')
    bar(a,0,0,m['stages_us']['arrival']+m['stages_us']['control_to_first_read'],'launch',.25)
    for i,row in enumerate(m['timeline'],1):
        for j,(lo,hi) in enumerate(row['reads_us']):bar(a,i-.28+j*.12,lo,hi,'read',.10)
        ws,we=row['write_us'];bar(a,i+.13,max(v[1] for v in row['reads_us']),ws,'handoff',.09);bar(a,i+.27,ws,we,'write')
    a.set_yticks(range(3),['Dispatch to first read']+[f"Token {row['token']} · full row" for row in x['items']]);a.set_ylim(2.6,-.54)
    b.set_title('CA · strided four-stream fragments on '+x['core'],loc='left',fontsize=13,weight='bold');b.set_title(f"{x['CA_us']:.3f} µs",loc='right',fontsize=13,weight='bold')
    desc(b,0,x['metadata'],'read');labels=['Metadata · 96 B'];y=1
    for row in x['items']:
        labels.append(f"Token {row['token']} · weights")
        for j,v in enumerate(row['weights']):desc(b,y-.1+j*.2,v,'read')
        y+=1
        for p in row['parts']:
            labels.append(f"{row['token']} · D {p['col']}–{p['col']+x['tiles']['inner']-1}")
            desc(b,y-.16,p['h'],'read');desc(b,y,p['x'],'read');desc(b,y+.17,p['w'],'write');y+=1
    for s,e in x['vector']:bar(b,y,s,e,'vector',.27)
    labels.append('AIV vector activity');b.set_yticks(range(y+1),labels);b.set_ylim(y+.55,-.5);b.set_xlabel('Device time from earliest AIV instruction (µs)')
    for ax,us in zip((a,b),(m['us'],x['CA_us'])):
        ax.axvline(us,ls='--',lw=1,color='#334e5c');ax.set_axisbelow(True);ax.grid(axis='x',color='#d6e2e7');ax.tick_params(axis='y',pad=7,length=0);ax.set_xlim(0,1.06*max(m['us'],x['CA_us']))
    fig.text(.215,.065,'Each CA write contains four 5,120-byte bursts. Adjacent fragments cover all 5,120 columns of each output stream.',fontsize=9)
    fig.text(.215,.042,'The model combines the two channel fragments into one row response; individual fragment timings remain unmodeled.',fontsize=9)
    fig.savefig(OUT/'split-width-repeat.png',dpi=155);plt.close(fig)
if __name__=='__main__':
    data=measure();render(data);write(OUT/'split-width-repeat.json',data)
