"""Map ordinary _6 writes back to logical output coordinates in each original full-chip trace."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.post24.collect import events, digest
from operator_model import OperatorStream
from operator_model.post_history import estimate_post_history
from operator_model.post_repeat import channel_tiles

OUT = ROOT/'results/v0.29/post'
SHAPE = [65,4,5120]


def union(parts):
    acc = []
    for a,b in sorted(parts):
        if acc and a <= acc[-1][1]:
            acc[-1][1] = max(acc[-1][1], b)
        else:
            acc.append([a,b])
    return acc


def collect(sku):
    src = ROOT/f'results/v0.25/post/receipts/width-{sku}.json'
    rows = json.loads(src.read_text())['rows']
    r, = (r for r in rows if r['shape'] == SHAPE and r['role'] == 'entry')
    st = r['stage']; p = ROOT/st['trace']
    assert r['timing_qualified'] and digest(p) == st['trace_sha256']
    core = max(st['windows'],key=lambda k:st['windows'][k][1])
    origin = st['first_tick']
    def copies(path):
        for e in events(path):
            a = json.loads(e['args']['detail'])
            if any(a.get(k) for k in ('is_sync','is_wait','is_duration_excluded')):
                continue
            if e['name'].startswith('MOV_SRC_TO_DST') or a['queue_type']=='SIMD':
                yield e,a
    t,n,d = SHAPE
    spans = []
    for e,a in copies(p):
        if a['queue_type']!='MTE3' or a.get('dst_mem')!='OUT':
            continue
        assert a['burst_number']==n and a['dst_stride']==2*d and a['processed_bytes']==n*a['burst_len']
        for j in range(n):
            lo = a['dst_addr'] + j*a['dst_stride']
            spans.append((lo, lo+a['burst_len']))
    spans.sort(); base = spans[0][0]; end = base
    for lo,hi in spans:
        assert lo == end
        end = hi
    assert end-base == 2*t*n*d
    rd=[];wr=[];vec=[]
    for e,a in copies(p.parent/core/'trace.json'):
        lo = (round(e['ts']*1650)-origin)/1650
        hi = (round((e['ts']+e['dur'])*1650)-origin)/1650
        if hi <= lo:continue
        if a['queue_type']=='SIMD':
            vec.append((lo,hi));continue
        v = dict(start_us=lo,end_us=hi,src=a['src_addr'],dst=a['dst_addr'],bytes=a['processed_bytes'])
        if a['queue_type']=='MTE2':rd.append(v)
        if a['queue_type']=='MTE3':wr.append(v)
    rd.sort(key=lambda v:v['start_us']);wr.sort(key=lambda v:v['start_us'])
    assert [v['bytes'] for v in rd] == [96]+2*[64,16,5120,20480,5120,20480]
    assert [v['bytes'] for v in wr] == 4*[20480]
    pieces=[]
    for i in range(2):
        desc=rd[1+6*i:7+6*i]; parts=[]
        for f in range(2):
            h,x=desc[2+2*f:4+2*f];w=wr[2*i+f]
            row, offset=divmod((w['dst']-base)//2,n*d)
            assert offset==2560*f
            if f:assert h['src']-parts[0]['h']['src']==x['src']-parts[0]['x']['src']==5120
            parts.append(dict(h=h,x=x,y=w,token=row,first_column=offset,columns=2560))
        assert parts[0]['token']==parts[1]['token']
        pieces.append(dict(token=parts[0]['token'],coeff=desc[:2],fragments=parts))
    assert pieces[1]['token']==pieces[0]['token']+1
    stream=OperatorStream(sku,track_post_history=True)
    for e in rows[:r['i']+1]: public=stream.advance('mhc_post',e['shape'])
    alternative=estimate_post_history(SHAPE,chip=sku).for_predecessor(r['prior_aiv'],r['previous_shape'])
    stage=dict(alternative.stages_us); ctrl=stage['arrival']+stage['control_to_first_read']
    ready=ctrl+stage['first_read_to_ready'];first=ready+stage['ready_to_first_write']
    last=first+stage['overlapping_rows'];finish=last+stage['last_write_response']
    assert abs(finish+stage['exit_after_write']+stage['shared_link_floor']-alternative.latency_us)<1e-10
    intervals=[(v['start_us'],v['end_us'])for v in rd[1:]]
    read_union=union(intervals)
    read_sum=sum(b-a for a,b in intervals); read_occupied=sum(b-a for a,b in read_union)
    return dict(chip=sku,shape=SHAPE,previous_shape=r['previous_shape'],active=st['active_aiv'],core=core,
                stream_order_digest=digest(src),input_trace=st['trace'],input_trace_sha256=st['trace_sha256'],
                ca_us=st['ca_us'],first_global_tick=origin,terminal_global_tick=st['last_tick'],
                output_base=base,total_written=end-base,dense_output_spans=len(spans),
                experimental_aggregate_us=alternative.latency_us,public_us=public.latency_us,
                scenario='reuse_after_token',model_stages_us=stage,model_marks_us=[0,ctrl,ready,first,last,finish,alternative.latency_us],
                ca_metadata=rd[0],rows=pieces,vector_union_us=union(vec),
                read_response_sum_us=read_sum,read_response_union_us=read_occupied,
                overlapping_read_response_us=read_sum-read_occupied,
                row2_issued_before_row1_last_write_retired_us=wr[1]['end_us']-rd[7]['start_us'],
                tiling=channel_tiles(d,stream.chip))


def plot(j):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    c=['#9daab3','#127e85','#8b73b3','#b6522d']
    fig,(model,ca)=plt.subplots(2,1,sharex=True,figsize=(12,8.3),gridspec_kw={'height_ratios':[.82,1.3]})
    fig.subplots_adjust(left=.22,right=.955,top=.815,bottom=.125,hspace=.47)
    fig.suptitle(j['chip'].split('_')[0]+'  |  mHC Post  ·  65 × 4 × 5120',x=.22,y=.975,ha='left',weight='bold',fontsize=21)
    fig.text(.22,.918,'[121, 4, 1216] → [65, 4, 5120]   ·   33 AIVs   ·   two 2,560-column fragments per row',fontsize=11.5)
    fig.legend(handles=[Patch(color=z,label=s) for z,s in zip(c,['Dispatch','MTE2 reads','Compute / issue interval','MTE3 writes'])],loc='upper left',bbox_to_anchor=(.211,.897),ncol=4,frameon=False)
    def bar(ax,y,a,b,color,height=.3):
        ax.broken_barh([(a,b-a)],(y-height/2,height),facecolors=c[color])
    m=j['model_marks_us'];tokens=[r['token']for r in j['rows']]
    model.set_title('Experimental row-aggregate schedule',loc='left',weight='bold')
    model.set_title(f"{j['experimental_aggregate_us']:.3f} µs",loc='right',weight='bold')
    for y,a,b,color in [(0,m[0],m[1],0),(1,m[1],m[2],1),(1,m[2],m[3],2),(2,m[3],m[4],2),(3,m[4],m[5],3)]:
        bar(model,y,a,b,color)
    model.set_yticks(range(4),['Dispatch to first read',f'Token {tokens[0]} · read to write issue','Next row · write-issue interval',f'Token {tokens[1]} · final write']);model.set_ylim(3.55,-.55)
    ca.set_title('CA · '+j['core']+' (last AIV to retire)',loc='left',weight='bold')
    ca.set_title(f"{j['ca_us']:.3f} µs",loc='right',weight='bold')
    def desc(y,v,kind,h=.18):bar(ca,y,v['start_us'],v['end_us'],kind,h)
    desc(0,j['ca_metadata'],1);labels=['Metadata · 96 B'];y=1
    for r in j['rows']:
        labels.append(f"Token {r['token']} · coefficients")
        for k,v in enumerate(r['coeff']):desc(y-.12+.24*k,v,1)
        y+=1
        for f in r['fragments']:
            labels.append(f"Token {r['token']} · D {f['first_column']}–{f['first_column']+2559}")
            desc(y-.22,f['h'],1);desc(y,f['x'],1);desc(y+.22,f['y'],3);y+=1
    for a,b in j['vector_union_us']:bar(ca,y,a,b,2,.28)
    labels.append('Vector · issued activity');ca.set_yticks(range(y+1),labels);ca.set_ylim(y+.55,-.55)
    ca.set_xlabel('Device time from earliest AIV instruction (µs)')
    for ax,total in ((model,j['experimental_aggregate_us']),(ca,j['ca_us'])):
        ax.axvline(total,ls='--',color='#3b5662',lw=1);ax.set_axisbelow(True);ax.grid(axis='x',color='#d3dfe3')
        ax.spines[['top','left','right']].set_visible(False);ax.tick_params(axis='y',length=0,pad=8)
    model.set_xlim(0,1.045*max(j['experimental_aggregate_us'],j['ca_us']))
    fig.text(.22,.052,f"Public occupancy model: {j['public_us']:.3f} µs. Upper panel combines both fragments. Lower panel identifies each write by GM address.",size=9)
    fig.text(.22,.028,'Purple above includes issue and dependency waits. Purple below is the union of active SIMD intervals. Dashed lines mark the full kernel window.',size=9)
    fig.savefig(OUT/('wide-'+j['chip'].split('_')[0][-2:].lower()+'.png'),dpi=155);plt.close(fig)


if __name__=='__main__':
    for sku in ('Ascend950DT_9582','Ascend950PR_957b'):
        row=collect(sku);plot(row)
        (OUT/('wide-'+sku.split('_')[0][-2:].lower()+'.json')).write_text(json.dumps(row,indent=2)+'\n')
