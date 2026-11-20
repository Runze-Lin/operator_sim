"""Render a registered single-rectangle critical-path budget and actual device issues."""
import hashlib
import json
from math import ceil, isclose
from pathlib import Path
from statistics import median
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from development.validation.matmul31.device.campaign import ca, OUT, REG
from operator_model import load_chip

LANES=['Control / launch','MTE2 / A and B','MTE1 / L1 to L0','Cube / MMAD','FIXP / partial sums','MTE2 / reduction reads','Vector / reduction','MTE3 / final write']
COLORS=['#7a8f99','#147b82','#5076b3','#7253a9','#b55436','#147b82','#935f9f','#bd672d']


def budget(job,sku):
    shape=job['shape'];m,n,_=shape; pred=job['predictions'][sku]; g=pred['geometry']; ch=load_chip(sku); hw=ch.details
    rates=ca.read(ROOT/'results/v0.16/matmul.json')['parameters']; clocks=1000*ch.frequency_ghz
    per_port=lambda port,key:hw[port]/ch.frequency_ghz if hw.get(port) else rates[key]
    delay=lambda key:hw.get('matmul_'+key) or rates[key]
    mte2=per_port('mte2_bandwidth_gbs','mte2_bytes_per_cycle');mte1=per_port('mte1_bandwidth_gbs','mte1_bytes_per_cycle');wr=per_port('mte3_bandwidth_gbs','write_bytes_per_cycle')
    am,an,k,bk=g['base_m'],g['base_n'],g['max_segment_k'],g['base_k']; assert g['output_tiles']==1 and k<=g['page_k']
    peak=ch.cube_tops['16']*1000/(ch.cube_cores*ch.frequency_ghz);times=[];t=0.
    def append(lane,duration,label):
        nonlocal t
        times.append(dict(lane=lane,start_us=t,end_us=t+duration,label=label));t+=duration
    append(0,g['launch_us'],'launch & arrival budget')
    append(1,(delay('read_latency_cycles')+2*k*max(m,n)/mte2)/clocks,'initial GM page')
    append(2,(delay('l0_latency_cycles')+2*min(k,bk)*max(m,n)/mte1)/clocks,'L0 lead')
    for i in range(ceil(k/bk)):
        part=min(k-i*bk,bk);cube=2*am*an*16*ceil(part/16)/peak; load=2*part*max(m,n)/mte1
        cycles=delay('mmad_latency_cycles')+cube if i+1==ceil(k/bk) else max(delay('mmad_ii_cycles')+cube,load)
        append(3,cycles/clocks,f'MMAD budget, K={part}')
    append(4,(delay('drain_cycles')+4*m*n/wr)/clocks,'partial write budget')
    if not isclose(t,g['launch_us']+g['partial_pipeline_us']):raise ValueError('Cumulative estimate disagrees with frozen partial pipeline')
    if g['partial_pipeline_us']<max(g['cold_floor_us'],g['l2_floor_us']):raise ValueError('Example needs an exposed shared-memory interval')
    append(6,g['merge_us'],'combined merge budget (MTE2, vector, writeback)')
    times[-1].update(scope='aggregate_merge',issue_budget_us=g['merge_issue_us'],shared_traffic_budget_us=g['merge_us']-g['merge_issue_us'])
    if not isclose(t,pred['us']):raise ValueError('Budget disagrees with frozen prediction')
    return times


def observed(stage):
    base=(ROOT/stage['trace']).parent;result=[];basis=[];populations={}
    # For each engine select its latest-finishing productive participant. The
    # all-core finish line independently includes passive participants as well.
    for kind in ('cube','vec'):
        selected=None
        for core,win in sorted(stage['windows'].items(),key=lambda r:-r[1][1]):
            if kind not in core:continue
            shard=base/core/'trace.json';raw=shard.read_bytes();xs=[e for e in json.loads(raw)['traceEvents'] if e.get('ph')=='X'];zero=min(e['ts'] for e in xs)-win[0]
            ev=[]
            for e in xs:
                name=e['name'];q=json.loads(e['args']['detail']);l=None
                if any(q.get(s) for s in ('is_wait','is_sync','is_duration_excluded')):continue
                if name=='MOV_OUT_TO_L1_MULTI_ND2NZ':l=1
                elif name=='LOAD_2Dv2':l=2
                elif name=='MMAD':l=3
                elif name=='FIX_L0C_TO_DST':l=4
                elif kind=='vec' and name=='VF':l=6
                elif kind=='vec' and name=='MOV_SRC_TO_DST_ALIGNv2':
                    if (q['src_mem'],q['dst_mem'])==('UB','OUT'):l=7
                    elif (q['src_mem'],q['dst_mem'])==('OUT','UB') and q['burst_number']>1:l=5
                if l is not None:ev.append(dict(lane=l,start_us=e['ts']-zero,end_us=e['ts']+e['dur']-zero,label=name,mmad_depth=q.get('k'),bytes=q.get('processed_bytes')))
            if not any(v['lane']==(3 if kind=='cube' else 7) for v in ev):continue
            for l in sorted({v['lane'] for v in ev}):
                same=[v for v in ev if v['lane']==l]; pop=populations.setdefault(LANES[l],[])
                pop.append((min(v['start_us'] for v in same),max(v['end_us'] for v in same)))
            if selected is None:
                selected=dict(core=core,start_us=win[0],end_us=win[1],file_sha256=hashlib.sha256(raw).hexdigest(),file=str(shard.relative_to(ROOT)),global_origin_us=zero)
                result.extend(ev);basis.append(selected)
        if not selected:raise ValueError(f'No productive {kind} shard')
    stats=lambda v:dict(min=min(v),median=median(v),max=max(v))
    phases={l:dict(cores=len(p),first_issue_us=stats([x for x,_ in p]),last_retire_us=stats([x for _,x in p])) for l,p in populations.items()}
    return result,basis,phases


def build(chip=ca.DT):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    reg=ca.read(REG);job=reg['rows'][0];assert job['shape']==[99,64,7168]
    if chip not in (ca.DT,ca.PR):raise ValueError('No physical target for this example')
    tag='dt' if chip==ca.DT else 'pr';stem='pro-b11-mtp8' if chip==ca.DT else 'pro-b11-mtp8-pr'
    d=ca.read(OUT/f'99-64-7168-{tag}.json');assert d['qualified'] and all(d['mechanism_checks'].values());stage,=d['stages']
    m=budget(job,chip);a,shards,phases=observed(stage);reg_time=job['predictions'][chip]['us'];measured=d['ca_us'];splits=job['predictions'][chip]['geometry']['k_splits']
    fig,axes=plt.subplots(2,1,sharex=True,figsize=(11,7.7));fig.subplots_adjust(left=.23,right=.97,top=.79,bottom=.13,hspace=.47)
    fig.text(.23,.95,f'{chip[6:11]} MatMulV3   |   99 × 64 × 7168',fontsize=21,weight='bold',color='#203641');fig.text(.23,.9,f'DeepSeek-V4-Pro indexer, B11 × Q9    ·    1 output rectangle × {splits} K partitions',fontsize=12,color='#395764')
    last=max(stage['windows'],key=lambda c:stage['windows'][c][1]);names=' / '.join(x['core'] for x in shards)
    totals=[reg_time,measured];titles=['Resource model · critical-partition budgets',f'CA trace · {names}']
    for i,(ax,ev,total,title) in enumerate(zip(axes,(m,a),totals,titles)):
        for v in sorted(ev,key=lambda q:q['start_us']):
            if v.get('scope')=='aggregate_merge':
                start=v['start_us'];end=v['end_us'];ax.fill_between([start,end],4.74,7.23,color='#d7e0e4',alpha=.68,zorder=2)
                ax.text((start+end)/2,6.,'Aggregate merge budget',fontsize=9,ha='center',va='center',color='#334f5b');continue
            l=v['lane'];start=v['start_us'];ax.barh(l,v['end_us']-start,left=start,height=.38,color=COLORS[l],alpha=.55 if i and l in (1,4,5) else .95,zorder=3);ax.vlines(start,l-.24,l+.24,lw=1,color=COLORS[l],zorder=4)
        ax.axvline(total,ls=(0,(3,3)),color='#30495b',lw=1);ax.set_title(title,loc='left',fontsize=12,weight='bold');ax.set_title(f'{total:.3f} µs',loc='right',fontsize=12,weight='bold');ax.set_ylim(7.7,-.7);ax.set_yticks(range(len(LANES)),LANES,fontsize=9);ax.tick_params(axis='y',length=0);ax.spines[['top','left','right']].set_visible(False);ax.grid(axis='x',color='#d3dfe5',lw=.7)
    axes[-1].set_xlim(0,max(totals)*1.04);axes[-1].set_xlabel('Device time since the first instruction on any core (µs)')
    fig.text(.23,.055,'CA ticks mark issues; bars include dependency wait. The shaded model box has no calibrated internal schedule. Dashed lines mark full windows.',fontsize=8,color='#415966');fig.text(.23,.032,f'CA selected the last productive AIC and AIV on each engine. Last participant of all cores: {last}.',fontsize=8,color='#415966')
    fig.savefig(OUT/(stem+'.png'),dpi=150);plt.close(fig)
    ca.save(OUT/(stem+'.json'),dict(shape=job['shape'],sku=chip,model_us=reg_time,ca_us=measured,model=m,ca=a,lanes=LANES,origin='Earliest instruction across all participating cores',productive_shards=shards,productive_core_distributions=phases,last_core=last,trace_sha256=stage['trace_sha256'],source_freeze_sha256=ca.digest(REG)))

if __name__=='__main__':build(ca.PR if '--pr' in sys.argv else ca.DT)
