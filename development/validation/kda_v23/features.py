"""Extract per-core transfer latency and issue spacing from existing true instruction spans."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import re
import statistics
import sys

from series import OUT,probe,read,save
from verify import events


def distribution(values):
    if not values:return None
    return dict(count=len(values),minimum=min(values),median=statistics.median(values),maximum=max(values))


def queue_stats(transfers):
    ordered=sorted(transfers)
    ranges=[]
    for start,end,size in ordered:
        if ranges and start<=ranges[-1][1]:ranges[-1][1]=max(ranges[-1][1],end)
        else:ranges.append([start,end])
    return dict(count=len(ordered),bytes=sum(t[2] for t in ordered),first_issue=ordered[0][0],
        last_retire=max(t[1] for t in ordered),descriptor_latency_ticks=distribution([t[1]-t[0] for t in ordered]),
        consecutive_issue_ticks=distribution([b[0]-a[0] for a,b in zip(ordered,ordered[1:])]),
        busy_union_ticks=sum(e-s for s,e in ranges))


def core_data(path,base,lo):
    mma=[];transfers=defaultdict(list);vec=Counter();scalar=0
    for e in events(path):
        d=json.loads(e['args']['detail'] or '{}')
        if d.get('is_duration_excluded') or d.get('is_sync') or d.get('is_wait'):continue
        a=round(e['ts']*1650)-lo;b=round((e['ts']+e['dur'])*1650)-lo
        if e['name']=='MMAD':
            mma.append(dict(start_tick=a,end_tick=b,m=d['m'],n=d['n'],k=d['k'],pc_offset=hex(int(e['args']['pc_addr'],16)-base)))
        if all(k in d for k in ('processed_bytes','src_mem','dst_mem')):
            transfers[d['src_mem']+'->'+d['dst_mem']].append([a,b,d['processed_bytes']])
        if d.get('instr_type','').startswith('RVEC'):vec[d['instr_type']]+=1
        if d.get('instr_type')=='SCALAR':scalar+=1
    mma.sort(key=lambda m:m['start_tick'])
    return dict(mmad=mma,mmad_issue_intervals_ticks=distribution([b['start_tick']-a['start_tick'] for a,b in zip(mma,mma[1:])]),
                memory={k:queue_stats(v) for k,v in sorted(transfers.items())},vector_subpipe_instructions=dict(vec),scalar_instructions=scalar)


def write(receipt):
    data=read(receipt)
    if not data.get('qualified'):return False
    target=OUT/'resources';target.mkdir(exist_ok=True);dest=target/receipt.name
    if dest.exists() and read(dest).get('receipt_sha256')==probe.digest(receipt):return True
    all_stages=[]
    for stage in data['stages']:
        sim=(probe.ROOT/stage['aggregate']).parent
        base=int((sim.parent/'dump/pc_start_addr.txt').read_text(),16)
        cores={c:core_data(sim/c/'trace.json',base,stage['first_tick']) for c in stage['cores']}
        assert sum(len(v['mmad']) for v in cores.values())==stage['mmads']
        all_stages.append(dict(kernel=stage['kernel'],family=stage['family'],active_aic=stage['active_aic'],active_aiv=stage['active_aiv'],
          ca_ticks=stage['last_tick']-stage['first_tick'],last_cores=[c for c,s in stage['cores'].items() if s['last_tick']==stage['last_tick']],cores=cores))
    save(dest,dict(case_id=data['case']['id'],sku=data['case']['sku'],shape=data['case']['shape'],receipt_sha256=probe.digest(receipt),
        unit='tick, 1650 ticks/us; all per-instruction timestamps relative to earliest core event in the same stage',
        traffic_scope='OUT is a GM instruction address space; no claim that these bytes all miss L2 or reach physical HBM/LPDDR. pipelined descriptor lifetime is distinct from issue spacing.',
        stages=all_stages))
    return True


if __name__=='__main__':
    for source in [OUT/'reused-ascend950dt-t2-h1.json',*sorted(OUT.glob('ascend950*.json'))]:
        if source.exists():write(source)
