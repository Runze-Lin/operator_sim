"""Summarize official trace data offline, distinguishing overlapping queue windows from latency."""
from collections import Counter, defaultdict
import json
from statistics import median
import sys
from register import OUT, ROOT, read, write
sys.path.insert(0,str(ROOT/'development/validation/sku17'))
from campaign_collect import events


def merged(seq):
    combined=[]
    for start,end in sorted(seq):
        if combined and start<=combined[-1][1]:
            combined[-1][1]=max(end,combined[-1][1])
        else:combined.append([start,end])
    return combined


def width(seq):
    return sum(b-a for a,b in seq)


def reduce(record):
    stage,=record['stages']
    core=max(stage['windows'],key=lambda c:stage['windows'][c][1])
    intervals=defaultdict(list);dmabytes=Counter();dmacounts=Counter();descriptors=defaultdict(list)
    first=float('inf')
    for event in events(ROOT/stage['trace']):
        if event['ph']!='X':continue
        t=event['ts'];first=min(t,first);q=event['tid'];name=event['name']
        dma=(q in ('MTE2','MTE3') and name=='MOV_SRC_TO_DST_ALIGNv2')
        critical=(event['pid']==core)
        if not(critical or dma):continue
        if not dma and event['dur']<=0:continue
        detail=event.get('args',{}).get('detail') or '{}'
        info=json.loads(detail) if isinstance(detail,str) else detail
        if not isinstance(info,dict):info={}
        if any(info.get(k,False) for k in ('is_wait','is_sync','is_duration_excluded')):continue
        if dma:
            dmabytes[q]+=info.get('processed_bytes',0);dmacounts[q]+=1
            if critical:descriptors[q,event['args']['pc_addr']].append((t,event['dur'],info.get('processed_bytes',0)))
        if critical and event['dur']>0:
            group='SIMD' if q.startswith(('SIMD','RVEC','VEC')) else q
            intervals[group].append((t,t+event['dur']))
    unions={q:merged(v)for q,v in intervals.items()};sw=unions.get('SIMD',[]);wr=unions.get('MTE3',[])
    streams=[]
    for (pipe,pc),xs in sorted(descriptors.items()):
        xs.sort();gaps=[b[0]-a[0]for a,b in zip(xs,xs[1:])]
        streams.append(dict(queue=pipe,pc=pc,count=len(xs),first_us=xs[0][0]-first,last_end_us=max(t+d for t,d,_ in xs)-first,
            duration_median_us=median(d for _,d,_ in xs),issue_gap_median_us=median(gaps)if gaps else None,
            raw_processed_bytes=sum(v for _,_,v in xs)))
    return dict(op=record['op'],shape=record['shape'],chip=record['chip'],trace=stage['trace'],trace_sha256=stage['trace_sha256'],
                ca_us=record['ca_us'],terminal_core=core,terminal_core_start_us=stage['windows'][core][0],queue_payload_union_us={q:width(v)for q,v in unions.items()},
                mte3_simd_overlap_us=width(wr)+width(sw)-width(merged(wr+sw)),critical_core_dma_streams=streams,
                all_cores_dma_counts=dict(dmacounts),all_cores_dma_reported_bytes=dict(dmabytes))


def main():
    target=OUT/'trace-evidence.json'
    old=read(target)['rows']if target.exists() else []
    cache={r['trace_sha256']:r for r in old}
    rows=[]
    for r in read(OUT/'records.json'):
        fingerprint=r['trace_sha256'][0]
        if fingerprint not in cache: cache[fingerprint]=reduce(r)
        rows.append(cache[fingerprint])
    write(target,dict(scope='Last-ending AIV active payload queue unions exclude waits, synchronization and excluded durations. Unions can overlap and need not sum to device latency. DMA counts and descriptor processed_bytes are observed, independent of physical DRAM bus transactions.',rows=rows))
    print('trace reductions',len(rows))

if __name__=='__main__':main()
