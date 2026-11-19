"""Offline CA windows for the worked examples. Hash-matched DB/timeline only."""
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import sqlite3
from campaign import ROOT, read, sha
from development.validation.sku15.fa_independent import events

BINS=100


def group(queue):
    if queue.startswith(('RVEC','SIMD','VEC')):
        return 'Vector'
    return {'FIXPIPE':'FIXP','CUBE':'Cube','SCALAR':'Scalar'}.get(queue,queue)


def timeline(rows,clock=1):
    """Union payload instruction intervals per queue on last-ending core in each domain."""
    by_queue=defaultdict(list)
    for k,q,start,end in rows:
        if end>start:
            by_queue[k,group(q)].append((start/clock,end/clock))
    return by_queue


@lru_cache(maxsize=100)
def reduce_trace(filename,digest):
    path=Path(filename)
    if not path.is_absolute():path=ROOT/path
    assert sha(path)==digest,(path,digest)
    if path.suffix=='.db':
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
            freq=float(dict(db.execute('select KernelInfoAttr,KernelInfoVal from KernelInfo'))['frequency_ghz'])
            wins=list(db.execute('select ExecInstrCoreId,CoreTypeId,min(ExecInstrTickStart),max(ExecInstrTickEnd) from ExecutedInstructions group by 1,2'))
            first=min(x[2] for x in wins);end=max(x[3] for x in wins); units=freq*1000
            traces={}
            for kind,dom in [('AIC',False),('AIV',True)]:
                selected=[x for x in wins if (x[1]!=1)==dom]
                if not selected:continue
                cid,typ,*_=max(selected,key=lambda x:x[3])
                query="""select json_extract(ExtendParams,'$.queue_type'),ExecInstrTickStart,ExecInstrTickEnd
                         from ExecutedInstructions where ExecInstrCoreId=? and CoreTypeId=?
                         and ExecInstrIsDurationExcluded=0 and ExecInstrIsWait=0 and ExecInstrIsSynchronization=0"""
                traces.update(timeline(((kind,q or 'unknown',s-first,t-first) for q,s,t in db.execute(query,(cid,typ))),units))
            duration=(end-first)/units
    else:
        wins={};interesting=defaultdict(list)
        for e in events(path):
            if e['ph']!='X':continue
            k=e['pid'];s=e['ts'];t=s+e['dur']
            lo,hi=wins.get(k,(s,t));wins[k]=min(s,lo),max(t,hi)
            if t<=s:continue
            detail=e.get('args',{}).get('detail') or '{}'
            meta=json.loads(detail) if detail.startswith('{') else {}
            if t>s and not any(meta.get(flag) for flag in ('is_sync','is_wait','is_duration_excluded')):
                interesting[k].append((e['tid'],s,t))
        first=min(s for s,t in wins.values());duration=max(t for s,t in wins.values())-first;traces={}
        for kind,tag in [('AIC','cubecore'),('AIV','veccore')]:
            chosen=[core for core in wins if tag in core]
            if chosen:
                core=max(chosen,key=lambda k:wins[k][1])
                traces.update(timeline((kind,q,s-first,t-first) for q,s,t in interesting[core]))
    tracks={}
    for (kind,q),intervals in traces.items():
        merged=[]
        for a,b in sorted(intervals):
            if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
            else:merged.append([a,b])
        size=duration/BINS;bins=[0.]*BINS
        for a,b in merged:
            for i in range(max(0,int(a/size)),min(BINS,int(b/size)+1)):
                bins[i]+=max(0,min(b,(i+1)*size)-max(a,i*size))/size
        tracks[kind+'/'+q]=dict(union_us=sum(b-a for a,b in merged),density=[round(min(1,x),3) for x in bins])
    return dict(trace=str(path.relative_to(ROOT)),sha256=digest,span_us=duration,tracks=tracks)


def discover(obj):
    if isinstance(obj,list):
        for val in obj:yield from discover(val)
    elif isinstance(obj,dict):
        if obj.get('db_path') and obj.get('db_sha256'):yield obj['db_path'],obj['db_sha256']
        if obj.get('report_path') and obj.get('trace_sha256'):yield str(Path(obj['report_path'])/'simulator/trace.json'),obj['trace_sha256']
        for v in obj.values():
            if isinstance(v,(dict,list)):yield from discover(v)


def pipelines(record):
    if record.get('stages') and isinstance(record['stages'][0],dict) and record['stages'][0].get('trace'):
        files=[(s['trace'],s['trace_sha256']) for s in record['stages']]
    else:
        paths=record['trace']; paths=[paths] if isinstance(paths,str) else paths
        digests=record.get('trace_sha256');digests=[digests] if isinstance(digests,str) else digests
        files=[]
        for path in paths or []:
            path=ROOT/path
            if path.name=='trace.json' or path.suffix=='.db':
                files.append((path,digests[len(files)]))
            elif path.is_file() and digests:
                candidates={digest:name for name,digest in discover(read(path))}
                files.extend((candidates[d],d) for d in digests if d in candidates)
    reduced=[reduce_trace(str(p),d) for p,d in files]
    if reduced and abs(sum(x['span_us'] for x in reduced)-record['ca_us'])>max(.005,1e-4*record['ca_us']):
        raise ValueError(('CA trace and reference time differ',record['op'],record['shape']))
    return reduced


def issue_landmarks(record,op):
    """First MMAD and descriptor spacing on the same last-ending core. Official trace."""
    from statistics import median
    stage=(record.get('stages') or [{}])[0]
    if not isinstance(stage,dict) or not stage.get('windows') or not stage.get('trace'):
        return None
    kind='veccore' if op=='quant' else 'cubecore'
    wins=stage['windows'];core=max((c for c in wins if kind in c),key=lambda c:wins[c][1])
    all_mmad=0;first=None;series=defaultdict(list)
    for event in events(ROOT/stage['trace']):
        if event['ph']!='X':continue
        t=event['ts'];first=t if first is None else min(first,t)
        name=event['name']
        if name=='MMAD':all_mmad+=1
        if event['pid']!=core:continue
        queue=event['tid']
        selected=(name=='MMAD' or (op=='quant' and name=='MOV_SRC_TO_DST_ALIGNv2' and queue in ('MTE2','MTE3'))
                  or (op!='quant' and name=='MOV_OUT_TO_L1_MULTI_ND2NZ'))
        if selected:
            # The same mnemonic can describe Q, K and V or payload and scale transfers.
            series[(queue,name,event['args']['pc_addr'])].append((t,event['dur']))
    lines=[f'取 {core}。全核 MMAD 计数为 {all_mmad}。']
    for (q,opname,pc),xs in sorted(series.items()):
        xs.sort();starts=[t for t,d in xs];gaps=[b-a for a,b in zip(starts,starts[1:])]
        cadence=f'{median(gaps):.4f} µs' if gaps else 'N/A（只有一次）'
        lines.append(f'| {q} | `{opname}` @ `{pc}` | {len(xs)} | {starts[0]-first:.4f} | {median(d for t,d in xs):.4f} | {cadence} |')
    return lines
