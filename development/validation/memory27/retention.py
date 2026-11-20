"""Bound dirty occupancy from the same measured DMA addresses with configurable cache organization."""
import argparse
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.27/memory'
sys.path.insert(0,str(ROOT/'development/validation/cache25'))
from descriptors import extract
from operator_model import load_chip


def read(path):
    return json.loads(path.read_bytes())


def payload(d):
    """The v0.25 digest covers every DMA including the separately allocated tiling data."""
    base=(ROOT/d['raw_trace']).parent
    regions={k:d[k] for k in ('read','write')}
    counts={'read':0,'write':0}
    fp=sha256()
    events=[]
    for core,shift in sorted(d['global_origins_us'].items()):
        path=base/core/'trace.json'
        all_events=[(kind,(addr,n,a-shift,b-shift)) for kind,(addr,n,a,b) in extract(path)]
        fp.update(json.dumps((core,all_events),separators=(',',':')).encode())
        for kind,(addr,n,a,b) in all_events:
            r=regions[kind]
            if r['gm_base']<=addr and addr+n<=r['gm_end']:
                events.append((kind=='write',addr,n,a,b))
                counts[kind]+=n
    if fp.hexdigest()!=d['dma_records_sha256'] or any(counts[k]!=v['bytes'] for k,v in regions.items()):
        raise ValueError('Changed DMA records or incomplete coverage')
    return events


def ordered(events,mode):
    """Latest first; all reads issue first and writes retire last for the maximum dirty case."""
    def stamp(e):
        wr,_,_,a,b=e
        if mode=='maximum_dirty':
            return (b if wr else a,wr)
        if mode=='minimum_dirty':
            return (a if wr else b,not wr)
        if mode=='issue':
            return (a,wr)
        return (b,wr)
    return sorted(events,key=stamp,reverse=True)


def byte_suffix(events,capacity):
    free=capacity
    dirty=0
    for wr,addr,n,a,b in events:
        take=min(n,free)
        if wr:
            dirty+=take
        free-=take
        if free==0:
            break
    return dirty


def sector_suffix(desc,regions,line,ways,capacity):
    """Reverse traversal: the last W distinct sectors in each set are the ones resident.

    Contiguous descriptors often straddle a sector at core or UB boundaries.
    Mark each physical line once, resolving repeat touches at the sorted access
    times. Use the actual absolute GM addresses for the set index.
    """
    sets=capacity//(line*ways)
    assert sets*line*ways==capacity
    slots=np.full(sets,ways,dtype=np.int16)
    # Two disjoint sector regions, verified before constructing this key.
    seen={wr:np.zeros((reg['gm_end']-1)//line-reg['gm_base']//line+1,dtype=bool)
          for wr,reg in regions.items()}
    dirty=total=0
    for wr,addr,n,a,b in desc:
        start,end=addr//line,(addr+n-1)//line+1
        offset=start-regions[wr]['gm_base']//line
        done=seen[wr][offset:offset+end-start]
        idx=np.arange(start,end,dtype=np.int64)[~done]%sets
        done[:]=True
        if not len(idx):
            continue
        if end-start>sets:
            raise ValueError('descriptor exceeds one set-index rotation')
        target=idx[slots[idx]>0]
        slots[target]-=1
        total+=len(target)
        if wr:dirty+=len(target)
        if total==sets*ways:
            break
    return dirty*line


def run(source):
    d=read(source)
    events=payload(d)
    chip=load_chip(d['chip'])
    cap=chip.details['l2_bytes']
    regions={False:d['read'],True:d['write']}
    rows={}
    for line in (64,128):
        r,w=regions.values()
        if not ((r['gm_end']-1)//line<w['gm_base']//line or (w['gm_end']-1)//line<r['gm_base']//line):
            raise ValueError('read/write areas share a line')
        for mode in ('minimum_dirty','issue','retire','maximum_dirty'):
            order=ordered(events,mode)
            rows.setdefault(mode,{})['fully_associative_bytes']=byte_suffix(order,cap)
            rows[mode].update({f'line{line}_ways{ways}':sector_suffix(order,regions,line,ways,cap)
                               for ways in (1,4,8,16)})
    dt_data=read(ROOT/'results/cast-capacity/controller-periods.json')
    phys=next((s for s in dt_data if s['shape']==d['shape']),None) if 'DT' in chip.name and d['op'].startswith('cast') else None
    r_bytes,w_bytes=d['read']['bytes'],d['write']['bytes']
    allowance=max(d['read']['last_retire'],d['write']['last_retire'])-d['read']['first_issue']
    cap_bytes=allowance*1000*chip.hbm_bandwidth_gbs
    observed=None if phys is None else dict(physical_reads=phys['read_bytes'],physical_writes=phys['write_bytes'],
               retained_dirty_equivalent=w_bytes-phys['write_bytes'],scope='128 controller sums, complete CA process including metadata')
    result=dict(op=d['op'],chip=chip.name,shape=d['shape'],ca_us=d['ca_us'],source=str(source.relative_to(ROOT)),
                dma_records_sha256=d['dma_records_sha256'],trace_sha256=d['aggregate_trace_sha256'],elf=d['binary_sha256'],
                read_bytes=r_bytes,write_bytes=w_bytes,L2_bytes=cap,descriptors=len(events),dirty_bytes=rows,
                physical_DT=observed,maximum_external_bytes_if_all_service_within_DMA=cap_bytes,
                min_required_dirty_if_cold_at_nominal_B=w_bytes-(cap_bytes-r_bytes),nominal_GBps=chip.hbm_bandwidth_gbs,
                controllers='DT measurements' if observed else 'not observed')
    for key in rows['minimum_dirty']:
        assert all(rows['minimum_dirty'][key] <= rows[stamp][key] <= rows['maximum_dirty'][key]
                   for stamp in ('issue', 'retire'))
    name=source.name.replace('-traffic','-retention')
    (OUT/name).write_text(json.dumps(result,indent=2)+'\n')
    print(name,flush=True)


if __name__=='__main__':
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument('paths',type=Path,nargs='+')
    for p in args.parse_args().paths:run(p.resolve())
