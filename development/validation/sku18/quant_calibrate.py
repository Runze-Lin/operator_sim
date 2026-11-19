"""Fit service and queue intervals on archived DT small scale requests, then audit DT/PR."""
from collections import defaultdict
from hashlib import sha256
import json
from math import ceil,log2
from pathlib import Path
from statistics import median
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'development/validation/sku17'))
from campaign_collect import events
from operator_model.target_tiling import quant_plan

OUT=ROOT/'results/v0.18'
FREQ=1650.


def read(file):
    return json.loads(file.read_text())


def inspect(path):
    starts={}; ends={}; first_simd={}; dma=defaultdict(list)
    for e in events(path):
        if e['ph']!='X':continue
        c=e['pid'];start=e['ts'];end=start+e['dur']
        starts[c]=min(starts.get(c,start),start)
        ends[c]=max(ends.get(c,end),end)
        if e['tid'].startswith('RVEC'):
            first_simd[c]=min(first_simd.get(c,start),start)
        if e['name']=='MOV_SRC_TO_DST_ALIGNv2':
            d=json.loads(e['args']['detail'])
            dma[c].append((start,end,d))
    origin=min(starts.values());rows=[]
    for c,ev in dma.items():
        rd=sorted((x for x in ev if x[2]['instr_type']=='MTE2'),key=lambda e:e[0])
        wr=sorted((x for x in ev if x[2]['instr_type']=='MTE3'),key=lambda e:e[0])
        assert len(rd)==len(wr)==2 and rd[0][2]['processed_bytes']==160
        r=rd[1];y,s=wr
        # Same-cycle ordering can differ across exports. Identify the FP8 vs
        # scale descriptor by payload size, not enumeration within the cycle.
        if y[2]['processed_bytes']<s[2]['processed_bytes']:y,s=s,y
        d=s[2];grains=set()
        for b in range(d['burst_number']):
            addr=d['dst_addr']+b*d['dst_stride']
            grains.update(range(addr//32,(addr+d['burst_len']-1)//32+1))
        row=dict(core=c,read_bytes=r[2]['processed_bytes'],scale_bytes=d['processed_bytes'],addr=d['dst_addr'],
            issue=(r[0]-origin)*FREQ,read=(r[1]-r[0])*FREQ,simd=(first_simd[c]-origin)*FREQ,
            data_issue=(y[0]-origin)*FREQ,scale_issue=(s[0]-origin)*FREQ,scale_end=(s[1]-origin)*FREQ,
            global_end=(ends[c]-origin)*FREQ,lines=sorted(grains))
        rows.append(row)
    base=min(x['addr'] for x in rows)
    for x in rows:
        x['offset']=x.pop('addr')-base;x['lines']=[g-base//32 for g in x['lines']]
    return sorted(rows,key=lambda x:x['offset'])


def observe(rs,loop_price):
    cohorts=defaultdict(list)
    for r in rs:
        for line in r['lines']:cohorts[line].append(r)
    shared=[g for g in cohorts.values() if len(g)>1]
    if not shared:return
    # An interval between completions can only proxy an issue interval if
    # the following request was already outstanding at the earlier completion.
    def spacings(g):
        g=sorted(g,key=lambda x:x['scale_end'])
        return [b['scale_end']-a['scale_end'] for a,b in zip(g,g[1:]) if b['scale_issue']<=a['scale_end']]
    def span(g):
        return (max(x['scale_end'] for x in g)-min(x['scale_end'] for x in g))/(len(g)-1)
    long_lines=[g for g in shared if len(g)==max(map(len,shared))]
    loops=lambda r:ceil(r['scale_bytes']/8)+ceil(r['scale_bytes']/128)
    return dict(active=len(rs),peak=max(map(len,shared)),launch=min(r['issue'] for r in rs),
                spread=(max(r['issue'] for r in rs)-min(r['issue'] for r in rs))/log2(len(rs)),
                frontend=median(r['simd']-r['issue'] for r in rs),
                vector=median(r['data_issue']-r['simd']-loops(r)*loop_price for r in rs),
                grain_ii=median(d for g in long_lines for d in spacings(g)) if any(spacings(g) for g in long_lines) else median(span(g) for g in long_lines),
                fabric_ii=median(spacings(rs)) if spacings(rs) else None)


def main():
    old=read(ROOT/'results/v0.4/dataset.json')['records'];paths={}
    for r in old:
        if r['case']['op']=='quant':
            m=r['measurement'];paths[tuple(r['case']['shape'])]=(Path(m['report_path'])/'simulator/trace.json',m['trace_sha256'])
    pairs=[r for r in read(ROOT/'results/v0.17/sku/pairs.json') if r['op']=='quant']
    for r in pairs:
        path=ROOT/r['dt']['trace'][0]
        if path.suffix=='.json':paths[tuple(r['shape'])]=(path,r['dt']['trace_sha256'][0])
    p=read(ROOT/'results/v0.17/quant.json')['parameters']
    observations=[]
    for shape,(path,digest) in paths.items():
        plan=quant_plan(shape,64,253952)
        if plan['iterations']!=1 or plan['rows_per_core']*ceil(plan['cols_per_core']/32)>32:continue
        rows=inspect(path);o=observe(rows,p['vec_loop_cycles'])
        if not o:continue
        assert sha256(path.read_bytes()).hexdigest()==digest
        observations.append(dict(shape=shape,sha256=digest,trace=str(path.relative_to(ROOT)),**o))
    # Preserve [5,320] as diagnostic. Both SKU timelines are replayed later.
    train=[o for o in observations if o['shape']!=(5,320)]
    defaults={k:median(o[k] for o in train if o[k] is not None)
              for k in ('launch','spread','frontend','vector','grain_ii','fabric_ii')}
    params={k+'_cycles':v for k,v in defaults.items()}
    (OUT/'quant-fragment.json').write_text(json.dumps(dict(training_soc='Ascend950DT_9582',parameters=params),indent=2)+'\n')
    diagnostic=next(r for r in pairs if r['shape']==[5,320])
    trace_diag={chip:dict(sha256=diagnostic[chip]['trace_sha256'][0],cores=inspect(ROOT/diagnostic[chip]['trace'][0])) for chip in ('dt','pr')}
    (OUT/'quant-calibration.json').write_text(json.dumps(dict(scope='DT stages only; [5,320] excluded, PR diagnostic only',training_shapes=len(train),samples=observations,diagnosis=trace_diag),indent=2)+'\n')
    print(defaults,[(r['shape'],round(r['grain_ii']),round(r['fabric_ii'] or 0)) for r in train])


if __name__=='__main__':main()
