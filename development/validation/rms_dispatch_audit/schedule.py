"""Experimental per-AIV issue and sector retirement schedule for one full-row UB tile."""
from collections import defaultdict
from math import ceil
from statistics import median

from operator_model.rms_v12 import parameters, tile_capacity
from operator_model.rms_resource import READ_PER_KIB, WRITE_PER_KIB, ARRIVAL_PER_AIV


def launch_class(block, active, high, islands=2, lanes=2):
    """The observed scheduler assigns equal prefixes of cores to its two partitions."""
    start=0
    for i in range(islands):
        n=(active+islands-i-1)//islands
        if start <= block < start+n:
            local=block-start
            hot=(high+islands-i-1)//islands
            if high==0:
                return 'first',i,local
            if local<hot:
                return 'resident',i,local
            if hot==0:
                return 'partition',i,local
            if local//lanes >= ceil(hot/lanes):
                return 'core',i,local
            return 'sibling',i,local
        start+=n
    raise ValueError('Block outside launch')


def useful(width,rows,read_price=READ_PER_KIB,vec_ratio=1.0):
    groups=rows*ceil(width/128)
    # Reduction precedes the statistic write. Gamma and x traverse MTE2.
    return max(2*width*(1+rows)*read_price/1024, parameters()['reduce'][1]*groups*vec_ratio)


def fit(examples):
    """Fit instruction landmarks and congested sector service, never kernel duration."""
    by_case=defaultdict(list)
    solo=[]; gaps=[];drain=[]
    y_responses=defaultdict(list)
    spreads=defaultdict(list)
    for r in examples:
        tokens,width=r['shape']
        blocks=r['blocks']
        stride=ceil(tokens/64)
        if any(len(b['stats'])!=1 for b in blocks):
            continue
        phases=defaultdict(list);sectors=defaultdict(list)
        for b in blocks:
            k=b['block']; nr=min(stride,tokens-k*stride)
            state,island,local=launch_class(k,len(blocks),r['prior'])
            arrival=b['begin']
            phases[('arrival',state)].append(arrival)
            e=b['stats'][0]
            phases[('lead',state)].append(e['issue']-arrival-useful(width,nr))
            if e['y_done'] > e['done'] + 2:
                y_responses[e['y_bytes']/1024].append(e['y_done']-e['y_issue'])
            sectors[(e['offset']+r['base_mod128'])//128].append(e)
            if b['end']==r['window'] and e['done']==max(x['done'] for x in b['stats']):
                drain.append(b['end']-e['done'])
        for key,values in phases.items():
            by_case[key].append((len(blocks),median(values)))
            if key[0]=='lead' and len(values)>=2:
                # Expected range in n uniform observations is (n-1)/(n+1)
                # of population support. Only measure within one launch/state.
                n=len(values)
                spreads[key[1]].append((max(values)-min(values))*(n+1)/(n-1))
        for entries in sectors.values():
            if len(entries)==1:
                e=entries[0];solo.append(e['done']-e['issue'])
            if len(entries)<4:
                continue
            issue_end=max(e['issue'] for e in entries)
            done=sorted(e['done'] for e in entries)
            gaps.extend(v-u for u,v in zip(done,done[1:]) if u>=issue_end)
    # The first input and reduction share chip resources. Estimate additional
    # admission cycles per participating AIV on the reusable path from phase
    # timestamps, giving each launch one median observation.
    samples=by_case['lead','resident']
    pairs=[(ty-tx)/(ny-nx) for i,(nx,tx) in enumerate(samples) for ny,ty in samples[:i] if ny!=nx]
    per_core=max(0.,median(pairs))
    leads={s:median(y for _,y in v) for (k,s),v in by_case.items() if k=='lead'}
    leads['resident']=median(t-per_core*(n-1) for n,t in samples)
    sizes=sorted((kb,median(vals)) for kb,vals in y_responses.items())
    slope=max(0.,median((ty-tx)/(y-x) for i,(x,tx) in enumerate(sizes) for y,ty in sizes[:i]))
    offset=median(t-kb*slope for kb,t in sizes)
    return dict(y_cycles=[offset,slope],arrival={s:median(y for _,y in v) for (k,s),v in by_case.items() if k=='arrival'},
                lead=leads,spread={k:median(v) for k,v in spreads.items()},resident_issue_cycles_per_active=per_core,
                solo=median(solo),sector_interval=median(gaps),drain=median(drain))


def price(shape, chip, previous, model, original):
    """Return a conditional estimate. No CA input at query time."""
    tokens,width=shape
    c=chip.vector_cores;stride=ceil(tokens/c);active=ceil(tokens/stride)
    d=original.projection['timing_diagnostics']
    cycles=1000*chip.frequency_ghz
    stage='first' if previous==0 else 'expand' if previous<active else 'reuse'
    if previous==0 or stride>tile_capacity(width, chip.details.get('ub_bytes') or 253952):
        return d['conditional_us'][stage]
    lanes=max(1,ceil(c/chip.cube_cores))
    sectors=defaultdict(list);end_y=0
    for k in range(active):
        state,island,local=launch_class(k,active,previous,lanes=lanes)
        rows=min(stride,tokens-k*stride)
        part=d['vector_price_cycles_per_reference_cycle']
        # Source coverage requires at least one example per physical frontend state.
        start=model['arrival'][state]
        contention=(active-1)*model['resident_issue_cycles_per_active'] if state=='resident' else 0.
        issue=start+model['lead'][state]+contention+useful(width,rows,d['read_cycles_per_kib'],part)
        vec=parameters()['normalize']
        fixed, per_kib = model['y_cycles']
        if chip.details.get('mte3_bandwidth_gbs'):
            per_kib=d['write_cycles_per_kib']
        y_latency=vec[0]+vec[1]*part*rows*ceil(width/128)+fixed+per_kib*2*width*rows/1024
        lo=k*4*stride//128;hi=(k*4*stride+4*rows-1)//128
        for sector in range(lo,hi+1):
            sectors[sector].append((state,issue,y_latency))
    end_stat=0
    for group in sectors.values():
        per_state=defaultdict(list)
        for state,issue,latency in group:
            per_state[state].append((issue,latency))
        issues=[]
        for state,requests in per_state.items():
            n=len(requests)
            spread=model['spread'].get(state,0)
            for rank,(issue,latency) in enumerate(sorted(requests)):
                # Expected order statistics of arrival within the sector.
                ready=issue+spread*((rank+1)/(n+1)-.5)
                end_y=max(end_y,ready+latency)
                issues.append(ready)
        done=0
        for arrival in sorted(issues):
            done=max(arrival+model['solo'],done+model['sector_interval'])
        end_stat=max(end_stat,done)
    total=(max(end_stat,end_y)+model['drain'])/cycles
    common=d['conditional_stages_us'][stage]
    shared_bound=common['SCALAR']*2970/2933.5+d['shared_io_floor_cycles']/cycles
    return max(total,shared_bound)


def both(shape,chip,high,model,r):
    active=r.active_aiv
    hot=price(shape,chip,active,model,r)
    first=price(shape,chip,0,model,r)
    expand=price(shape,chip,max(1,active//2),model,r)
    rare=price(shape,chip,1,model,r)
    upper=max(first,rare,expand)
    unknown=2*upper*hot/(upper+hot)
    return price(shape,chip,high,model,r),unknown
