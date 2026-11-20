"""Exact simultaneous, protocol-wise error feasibility for identical visible inputs."""
from collections import defaultdict
from math import ceil
from operator_model import load_chip


def limits(rows, cap=.50, wape=.25):
    by_shape=defaultdict(dict)
    for r in rows:
        by_shape[tuple(r['shape'])][r['role']]=r['ca_us']
    pairs=[(q['entry'],q['repeat']) for q in by_shape.values() if 'entry'in q and 'repeat'in q]
    if not pairs:return dict(n=0)
    e_total=sum(x for x,y in pairs);r_total=sum(y for x,y in pairs)
    delta=sum(abs(x-y) for x,y in pairs)
    min_entry=max_entry=0.
    violation=0.
    for x,y in pairs:
        low=min(x,y);high=max(x,y)
        left=max(low,(1-cap)*high);right=min(high,(1+cap)*low)
        if left>right:violation=max(violation,left-right)
        d1,d2=sorted([abs(left-x),abs(right-x)])
        min_entry+=d1;max_entry+=d2
    best_entry=min(max_entry,max(min_entry,delta*e_total/(e_total+r_total)))
    constrained=max(best_entry/e_total,(delta-best_entry)/r_total)
    return dict(n=len(pairs),max_error_lower_pct=100*max(abs(x-y)/(x+y) for x,y in pairs),
                wape_lower_pct=100*delta/(e_total+r_total),
                simultaneous_feasible=violation==0 and constrained<=wape,
                constrained_wape_lower_pct=100*constrained if not violation else None,
                per_shape_gap_us=violation,entry_absolute_error_sum_us=[min_entry,max_entry],
                goal_entry_absolute_error_sum_us=[delta-wape*r_total,wape*e_total])


def group(rows,skus):
    result={}
    for sku in skus:
        chip=load_chip(sku);limit=2*ceil(chip.vector_cores/chip.cube_cores)
        same=[v for v in rows if v['chip']==sku];low=[];high=[]
        for v in same:
            a=v.get('active_aiv')
            if a is None:
                t=v['shape'][0];a=ceil(t/ceil(t/chip.vector_cores))
            (low if a<=limit else high).append(v)
        result[sku]={'all':limits(same),'first physical core per partition':limits(low),'larger footprint':limits(high)}
    return result
