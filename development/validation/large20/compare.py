"""Aggregate held out, concurrent and historical Cast experiments separately."""
from statistics import median
from large_cast import DT, PR, ROOT, OUT, estimate, read, write


def stats(xs):
    return None if not xs else dict(n=len(xs),wape_pct=100*sum(abs(p-y) for p,y in xs)/sum(y for p,y in xs),
                                    max_abs_pct=max(abs(100*(p/y-1)) for p,y in xs))


def summary(rows):
    answer={}
    for key in ('dt','pr','speedup'):
        for version in ('old','current'):
            pairs=[(row[key][version],row[key]['ca']) for row in rows if row.get(key)]
            answer[key+'_'+version]=stats(pairs)
    return answer


def compare(data):
    history=[]
    historical=read(ROOT/'results/v0.19/workloads/cast-evaluation.json')
    for p in historical['pairs']:
        d=dict(shape=p['shape'])
        for side,chip in [('dt',DT),('pr',PR)]:
            new=estimate('cast_bf16_fp32',p['shape'],chip=chip,timing_mode='resource').latency_us
            d[side]=dict(old=p[side+'_us'],ca=p[side+'_ca_us'],current=new)
        d['speedup']={key:d['dt'][key]/d['pr'][key] for key in d['dt']}
        history.append(d)
    groups={'prior_holdout_replay':history,'development':[],'parallel_replay':[],'independent':[]}
    for r in data['cases']:
        row={'shape':r['shape']};roles=set()
        for side,chip in [('dt',DT),('pr',PR)]:
            actual=r['devices'][chip].get('ca_us')
            if not actual:continue
            row[side]=dict(old=r['targets'][chip]['us'],current=r['current'][chip]['us'],ca=actual)
            roles.add(r['roles'][chip].split(':')[0].replace(' ','_'))
        if len(row)==1:continue
        if 'dt' in row and 'pr' in row:
            row['speedup']={key:row['dt'][key]/row['pr'][key] for key in row['dt']}
        assert len(roles)==1
        groups[roles.pop()].append(row)
    result={k:{'rows':v,'summary':summary(v)} for k,v in groups.items()}
    write(OUT/'paired-comparison.json',result)
    return result
