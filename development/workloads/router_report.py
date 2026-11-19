"""Score paired frozen Kimi router holdouts and retain stages, origins and failed costs."""
from collections import defaultdict
from math import ceil
from statistics import median
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'development/validation/sku18'))
from rms_evidence import spans
from router_validate import DT,PR,FUT,OUT,RAW,digest,write


def boundaries(trace,experts):
    """The two calibration traces have one row, one logits DMA and one bias DMA."""
    origin,start,end,events=spans(trace)
    table=json.loads(trace.read_text())['traceEvents']
    assert len({x['pid'] for x in table if x['ph']=='X'})==1
    payload=[]
    for r in table:
        if r['ph']!='X': continue
        attrs=json.loads(r['args']['detail'])
        if attrs.get('burst_len')==4*experts and attrs['queue_type']=='MTE2':
            payload.append(r)
    assert len(payload)==2
    issue=min(r['ts'] for r in payload)
    ready=max(r['ts']+r['dur'] for r in payload)
    store=min(a for q,a,b,op in events if q=='MTE3' and op.startswith('MOV_SRC'))
    return [round((b-a)*1650,2) for a,b in zip((origin,issue,ready,store),(issue,ready,store,end))]


def stats(predicted,observed):
    return {'n':len(observed),'wape':sum(abs(x-y) for x,y in zip(predicted,observed))/sum(observed),
            'max':max(abs(x/y-1) for x,y in zip(predicted,observed))} if observed else {'n':0,'wape':None,'max':None}


def main():
    hold=json.loads((OUT/'router-holdout.json').read_text())
    recovery=OUT/'router-recovery.json'
    if recovery.exists():
        hold.extend(json.loads(recovery.read_text()))
    reg=json.loads((OUT/'router-holdout-freeze.json').read_text())
    allrows=hold.copy()
    calibration=[]
    for file in ['router-calibration.json','router-calibration-ipc.json']:
        for record in json.loads((OUT/file).read_text()):
            allrows.append(record)
            if record['qualified']:
                k=record['kernel'];tr=ROOT/k['trace'];assert digest(tr)==k['trace_sha256']
                calibration.append(dict(shape=record['shape'],ca_us=k['ca_us'],trace_sha256=k['trace_sha256'],
                                        control_read_rank_drain_cycles=boundaries(tr,record['shape'][1])))
    exported=json.loads((OUT/'router-window-recovery.json').read_text())['recovered']
    by={}
    qualified=[]
    for r in [*hold,*exported]:
        if r['qualified']:
            key=tuple(r['shape']),r['sku']
            assert key not in by
            by[key]=r['kernel']
            if 'cost' not in r:
                original=next(x for x in hold if (tuple(x['shape']),x['sku'])==key)
                r=dict(r,cost=original['cost'])
            qualified.append(dict(r,role=r.get('role','fresh frozen holdout, complete official export')))
    write(OUT/'router-qualified.json',qualified)
    pairs=[]
    metric=defaultdict(lambda:([],[]))
    for j in reg['jobs']:
        shape=tuple(j['shape'])
        p=j['predictions'];r=dict(shape=shape,dt_us=p[DT],pr_us=p[PR],future_us=p[FUT])
        for sku,side in [(DT,'dt'),(PR,'pr')]:
            observed=by.get((shape,sku))
            r[side+'_ca_us']=observed['ca_us'] if observed else None
            if observed:
                r[side+'_active_aiv']=observed['active_aiv']
                metric[side][0].append(p[sku]);metric[side][1].append(observed['ca_us'])
        if r['dt_ca_us'] and r['pr_ca_us']:
            r['speed'],r['ca_speed']=p[DT]/p[PR],r['dt_ca_us']/r['pr_ca_us']
            metric['speed'][0].append(r['speed']);metric['speed'][1].append(r['ca_speed'])
        pairs.append(r)
    from operator_model import estimate
    from time import perf_counter_ns
    samples=[]
    for j in reg['jobs']:
        for chip,expected in j['predictions'].items():
            t0=perf_counter_ns()
            result=estimate('moe_gating_top_k',tuple(j['shape']),dtype='fp32',chip=chip)
            samples.append((perf_counter_ns()-t0)/1e6)
            assert abs(result.latency_us-expected)<1e-10
    costs=[r['cost'] for r in allrows]
    out=dict(summary={k:stats(*v) for k,v in metric.items()},pairs=pairs,calibration=calibration,
             query_ms=dict(measurement='source API after import, unique shape/chip requests in one process',
                           n=len(samples),median=median(samples),p95=sorted(samples)[ceil(.95*len(samples))-1],
                           maximum=max(samples)),
             costs=dict(physical_attempts=len(costs),qualified=sum(r['qualified'] for r in allrows),
                        recovered_complete_windows=len(exported),
                        simulator_wall_sum_s=sum(c['seconds'] for c in costs),
                        cpu_s_lower_bound=sum((c['cpu_seconds'] or 0) for c in costs),
                        retained_bytes=sum(c['raw_bytes'] for c in costs),
                        first_started=min(c['started_utc'] for c in costs),
                        last_finished=max(c['finished_utc'] for c in costs)))
    write(OUT/'router-evaluation.json',out)
    plot(pairs)


def plot(pairs):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.6),sharey=True,layout='constrained')
    for ax,side in zip(axes,['dt','pr']):
        xs=list(range(len(pairs)))
        ax.plot(xs,[r[side+'_us'] for r in pairs],'D--',ms=4,color='#734a96',label='Frozen prediction')
        labels=[(i,r[side+'_ca_us']) for i,r in enumerate(pairs) if r[side+'_ca_us'] is not None]
        if labels:
            ids,ys=zip(*labels)
            ax.scatter(ids,ys,s=30,c='#1b6956',label='CA measurement')
        ax.set_xticks(xs,['×'.join(map(str,r['shape'])) for r in pairs],rotation=54,ha='right',fontsize=8)
        ax.set_title('950'+side.upper());ax.grid(axis='y',alpha=.22)
        ax.set_xlabel('tokens × experts × selected')
    axes[0].set_ylabel('Kernel time (µs)');axes[1].legend(frameon=False)
    fig.savefig(ROOT/'docs/examples/figures/moe_gating_top_k.png',dpi=135)
    plt.close(fig)

if __name__=='__main__':
    main()
