"""Report frozen standalone Cast holdouts and cost of all physical attempts."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/v0.19/workloads'
DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'
def load(name):
    return json.loads((OUT/name).read_text())


def chart(pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=sorted(pairs,key=lambda r:r['shape'][0]*r['shape'][1])
    fig,axes=plt.subplots(1,2,figsize=(9.1,3.6),layout='constrained',sharey=True)
    for ax,side in zip(axes,('dt','pr')):
        xs=[r['shape'][0]*r['shape'][1] for r in rows]
        yes=[r for r in rows if r[side+'_ca_us']]
        ax.plot(xs,[r[side+'_us'] for r in rows],color='#1e6d8c',marker='o',markersize=4,label='Frozen model')
        ax.scatter([r['shape'][0]*r['shape'][1] for r in yes],[r[side+'_ca_us'] for r in yes],color='#a65b24',marker='x',label='CA measurement',zorder=4)
        ax.set_xscale('log')
        ax.grid(alpha=.22)
        ax.set_xlabel('Contiguous BF16 elements')
        ax.set_title('950'+side.upper())
    axes[0].set_ylabel('Single-kernel time (µs)')
    axes[0].legend(fontsize=8)
    fig.suptitle('BF16 to FP32 Cast   |   Independent holdouts')
    fig.savefig(ROOT/'docs/examples/figures/cast_bf16_fp32.png',dpi=150)
    plt.close(fig)


def report():
    freeze=load('cast-holdout-freeze.json')
    attempts=[]
    observed={}
    for stage in ('calibration','holdout'):
        for suffix in ('','-retry'):
            path=f'cast-{stage}{suffix}.json'
            if not (OUT/path).exists():continue
            for record in load(path):
                attempts.append(record)
                if record['qualified'] and stage=='holdout':
                    observed[tuple(record['shape']),record['sku']]=record
    qualified=[]
    for job in freeze['jobs']:
        for sku in (DT,PR):
            rec=observed.get((tuple(job['shape']),sku))
            qualified.append(rec or dict(shape=job['shape'],sku=sku,qualified=False,kernel=None))
    (OUT/'cast-qualified.json').write_text(json.dumps(qualified,indent=2)+'\n')
    pairs=[]
    for job in freeze['jobs']:
        p=job['predictions'];shape=tuple(job['shape'])
        result=dict(shape=list(shape),dt_us=p[DT],pr_us=p[PR],future_us=p[FUT],
                    dt_ca_us=None,pr_ca_us=None,dt_aiv_ca=None,pr_aiv_ca=None,
                    speedup_pred=p[DT]/p[PR],speedup_ca=None,dt_kernel=None,pr_kernel=None)
        for name,sku in [('dt',DT),('pr',PR)]:
            if (shape,sku) in observed:
                stage=observed[shape,sku]['kernel']
                result[f'{name}_ca_us']=stage['ca_us']
                result[f'{name}_aiv_ca']=stage['active_aiv']
                result[f'{name}_kernel']=stage['kernel']
        if result['dt_ca_us'] and result['pr_ca_us']:
            result['speedup_ca']=result['dt_ca_us']/result['pr_ca_us']
        pairs.append(result)
    # For diagnosis only, re-run the frozen code to get the planned grid.
    # The public forecast and all error scores above use the committed freeze.
    import sys
    sys.path.insert(0,str(ROOT))
    from operator_model.cast_resource import resource_cast
    from operator_model import load_chip
    for p in pairs:
        for k,sku in [('dt',DT),('pr',PR)]:
            t,_,a,diag=resource_cast(p['shape'],load_chip(sku))
            assert abs(t-p[k+'_us'])<1.e-12
            p[k+'_aiv_model']=a
    summary={}
    for name,truth,predict in [('dt','dt_ca_us','dt_us'),('pr','pr_ca_us','pr_us'),('speed','speedup_ca','speedup_pred')]:
        selected=[p for p in pairs if p[truth] is not None]
        summary[name]=dict(n=len(selected),wape=sum(abs(p[predict]-p[truth]) for p in selected)/sum(p[truth] for p in selected) if selected else None,
                           max=max((abs(p[predict]/p[truth]-1) for p in selected),default=None))
    coverage = load('cast-dma-coverage.json') if (OUT/'cast-dma-coverage.json').exists() else []
    for row in pairs:
        for side,sku in [('dt',DT),('pr',PR)]:
            match = next((r for r in coverage if r['shape']==row['shape'] and r['sku']==sku),None)
            row[side+'_complete_dma'] = None if not match else all(
                match[k+'_bytes_observed']==match['logical_'+k+'_bytes'] for k in ('read','write'))
    repeat = load('cast-dma-recheck.json') if (OUT/'cast-dma-recheck.json').exists() else None
    if repeat is not None:
        attempts.append(repeat)
    from datetime import datetime
    costs=[p['cost'] for p in attempts]
    from cast_phases import profile
    qualified_pairs=[p for p in pairs if p['speedup_ca']]
    extremes=[]
    if qualified_pairs:
        for side,sku in [('dt',DT),('pr',PR)]:
            worst=max(qualified_pairs,key=lambda p:abs(p[side+'_us']/p[side+'_ca_us']-1))
            diag=profile(observed[tuple(worst['shape']),sku])
            extremes.append(dict(sku=sku,**diag))
    out=dict(worst_case_payload_windows=extremes,pairs=pairs,summary=summary,role='fresh frozen isolated Cast component, independent of RMSNorm forecast set',
             cost=dict(attempts=len(attempts),qualified=sum(p['qualified'] for p in attempts),
                       raw_bytes=sum(p['raw_bytes'] for p in costs),simulator_wall_s_sum=sum(p['seconds'] for p in costs),
                       cpu_s_lower_bound=sum((p['cpu_seconds'] or 0) for p in costs),
                       elapsed_s=(max(datetime.fromisoformat(p['finished_utc']) for p in costs)-min(datetime.fromisoformat(p['started_utc']) for p in costs)).total_seconds()))
    if repeat is not None:
        out['dma_recheck']=dict(shape=repeat['shape'],sku=repeat['sku'],qualified=repeat['qualified'],
                                complete_dma=repeat['complete_dma'],new_ca_us=repeat.get('kernel',{}).get('ca_us'),
                                primary_statistics_includes_repeat=False)
        # Provide a second view only if all DMA bytes are observed in the repeat.
        if repeat['complete_dma']:
            strict = [(r['pr_us'],repeat['kernel']['ca_us'] if r['shape']==repeat['shape'] else r['pr_ca_us']) for r in pairs]
            out['dma_recheck']['strict_pr_wape']=sum(abs(x-y) for x,y in strict)/sum(y for x,y in strict)
            out['dma_recheck']['strict_pr_max']=max(abs(x/y-1) for x,y in strict)
            speed=[(r['speedup_pred'],r['dt_ca_us']/actual) for r,(_,actual) in zip(pairs,strict)]
            out['dma_recheck']['strict_speed_wape']=sum(abs(x-y) for x,y in speed)/sum(y for x,y in speed)
    (OUT/'cast-evaluation.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    chart(pairs)
    print(json.dumps(summary))
if __name__=='__main__':
    report()
