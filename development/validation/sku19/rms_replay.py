"""Reprice v0.18 and historical compact traces and draw the RMSNorm worked example."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from dataclasses import replace
from operator_model import estimate,load_chip
from rms_calibrate import calibrate

def base_chip(name):
    """Reproduce first-freeze forecasts after supplemental prices are enabled."""
    c=load_chip(name)
    return replace(c,details={k:v for k,v in c.details.items() if not k.startswith('rms_first_')})


def load(f):
    return json.loads((ROOT/f).read_text())


def wape(pred, truth):
    return sum(abs(x-y) for x,y in zip(pred,truth)) / sum(truth)


def replay():
    pairs = [p for p in load('results/v0.18/sku/pairs.json') if p['op']=='rmsnorm']
    previous = {tuple(item['shape']):item for item in load('results/v0.18/sku/v018-forecasts.json')['cases'] if item['op']=='rmsnorm'}
    rows = []
    for pair in pairs:
        row = {'shape':pair['shape']}
        for side in ('dt','pr'):
            y=estimate('rmsnorm',pair['shape'],chip=base_chip(pair[side]['chip']),timing_mode='resource')
            row[side]=dict(ca=pair[side]['ca_us'],v018=pair[f'{side}_us'],v019=y.latency_us,
                           conditional=y.projection['timing_diagnostics']['conditional_us'],
                           plan=y.projection['target_plan'],stages_us=y.projection['stages_us'])
            diag = y.projection['timing_diagnostics']
            t = diag['stages_us']
            old_floor = max(0.,diag['shared_io_floor_cycles']/1650-t['MTE2']-t['MTE3']-t['SIMD'])
            old_diag = previous[tuple(row['shape'])]['targets'][side]['projection']['timing_diagnostics']
            row[side].update(v018_first=old_diag['conditional_us']['first'],
                             v019_with_v018_shared_window=y.latency_us-t['shared_io']+old_floor)
        row['speedup']={k:row['dt'][k]/row['pr'][k] for k in ('ca','v018','v019')}
        rows.append(row)
    summary={k:{v:wape([r[k][v] for r in rows],[r[k]['ca'] for r in rows]) for v in ('v018','v019')}
             for k in ('dt','pr','speedup')}
    aggregate={v:sum(r['dt'][v] for r in rows)/sum(r['pr'][v] for r in rows) for v in ('ca','v018','v018_first','v019_with_v018_shared_window','v019')}
    old=(load('results/v0.12/rms/development-evidence.json')+load('results/v0.12/rms/holdout-evidence.json'))
    compact=[]
    for r in old:
        args=(r['shape'],r['prior'])
        without=estimate('rmsnorm',args[0],timing_mode='resource')
        with_h=estimate('rmsnorm',args[0],{'resident_aiv':args[1]},timing_mode='resource')
        compact.append(dict(shape=args[0],resident_aiv=args[1],state=with_h.projection['timing_diagnostics']['residency'],
                            ca=sum(r['phases'].values()),without_h=without.latency_us,with_h=with_h.latency_us))
    state_summary={}
    for state in ('first','expand','reuse'):
        g=[r for r in compact if r['state']==state]
        state_summary[state]=dict(n=len(g),**{k:wape([r[k] for r in g],[r['ca'] for r in g]) for k in ('with_h','without_h')})
    out=dict(role='Development replay only: v0.18 isolated pairs and older v0.12 DT compact sequences.',
             parameters=calibrate(),summary=summary,aggregate_ratio=aggregate,isolated_pairs=rows,
             compact_states=state_summary,compact_rows=compact)
    (ROOT/'results/v0.19/rms-replay.json').write_text(json.dumps(out,indent=2)+'\n')
    return next(p for p in pairs if p['shape']==[57,1536])


def chart(pair):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(9,5.4),sharex=True,layout='constrained')
    colors=['#52789a','#24877e','#b97732','#c24d54','#334e68']
    rows=load('results/v0.18/sku/trace-evidence.json')['rows']
    for ax,side in zip(axes,('dt','pr')):
        r=next(t for t in rows if t['shape']==pair['shape'] and t['chip']==pair[side]['chip'])
        counts=sorted(r['critical_core_dma_streams'],key=lambda q:q['first_us'])
        assert len(counts)==5 and all(q['count']==1 for q in counts)
        for i,s in enumerate(counts):
            a,b=s['first_us'],s['last_end_us']
            ax.barh(i,b-a,.60,left=a,color=colors[i])
        ax.set_yticks(range(5),['Tiling read','Gamma read','Input read','rstd write','y write'])
        ax.invert_yaxis()
        ax.grid(axis='x',alpha=.18)
        ax.axvline(pair[side]['ca_us'],lw=1.2,color='#303030',ls='--')
        y=estimate('rmsnorm',pair['shape'],chip=pair[side]['chip'],timing_mode='resource')
        ax.axvline(y.latency_us,lw=1.2,color='#7b529b')
        ax.set_title(f'950{side.upper()}   |   {y.active_aiv} AIVs   |   CA {r["ca_us"]:.2f} µs (dashed), model {y.latency_us:.2f} µs (purple)',fontsize=11,loc='left')
    fig.suptitle('RMSNorm [57, 1536]   |   DMA intervals on the last-finishing AIV',fontsize=14)
    axes[-1].set_xlabel('Time since earliest instruction across all AIVs (µs)')
    axes[-1].set_xlim(0,6.7)
    fig.savefig(ROOT/'docs/examples/figures/rmsnorm.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    chart(replay())
