"""Compare identical byte streams with different model shapes; make forward cost forecasts."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.26/memory'
PRO=ROOT/'results/v0.23/pro-prefill'
sys.path.insert(0,str(ROOT/'development/validation/cache25'))
from prefix import prefix
from shadow import predict
SKUS=('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT')
POLICIES=('uniform','pressure')
COLORS={'Read':'#167886','Vector':'#7655a6','Write':'#b84c2b'}


def read(p):
    return json.loads(p.read_text())


def write(p,d):
    p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def freeze():
    target=OUT/'pro-256ki-prefix-forecast.json'
    if target.exists():
        raise FileExistsError(target)
    if any((PRO/f'pro-b8-256ki-{s}.json').exists() for s in ('dt','pr')):
        raise ValueError('terminal outcome already visible')
    j=next(j for j in read(PRO/'freeze.json')['jobs'] if j['id']=='pro-b8-256ki-dt')
    cases={}
    for sku in SKUS:
        totals=predict(j['op'],j['shape'],sku)
        curves={p:prefix(j['shape'],sku,p) for p in POLICIES}
        for p,series in curves.items():
            assert abs(series[-1]['completion_us']-totals[p]['us'])<1e-8
        cases[sku]=dict(totals=totals,prefixes=curves)
    sources=[PRO/'freeze.json',ROOT/'development/validation/cache25/prefix.py',ROOT/'development/validation/cache25/shadow.py',
             Path(__file__),*[ROOT/f'configs/chips/{s}.json' for s in SKUS]]
    write(target,dict(utc=datetime.now(timezone.utc).isoformat(),shape=j['shape'],op=j['op'],batch=j['batch'],sequence=j['sequence'],
                      qualified_scope='Supplemental prefix/retention forecasts before any full 256ki outcome. DT collection was already in flight; PR had not started. Existing v0.23 forecast and run budget take precedence.',
                      sources={str(p.relative_to(ROOT)):digest(p) for p in sources},cases=cases))


def summarize(d,src,stage):
    rows=d['ca_tiles']
    m=d['model_tiles']
    assert len(m)==len(rows)==57 and d['ca_tile_elements']==d['model_tile_elements']
    delta=lambda i,k,n:m[i][k][n]-rows[i][k][n]
    return dict(shape=d['shape'],origin=d['origin'],source=str(src.relative_to(ROOT)),sha256=digest(src),trace_sha256=d['trace_sha256'],
                tiles=len(rows),tile_elements=d['ca_tile_elements'],core=d['ca_core'],active=d['active_aiv'],stage=stage,
                full_model_us=d['model_us'],full_ca_us=d['ca_us'],future_us=d['projection_970dt_us'],
                first_two_model_read_gap=m[1]['Read'][0]-m[0]['Read'][0],first_two_ca_read_gap=rows[1]['Read'][0]-rows[0]['Read'][0],
                first_ca_read_latency=rows[0]['Read'][1]-rows[0]['Read'][0],
                middle_model_minus_ca_read_start_us=delta(28,'Read',0),middle_model_minus_ca_write_end_us=delta(28,'Write',1),
                model=m[:4],ca=rows[:4],last_block_write_model_minus_ca_us=delta(-1,'Write',1))


def draw(name,d,xmax):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/operator-sim-fig26')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    fig,axs=plt.subplots(2,1,figsize=(11,7.6),sharex=True)
    left=.168
    fig.subplots_adjust(left=left,right=.975,top=.765,bottom=.14,hspace=.55)
    ink='#182e3b'
    fig.text(left,.945,f"950PR Cast  |  {d['stage']} prefill",fontsize=23,weight='bold',color=ink)
    fig.text(left,.895,f"BF16 → FP32   ·   [{d['shape'][0]:,}, {d['shape'][1]}]   ·   56 AIVs   ·   first four of 57 UB blocks",fontsize=12,color='#405e6a')
    fig.legend([Patch(color=c) for c in COLORS.values()],['MTE2 / GM → UB','Vector / BF16 → FP32','MTE3 / UB → GM'],
               frameon=False,loc='upper left',bbox_to_anchor=(left-.008,.88),ncol=3,fontsize=11)
    for ax,key,title,stop in zip(axs,('model','ca'),('Resource model · shared-link service',f"CA trace · {d['core']} (last to finish)"),(d['full_model_us'],d['full_ca_us'])):
        for i,row in enumerate(d[key]):
            for off,(stage,col) in zip((-.21,0,.21),COLORS.items()):
                a,b=row[stage]
                ax.barh(i+off,b-a,left=a,height=.15,color=col,zorder=3)
                if stage=='Read':
                    ax.vlines(a,i+off-.12,i+off+.12,color=col,lw=.8)
        ax.set_title(title,loc='left',fontsize=14,color=ink,weight='bold',pad=7)
        ax.set_title(f'full {stop:.3f} µs',loc='right',fontsize=13,weight='bold',color=ink,pad=7)
        ax.set_yticks(range(4),[f'UB {i+1}  ·  slot {i%2}' for i in range(4)],fontsize=11)
        ax.set_ylim(3.48,-.48);ax.set_xlim(0,xmax)
        ax.grid(axis='x',color='#d3dde3',lw=.8);ax.tick_params(length=0,pad=5);ax.set_axisbelow(True);ax.spines[:].set_visible(False)
        for i in (.5,1.5,2.5):
            ax.axhline(i,color='#e0e7eb',lw=.8,zorder=1)
    axs[-1].set_xlabel('Device time from the first instruction on any AIV (µs)',labelpad=8,fontsize=11)
    fig.text(left,.048,'Top prices each block at its share of sustained service. Bottom follows real GM/UB descriptors and conversion instructions.',fontsize=10,color='#46626f')
    fig.text(left,.019,'Read ticks mark issue. Wide CA bars include waiting until retirement. Both panels use the same time origin and scale.',fontsize=10,color='#46626f')
    fig.savefig(OUT/(name+'.png'),dpi=150,facecolor='white');plt.close(fig)


def replay():
    by={}
    for name,stage,src in [('flash-256ki-pr-head','Flash B8 × 256Ki',ROOT/'results/v0.23/prefill/flash-256ki-pr.json'),
                           ('pro-128ki-pr-head','Pro B8 × 128Ki',PRO/'pro-b8-128ki-pr-blocks.json')]:
        by[name]=summarize(read(src),src,stage)
    a,b=by.values()
    assert a['model']==b['model'] and a['tile_elements']==b['tile_elements'] and a['full_model_us']==b['full_model_us']
    end=max(x[1] for d in by.values() for key in ('model','ca') for r in d[key] for x in r.values())
    xmax=2*int(end//2+1)
    for name,d in by.items():
        draw(name,d,xmax)
    write(OUT/'equal-payload.json',by)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze','replay'))
    {'freeze':freeze,'replay':replay}[parser.parse_args().action]()
