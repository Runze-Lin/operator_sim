"""Align the last two full KV pages and the single-token tail to a frozen FA forecast."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from math import ceil, floor, isclose
from pathlib import Path
from statistics import median
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model import estimate, load_chip

SUP = ROOT/'results/v0.25/coverage/supplement'
OUT = ROOT/'results/v0.27'
PALETTE = {'K read':'#147988','V read':'#5078b5','QK':'#7053a8','PV':'#b44a2b'}


def load(path):
    return json.loads(path.read_bytes())


def collect(sku):
    name = 'fa-supp-'+sku+'-9-32769-64-1-1'
    receipt = load(SUP/(name+'.json'))
    proof = load(SUP/'physical'/(name+'.json'))
    entry = next(r for r in load(SUP/'score.json')['rows'] if r['id']==name)
    assert proof['qualified'] and receipt['qualified'] and entry['shape']==receipt['shape']
    stage, = receipt['stages']
    core, = (c for c in proof['cores'] if 'cube' in c)
    shard = (ROOT/stage['trace']).parent / core / 'trace.json'
    raw = shard.read_bytes()
    assert sha256(raw).hexdigest()==proof['cores'][core]['shard_sha256']
    origin = proof['earliest_global_us']
    stop = max(proof['cores'], key=lambda c: proof['cores'][c]['last'])
    assert isclose(proof['cores'][stop]['last']-origin, stage['ca_us'], abs_tol=.001)
    dmas, cubes = defaultdict(list), defaultdict(list)
    for e in json.loads(raw)['traceEvents']:
        if e['ph']!='X':
            continue
        detail = json.loads(e['args'].get('detail') or '{}')
        span = [e['ts']-origin,e['ts']+e['dur']-origin]
        if (detail.get('src_mem'),detail.get('dst_mem'))==('OUT','L1'):
            dmas[e['args']['pc_addr']].append((span,detail['src_addr'],detail['processed_bytes']))
        if e['name']=='MMAD':
            cubes[e['args']['pc_addr']].append((span,tuple(detail.get(k) for k in ('m','n','k'))))
    prediction = estimate('fa',receipt['shape'],chip=sku)
    assert isclose(prediction.latency_us,entry['candidate']) and stage['mmad']==514
    wave, = prediction.projection['timing_diagnostics']['wave_types']
    clock = load_chip(sku).frequency_ghz*1000
    launch = prediction.projection['stages_us']['launch']
    params = load(ROOT/'results/v0.16/fa.json')['routes']['wide']
    nrows = [256]*128+[1]
    counts = [ceil(n/128) for n in nrows]
    model, actual = {k:[] for k in PALETTE}, {k:[] for k in PALETTE}
    pc = {}
    streams = sorted(((k,sorted(v)) for k,v in dmas.items() if len(v)==len(nrows)),key=lambda kv:kv[1][0][0][0])
    assert len(streams)==2 and streams[0][1][-1][1]+128<=streams[1][1][0][1]
    for key,(addr,items) in zip(('K read','V read'),streams):
        base = items[0][1]
        pc[key] = addr
        for i,(span,src,nbytes) in enumerate(items):
            assert src==base+i*256*64*2 and nbytes==nrows[i]*64*2
            actual[key].append([span])
    missing_dims = {}
    for addr,rows in cubes.items():
        rows.sort()
        dims = {v for _,v in rows if v!=(None,None,None)}
        assert len(dims)==1
        dim, = dims
        key,tail = {(9,128,64):('QK',False),(9,64,128):('PV',False),
                    (9,1,64):('QK',True),(9,64,1):('PV',True)}[dim]
        group = key+(' tail' if tail else ' full')
        assert group not in pc and len(rows)==(1 if tail else 256)
        pc[group]=addr
        missing_dims[group]=sum(d==(None,None,None) for _,d in rows)
        if not actual[key]:
            actual[key]=[None]*len(nrows)
        for i in ([128] if tail else range(128)):
            pairs=rows if tail else rows[2*i:2*i+2]
            assert pairs[0][1]==dim and len(pairs)==counts[i]
            actual[key][i]=[x for x,d in pairs]
    cursor = 0
    for i,n in enumerate(nrows):
        tile = wave['first'] if i<128 else wave['last']
        if i: cursor+=tile['ii_cycles']
        for key in PALETTE:
            if key in ('QK','PV'):
                begin=wave[key.lower()+'_start_cycles']
                duration=tile[key.lower()]
            else:
                begin=params['k_lag']+(params['v_lag'] if key=='V read' else 0.)
                duration=params['read_intercept']+2*n*64/wave['read_bytes_per_cycle']
            start=launch+(begin+cursor)/clock
            model[key].append([[start,start+duration/clock]])
        assert actual['K read'][i][0][0] < actual['QK'][i][0][0]
        assert actual['V read'][i][0][0] < actual['PV'][i][0][0]
    return dict(shape=receipt['shape'], chip=sku, page_tokens=nrows, last_core=stop, cube_core=core,
                full_model_us=entry['candidate'], full_ca_us=stage['ca_us'], future_970_us=entry['future_us'],
                model=model, ca=actual, pc=pc, missing_dims_by_pc=missing_dims, cube_shard_sha256=proof['cores'][core]['shard_sha256'],
                receipt=str((SUP/(name+'.json')).relative_to(ROOT)), physical=str((SUP/'physical'/(name+'.json')).relative_to(ROOT)),
                model_ii_us=wave['first']['ii_cycles']/clock,
                measured_qk_ii_us=median(actual['QK'][i][0][0]-actual['QK'][i-1][0][0] for i in range(4,126)),
                kernel=stage['kernel'], semantics='Later modeled read/MMAD windows advance by scheduled page II; full kernel includes Vector and control.')


def draw(j, dest):
    indices = range(126,129)
    names = ('model','ca')
    ends = [x for name in names for key in PALETTE for i in indices for span in j[name][key][i] for x in span]
    left=floor(min(ends))-.1
    right=ceil(max(max(ends),j['full_model_us'],j['full_ca_us']))+.1
    fig,axes=plt.subplots(2,1,figsize=(11.5,7.5),sharex=True)
    fig.subplots_adjust(left=.19,right=.966,top=.765,bottom=.14,hspace=.52)
    ink='#182e3b'
    fig.text(.19,.94,f"{j['chip'][6:11]} FlashAttentionScore  |  Q = 9, KV = 32,769",size=21,weight='bold',color=ink)
    fig.text(.19,.89,'One head  ·  D = 64  ·  128 full KV pages and one final token  ·  1 AIC + 2 AIVs',size=12,color='#405b6b')
    fig.legend([Patch(color=v) for v in PALETTE.values()], ['MTE2 / K','MTE2 / V','Cube / QK','Cube / PV'],
               loc='upper left',bbox_to_anchor=(.18,.875),ncol=4,frameon=False)
    for ax,name,title,t in zip(axes,names,('Resource model · KV page schedule','CA trace · '+j['cube_core']),
                               (j['full_model_us'],j['full_ca_us'])):
        for row,i in enumerate(indices):
            for off,(key,color) in zip((-.27,-.09,.09,.27),PALETTE.items()):
                for a,b in j[name][key][i]:
                    ax.barh(row+off,b-a,left=a,height=.14,color=color,zorder=3)
                    if name=='ca':ax.vlines(a,row+off-.10,row+off+.10,color=color,lw=1,zorder=4)
        ax.set(ylim=(2.48,-.48),xlim=(left,right),yticks=[0,1,2],
               yticklabels=['KV 32,257–32,512','KV 32,513–32,768','KV 32,769  ·  tail'])
        ax.set_axisbelow(True)
        ax.grid(axis='x',color='#d3dfe5')
        for y in (.5,1.5): ax.axhline(y,color='#dce5e9',lw=.8)
        ax.axvline(t,ls=(0,(3,3)),color='#496273',lw=1.1)
        ax.set_title(title,loc='left',size=15,weight='bold',color=ink)
        ax.set_title(f'full {t:.3f} µs',loc='right',size=15,weight='bold',color=ink)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0,pad=5)
    axes[-1].set_xlabel('Time since first device instruction (µs)',labelpad=8)
    fig.text(.19,.046,'Ticks mark descriptor and MMAD issue. Full pages have two MMAD fragments per product; the one-token tail has one.',size=9.6,color='#455e6c')
    fig.text(.19,.018,'Dashed lines mark whole-kernel end, including Vector drain. Last measured core: '+j['last_core']+'.',size=9.6,color='#455e6c')
    fig.savefig(dest,dpi=150,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sku',choices=('dt','pr'),default='dt')
    parser.add_argument('--replot',action='store_true')
    args=parser.parse_args()
    path=OUT/f'fa-32769-{args.sku}-tail.json'
    if args.replot:
        j=load(path)
    else:
        j=collect({'dt':'Ascend950DT_9582','pr':'Ascend950PR_957b'}[args.sku])
        path.write_text(json.dumps(j,indent=2)+'\n')
    draw(j,path.with_suffix('.png'))
