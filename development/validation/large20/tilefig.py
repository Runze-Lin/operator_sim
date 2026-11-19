"""Compare the exposed resource schedule with the terminal AIV on the same axis."""
from statistics import median
from pathlib import Path
from large_cast import DT, PR, OUT, ROOT, read, write
from operator_model import load_chip
from figures import STAGE_GROUPS, merged
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from campaign_collect import events


def queue_data(stage,terminal):
    # Per-core schema 1 maps queue ids to process names. Decode each event once,
    # then collapse concurrent lane events. This avoids loading the full-chip
    # multi-GiB JSON when rendering a long-context single-core example.
    trace=(ROOT/stage['trace']).parent/terminal/'trace.json'
    first=float('inf');names={};compute=[];scalar=[]
    for e in events(trace):
        if e['ph']=='M' and e['name']=='process_name':
            names[e['pid']]=e['args']['name']
        elif e['ph']=='X':
            first=min(first,e['ts'])
            target=names.get(e['pid'])
            if target=='SCALAR':
                scalar.append((e['ts'],e['ts']+e['dur']))
            elif e['name']=='RV_VCVT_F2F':
                compute.append((e['ts'],e['ts']+e['dur']))
    assert compute and scalar
    origin=first-stage['windows'][terminal][0]
    def relative(v):return [[a-origin,width] for a,width in merged(v)]
    return origin,relative(compute),relative(scalar)


def evidence(case,chip):
    t,h=case['shape'];key=f'cast_{t}_{h}_{chip}'
    stage=read(OUT/(key+'.json'))['stage']
    audit=read(OUT/(key+'-dma.json'))
    current=case['current'][chip];diag=current['diagnostics'];plan=diag['target_plan']
    cap=plan['ub_tile_elements'];extra=plan['busy_core_elements']-cap
    model_interval=diag['stages_us']['steady_tiles']*cap/extra if extra>0 else None
    origin,compute,scalar=queue_data(stage,audit['terminal'])
    dma={};gaps={}
    for q,size in [('MTE2',2*cap),('MTE3',4*cap)]:
        seq=sorted((e for e in audit['terminal_dma'] if e['queue']==q),key=lambda x:x['ts_us'])
        full=[e for e in seq if e['bytes']==size]
        intervals=[b['ts_us']-a['ts_us'] for a,b in zip(full,full[1:])]
        gaps[q]=median(intervals) if intervals else None
        dma[q]=[dict(start=e['ts_us']-origin,length=e['duration_us'],bytes=e['bytes']) for e in seq]
    ticks_per_us=1000*load_chip(chip).frequency_ghz
    # An equivalent number of full tiles avoids rounding the final partial
    # tile into a full interval. The whole-chip service limit acts on all AIVs.
    equivalent_tiles=(t*h)/(plan['active_aiv']*cap)
    shared_tile_us=max(diag['external_floor_cycles'],diag['l2_floor_cycles'] or 0)/ticks_per_us/equivalent_tiles
    implied_interval=max(model_interval or 0,shared_tile_us)
    return dict(schema=2,implied_full_interval_us=implied_interval,shared_full_interval_us=shared_tile_us,
                terminal=audit['terminal'],ca_us=stage['ca_us'],prediction_us=current['us'],
                stages=diag['stages_us'],tile_elements=cap,waves=plan['busy_core_waves'],
                model_full_interval_us=model_interval,median_full_issue_interval_us=gaps,
                dma=dma,conversion_union=compute,scalar_union=scalar,
                trace_sha256=stage['trace_sha256'])


def render(case,label,file_stem):
    t,h=case['shape'];bundle={}
    for c in (DT,PR):
        if 'ca_us' not in case['devices'][c]:continue
        cache=OUT/f'{file_stem}-{c}-plot.json'
        if cache.exists() and read(cache).get('schema')==2:
            bundle[c]=read(cache)
        else:
            bundle[c]=evidence(case,c);write(cache,bundle[c])
    if not bundle:return
    png=OUT/(file_stem+'.png')
    if png.exists() and all((OUT/f'{file_stem}-{c}-plot.json').stat().st_mtime_ns<=png.stat().st_mtime_ns for c in bundle):return
    fig,axes=plt.subplots(2,len(bundle),squeeze=False,figsize=(6.7*len(bundle),6.0),gridspec_kw={'height_ratios':[1,3.15]})
    fig.subplots_adjust(left=.115 if len(bundle)==1 else .085,right=.985,top=.795,bottom=.15,wspace=.33,hspace=.54)
    fig.suptitle(f'{label}   |   Cast BF16 → FP32, {t:,} × {h:,}',y=.988,fontsize=17,fontweight='bold')
    fig.legend(handles=[Patch(color=color,label=name) for name,_,color in STAGE_GROUPS],ncol=3,loc='upper center',bbox_to_anchor=(.52,.942),fontsize=9,frameon=False)
    maximum=1.04*max(max(x['ca_us'],x['prediction_us']) for x in bundle.values())
    colors={'MTE2':'#087c82','MTE3':'#bc6738'}
    for col,(chip,r) in enumerate(bundle.items()):
        top,ca=axes[:,col]
        pred=r['prediction_us'];actual=r['ca_us'];left=0
        for _,keys,color in STAGE_GROUPS:
            duration=sum(r['stages'][k] for k in keys)
            top.barh(0,duration,left=left,color=color,height=.50);left+=duration
        top.axvline(pred,color='#374b53',ls='--',lw=1.1)
        top.set_yticks([]);top.set_ylim(-.65,.65);top.tick_params(labelbottom=False)
        alias='950DT' if chip==DT else '950PR'
        top.set_title(f"{alias} model  ·  {pred:.2f} µs  ·  {r['waves']} UB tiles",loc='left',weight='bold',fontsize=11)
        off=r['model_full_interval_us']
        top.set_xlabel(f"Model full-tile rate-equivalent interval   R/W ≈ {r['implied_full_interval_us']:.2f} µs\nLocal ≈ {off:.2f} µs  ·  shared ≈ {r['shared_full_interval_us']:.2f} µs",fontsize=8.6)
        ca.set_title(f"{alias} CA  ·  {actual:.2f} µs  ·  {r['terminal']}",loc='left',weight='bold',fontsize=11)
        for q,base in [('MTE2',0),('MTE3',4)]:
            seq=r['dma'][q]
            for i,e in enumerate(seq):
                lane=base+(0 if i<2 else 2 if i>=len(seq)-2 else 1)
                alpha=.60 if lane==base+1 else .88
                ca.broken_barh([(e['start'],e['length'])],(lane-.26,.52),facecolors=colors[q],alpha=alpha)
                ca.vlines(e['start'],lane-.35,lane+.35,color='#28363e',lw=.65)
        ca.broken_barh(r['conversion_union'],(2.76,.48),facecolors='#7755a1')
        ca.broken_barh(r['scalar_union'],(6.76,.48),facecolors='#637688')
        ca.set_ylim(7.55,-.65);ca.set_yticks(range(8),['Read first 2','Read middle','Read last 2','Convert','Write first 2','Write middle','Write last 2','Scalar'],fontsize=8)
        ca.axvline(actual,color='#263941',lw=1.2);ca.axvline(pred,color='#374b53',ls='--',lw=1.)
        ix=r['median_full_issue_interval_us']
        ca.set_xlabel(f"Elapsed device time (µs)\nFull-tile issue interval   read {ix['MTE2']:.2f} µs   ·   write {ix['MTE3']:.2f} µs",fontsize=9)
        for axis in (top,ca):
            axis.set_xlim(0,maximum);axis.grid(axis='x',alpha=.18);axis.set_axisbelow(True);axis.spines[['right','top']].set_visible(False)
    fig.text(.085,.018,'Ticks mark observed DMA starts. First/last refer to visible descriptors, including the tail. Full-tile medians exclude tails. Model bars show exposed costs.',fontsize=8)
    fig.savefig(png,dpi=155);plt.close(fig)


def draw_tile_windows(data):
    for case in data['cases']:
        t,h=case['shape']
        if h in (32,64):
            size=t//8192;model='Flash' if h==32 else 'Pro'
            render(case,f'{model}   ·   8 × {size}ki prefill requests',f'{model.lower()}-{size}ki-tiles')
