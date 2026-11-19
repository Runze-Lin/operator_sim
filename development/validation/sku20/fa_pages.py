"""Plot PR 32k or DT 128k KV pages with recorded MMAD PCs and a common device clock."""
from collections import defaultdict
import hashlib
import json
from math import isclose
from pathlib import Path
from statistics import median
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate, load_chip

OUT=ROOT/'results/v0.20/examples/fa-pr-32k-pages.json'


def archived_dt():
    import sqlite3
    history=json.loads((ROOT/'results/v0.5/evaluation.json').read_text())['rows']
    old=next(r for r in history if r['id']=='f5_holdout_128k')
    path,=(ROOT/'runs/model-v5/f5_holdout_128k').rglob('*.db')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):
            h.update(block)
    assert h.hexdigest()==old['db_sha256']
    clock=1000*load_chip('Ascend950DT_9582').frequency_ghz
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        spans=list(db.execute('SELECT CoreTypeId, min(ExecInstrTickStart), max(ExecInstrTickEnd) '
                              'FROM ExecutedInstructions GROUP BY CoreTypeId'))
        origin=min(a for _,a,_ in spans)
        ids={1:'core0.cubecore0',2:'core0.AIV(type2)',3:'core0.AIV(type3)'}
        cores=[dict(core=ids[c],start_us=(a-origin)/clock,end_us=(b-origin)/clock) for c,a,b in spans]
        by_pc=defaultdict(list)
        for pc,start,end,detail in db.execute("SELECT SourceInstrAddr,ExecInstrTickStart,ExecInstrTickEnd,ExtendParams "
                                              "FROM ExecutedInstructions WHERE ExecInstrName='MMAD' AND CoreTypeId=1"):
            by_pc[hex(pc)].append(dict(ts=(start-origin)/clock,dur=(end-start)/clock,args=dict(detail=detail)))
    assert sum(map(len,by_pc.values()))==old['observed_mmad']==2048
    assert isclose(max(c['end_us'] for c in cores),old['reference_us'])
    ca=dict(soc='Ascend950DT_9582',core_windows=cores,ca_us=old['reference_us'],kernel=old['kernel'],
            correctness=old['correctness'],trace=str(path.relative_to(ROOT)),trace_sha256=old['db_sha256'])
    return dict(shape=old['shape'],contract={'api':'aclnnFlashAttentionScoreV2'}),ca,by_pc


def load_receipt(dt128=False):
    if dt128:
        item,ca,by_pc=archived_dt()
        origin=0.0  # archived_dt returned times relative to earliest instruction
    else:
        history=json.loads((ROOT/'results/v0.15/sku/holdout.json').read_text())
        item=next(r for r in history if r['op']=='fa' and r['shape']==[9,32768,64,1,1])
        ca=item['pr']
    model=estimate('fa',item['shape'],chip=ca['soc'],timing_mode='resource')
    clock=1000*load_chip(ca['soc']).frequency_ghz
    diag=model.projection['timing_diagnostics'];wave,=diag['wave_types']
    tile=wave['first'];count=diag['target_plan']['iterations']
    assert count==item['shape'][1]//256 and diag['mmads_total']==4*count and wave['last']==tile
    qk,pv=wave['qk_start_cycles'],wave['pv_start_cycles']
    launch=model.projection['stages_us']['launch'];step=tile['ii_cycles']/clock
    phases={name:[[launch+start/clock+i*step,launch+(start+duration)/clock+i*step] for i in range(count)]
            for name,start,duration in [('QK',qk,tile['qk']),('PV',pv,tile['pv'])]}
    assert isclose(phases['PV'][-1][1]+model.projection['stages_us']['drain'],model.latency_us)
    cores=ca['core_windows'];cube,= [g for g in cores if 'cube' in g['core']]
    if not dt128:
        path=(ROOT/ca['trace']).parent/cube['core']/'trace.json'
        raw=path.read_bytes()
        events=[e for e in json.loads(raw)['traceEvents'] if e['ph']=='X']
        origin=min(e['ts'] for e in events)-cube['start_us']
        assert isclose(max(e['ts']+e['dur'] for e in events)-origin,cube['end_us'],abs_tol=.001)
        by_pc=defaultdict(list)
        for e in events:
            if e['name']=='MMAD':
                by_pc[e['args']['pc_addr']].append(e)
    assert len(by_pc)==2
    groups={};identifiers={}
    for pc,group in by_pc.items():
        group.sort(key=lambda e:e['ts'])
        shapes=[tuple(json.loads(e['args']['detail']).get(k) for k in ('m','n','k')) for e in group]
        known=set(shapes)-{(None,None,None)}
        assert len(known)==1
        name={(9,128,64):'QK',(9,64,128):'PV'}[known.pop()]
        assert len(group)==2*count
        if name=='PV':
            assert all(shapes[i]==(9,64,128) for i in range(0,len(shapes),2))
        identifiers[name]=dict(pc=pc,missing_dimensions=shapes.count((None,None,None)),
                              fragments_per_page=2)
        groups[name]=[[[e['ts']-origin,e['ts']+e['dur']-origin] for e in group[i:i+2]]
                      for i in range(0,len(group),2)]
    gaps={k:median(b[0][0]-a[0][0] for a,b in zip(vals[3:-3],vals[4:-2])) for k,vals in groups.items()}
    return dict(shape=item['shape'],sku=ca['soc'],model_us=model.latency_us,ca_us=ca['ca_us'],
                chip_clock_cycles_per_us=clock,api=item['contract']['api'],kernel=ca['kernel'],
                model=phases,ca=groups,pc=identifiers,model_ii_us=step,
                ca_midrun_first_fragment_gap_us=gaps,cube=cube['core'],terminal=max(cores,key=lambda g:g['end_us']),
                correctness=ca['correctness'],scope='One dense SBH noncausal Q9 head; two 128-token MMADs per 256-token page per stream. Later model service windows advance by the resource II.',
                merged_trace=ca['trace'],merged_sha256=ca['trace_sha256'],cube_trace=ca['trace'] if dt128 else str(path.relative_to(ROOT)),
                cube_sha256=ca['trace_sha256'] if dt128 else hashlib.sha256(raw).hexdigest())


def draw(r,slug='fa-pr-32k-pages'):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D
    colors={'QK':'#67519c','PV':'#167d85'}
    fig=plt.figure(figsize=(12,6.4))
    gs=fig.add_gridspec(3,3,height_ratios=[1,1,.8],left=.105,right=.978,top=.76,bottom=.19,wspace=.25,hspace=.61)
    q,kv,d,_,_=r['shape'];count=kv//256;sku=r['sku'].split('_')[0].removeprefix('Ascend')
    fig.text(.105,.949,f'{sku}  FlashAttentionScoreV2  |  Q = {q}, KV = {kv:,}, D = {d}',weight='bold',fontsize=17,color='#172e3b')
    fig.text(.105,.898,f'{count} KV pages · one AIC + two AIVs · zooms share the device clock for both rows',fontsize=11,color='#425a68')
    fig.legend([Patch(color=colors[n]) for n in colors]+[Line2D([],[],ls='',marker='|',color='#273448',ms=9)],
               ['QK / score','PV / output','CA MMAD issue'],loc='upper left',bbox_to_anchor=(.097,.883),ncol=3,frameon=False)
    selected=[(0,1),(count//2-1,count//2),(count-2,count-1)]
    for col,pair in enumerate(selected):
        ranges=[span for i in pair for n in colors for span in [r['model'][n][i],*r['ca'][n][i]]]
        lo=min(x[0] for x in ranges)-.25; hi=max(x[1] for x in ranges)+.25
        for row,mode in enumerate(('model','ca')):
            ax=fig.add_subplot(gs[row,col])
            for j,i in enumerate(pair):
                for n,delta in [('QK',.11),('PV',-.11)]:
                    spans=[r['model'][n][i]] if mode=='model' else r['ca'][n][i]
                    for a,b in spans:
                        ax.broken_barh([(a,b-a)],(1-j+delta-.085,.17),facecolors=colors[n],alpha=1 if mode=='ca' or i==0 else .60)
                        if mode=='ca': ax.vlines(a,1-j+delta-.14,1-j+delta+.14,color=colors[n],lw=.85)
            ax.set(xlim=(lo,hi),ylim=(-.45,1.45),yticks=[0,1],yticklabels=[f'KV {pair[1]+1}',f'KV {pair[0]+1}'])
            ax.grid(axis='x',alpha=.19);ax.set_axisbelow(True);ax.xaxis.set_major_locator(MaxNLocator(5));ax.tick_params(length=0,labelsize=9)
            for spine in ax.spines.values():spine.set_visible(False)
            if row==0:ax.set_title(['Pipeline fill','Steady state','Tail pages'][col],loc='left',fontsize=10.5,weight='bold')
            if col==0:ax.set_ylabel('Model' if row==0 else 'CA · AIC',weight='bold',labelpad=7)
    ax=fig.add_subplot(gs[2,:])
    for y,n in enumerate(('PV','QK')):
        for page in r['ca'][n]:
            ax.vlines(page[0][0],y-.24,y+.24,color=colors[n],lw=.6)
    for key,color in [('model_us','#67519c'),('ca_us','#364651')]:
        ax.axvline(r[key],color=color,ls='--',lw=1)
    ax.set_yticks([0,1],['CA · PV','CA · QK']);ax.set_ylim(-.65,1.65);ax.set_xlim(0,max(r['ca_us'],r['model_us'])*1.025)
    ax.set_xlabel('Device time since first instruction on any core (µs)');ax.grid(axis='x',alpha=.17)
    ax.set_title(f"Full window     model {r['model_us']:.3f} µs     CA {r['ca_us']:.3f} µs     last core {r['terminal']['core']}",loc='left',weight='bold',fontsize=10.5)
    ax.tick_params(length=0);[s.set_visible(False) for s in ax.spines.values()]
    fig.text(.105,.055,'Model: page 1 uses QK/PV readiness; faded later bars use the frozen steady issue interval.  CA: each zoom shows both fragment instructions.',fontsize=9)
    fig.text(.105,.025,'Overview: each tick is the first fragment of a KV page. The complete device window includes the Vector drain after the last Cube instruction.',fontsize=9)
    fig.savefig(ROOT/'docs/examples/figures'/(slug+'.png'),dpi=155,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    dt128='--dt128k' in sys.argv
    path=OUT.with_name('fa-dt-128k-pages.json') if dt128 else OUT
    if '--replot' in sys.argv:
        data=json.loads(path.read_text())
    else:
        data=load_receipt(dt128)
        path.write_text(json.dumps(data,separators=(',',':'))+'\n')
    draw(data,path.stem)
    print(data['model_us'],data['ca_us'],data['model_ii_us'],data['ca_midrun_first_fragment_gap_us'])
