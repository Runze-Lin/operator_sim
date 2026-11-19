"""Extract small reproducible last-core plots, preserving the all-core time origin."""
import argparse
from collections import defaultdict
import hashlib
import json
from math import isclose
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate, load_chip
from operator_model.quant_resource import resource_quant

OUT = ROOT/'results/v0.20/examples'
COLORS={'MTE2':'#147786','MTE1':'#5275ae','CUBE':'#7150a0','FIXP':'#ac4929',
        'MTE3':'#be642c','SIMD':'#6c4d9f','SCALAR':'#617687','control':'#8597a1',
        'fabric':'#407fa0','phase':'#4d7083','steady':'#297b7a'}
PRIMARY=('MTE2','MTE1','CUBE','FIXP','SIMD','MTE3','SCALAR')
FAST={'quant','rmsnorm','fa','matmul'}


def read(path):
    return json.loads(path.read_bytes())


def union(seq):
    merged=[]
    for a,b in sorted(seq):
        if merged and a<=merged[-1][1]:
            merged[-1][1]=max(b,merged[-1][1])
        else:
            merged.append([a,b])
    return merged


def windows(stage,path):
    if stage.get('windows'):
        return stage['windows']
    for parent in path.parents:
        if parent.name in ('runs','profile'):
            continue
        qualified=parent/'qualified.json'
        if not qualified.is_file():
            continue
        docs=read(qualified)
        for doc in docs if isinstance(docs,list) else [docs]:
            for st in doc.get('stages',[]):
                if st.get('trace')==str(path.relative_to(ROOT)) and st.get('windows'):
                    return st['windows']
    # Slim manifests retain all-core aggregates. Recover starts/ends from
    # the individual core shards: those timestamps share a device clock.
    spans={}
    for shard in sorted(path.parent.glob('core*/trace.json')):
        events=(e for e in read(shard)['traceEvents'] if e['ph']=='X')
        first=float('inf');last=-first
        for e in events:
            first=min(first,e['ts']);last=max(last,e['ts']+e['dur'])
        if last>=first:
            spans[shard.parent.name]=[first,last]
    start=min(a for a,_ in spans.values())
    return {c:[a-start,b-start] for c,(a,b) in spans.items()}


def trace_lanes(stage,path,offset,check_us):
    spans=windows(stage,path)
    assert isclose(max(b for a,b in spans.values()),check_us,abs_tol=.002)
    groups=[]
    for family in ('cube','vec'):
        ids=[c for c in spans if family+'core' in c]
        if not ids:continue
        core=max(ids,key=lambda c:spans[c][1])
        shard=path.parent/core/'trace.json'
        raw=shard.read_bytes()
        events=json.loads(raw)['traceEvents']
        origin=min(e['ts'] for e in events if e['ph']=='X')-spans[core][0]
        intervals=defaultdict(list);issues=defaultdict(list)
        for e in events:
            if e['ph']!='X' or e['dur']<=0:continue
            args=e.get('args',{});detail=args.get('detail') or '{}'
            detail=json.loads(detail) if isinstance(detail,str) else detail
            if not isinstance(detail,dict) or any(detail.get(k) for k in ('is_wait','is_sync','is_duration_excluded')):continue
            q=detail.get('instr_type','')
            if q.startswith(('SIMD','RVEC','VEC')):q='SIMD'
            if q not in PRIMARY:continue
            a=e['ts']-origin+offset;b=a+e['dur']
            intervals[q].append([a,b])
            if e['name']=='MMAD' or (q in ('MTE2','MTE3','FIXP') and ('MOV' in e['name'] or 'FIX_' in e['name'])):
                issues[q].append(a)
        groups.append(dict(core=core,window=[v+offset for v in spans[core]],file=str(shard.relative_to(ROOT)),
            sha256=hashlib.sha256(raw).hexdigest(),lanes={q:union(v) for q,v in intervals.items()},
            issues={q:sorted(v) for q,v in issues.items()}))
    return groups


def interval(row,a,b,color):
    return [row,float(a),float(b),color]


def upper(op,shape,attrs,chip):
    e=estimate(op,shape,attrs,chip=chip,timing_mode='resource' if op in FAST else 'auto')
    stages=e.projection['stages_us']
    data=e.projection.get('timing_diagnostics') or {}
    plan=e.projection.get('target_plan') or {}
    segments=[];marks=[];caption='Modeled duration budget · cumulative contributions'
    clock=1000*load_chip(chip).frequency_ghz
    if op=='fa':
        assert plan['waves']==1
        w,=data['wave_types'];t=w['first'];launch=stages['launch']
        point=lambda v: launch+v/clock
        r=w['read_bytes_per_cycle']
        segments.extend([interval('Launch',0,launch,'control'),
            interval('KV page 1 · K read',point(w['k_ready_cycles']-t['k_bytes']/r),point(w['k_ready_cycles']),'MTE2'),
            interval('KV page 1 · V read',point(w['v_ready_cycles']-t['v_bytes']/r),point(w['v_ready_cycles']),'MTE2'),
            interval('KV page 1 · QK',point(w['qk_start_cycles']),point(w['qk_start_cycles']+t['qk']),'CUBE'),
            interval('KV page 1 · Vec1',point(w['qk_start_cycles']+t['qk']),point(w['qk_start_cycles']+t['qk']+t['vec1']),'SIMD'),
            interval('KV page 1 · PV',point(w['pv_start_cycles']),point(w['pv_start_cycles']+t['pv']),'CUBE')])
        s=launch+stages['first_tile']
        for i in range(1,plan['iterations']):
            end=s+(w['last'] if i==plan['iterations']-1 else t)['ii_cycles']/clock
            segments.append(interval('KV pages 2–%d · II'%plan['iterations'],s,end,'steady'))
            marks.append([segments[-1][0],s]);s=end
        segments.append(interval('Drain',s,e.latency_us,'MTE3'))
        caption='Resource model · first page, steady issue interval and drain'
    elif op=='quant':
        _,_,_,q=resource_quant(shape,load_chip(chip),explain_tiles=True)
        tiles=q['tile_intervals_us']
        segments.append(interval('Dispatch',0,stages['dispatch'],'control'))
        for i,t in enumerate(tiles):
            for unit in ('MTE2','SIMD','MTE3'):
                segments.append(interval(f'UB tile {i+1} · {t["rows"]}×{t["cols"]}',*t[unit],unit))
        mx=max(x[2] for x in segments)
        finish=e.latency_us-stages['drain']
        assert mx<=finish+1e-7
        if mx<finish:
            segments.append(interval('Shared link · exposed',mx,finish,'fabric'))
        segments.append(interval('Drain',finish,e.latency_us,'MTE3'))
        caption='Resource model · two-slot UB schedule and exposed link floor'
    elif op=='rmsnorm':
        # Two physically independent clocks. Mutually exclusive core costs
        # determine one finish; the shared traffic bound starts at P_release.
        local=0.
        for label,delta in stages.items():
            if label=='shared_io' or delta<=0:continue
            color=label if label in COLORS else 'control'
            segments.append(interval('Terminal core · exposed costs',local,local+delta,color))
            marks.append(['Terminal core · exposed costs',local])
            local+=delta
        from operator_model.rms_resource import FIRST_PAYLOAD_RELEASE
        release=data.get('first_dispatch_cycles',{}).get('payload',FIRST_PAYLOAD_RELEASE)/clock
        shared=release+data['shared_io_floor_cycles']/clock
        segments.append(interval('Device · memory path',0,release,'control'))
        segments.append(interval('Device · memory path',release,shared,'fabric'))
        assert isclose(e.latency_us,max(shared,local),abs_tol=1e-6)
        caption='Resource model · latest of core finish and shared transfer'
    else:
        cursor=0.
        for label,value in stages.items():
            if value<=0:continue
            segments.append(interval(label.replace('_',' '),cursor,cursor+value,label if label in COLORS else 'phase'))
            cursor+=value
        assert isclose(cursor,e.latency_us,abs_tol=1e-5)
        if op in ('mhc','moe_ffn'):
            caption='Resource projection · serial kernel windows'
    return dict(latency_us=e.latency_us,plan=plan,segments=segments,marks=marks,caption=caption)


def build(op,row,side):
    obs=row[side+'_observation']
    chip='Ascend950DT_9582' if side=='dt' else 'Ascend950PR_957b'
    top=upper(op,row['shape'],row['attrs'],chip); bottom=[];offset=0.
    paths=obs['trace'];paths=paths if isinstance(paths,list) else [paths]
    sts=obs.get('stages') or [obs]
    for i,(p,st) in enumerate(zip(paths,sts)):
        groups=trace_lanes(st,ROOT/p,offset,st['ca_us'])
        terminal=max(groups,key=lambda g:g['window'][1])['core']
        bottom.append(dict(groups=groups,terminal=terminal,kernel=st['kernel'],start=offset,us=st['ca_us'],
            trace=p,trace_sha256=(obs['trace_sha256'][i] if isinstance(obs['trace_sha256'],list) else obs['trace_sha256'])))
        offset+=st['ca_us']
    assert isclose(offset,obs['ca_us'],abs_tol=1e-5)
    return dict(op=op,shape=row['shape'],attrs=row['attrs'],chip=chip,api=row['api'],correctness=obs['correctness'],
        ca_us=offset,model=top,ca=bottom,model_scope=top['caption'],
        ca_scope='Global earliest instruction to final retirement; for serial launches concatenate net windows without host gaps. Ranges union non-wait, non-sync instruction descriptors on the last core of each family.')


def draw(d,target):
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    top=d['model'];groups=d['ca'];segments=top['segments']
    labels=list(dict.fromkeys(x[0] for x in segments))
    lanes=[];ticks=[]
    for i,s in enumerate(groups):
        prefix=f'K{i+1} ' if len(groups)>1 else ''
        for g in s['groups']:
            family='AIC' if 'cubecore' in g['core'] else 'AIV'
            for q in PRIMARY:
                if q not in g['lanes']:continue
                label=f'{prefix}{family} · {q}'
                lanes.append((label,g['lanes'][q],q))
                ticks.extend((label,t) for t in g['issues'].get(q,[]))
    n0=len(labels);n1=len(lanes)
    fig,axes=plt.subplots(2,1,sharex=True,figsize=(10.5,max(5.6,2.8+.255*(n0+n1))),
        gridspec_kw={'height_ratios':[max(3,n0),max(3,n1)]})
    fig.subplots_adjust(left=.245,right=.965,top=.825,bottom=.13,hspace=.52)
    short='950DT' if '950DT' in d['chip'] else '950PR'
    shape=' × '.join(map(str,d['shape']))
    fig.suptitle(f"{short}  {d['op']}  |  {shape}",x=.245,y=.977,ha='left',fontsize=18,weight='bold',color='#192f3d')
    display={'CUBE':'Cube / MMAD','SIMD':'Vector / SIMD','fabric':'Shared link','phase':'Stage budget','steady':'Steady II','control':'Control'}
    used=set(x[-1] for x in segments)|set(q for _,_,q in lanes)
    fig.legend([Patch(color=COLORS[q]) for q in COLORS if q in used],
        [display.get(q,q) for q in COLORS if q in used],loc='upper left',bbox_to_anchor=(.237,.932),
        ncol=6,fontsize=8,frameon=False)
    terminal=', '.join(s['terminal'].replace('.cubecore0',' AIC').replace('.veccore',' V') for s in groups)
    for ax,rows,title,end in zip(axes,(labels,[l for l,v,q in lanes]),
        (top['caption'],'CA instructions · last to finish: '+terminal),(top['latency_us'],d['ca_us'])):
        ax.set_title(title,loc='left',fontsize=10.5,weight='bold',color='#233c49',pad=7)
        ax.text(1,1.025,f'{end:.3f} µs',transform=ax.transAxes,ha='right',va='bottom',weight='bold',color='#233c49')
        ax.set_yticks(range(len(rows)),rows,fontsize=8.5);ax.set_ylim(len(rows)-.45,-.55)
        ax.tick_params(axis='y',length=0,pad=7);ax.grid(axis='x',c='#dae3e8',lw=.7);ax.set_axisbelow(True)
        ax.axvline(end,c='#425963',lw=1,ls=(0,(3,3)))
        for edge in ax.spines.values():edge.set_visible(False)
    for label,a,b,q in segments:
        y=labels.index(label)
        pos={'MTE2':-.23,'SIMD':0,'MTE3':.23}.get(q,0) if label.startswith('UB tile') else 0
        axes[0].barh(y+pos,b-a,left=a,height=.18 if label.startswith('UB tile') else .46,color=COLORS[q])
    for label,t in top['marks']:axes[0].vlines(t,labels.index(label)-.28,labels.index(label)+.28,lw=.5,color='white')
    bottom_labels=[x[0] for x in lanes]
    for i,(label,ranges,q) in enumerate(lanes):
        axes[1].broken_barh([(a,b-a) for a,b in ranges],(i-.24,.48),facecolors=COLORS[q],linewidths=0)
    for label in dict.fromkeys(k for k,_ in ticks):
        y=bottom_labels.index(label); axes[1].vlines([v for k,v in ticks if k==label],y-.33,y+.33,color='#233d4c',lw=.38,alpha=.8)
    axes[1].set_xlim(0,max(top['latency_us'],d['ca_us'])*1.025);axes[1].set_xlabel('Device time from first instruction (µs)')
    fig.text(.245,.034,'CA ticks: actual DMA / MMAD issues. Bars show non-wait descriptor windows; queues can overlap.',fontsize=8,color='#435c69')
    fig.savefig(target,dpi=150,facecolor='white');plt.close(fig)


def cases():
    previous=read(ROOT/'results/v0.19/evaluation.json')['rows']
    snapshots=read(ROOT/'results/v0.19/sku/final-records.json')
    for op in ('quant','rmsnorm'):
        records=[r for r in snapshots if r['op']==op and r['shape']==[2048,5120]]
        row=dict(op=op,shape=[2048,5120],attrs={},api=records[0]['api'])
        for rec in records:
            side='pr' if '950PR' in rec['chip'] else 'dt'
            row[side+'_observation']=rec
        for side in ('dt','pr'):yield f'{op}-chunk-{side}',op,row,side
    registered=ROOT/'results/v0.20/rms/records.json'
    for rec in read(registered):
        if rec['shape']!=[1024,5120]:continue
        side='dt' if '950DT' in rec['chip'] else 'pr'
        item=dict(op='rmsnorm',shape=rec['shape'],attrs={},api=rec['api'])
        item[side+'_observation']=rec
        yield 'rmsnorm-new-'+side,'rmsnorm',item,side
    for op in sorted({r['op'] for r in previous}-{'matmul','quant','rmsnorm','kda_chunk'}):
        choices=[r for r in previous if r['op']==op and r.get('pr_ca_us')]
        if op=='fa':choices=[r for r in choices if r['shape']==[9,8191,64,1,1]]
        else:
            # Match the shape printed by an existing worked example when available.
            import re,ast
            text=(ROOT/'docs/examples'/(op+'.md')).read_text()
            m=re.search(r'^shape = (\[[^\n]+?\])$',text,re.M)
            if m:
                same=[r for r in choices if r['shape']==ast.literal_eval(m[1])]
                choices=same or choices
        row=choices[0]
        yield op+'-pr',op,row,'pr'
        if op=='fa':yield op+'-dt',op,row,'dt'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract',action='store_true',help='Use locally retained original raw CA traces')
    parser.add_argument('--case',action='append',default=[])
    args=parser.parse_args()
    for name,op,row,side in cases():
        if args.case and name not in args.case:continue
        evidence=OUT/(name+'.json')
        if args.extract:
            evidence.write_text(json.dumps(build(op,row,side),separators=(',',':'))+'\n')
        draw(read(evidence),ROOT/'docs/examples/figures'/(name+'-pipeline.png'))
        print(name,flush=True)
