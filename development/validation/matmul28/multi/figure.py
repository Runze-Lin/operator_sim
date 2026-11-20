"""Whole-kernel, shared-origin comparison of the retiled [144,64,7168] pipeline."""
import hashlib
import json
from math import ceil,isclose
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from operator_model import load_chip
from operator_model.matmul_resource import _costs
from operator_model.matmul_multisplit import estimate_matmul_retile

def figure():
    base=ROOT/'results/v0.28/matmul';out=base/'multi';receipt=base/'144-64-7168-dt-drain.json';d=json.loads(receipt.read_bytes());s=d['stages'][0]
    root=(ROOT/s['trace']).parent;shape=[144,64,7168];chip=load_chip()
    v=estimate_matmul_retile(shape,chip=chip);c=_costs();hz=chip.frequency_ghz;fc=hz*1000
    m,n,_=shape;k=v['max_segment_k'];bk=v['base_k'];mmads=ceil(k/bk);assert k<=v['page_k'] and v['output_tiles']==1
    lanes=['Control / launch','MTE2 · A/B input','MTE1 · L1 to L0','Cube · MMAD','FIXP · partial output','AIV · read partials','AIV · reduce / cast','MTE3 · final output']
    colors=['#738894','#117E82','#5376B0','#7153AE','#BC5D36','#298A8C','#915EAB','#B1512F']
    predicted=[];actual=[]
    def add(seq,l,a,b,name,**data):seq.append(dict(lane=l,begin_us=a,end_us=b,event=name,**data))
    t=v['launch_us'];add(predicted,0,0,t,'launch')
    rd=(c['read_latency_cycles']+2*k*max(m,n)/c['mte2_bytes_per_cycle'])/fc
    lead=(c['l0_latency_cycles']+2*bk*max(m,n)/c['mte1_bytes_per_cycle'])/fc
    add(predicted,1,t,t+rd,'critical input fill budget');t+=rd
    add(predicted,2,t,t+lead,'initial L0 lead budget');t+=lead
    peak=chip.cube_tops['16']*1000/(hz*chip.cube_cores)
    for j in range(mmads):
        depth=min(bk,k-j*bk);cycles=2*v['base_m']*v['base_n']*16*ceil(depth/16)/peak
        duration=(max(c['mmad_ii_cycles']+cycles,2*bk*max(m,n)/c['mte1_bytes_per_cycle']) if j+1<mmads else c['mmad_latency_cycles']+cycles)/fc
        add(predicted,3,t,t+duration,'MMAD budget',k=depth);t+=duration
    fix=(c['drain_cycles']+4*m*n/c['write_bytes_per_cycle'])/fc
    add(predicted,4,t,t+fix,'partial FIXP drain budget');t+=fix
    assert isclose(t,v['launch_us']+v['partial_pipeline_us'])
    assert v['partial_pipeline_us']>=max(v['cold_floor_us'],v['l2_floor_us'])
    add(predicted,6,t,t+v['merge_issue_us'],'aggregate merge issue budget');t+=v['merge_issue_us']
    add(predicted,5,t,v['latency_us'],'aggregate extra partial traffic budget')
    trace_files=[];cores=[]
    for cls in ('cube','vec'):
        core,win=max(((a,b) for a,b in s['windows'].items() if cls in a),key=lambda r:r[1][1]);cores.append(core)
        p=root/core/'trace.json';raw=p.read_bytes();ev=[e for e in json.loads(raw)['traceEvents']if e.get('ph')=='X'];zero=min(e['ts'] for e in ev)-win[0]
        trace_files.append(dict(core=core,core_start_us=win[0],core_end_us=win[1],global_origin_us=zero,path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest()))
        for e in ev:
            a=e['ts']-zero;b=a+e['dur'];z=json.loads(e['args']['detail']);name=e['name'];l=None
            if any(z.get(h) for h in ('is_sync','is_wait','is_duration_excluded')):continue
            if name=='MOV_OUT_TO_L1_MULTI_ND2NZ':l=1
            elif name=='LOAD_2Dv2':l=2
            elif name=='MMAD':l=3
            elif name=='FIX_L0C_TO_DST':l=4
            elif name=='VF' and cls=='vec':l=6
            elif name=='MOV_SRC_TO_DST_ALIGNv2' and cls=='vec':
                if z['src_mem']=='UB' and z['dst_mem']=='OUT':l=7
                elif z['src_mem']=='OUT' and z['dst_mem']=='UB' and z['burst_number']>1:l=5
            if l is not None:add(actual,l,a,b,name,pc=e['args']['pc_addr'],k=z.get('k'),bytes=z.get('processed_bytes'))
    assert len([x for x in actual if x['lane']==3])==mmads
    assert abs(max(z['core_end_us']for z in trace_files)-s['ca_us'])<1e-4
    fig,axes=plt.subplots(2,1,sharex=True,figsize=(12,7.8));fig.subplots_adjust(left=.20,right=.955,top=.80,bottom=.17,hspace=.42)
    fig.text(.20,.944,'950DT MatMulV3  |  144 × 64 × 7168',fontsize=23,weight='bold',color='#203541')
    fig.text(.20,.893,'One output rectangle  ·  32 K partitions  ·  224 values per AIC  ·  MMAD depths 64, 64, 64, 32',fontsize=12,color='#45606C')
    fig.text(.20,.848,'Registered ordinary tiling  15.309 µs       Retiled resource candidate  7.080 µs',fontsize=11,color='#48596C')
    for iy,(ax,events,end,title) in enumerate(zip(axes,(predicted,actual),(v['latency_us'],s['ca_us']),('Retiled resource model · critical-path budgets','CA trace · '+cores[0]+' and '+cores[1]))):
        ax.set_title(title,loc='left',weight='bold',fontsize=14,color='#203541');ax.set_title(f'{end:.3f} µs',loc='right',weight='bold',fontsize=13,color='#203541')
        for x in sorted(events,key=lambda z:z['begin_us']):
            l=x['lane'];a=x['begin_us'];b=x['end_us'];ax.barh(l,b-a,left=a,height=.37,color=colors[l],alpha=.52 if iy and l in (1,4,5) else .90,zorder=3)
            ax.vlines(a,l-.22,l+.22,color=colors[l],lw=1.2,zorder=4)
        ax.axvline(end,ls=(0,(3,3)),lw=1.2,color='#334957');ax.set_yticks(range(8),lanes,fontsize=9.5);ax.set_ylim(7.68,-.65);ax.tick_params(axis='y',length=0)
        ax.grid(axis='x',color='#D3DFE5',lw=.7);ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color('#b0c0c7')
    axes[1].set_xlim(0,max(v['latency_us'],s['ca_us'])*1.035);axes[1].set_xlabel('Device time from first instruction on any core (µs)',fontsize=12,labelpad=8)
    fig.text(.20,.073,'Top: merge and extra-traffic budgets span the final reduction; no separate final-store issue time is modeled.',size=9,color='#455f6e')
    fig.text(.20,.047,'Bottom: ticks mark real issues, bars the full in-flight descriptor interval, including dependency waits. Dashes mark the all-core end.',size=9,color='#455f6e')
    stem=out/'dt-144-retile';fig.savefig(stem.with_suffix('.png'),dpi=160);plt.close(fig)
    stem.with_suffix('.json').write_text(json.dumps(dict(shape=shape,role='development trace used for inspecting the new retile mechanism',lanes=lanes,model_us=v['latency_us'],ca_us=s['ca_us'],model=predicted,ca=actual,shards=trace_files,trace_sha256=s['trace_sha256'],receipt=str(receipt.relative_to(ROOT))),indent=2)+'\n')
if __name__=='__main__':figure()
