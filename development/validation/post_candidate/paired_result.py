"""Verify the two frozen mHC Post ordinal pairs and plot actual last-finishing cores."""
from collections import defaultdict
from math import isclose
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.post_candidate.repeat_ca import OUT, RAW, REG, DT, PR, digest, read, write, qualify
from development.validation.post_candidate.extract import extract
from campaign_collect import events
from operator_model import load_chip
from operator_model.post_resource import resource_post

FIGURE = ROOT/'docs/examples/figures/mhc-post-repeat-paired.png'
PHASES = ['arrival','SCALAR','MTE2','SIMD','MTE3','other','shared_link_floor']
PALETTE = dict(arrival='#bbcad3',SCALAR='#55708e',MTE2='#127f84',SIMD='#7854ac',MTE3='#bd6132',
               other='#7c8790',shared_link_floor='#7c8790')
QUEUES = ('SCALAR','MTE2','SIMD','MTE3')


def merged(intervals):
    # Union of true interval spans, with no time binning or artificial widening.
    out=[]
    for a,b in sorted(intervals):
        if out and a<=out[-1][1]:
            out[-1][1]=max(out[-1][1],b)
        else:
            out.append([a,b])
    return out


def intervals(stage):
    core=max(stage['windows'],key=lambda name:stage['windows'][name][1])
    tr=ROOT/Path(stage['trace']).parent/core/'trace.json'
    segments=defaultdict(list)
    first=None
    for e in events(tr):
        if e.get('ph')!='X':
            continue
        t=e['ts']
        first=t if first is None else min(first,t)
        d=json.loads(e['args']['detail'])
        if e['dur']<=0 or any(d.get(flag) for flag in ('is_wait','is_sync','is_duration_excluded')):
            continue
        q='SIMD' if d['instr_type'].startswith(('SIMD','RVEC')) else d['queue_type']
        if q in ('MTE2','MTE3') and not e['name'].startswith('MOV_SRC_TO_DST'):
            continue
        if q in QUEUES:
            segments[q].append((t,t+e['dur']))
    origin=first-stage['windows'][core][0]
    by_queue={q:merged([(a-origin,b-origin) for a,b in segments[q]]) for q in QUEUES}
    return dict(terminal_core=core,terminal_trace=str(tr.relative_to(ROOT)),terminal_trace_sha256=digest(tr),
                terminal_arrives_us=stage['windows'][core][0],nonwait_queue_union_us=by_queue)


def collect():
    f=read(REG)
    assert digest(ROOT/f['runner'])==f['runner_sha256']
    assert digest(ROOT/'operator_model/post_resource.py')==f['public_model_sha256']
    assert digest(Path(__file__).with_name('repeat_ca.py'))==f['source_sha256']
    assert digest(OUT/'prospective.json')==f['candidate_source']
    assert all(digest(ROOT/name)==sha for name,sha in f['runner_sources'].items())
    shape=f['shape']
    result={}
    for chip in (DT,PR):
        receipt=OUT/f'paired-{chip}.json'
        old=read(receipt)
        work=RAW/chip
        assert old['qualified'] and old['shape']==shape and old['chip']==chip
        assert old['cost']==read(work/'cost.json') and old['cost']['started_utc']>f['utc']
        assert old['cost']['exit_code']==0 and old['cost']['stop_reason'] is None
        assert (work/'cases.txt').read_text()==f['batch_text']
        verified=qualify(chip,work,f)
        assert verified==old['rows']
        _,_,active,diag=resource_post(shape,load_chip(chip),{'has_h_res':True})
        assert diag['target_plan']==f['targets'][chip]['prior_registered_shape']['target_plan'] and active==48
        rows=[]
        for i,r in enumerate(verified):
            cond=r['condition']
            observed=r['stage']
            u=observed['ca_us']
            assert observed['binary_sha256']==f['expected_kernel_sha256'] and len(observed['windows'])==48
            assert 'bad=0 max_abs_error=0 ' in r['correctness'] and r['ordinal']==i
            assert observed['active_aic']==0 and observed['active_aiv']==48
            assert isclose(max(w[1] for w in observed['windows'].values()),u,abs_tol=1e-8)
            assert min(w[0] for w in observed['windows'].values())==0
            fr=f['targets'][chip]
            assert r['current_unknown_us']==fr['v021_us']['unknown'] and r['current_us']==fr['v021_us'][cond]
            assert r['candidate_unknown_us']==fr['candidate_us']['shape_only'] and r['candidate_us']==fr['candidate_us'][cond]
            model=diag['state_phases_us'][cond]
            assert isclose(sum(model.values()),r['current_us'],abs_tol=1e-10)
            phase=extract({'us':u,'stage':observed})
            costs={k:r[k] for k in ('current_us','current_unknown_us','candidate_us','candidate_unknown_us')}
            rows.append(dict(ordinal=i,condition=cond,ca_us=u,predictions_us=costs,
                             signed_error_pct={k:100*(v/u-1) for k,v in costs.items()},
                             model_contributions_us=model,ca_milestones=phase,**intervals(observed),
                             correctness=r['correctness'],kernel=observed['kernel'],binary_sha256=observed['binary_sha256'],
                             trace=observed['trace'],trace_sha256=observed['trace_sha256'],cores=sorted(observed['windows'])))
        first,repeat=rows
        assert first['cores']==repeat['cores'] and first['kernel']==repeat['kernel'] and first['trace']!=repeat['trace']
        for key in ('read_bytes','write_bytes','raw_queue_instruction_counts','simd_core_instructions'):
            assert first['ca_milestones'][key]==repeat['ca_milestones'][key]
        b=first['ca_us']
        a=repeat['ca_us']
        assert b>a
        tolerance_intervals=[(.6*u,1.4*u) for u in (b,a)]
        lower=2*a*b/(a+b)
        wapes={}
        for k in first['predictions_us']:
            errors=[abs(row['predictions_us'][k]-row['ca_us']) for row in rows]
            wapes[k]=dict(wape_pct=100*sum(errors)/(a+b),
                          max_absolute_error_pct=max(abs(row['signed_error_pct'][k]) for row in rows))
        result[chip]=dict(shape=shape,target_plan=diag['target_plan'],rows=rows,
            unknown_model_budget_us=diag['stages_us'],cost=old['cost'],first_over_repeat=b/a,
            pair_only_metrics=wapes,first_40pc_interval_us=tolerance_intervals[0],repeat_40pc_interval_us=tolerance_intervals[1],
            intersects=tolerance_intervals[0][0]<=tolerance_intervals[1][1],minimax_unknown_us=lower,
            min_possible_max_relative_error_pct=100*(b-a)/(b+a),
            before_first_vector_saving_us=first['ca_milestones']['first_vector_operand_us']-repeat['ca_milestones']['first_vector_operand_us'],
            input_sha_check='same host ELF and source; batch ordinal only printed as ID; same per-element functions in post(stream,T,N,D), separate tensor allocation for each call',
            receipt= str(receipt.relative_to(ROOT)),receipt_sha256=digest(receipt),run_log_sha256=digest(work/'run.log'))
    return dict(preregistered=str(REG.relative_to(ROOT)),registration_sha256=digest(REG),registered_utc=f['utc'],
                sample_role='Frozen unseen shape; one paired process per 950 SKU. These 2 ordinals per SKU are not independent shapes.',
                kernel_correctness='Same package ELF and CANN host executable. Complete per-ordinal numerical golden and all vendor core windows rechecked from raw files.',
                proof='For first b > repeated a, tolerance e requires b*(1-e)<=p<=a*(1+e), hence e >= (b-a)/(b+a), equality p=2ab/(a+b).',
                physical_scope='Per-launch origin = earliest device instruction, terminal core = latest to finish. Model stage durations are additive contributions only. CA queue spans union of actual service descriptors/instructions, removing waits/sync/excluded durations; individual queues can overlap.',
                chips=result)


def plot(data):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/operator-sim-post-repeat-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator
    from matplotlib.lines import Line2D
    fig=plt.figure(figsize=(14.0,8.0))
    axes=fig.subplots(3,2,sharex=True,gridspec_kw={'height_ratios':[1,1.2,1.2],'wspace':.22,'hspace':.68})
    for col,chip in enumerate((DT,PR)):
        rec=data['chips'][chip]
        rows=rec['rows']
        model=axes[0,col]
        for y,components in enumerate([row['model_contributions_us'] for row in rows]+[rec['unknown_model_budget_us']]):
            pos=0.
            for q in PHASES:
                w=components[q]
                model.barh(y,w,left=pos,color=PALETTE[q],edgecolor='none',height=.43,zorder=3)
                pos+=w
            model.text(pos+.065,y,f'{pos:.3f}',va='center',fontsize=10,color='#162e42')
        model.set_yticks([0,1,2],['First','Repeat','History unknown'])
        model.set_ylim(2.55,-.55)
        model.set_title('v0.21 model  ·  critical-path cost contributions',loc='left',fontsize=11,fontweight='bold')
        model.text(0,1.49,chip[6:11],transform=model.transAxes,fontsize=17,fontweight='bold',color='#1c354b')
        for ordinal,row in enumerate(rows):
            ax=axes[ordinal+1,col]
            ax.axvspan(0,row['ca_us'],color='#eff3f6',zorder=0)
            for lane,q in enumerate(QUEUES):
                spans=row['nonwait_queue_union_us'][q]
                ax.broken_barh([(a,b-a) for a,b in spans],(lane-.18,.36),facecolors=PALETTE[q],edgecolors='none',zorder=3)
            rv=row['ca_milestones']['first_vector_operand_us']
            ax.scatter([rv],[2],marker='v',color='#1e1435',s=27,zorder=5)
            for q,field,idx in [('MTE2','dma_read',1),('MTE3','dma_write',3)]:
                for d in row['ca_milestones'][field]:
                    ax.vlines(d['start_us'],idx-.24,idx+.24,color='#213b45',lw=.7,zorder=4)
            ax.axvline(row['ca_us'],ls='--',lw=1.05,color='#304559',zorder=4)
            ax.set_title(f"CA {'first' if ordinal==0 else 'repeat'}  ·  {row['terminal_core']}  ·  {row['ca_us']:.3f} µs",
                         loc='left',fontsize=11,fontweight='bold')
            ax.set_yticks(range(4),['Scalar','MTE2 read','SIMD','MTE3 write'])
            ax.set_ylim(3.5,-.55)
        for ax in axes[:,col]:
            ax.set_xlim(0,5.65)
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.xaxis.grid(color='#d7e0e7',lw=.8,zorder=.5)
            ax.tick_params(axis='both',labelsize=10,colors='#334b5b')
            for side in ['top','right','left']: ax.spines[side].set_visible(False)
    fig.suptitle('mHC Post  |  BF16 [3, 4, 5376]  |  Same-stream repeat',
                 x=.07,y=.985,ha='left',fontsize=20,fontweight='bold',color='#192f40')
    handles=[Patch(facecolor=PALETTE[q],label=l) for q,l in [('arrival','Arrival'),('SCALAR','Scalar'),
                ('MTE2','MTE2'),('SIMD','SIMD'),('MTE3','MTE3'),('other','Other')]]
    handles.append(Line2D([],[],marker='v',ls='None',color='#1e1435',label='First operand RV_VLD'))
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.066,.938),frameon=False,ncol=7,fontsize=10)
    fig.subplots_adjust(top=.765,bottom=.165,left=.113,right=.975)
    fig.supxlabel('Device time from the earliest instruction in each call (µs)',y=.091,fontsize=12)
    fig.text(.07,.041,'Model rows stack additive budgets in display order. CA rows show actual non-wait service on the last core to finish; queues can overlap.',fontsize=10,color='#395364')
    fig.text(.07,.018,'Shading marks the full 48-AIV window, vertical ticks mark DMA issues. The first MTE2 transfer reads 96 B of metadata.',fontsize=10,color='#395364')
    fig.savefig(FIGURE,dpi=180,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    results=collect()
    write(OUT/'paired-result.json',results)
    plot(results)
