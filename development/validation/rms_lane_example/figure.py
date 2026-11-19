"""Rebuild the RMSNorm v0.22 diagram from two independently verified ordered CA calls."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate,load_chip
from operator_model.rms_lane_schedule import parameters

CASES = ((4,4800),(34,1904))
CHIP = 'Ascend950DT_9582'
DATA = ROOT/'results/v0.22/rms-lane-example.json'
FIG = ROOT/'docs/examples/figures/rms-v022-lane-sector.png'
RECEIPT = ROOT/f'results/rms-dispatch-audit/verified-ordered-{CHIP}.json'


def merged(spans):
    result=[]
    for a,b in sorted(spans):
        if result and a<=result[-1][1]:
            result[-1][1]=max(result[-1][1],b)
        else:
            result.append([a,b])
    return result


def extract(archive):
    data=[]
    hz=load_chip(CHIP).frequency_ghz*1000
    runs=json.loads(RECEIPT.read_text())['rows']
    for shape in CASES:
        r,=[r for r in runs if tuple(r['shape'])==shape]
        assert r['timing_qualified'] and all('bad=0' in s for s in r['correctness'])
        s=r['stage']
        core=max(s['windows'],key=lambda c:s['windows'][c][1])
        trace=archive/s['trace']
        assert sha256(trace.read_bytes()).hexdigest()==s['trace_sha256']
        start=s['first_tick']/hz
        first_address=None
        statistics=[]
        phases={}
        for c,facts in s['cores'].items():
            shard=trace.parent/c/'trace.json'
            buf=shard.read_bytes()
            assert sha256(buf).hexdigest()==facts['shard_sha256']
            events=[e for e in json.loads(buf)['traceEvents'] if e.get('ph')=='X']
            assert abs(min(e['ts'] for e in events)-facts['first_tick']/hz)<1e-4
            active={q:[] for q in ('SCALAR','MTE2','SIMD','MTE3_rstd','MTE3_y')}
            stores=[]
            for e in events:
                d=json.loads(e['args']['detail'])
                if any(d.get(k) for k in ('is_wait','is_sync','is_duration_excluded')) or e['dur']<=0:
                    continue
                q=d['queue_type']
                t=e['ts']-start
                if q in ('MTE2','MTE3') and not e['name'].startswith('MOV_SRC_TO_DST'):
                    continue
                if q=='MTE3':
                    q='MTE3_rstd' if d['processed_bytes']==4 else 'MTE3_y'
                    stores.append((q,d['dst_addr'],d['processed_bytes'],t,t+e['dur']))
                if d['instr_type'].startswith(('RVEC','SIMD')):
                    q='SIMD'
                if c==core and q in active:
                    active[q].append((t,t+e['dur']))
            small,large=sorted(stores,key=lambda z:z[3])
            assert small[0]=='MTE3_rstd' and large[0]=='MTE3_y' and large[2]==2*shape[1]
            address=small[1]
            first_address=address if first_address is None else min(first_address,address)
            statistics.append((address,small[3],small[4]))
            if c==core:
                phases={q:merged(t) for q,t in active.items()}
        assert len(statistics)==shape[0] and first_address%128==0
        a=estimate('rmsnorm',shape,{'resident_aiv':r['prior_aiv']},chip=CHIP)
        assert abs(a.latency_us-r['candidate_known_us'])<1e-10
        diag=a.projection['timing_diagnostics']['lane_schedule']
        sector=diag['critical_sector']
        writes=sorted([(i,b,e) for i,b,e in statistics if (i-first_address)//128==sector])
        intervals={};t=0.
        for name,duration in a.projection['stages_us'].items():
            intervals[name]=[t,t+duration]
            t+=duration
        # Show actual rstd-sector retirements on CA; the model can identify
        # individual retirement cadence when that chain is critical.
        markers=[]
        if diag['critical_path']=='rstd':
            finish=intervals['MTE3_rstd'][1]
            markers=[finish+k*parameters()['sector_interval']/hz for k in range(diag['retirement_successors']+1)]
            assert abs(markers[-1]-intervals['sector_retirement'][1])<1e-10
        data.append(dict(shape=shape,H=r['prior_aiv'],chip=CHIP,model_us=a.latency_us,
                   measured_us=s['ca_us'],critical_path=diag['critical_path'],critical_sector=sector,
                   critical_lane_class=diag['lane_class'],old_known_us=r['previous_known_us'],
                   unknown_us=r['previous_unknown_us'],model_stages=intervals,model_retirements=markers,
                   terminal_core=core,terminal_start=s['windows'][core][0],ca_nonwait=phases,
                   measured_sector=[dict(offset=addr-first_address,issue=b,done=e) for addr,b,e in writes],
                   source_trace=s['trace'],source_sha256=s['trace_sha256'],kernel=s['kernel'],
                   binary_sha256=s['binary_sha256'],goldens=r['correctness']))
    DATA.write_text(json.dumps(data,indent=2)+'\n')


def plot():
    os.environ.setdefault('MPLCONFIGDIR','/tmp/rms-lane-example-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    data=json.loads(DATA.read_text())
    fig,axes=plt.subplots(2,2,figsize=(14,7.7),sharex=True,sharey=True)
    color=['#61778b','#16828a','#7b57a9','#426aa7','#b75a32','#426aa7']
    labels=['Front end / scalar','MTE2','SIMD','rstd write','y write','rstd retirements']
    names={'arrival':0,'frontend_to_rstd_issue':0,'MTE2':1,'SIMD_normalize':2,'MTE3_rstd':3,
           'sector_retirement':3,'MTE3_y':4,'drain':0}
    for col,row in enumerate(data):
        top,bottom=axes[:,col]
        for k,(a,b) in row['model_stages'].items():
            lane=names[k]
            c='#a6b2bd' if k in ('arrival','drain') else color[lane]
            top.barh(lane,b-a,left=a,height=.32,color=c,zorder=3)
        for t in row['model_retirements']:
            top.vlines(t,4.8,5.2,color=color[5],lw=1.1)
        for lane,q in enumerate(('SCALAR','MTE2','SIMD','MTE3_rstd','MTE3_y')):
            bottom.broken_barh([(a,b-a) for a,b in row['ca_nonwait'][q]],(lane-.16,.32),
                               facecolors=color[lane],edgecolors='none',zorder=3)
        for event in row['measured_sector']:
            bottom.vlines(event['done'],4.8,5.2,color=color[5],lw=1.1)
        for (ax,total,title) in ((top,row['model_us'],'v0.22 model  ·  '+row['critical_path']+' path'),
                    (bottom,row['measured_us'],'CA  ·  '+row['terminal_core'])):
            ax.axvline(total,ls='--',lw=1.1,color='#283f50')
            ax.set_title(title,loc='left',fontsize=12,weight='bold')
            ax.set_title(f'{total:.3f} µs',loc='right',fontsize=12,weight='bold')
            ax.grid(axis='x',color='#d5e0e7',lw=.8)
            ax.set_xlim(0,5.35)
            ax.set_yticks(range(6),labels,fontsize=10)
            ax.set_ylim(5.5,-.65)
            ax.tick_params(axis='y',length=0)
            for side in ('left','top','right'):ax.spines[side].set_visible(False)
        top.text(0,1.27,f"{row['shape'][0]} × {row['shape'][1]}   |   {row['H']} → {row['shape'][0]} AIVs",
                 transform=top.transAxes,fontsize=16,weight='bold',color='#19394a')
    fig.suptitle('950DT RMSNorm   |   A new sibling lane and a contested statistics sector',x=.055,y=.98,
                 ha='left',fontsize=19,weight='bold',color='#163446')
    fig.legend(handles=[Patch(color=color[1],label='MTE2'),Patch(color=color[2],label='SIMD'),
                       Patch(color=color[3],label='MTE3 / rstd'),Patch(color=color[4],label='MTE3 / y'),
                       Line2D([],[],ls='None',marker='|',color=color[5],label='Sector 0 retirement')],
               ncol=5,frameon=False,loc='upper left',bbox_to_anchor=(.049,.925))
    fig.subplots_adjust(left=.15,right=.97,top=.755,bottom=.18,hspace=.43,wspace=.20)
    fig.supxlabel('Device time from the earliest instruction in each launch (µs)',y=.095)
    fig.text(.055,.045,'Model bars show the governing dependency chain. CA bars show actual non-wait instructions and full write descriptors on the last-finishing core.',fontsize=10,color='#425b69')
    fig.text(.055,.02,'CA retirement ticks include all writers to the same 128 B rstd sector. Modeled retirement ticks appear only when this sector determines completion.',fontsize=10,color='#425b69')
    fig.savefig(FIG,dpi=170,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive-root',type=Path,help='Workspace containing the original runs/ tree. Omit to use the verified compact data.')
    args=parser.parse_args()
    if args.archive_root:
        extract(args.archive_root)
    plot()
