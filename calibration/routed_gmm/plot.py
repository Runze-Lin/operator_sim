"""Redraw the example's model-budget / physical CA timeline (matplotlib is optional)."""
import argparse
import json
from pathlib import Path

from calibration.mx_matmul.shards import trace_events
from operator_model import estimate

FOLDER=Path(__file__).parent
BUDGET=['control_to_first_payload','first_dependency','overlapped_service','retirement']
RESOURCES=['SCALAR','MTE2','MTE1','CUBE','FIXP']


def extract(raw):
    data=[]
    for sku in ('dt','pr'):
        directory=raw/f'dev-{sku}-00'
        r=json.loads((directory/'receipt.json').read_text())
        cores=r['cores']
        terminal=max(cores,key=lambda c:cores[c]['retire_tick'])
        zero=min(c['first_tick'] for c in cores.values())/1650
        path,=directory.glob(f'profile/OPPROF_*/simulator/{terminal}/trace.json')
        by_pipe={k:[] for k in RESOURCES}
        for event in trace_events(path):
            if event['ph']!='X' or not event['args'].get('detail'):continue
            info=json.loads(event['args']['detail'])
            pipe=info.get('instr_type')
            if (pipe not in by_pipe or not event['dur'] or
                    any(info.get(k) for k in ('is_wait','is_sync','is_duration_excluded'))):
                continue
            start=event['ts']-zero; end=start+event['dur']
            by_pipe[pipe].append([round(start,6),round(end,6)])
        # In each lane show union occupancy; overlapping descriptors are not additive.
        merged={}
        for pipe,intervals in by_pipe.items():
            spans=[]
            for lo,hi in sorted(intervals):
                if spans and lo<=spans[-1][1]:spans[-1][1]=max(spans[-1][1],hi)
                else:spans.append([lo,hi])
            merged[pipe]=spans
        projected=estimate('routed_grouped_matmul',r['shape'],{'expert_counts':r['expert_counts'],
                           'implementation':'aclnnGroupedMatmulV5'},dtype='mxfp4',chip=r['chip'])
        plan=projected.projection['target_plan']
        data.append(dict(chip=r['chip'],shape=r['shape'],counts=r['expert_counts'],critical_core=terminal,
                         model_us=projected.latency_us,ca_us=r['ca_us'],cycles=plan['stage_cycles'],
                         frequency_ghz=plan['target_frequency_ghz'],trace_us=merged))
    return data


def draw(data,output):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,5.2),sharex=True,gridspec_kw={'height_ratios':[1,1.2]})
    bcolors=['#a2b4bd','#678192','#176b76','#b25734']
    ccolors=['#556976','#156a7a','#5275b3','#7855a0','#b25734']
    for col,item in enumerate(data):
        top,low=axes[:,col]
        top.set_title(f"{item['chip']}    {item['model_us']:.3f} µs modeled",fontsize=11)
        start=0
        for idx,key in enumerate(BUDGET):
            span=item['cycles'][key]/(1000*item['frequency_ghz'])
            top.broken_barh([(start,span)],(idx-.24,.48),facecolor=bcolors[idx])
            start+=span
        low.set_title(f"{item['critical_core']}    {item['ca_us']:.3f} µs full CA",fontsize=11)
        for idx,pipe in enumerate(RESOURCES):
            low.broken_barh([(a,b-a) for a,b in item['trace_us'][pipe]],(idx-.25,.5),facecolor=ccolors[idx])
        for ax,lim,n in [(top,item['model_us'],len(BUDGET)),(low,item['ca_us'],len(RESOURCES))]:
            ax.axvline(lim,ls='--',color='#324952',lw=1)
            ax.set_ylim(n-.5,-.5);ax.set_xlim(0,6);ax.grid(axis='x',lw=.45,alpha=.45)
            ax.spines[['top','right']].set_visible(False)
        top.set_yticks(range(4),['Control','First dependency','Exposed service','Retirement'] if col==0 else ['']*4)
        low.set_yticks(range(5),RESOURCES if col==0 else ['']*5)
        low.set_xlabel('Time from earliest issued core event (µs)')
    fig.suptitle('Routed MXFP4 GMM  |  [3, 128, 256]  |  local expert rows [1, 2]',fontsize=15)
    fig.subplots_adjust(left=.15,right=.97,top=.84,bottom=.11,wspace=.16,hspace=.46)
    fig.savefig(output,dpi=140,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path)
    parser.add_argument('--data',type=Path,default=FOLDER/'example-timeline.json')
    parser.add_argument('--output',type=Path,default=FOLDER/'example-timeline.png')
    args=parser.parse_args()
    if args.raw:args.data.write_text(json.dumps(extract(args.raw),separators=(',',':'))+'\n')
    draw(json.loads(args.data.read_text()),args.output)
