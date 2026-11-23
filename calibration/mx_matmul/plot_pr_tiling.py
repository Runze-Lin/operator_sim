"""Redraw the 950PR M=2048 A8W4 worked example from its compact JSON.

python -m calibration.mx_matmul.plot_pr_tiling docs/examples/assets/mx-pr-tiling.json figure.png
Add --extract RAW_CASE --v12-src PATH_TO_1.2_SRC with 1.3 installed to refresh the JSON.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def extract(raw, previous_src):
    from calibration.__main__ import digest, encoded, read, require
    from calibration.msopprof import sha
    from operator_model import estimate

    receipt = read(raw/'receipt.json')
    case = receipt['case']
    m,n,k = case['shape']
    require((m,n,k)==(2048,128,256) and case['chip']=='Ascend950PR_957b'
            and case['contract']=='matmul_mx_mixed', 'Expected the 1.2 M=2048 A8W4 PR example')
    stage, = receipt['stages']
    simulator, = (raw/'profile').glob('OPPROF_*/simulator')
    shards = {p.parent.name:p for p in simulator.glob('core*.*core*/trace.json')}
    require(set(shards)==set(stage['cores']) and
            digest(encoded({name:sha(p) for name,p in shards.items()}))==stage['trace_sha256'],
            'Example trace fingerprint differs from its receipt')
    by_core = {}
    zero, finish = float('inf'), -float('inf')
    for name,p in shards.items():
        core, lane = name.split('.')
        entry = by_core.setdefault(core,{'aic':core})
        for e in read(p)['traceEvents']:
            if e.get('ph') != 'X':
                continue
            start,end = e['ts'], e['ts']+e['dur']
            zero,finish = min(zero,start),max(finish,end)
            inst = e['name']
            if inst not in ('MMAD_MX','MOV_UB_TO_L1','MOV_SRC_TO_DST_ALIGNv2','FIX_L0C_TO_DST'):
                continue
            info = json.loads(e['args']['detail'])
            if inst=='MMAD_MX' and lane=='cubecore0':
                entry.update(m=info['m'],n=info['n'],k=info['k'],mmad=[start,end])
            elif inst=='FIX_L0C_TO_DST' and lane=='cubecore0':
                entry['dst'] = info['dst_addr']
            elif inst=='MOV_UB_TO_L1':
                entry.update(handoff=[start,end],expanded_bytes=info['processed_bytes'])
            elif lane=='veccore0' and info.get('processed_bytes')==n*k//2:
                entry['weight_read'] = [start,end]
    rows=sorted(by_core.values(),key=lambda x:x['dst'])
    output_base=rows[0]['dst']
    last=0
    for r in rows:
        # A BF16 output row has 2*N bytes. Recover each true M offset from
        # the physical FIXP destination rather than assuming core id order.
        r['m_begin'],rem=divmod(r.pop('dst')-output_base,2*n)
        require(rem==0 and r['m_begin']==last and r['n']==n and r['k']==k,
                'Output rows or reduction fragments differ from the worked example')
        last+=r['m']
        for name in ('handoff','mmad','weight_read'):
            r[name]=[round(t-zero,6) for t in r[name]]
    require(last==m,'Incomplete output grid')
    current=estimate(case['op'],case['shape'],case['attrs'],dtype=case['dtype'],chip=case['chip'])
    old_code = ("import json,sys; from operator_model import estimate,__version__; "
                "c=json.load(sys.stdin); assert __version__.startswith('1.2.'); "
                "r=estimate(c['op'],c['shape'],c['attrs'],dtype=c['dtype'],chip=c['chip']); "
                "print(json.dumps([r.active_aic,r.projection['target_plan']['tile'],r.latency_us]))")
    old=json.loads(subprocess.check_output([sys.executable,'-c',old_code],input=json.dumps(case),text=True,
                                          env={**os.environ,'PYTHONPATH':str(previous_src.resolve())},cwd=raw))
    return {'chip':case['chip'],'shape':case['shape'],'kernel':stage['kernel'],
            'physical_shards_sha256':stage['trace_sha256'],
            'old':{'active_aic':old[0],'tile':old[1],'us':old[2]},
            'current':{'active_aic':current.active_aic,'tile':current.projection['target_plan']['tile'],
                       'us':current.latency_us}, 'ca_us':round(finish-zero,6),'rows':rows}


def render(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    m,n,k=data['shape']
    rows=data['rows']
    fig=plt.figure(figsize=(11,7.5),facecolor='white')
    grid=fig.add_gridspec(2,1,left=.205,right=.97,top=.818,bottom=.195,height_ratios=[1,2.35],hspace=.60)
    spatial=fig.add_subplot(grid[0]); timeline=fig.add_subplot(grid[1])
    ink='#193043'
    fig.text(.055,.933,f'950PR A8W4 Matmul   |   {m:,} × {n} × {k}',fontsize=20,weight='bold',color=ink)
    fig.text(.055,.89,'BF16 output   ·   One scaled FP8 MMAD per working AIC',fontsize=11,color='#415968')
    spatial.set_title('Output rows assigned to each working AIC',loc='left',color=ink,weight='bold',fontsize=12,pad=9)
    lines=[]
    for field,label in [('old','v1.2 model'),('current','v1.3 model')]:
        ref=data[field]
        h=ref['tile'][0]
        lines.append((f"{label}  ·  {ref['active_aic']} AICs", [(i,min(h,m-i)) for i in range(0,m,h)],ref['us']))
    lines.append((f'CA  ·  {len(rows)} AICs',[(r['m_begin'],r['m']) for r in rows],data['ca_us']))
    palettes=[('#74889a','#a5b3bf'),('#136f78','#62acb0'),('#136f78','#62acb0')]
    for i,(label,bands,us) in enumerate(lines):
        for j,(lo,h) in enumerate(bands):
            col='#bf6638' if h<bands[0][1] else palettes[i][j%2]
            spatial.barh(2-i,h,left=lo,height=.65,color=col,edgecolor='white',lw=.5)
        spatial.text(m+32,2-i,f'{us:.3f} µs',va='center',fontsize=9,color=ink)
    spatial.set_yticks([2,1,0],[line[0] for line in lines]); spatial.tick_params(axis='y',length=0,pad=9)
    spatial.set_xlim(0,m*1.12); spatial.set_xticks([0,512,1024,1536,2048])
    spatial.set_xlabel('M coordinate   ·   full N = 128 on every block',fontsize=10)
    spatial.spines[:].set_visible(False)
    colors={'weight_read':'#16818b','handoff':'#db9955','mmad':'#7353a5'}
    timeline.set_title('CA descriptors by output block',loc='left',color=ink,weight='bold',fontsize=12,pad=10)
    for row in rows:
        i=row['m_begin']
        for field,col in colors.items():
            start,end=row[field]
            timeline.broken_barh([(start,end-start)],(i+row['m']*.13,row['m']*.74),facecolors=col)
    timeline.set_ylim(m,0);timeline.set_xlim(0,6.1)
    timeline.set_yticks([0,512,1024,1536,2048])
    timeline.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{int(x):,}'))
    timeline.set_ylabel('M coordinate')
    timeline.set_xlabel('Device time from first physical-core event (µs)')
    timeline.axvline(data['ca_us'],ls='--',lw=1,color='#344650')
    timeline.text(data['ca_us']-.04,55,f"{data['ca_us']:.3f} µs",color=ink,rotation=90,ha='right',va='top',fontsize=8)
    timeline.grid(axis='x',color='#dde6e9',lw=.6)
    timeline.set_axisbelow(True);timeline.spines[['top','right','left']].set_visible(False)
    labels=['Packed W read   ·   MTE2','Expanded W to L1   ·   MTE3','Scaled product   ·   MMAD']
    fig.legend(handles=[Patch(color=col,label=label) for col,label in zip(colors.values(),labels)],
                    loc='lower left',bbox_to_anchor=(.195,.069),frameon=False,ncol=3,fontsize=9,
                    handlelength=1.15,columnspacing=1.65)
    fig.text(.055,.055,'Each CA band covers its output rows. The orange end block above has 48 rows; the other 25 have 80.',fontsize=9,color=ink)
    expanded=sum(r['expanded_bytes'] for r in rows)
    fig.text(.055,.031,f'Each producer hands {rows[0]["expanded_bytes"]//1024} KiB of FP8 weights to L1 ({expanded//1024} KiB total). Descriptor durations include dependency waits.',fontsize=9,color=ink)
    fig.savefig(output,dpi=150,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path)
    parser.add_argument('figure',type=Path)
    parser.add_argument('--extract',type=Path,help='Verified raw directory of the 1.2 M=2048 mixed PR case')
    parser.add_argument('--v12-src',type=Path,help='Source directory of the released 1.2 Python package')
    args=parser.parse_args()
    if args.extract:
        if not args.v12_src:
            parser.error('--extract requires --v12-src')
        data=extract(args.extract,args.v12_src)
        args.data.write_text(json.dumps(data,indent=2)+'\n')
    render(json.loads(args.data.read_bytes()),args.figure)
