"""Align the two RMSNorm output stores by logical row in model and CA."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model import load_chip
from operator_model.rms_dispatch import price_rms
from operator_model.rms_history import _lane_prices
from operator_model.rms_lane_schedule import _launch_class, parameters
from development.validation.rms_ordered_verify.verify import digest, events

OUT = ROOT/'results/v0.24/rms-prefix'
GOLDENS = ROOT/'results/rms-unknown23/verified'
RSTD, Y = 0x4f38, 0x5128


def extract(chip_name):
    chip = load_chip(chip_name)
    clock = chip.frequency_ghz * 1000
    receipt = GOLDENS/('sector-'+chip.name+'.json')
    r, = [r for r in json.loads(receipt.read_text())['rows'] if r['shape']==[34,1664] and r['role']=='entry']
    assert r['timing_qualified'] and all('bad=0' in s for s in r['correctness'])
    shape = r['shape']
    stage = r['stage']
    measured = {}
    trace = ROOT/stage['trace']
    assert digest(trace) == stage['trace_sha256']
    base = int((trace.parent.parent/'dump/pc_start_addr.txt').read_text(),16)
    for core,meta in stage['cores'].items():
        shard = trace.parent/core/'trace.json'
        assert digest(shard) == meta['shard_sha256']
        stores = {}
        for event in events(shard):
            pc = int(event['args']['pc_addr'],16)-base
            if pc not in (RSTD,Y) or not event['name'].startswith('MOV_SRC_TO_DST'):
                continue
            desc = json.loads(event['args']['detail'])
            assert pc not in stores
            assert desc['processed_bytes'] == (4 if pc==RSTD else shape[1]*2)
            issue = event['ts']-stage['first_tick']/clock
            stores[pc] = [desc['dst_addr'],issue,issue+event['dur']]
        measured[core] = stores
    rbase = min(s[RSTD][0] for s in measured.values())
    ybase = min(s[Y][0] for s in measured.values())
    assert rbase % 128 == 0
    actual = {}
    for core, stores in measured.items():
        row, remainder = divmod(stores[RSTD][0]-rbase,4)
        assert not remainder and stores[Y][0]-ybase == row*shape[1]*2 and row not in actual
        actual[row] = dict(core=core,rstd=stores[RSTD][1:],y=stores[Y][1:])

    total,_,active,diag = price_rms(shape,chip,{'resident_aiv':r['prior_aiv']})
    assert diag['target_plan']['rows_per_core']==1 and set(actual)==set(range(shape[0]))
    p = parameters()
    prices = _lane_prices(1,shape[1],active,chip,diag,p)
    groups = {}
    model = {}
    for row in sorted(actual):
        state = _launch_class(row,active,r['prior_aiv'],(chip.vector_cores+chip.cube_cores-1)//chip.cube_cores)
        groups.setdefault((row//32,state),[]).append(row)
    for (_,state),rows in groups.items():
        for rank,row in enumerate(rows,1):
            ready,y_delay = prices[state]
            issue = ready + p['spread'].get(state,0)*(rank/(len(rows)+1)-.5)
            y_latency = p['y_cycles'][0]+diag['write_cycles_per_kib']*2*shape[1]/1024 if chip.details.get('mte3_bandwidth_gbs') else p['y_cycles'][0]+p['y_cycles'][1]*2*shape[1]/1024
            model[row] = dict(state=state,rstd=[issue/clock,None],y=[(issue+y_delay-y_latency)/clock,(issue+y_delay)/clock])
    for sector in {row//32 for row in actual}:
        last = 0
        for row in sorted((row for row in actual if row//32==sector),key=lambda row:(model[row]['rstd'][0],row)):
            issue = model[row]['rstd'][0]*clock
            last = max(issue+p['solo'], last+p['sector_interval'])
            model[row]['rstd'][1] = last/clock
    end = max(max(l['y'][1],l['rstd'][1]) for l in model.values())+p['drain']/clock
    assert abs(total-end)<1e-9 and abs(total-r['known_us'])<1e-9
    assert abs(max(max(l['y'][1],l['rstd'][1]) for l in actual.values())-stage['ca_us']) < .1
    return dict(chip=chip.name,shape=shape,H=r['prior_aiv'],active_aiv=active,model_us=total,ca_us=stage['ca_us'],
                logical_rows=[dict(row=i,sector=i//32,model=model[i],ca=actual[i]) for i in sorted(actual)],
                receipt=str(receipt.relative_to(ROOT)),receipt_sha256=digest(receipt),trace=stage['trace'],trace_sha256=stage['trace_sha256'],
                kernel=stage['kernel'],goldens=r['correctness'],source='Archived development replay. Time zero is the earliest instruction on any AIV.')


def draw(data, path):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/operator-sim-rms-prefix24')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator
    selected = [0,7,15,23,30,31,32,33]
    blue,orange,ink = '#196e83','#bb612e','#182e3b'
    fig,axes = plt.subplots(2,1,figsize=(11.3,8.4),sharex=True)
    left = .19
    fig.subplots_adjust(left=left,right=.97,top=.79,bottom=.14,hspace=.38)
    fig.suptitle(f"{data['chip'][6:11]} RMSNorm  |  34 rows × 1,664 channels",x=left,y=.977,ha='left',color=ink,fontsize=21,fontweight='bold')
    fig.text(left,.911,'34 AIVs  ·  one row per AIV  ·  17 previously resident  ·  two output sectors for rstd',color='#425967',fontsize=11)
    fig.legend(handles=[Patch(color=blue,label='rstd · 4 B per row'),Patch(color=orange,label='y · 3,328 B per row'),
                        Line2D([],[],color=ink,ls='--',label='Full kernel end')],frameon=False,ncol=3,loc='upper left',bbox_to_anchor=(left-.008,.899))
    for (ax,side,total,title) in zip(axes,('model','ca'),(data['model_us'],data['ca_us']),('Resource model · issue to retirement','CA trace · same output rows by GM address')):
        ax.set_axisbelow(True)
        ax.grid(axis='x',color='#dce4ea')
        ax.set_yticks(range(len(selected)),[f'row {r}  ·  sector {r//32}' for r in selected],fontsize=9)
        ax.set_ylim(len(selected)-.45,-.55)
        ax.set_xlim(0, max(data['model_us'],data['ca_us'])*1.045)
        ax.xaxis.set_major_locator(MultipleLocator(.5))
        ax.axhline(5.5,color='#c1ced4',lw=.9)
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0,pad=6)
        ax.set_title(title,loc='left',fontsize=14,fontweight='bold',color=ink)
        ax.set_title(f'{total:.3f} µs',loc='right',fontsize=14,fontweight='bold',color=ink)
        ax.axvline(total,color=ink,ls=(0,(3,3)),lw=1)
        for i,row in enumerate(selected):
            events=data['logical_rows'][row][side]
            for shift,name,color in [(-.14,'rstd',blue),(.14,'y',orange)]:
                a,b=events[name]
                ax.barh(i+shift,b-a,left=a,height=.21,color=color)
    axes[-1].set_xlabel('Device time from the earliest instruction on any AIV (µs)')
    fig.text(left,.048,'Rows 0–31 share a 128 B statistics sector. Rows 32–33 write the next sector. Eight of 34 rows shown.\nBars span issue to completion, including waits. Per-row model order within each launch class is approximate.',fontsize=9,color='#425967')
    fig.savefig(path,dpi=150,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    for chip in ('Ascend950DT_9582','Ascend950PR_957b'):
        row = extract(chip)
        name = ('dt' if 'DT' in chip else 'pr')+'-rms-output-rows'
        (OUT/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n')
        draw(row,OUT/(name+'.png'))
