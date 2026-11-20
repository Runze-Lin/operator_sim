"""Illustrate one frozen MTP8 prediction and the actual last AIC, on the same device clock."""
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate
from development.validation.matmul_dispatch30.device import OUT,ca,save

COLORS={'Control':'#687681','MTE2':'#167b82','MTE1':'#4f77ae','Cube':'#7553a4','FIXP':'#b4502b'}
STAGES=(('launch','Control'),('read_fill','MTE2'),('l0_lead','MTE1'),
        ('page_intervals','Cube'),('last_mmad','Cube'),('write_tail','FIXP'))
NAMES={'MOV_OUT_TO_L1_MULTI_ND2NZ':'MTE2','LOAD_2Dv2':'MTE1','MMAD':'Cube','FIX_L0C_TO_DST':'FIXP'}


def build():
    receipt=OUT/'9-dt-retired.json'
    r=ca.read(receipt);s,=r['stages'];shape=r['shape']
    pred=estimate('matmul',shape,chip=ca.DT)
    assert isclose(pred.latency_us,r['model_us'])
    prices=pred.projection['stages_us']
    assert all(not v for k,v in prices.items() if k not in dict(STAGES))
    model=[]; end=0
    for label,pipe in STAGES:
        us=prices[label]
        if us:model.append(dict(name=label,pipe=pipe,start=end,end=end+us))
        end+=us
    assert isclose(end,pred.latency_us)
    trace=ROOT/s['trace'];assert sha256(trace.read_bytes()).hexdigest()==s['trace_sha256']
    events=[e for e in ca.read(trace)['traceEvents'] if e['ph']=='X']
    origin=min(e['ts'] for e in events)
    core=max(s['windows'],key=lambda c:s['windows'][c][1])
    actual=[dict(name=e['name'],pipe=NAMES[e['name']],start=e['ts']-origin,end=e['ts']+e['dur']-origin)
            for e in events if e['pid']==core and e['name'] in NAMES]
    assert sum(e['pipe']=='Cube' for e in actual)==2
    data=dict(chip=r['chip'],shape=shape,frozen_us=pred.latency_us,ca_us=r['ca_us'],
              trace_sha256=s['trace_sha256'],core=core,model_critical_costs=model,ca_instruction_events=actual,
              scope='Model bands decompose the critical time budget; CA bands show actual issue through retirement. Both start at the earliest instruction over all active cores.')
    save(OUT/'mtp8_example.json',data)
    fig,axes=plt.subplots(2,1,sharex=True,figsize=(10.4,6.4))
    fig.subplots_adjust(left=.17,right=.965,top=.79,bottom=.19,hspace=.42)
    fig.text(.17,.95,'950DT MatMulV3  |  9 × 128 × 512',weight='bold',fontsize=19,color='#23394a')
    fig.text(.17,.903,'Flash indexer.wk  ·  BF16  ·  2 AICs  ·  2 output tiles  ·  4 MMADs across the chip',fontsize=10)
    fig.legend([Patch(color=c) for c in COLORS.values()],COLORS,loc='upper left',bbox_to_anchor=(.16,.88),ncol=5,frameon=False)
    for ax,items,title,total,ys in [(axes[0],model,'Resource model  |  critical-path contributions',pred.latency_us,list(COLORS)),
                                    (axes[1],actual,'CA trace  |  '+core+' (last to finish)',r['ca_us'],list(COLORS)[1:])]:
        ax.set_title(title,loc='left',fontsize=11,weight='bold')
        ax.set_title(f'{total:.3f} µs',loc='right',fontsize=12,weight='bold')
        ax.set_yticks(range(len(ys)),ys);ax.set_ylim(len(ys)-.45,-.55)
        ax.grid(axis='x',color='#dce4e8',lw=.7);ax.set_axisbelow(True)
        ax.spines[['top','right','left']].set_visible(False)
        for i in items:
            y=ys.index(i['pipe'])
            ax.broken_barh([(i['start'],i['end']-i['start'])],(y-.20,.40),
                           facecolors=COLORS[i['pipe']],alpha=.73 if i['pipe']!='Cube' else 1)
            ax.vlines(i['start'],y-.28,y+.28,color=COLORS[i['pipe']],lw=1.25)
        ax.axvline(total,ls='--',lw=1.1,color='#37505b')
    axes[1].set_xlim(0,4.55);axes[1].set_xlabel('Device time since first instruction on any participating core (µs)')
    fig.text(.17,.066,'Top: exposed stage costs, accumulated in dependency order. Bottom: payload descriptors only.',fontsize=9)
    fig.text(.17,.035,'Vertical ticks indicate starts. Control setup, register commands and dependency waits are included in the CA window.',fontsize=9)
    fig.savefig(OUT/'mtp8_example.png',dpi=170);plt.close(fig)


if __name__=='__main__':build()
