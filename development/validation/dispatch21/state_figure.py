"""Compare pre-launch state forecasts with the measured terminal-AIV issue timeline."""
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import sys
from math import isclose

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate
from development.validation.sku19.campaign import digest,read,write
from development.validation.sku20.timelines import union
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

COLORS=dict(SCALAR='#607685',MTE2='#15778a',SIMD='#7453a6',MTE3='#b65327')


def extract(op):
    if op=='rmsnorm':
        row=next(r for r in read(ROOT/'results/v0.12/rms/rows.json') if r['shape']==[3,6656])
    else:
        row=next(r for r in read(ROOT/'results/v0.12/post/rows.json') if r['shape']==[16,4,256])
    path=ROOT/row['db_source'];sha=digest(path)
    assert sha==row['db_sha256']
    with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as db:
        info=dict(db.execute('SELECT KernelInfoAttr, KernelInfoVal FROM KernelInfo'))
        clock=info['exec_time_ticks']/(info['exec_time_ns']/1000)
        t,=db.execute('SELECT MIN(ExecInstrTickStart) FROM ExecutedInstructions').fetchone()
        core,kind,last=db.execute('SELECT ExecInstrCoreId,CoreTypeId,MAX(ExecInstrTickEnd) AS last FROM ExecutedInstructions GROUP BY ExecInstrCoreId,CoreTypeId ORDER BY last DESC LIMIT 1').fetchone()
        actual=defaultdict(list)
        for first,end,raw in db.execute('SELECT ExecInstrTickStart,ExecInstrTickEnd,ExtendParams FROM ExecutedInstructions WHERE ExecInstrCoreId=? AND CoreTypeId=? AND ExecInstrIsWait=0 AND ExecInstrIsSynchronization=0 AND ExecInstrIsDurationExcluded=0',(core,kind)):
            pipe=json.loads(raw or '{}').get('instr_type','')
            if pipe.startswith(('RVEC','SIMD')):pipe='SIMD'
            if pipe in COLORS and end>first:actual[pipe].append(((first-t)/clock,(end-t)/clock))
        ca=(last-t)/clock
        assert isclose(ca,row['reference_us'],abs_tol=.002)
    shape=row['shape']; attrs=row['attrs']; r=estimate(op,shape,attrs); diag=r.projection['timing_diagnostics']
    state=diag.get('conditional_us') or diag['context_predictions_us']
    frozen_h=row['prior_aiv']
    chosen=estimate(op,shape,{**attrs,'resident_aiv':frozen_h})
    file=ROOT/'results/v0.21/dispatch'/f'{op}-state.json'
    write(file,dict(op=op,shape=shape,model_us=r.latency_us,ca_us=ca,known_us=chosen.latency_us,
          states=state,prior_aiv=frozen_h,core=f'AIV {core}, type {kind}',kernel=info['name'],stages=diag['stages_us'],
          actual={k:union(v) for k,v in actual.items()},trace_sha256=sha,trace=str(path.relative_to(ROOT))))
    return file


def render(file):
    d=read(file); fig,(a,b)=plt.subplots(2,1,figsize=(11,6.8),sharex=True)
    fig.subplots_adjust(top=.82,left=.19,right=.96,bottom=.15,hspace=.6)
    fig.text(.19,.94,'950DT '+{'mhc_post':'mHC Post','rmsnorm':'RMSNorm'}[d['op']]+'  |  '+' × '.join(map(str,d['shape'])),fontsize=20,weight='bold',color='#1a323e')
    fig.text(.19,.89,f"Previous launches covered {d['prior_aiv']} AIVs in this stream. Unknown-history forecast uses only shape and chip.",fontsize=11,color='#435d6a')
    for ax in (a,b):
        ax.spines[:].set_visible(False);ax.grid(axis='x',color='#dae4e8',linewidth=.7);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
    a.set_title('Resource model · alternative dispatch windows',loc='left',weight='bold')
    a.set_title(f"shape only {d['model_us']:.3f} µs",loc='right',weight='bold')
    for y,k in enumerate(('reuse','expand','first')):
        a.barh(y,d['states'][k],height=.38,color=['#24758a','#7594a4','#9baebb'][y])
        a.text(d['states'][k]+.04,y,f"{d['states'][k]:.3f}",va='center',fontsize=9)
    a.set_yticks([0,1,2],['Hot reuse','Expand cores','First launch']);a.set_ylim(-.6,2.6)
    a.axvline(d['model_us'],color='#7352a0',ls='--',lw=1.4)
    a.axvline(d['known_us'],color='#18748b',ls=':',lw=1.6)
    b.set_title(f"CA trace · last to finish ({d['core']})",loc='left',weight='bold')
    b.set_title(f"{d['ca_us']:.3f} µs",loc='right',weight='bold')
    for y,k in enumerate(reversed(COLORS)):
        for st,end in d['actual'].get(k,[]):b.barh(y,end-st,left=st,height=.43,color=COLORS[k],linewidth=0)
    b.set_yticks(range(4),list(reversed(COLORS)));b.set_ylim(-.5,3.6);b.axvline(d['ca_us'],color='#455c68',ls='--',lw=1)
    b.set_xlim(0,max(*d['states'].values(),d['ca_us'])*1.14)
    b.set_xlabel('Device time since first instruction (µs)')
    fig.text(.19,.055,'Upper dashed: unknown history. Dotted: previous AIV count supplied. Lower bars: measured non-wait instruction intervals.',fontsize=9,color='#435d6a')
    fig.savefig(ROOT/'docs/examples/figures'/f"{d['op']}-v21-state.png",dpi=145,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    for op in ('rmsnorm','mhc_post'):
        f=extract(op) if '--extract' in sys.argv else ROOT/'results/v0.21/dispatch'/f'{op}-state.json'
        render(f)
