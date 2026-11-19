"""Hypothetical hardware sensitivities for the frozen graph; no observations or fitting."""
from copy import deepcopy
import json
from pathlib import Path

from extract import OUT
from model import ROOT, PARAMS, load, _core, predict

CHIP='Ascend970DT'


def evaluate(t,d,chip,prices):
    # Kernel allocations before alignment. A smaller UB cannot launch this schedule.
    if chip['details'].get('ub_bytes',253952)<213152:
        return None
    a=min(t,chip['vector_cores']);q,r=divmod(t,a);n,tail=divmod(d,4096)
    widths=[4096]*n+([tail] if tail else [])
    cohorts=[(q,a-1)]+([(q+1,r-1)] if r else [])
    core=max(_core(rows,widths,rank,chip,prices,False)[0] for rows,rank in cohorts)
    frequency=chip['frequency_ghz'];bw=chip['hbm_bandwidth_gbs']
    reads=t*(10*d+80);writes=t*8*d;l2=chip['details']
    hbm=(reads+max(0,writes-l2['l2_bytes']))*frequency/bw
    shared=(reads+writes)*frequency/l2['l2_bandwidth_gbs']
    return max(core,hbm,shared)/(frequency*1000)


def variations():
    base=load(ROOT/'configs/chips'/f'{CHIP}.json');prices=load(PARAMS)
    result=[]
    names=['base','local MTE bytes/cycle x2','local MTE response setup x0.5','FP32 Vec per-core peak x2',
           'offchip bandwidth x2','AIV count /2 with fixed per-core Vec','UB 192 KiB']
    for label in names:
        c,p=deepcopy(base),deepcopy(prices)
        if label==names[1]:
            for line in p['dma'].values():line['ticks_per_unit']/=2
        if label==names[2]:
            for line in p['dma'].values():line['setup']/=2
        if label==names[3]:c['vector_tops']['32']*=2
        if label==names[4]:c['hbm_bandwidth_gbs']*=2
        if label==names[5]:c['vector_cores']//=2;c['vector_tops']['32']/=2
        if label==names[6]:c['details']['ub_bytes']=196608
        for t,d in [(1,1280),(2,16448),(65,256),(8*32768,7168)]:
            base_us=predict(t,d,CHIP)['device_us']
            us=evaluate(t,d,c,p)
            if label=='base':assert abs(us-base_us)<1e-8
            result.append(dict(scenario=label,tokens=t,hidden=d,base_us=base_us,scenario_us=us,
                               speedup=base_us/us if us else None,measurement=False))
    return dict(target=CHIP,role='counterfactual sensitivity only; inherited 950 prices; no target CA',
                tile=4096,min_ub_bytes_excluding_alignment=213152,scenarios=result)


if __name__=='__main__':
    v=variations();(OUT/'sensitivity.json').write_text(json.dumps(v,indent=2)+'\n')
    for row in v['scenarios']:
        if row['tokens']==8*32768 or row['tokens']==1:
            print(row['scenario'],row['tokens'],round(row['speedup'],4) if row['speedup'] else 'unsupported')
