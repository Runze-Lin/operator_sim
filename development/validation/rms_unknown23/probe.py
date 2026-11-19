"""Enumerate native predecessor states, keeping truth and acquisition metadata outside prediction."""
from functools import lru_cache
import json
from pathlib import Path
import sys
from statistics import mean,median
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate, load_chip
from operator_model.rms_dispatch import price_rms
from operator_model.rms_lane_schedule import schedule


def harmonic(a,b):return 2*a*b/(a+b)


@lru_cache(maxsize=None)
def states(shape,name):
    chip=load_chip(name); old,_,active,d=price_rms(shape,chip,{})
    cold=d['conditional_us']['first'];time=[cold]
    if d['target_plan']['iterations']==1:
        for h in range(1,active+1):
            s,_=schedule(shape,chip,h,d);time.append(sum(s.values()))
    else:
        time.extend([d['conditional_us']['expand']]*(active-1)+[d['conditional_us']['reuse']]);
    return active,chip.vector_cores,time,old


def report(rows):
    groups=defaultdict(list)
    for r in rows:
        a,c,t,old=states(tuple(r['shape']),r['chip']); cold=t[0];re=t[-1]; hot=min(t);upper=max(t);h=r['prior_aiv'];times=t[1:]+[re]*(c-a)
        exp=mean(times)
        choices={'old':old,'H*':t[min(h,a)],'HM':harmonic(hot,upper),'HM_0h':harmonic(re,cold),'AM_0h':(re+cold)/2,'GM_0h':(re*cold)**.5,'AM_full':(hot+upper)/2,'GM_full':(hot*upper)**.5,'AM_avg':(cold+exp)/2,'HM_avg':harmonic(exp,cold),'HM_exp':harmonic(exp,upper),'GM_avg':(exp*cold)**.5,'GM_bound':max(harmonic(hot,upper),min((cold*exp)**.5,hot*1.4)),'clipped':max(.6*upper,min(1.4*hot,old)),'gmclip':max(.6*upper,min(1.4*hot,(cold*exp)**.5))}
        common= ('DT' if '950DT' in r['chip'] else 'PR')
        state='first' if h==0 else 'expand' if h<a else 'hot'
        for group in [common,common+' '+r['provider'],common+' '+state,common+' '+r['source'].split('/')[1]]:groups[group].append((r,choices))
    names=list(choices)
    for group,rr in groups.items():
        print(group,len(rr),' '.join(f'{n}={sum(abs(x[n]-r["ca_us"]) for r,x in rr)/sum(r["ca_us"] for r,x in rr)*100:.1f}/{max(abs(x[n]/r["ca_us"]-1) for r,x in rr)*100:.1f}' for n in names))

if __name__=='__main__':report(json.loads((ROOT/'results/v0.22/rms-unknown/audit.json').read_text())['rows'])
