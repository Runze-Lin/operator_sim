"""Replay with a suffix only, forgetting every arrival before the preceding shape."""
from collections import defaultdict
from math import ceil
from statistics import mean
from landmarks import ROOT,OUT,read,save
from schedule28 import estimate,load_chip
from score28 import metrics
def main():
    orig=ROOT/'results/v0.26/sink'
    jobs=read(orig/'register.json')['jobs'];receipt=read(orig/'independent.json')['rows'];rs=[]
    for x in receipt:
        seq=sorted((j for j in jobs if j['chip']==x['chip'] and j['group']==x['group']),key=lambda j:j['index'])
        j=seq[x['index']];chip=load_chip(x['chip']);t=j['shape'][0];history=j['prior_h'];k=estimate(j['shape'],j['iters'],chip)
        # Stream could have started earlier. No reset signal and no observed
        # elapsed/active-core measurement serve as pricing inputs.
        lower=0
        if j['index']:
            prev_t=seq[j['index']-1]['shape'][0];lower=ceil(prev_t/ceil(prev_t/chip.vector_cores))
        else:prev_t=None
        recent=estimate(j['shape'],j['iters'],chip,lower,chip.vector_cores)
        nonfirst=estimate(j['shape'],j['iters'],chip,int(bool(j['index'])),chip.vector_cores)
        known=estimate(j['shape'],j['iters'],chip,history,history)
        assert lower<=history
        rs.append(dict(sku=x['chip'],group=x['group'],index=j['index'],t=t,iters=j['iters'],h=history,
            ca_us=x['ca_us'],previous_shape=None if prev_t is None else [prev_t,4],lower_h=lower,upper_h=chip.vector_cores,
            new_possible_states=list(recent['states']),unknown=k['point_us'],nonfirst=nonfirst['point_us'],
            recent=recent['point_us'],known=known['point_us'],range_us=recent['range_us'],source=x['receipt']))
    result={}
    for sku in sorted({x['sku']for x in rs}):
        result[sku]={}
        for g in ('all','previous shape visible','bounded to single state','still ambiguous after a preceding launch'):
            group=[x for x in rs if x['sku']==sku and (g=='all' or x['previous_shape'] is not None and (g=='previous shape visible' or (len(x['new_possible_states'])==1)==(g=='bounded to single state')))]
            result[sku][g]={k:metrics(group,k) for k in ('unknown','nonfirst','recent','known')}
    save(OUT/'one-predecessor.json',dict(protocol='Each v0.26 stream replayed with only the immediately preceding same-provider geometry visible. Lower bounds come exclusively from predicted partition. No reset or CA values available to the price.',rows=rs,groups=result))
if __name__=='__main__':main()
