"""Exact feasibility for paired protocols with max relative error and per-protocol WAPE."""
from fractions import Fraction as F
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import read,write,DT,PR
OUT=ROOT/'results/post-candidate'


def solve(groups):
    # With low_i<=high_i a forecast outside the measured pair can be clamped
    # into it without hurting any criterion. Absolute errors then linearize.
    low=[F(str(g['low_us'])) for g in groups]
    high=[F(str(g['high_us'])) for g in groups]
    A=sum(low);B=sum(high);cap=F(2,5);w=F(1,5)
    lower=[max(a,(1-cap)*b) for a,b in zip(low,high)]
    upper=[min(b,(1+cap)*a) for a,b in zip(low,high)]
    L=sum(lower);U=sum(upper)
    target=(2*A*B)/(A+B)
    if any(l>u for l,u in zip(lower,upper)):
        raise ValueError('Max constraint itself infeasible; update audit if new archive includes such pair')
    minimax_sum=min(U,max(L,target))
    best=max((minimax_sum-A)/A,(B-minimax_sum)/B)
    oklo=max(L,(1-w)*B);okhi=min(U,(1+w)*A)
    # This witness only certifies existence for the observed set. Its labels
    # are unknown to deployment and must never be used to predict new shapes.
    f=(minimax_sum-L)/(U-L) if U>L else F(0)
    p=[l+(u-l)*f for l,u in zip(lower,upper)]
    assert max(sum(b-t for b,t in zip(high,p))/B,sum(t-a for t,a in zip(p,low))/A)==best
    assert all(abs(t-b)/b<=cap and abs(t-a)/a<=cap for t,a,b in zip(p,low,high))
    return dict(groups=groups,independent_shape_variables=len(groups),
                low_total_us=float(A),high_total_us=float(B),per_protocol_feasible=oklo<=okhi,
                required_sum_interval_us=list(map(float,[oklo,okhi])),
                signed_feasibility_margin_us=float(okhi-oklo),
                minimum_worst_protocol_wape_pct_at_max40=100*float(best),
                pooled_wape_pct_for_between_endpoints=100*float((B-A)/(B+A)),
                witness_sum_us=float(sum(p)),witness_us=list(map(float,p)),
                witness_low_wape_pct=100*float((sum(p)-A)/A),witness_high_wape_pct=100*float((B-sum(p))/B),
                witness_largest_err_pct=100*float(max(max(abs(t-b)/b,abs(t-a)/a) for t,a,b in zip(p,low,high))),
                min_worst_single_shape_relative_pct=[100*float((b-a)/(b+a)) for a,b in zip(low,high)])


def run():
    ev=read(OUT/'phase-evidence.json')
    result={}
    for sku in (DT,PR):
        rows=[r for r in ev if r['chip']==sku and r['binary']]
        groups=[]
        for shape in sorted(set(tuple(r['shape']) for r in rows)):
            same=[r for r in rows if tuple(r['shape'])==shape]
            if len(same)!=2:continue
            cold,chain=sorted(same,key=lambda r:r['state']=='after_pre_sinkhorn')
            assert cold['state']=='first' and chain['state']=='after_pre_sinkhorn'
            assert (cold['kernel'],cold['binary'],cold['active'],cold['backend'])==(chain['kernel'],chain['binary'],chain['active'],chain['backend'])
            assert cold['us']>=chain['us']
            groups.append(dict(shape=list(shape),low_us=chain['us'],high_us=cold['us'],
                               kernel=cold['kernel'],binary_sha256=cold['binary'],active_aiv=cold['active'],
                               low_trace_sha256=chain['sha256'],high_trace_sha256=cold['sha256']))
        result[sku]=dict(exact_pair_set=solve(groups),individual=[solve([g]) for g in groups])
    compact=[r for r in ev if r['chip']==DT and r['shape']==[2,4,6144]]
    assert len(compact)==2 and {r['state'] for r in compact}=={'first','reuse'}
    a,b=sorted(compact,key=lambda r:r['us'])
    c=solve([dict(shape=a['shape'],low_us=a['us'],high_us=b['us'],kernel=a['kernel'],binary_sha256=None,
                  active_aiv=a['active'],low_trace_sha256=a['sha256'],high_trace_sha256=b['sha256'])])
    write(OUT/'protocol-feasibility.json',dict(
          scope='v0.12/v0.17 archive only. Equal counts within each first vs after-producer-chain pair. Same chip, inputs/attrs, ELF hash and number of active AIV. CANN 9.2.1 BF16 N=4, mandatory residual. No v0.21 holdout labels used.',
          method='Exact rational arithmetic on archived microsecond decimal strings. For low=a, high=b, p clamp to [a,b]. Under max 40%, per-shape interval [l,u]=[max(a,.6b),min(b,1.4a)]. Dual WAPE <=20 iff max(sum(l),.8*sum(b)) <= min(sum(u),1.2*sum(a)). For minimax WAPE clamp 2AB/(A+B) to [sum(l),sum(u)], A=sum(a),B=sum(b). This ORACLE feasibility assigns one free p per observed shape and does not establish a generalizable estimator.',
          exact_same_elf_pairs=result,compact_post_only_separate=c,
          limitations='A producer chain Post is a distinct launch history from repeated identical Post. No same-ELF repeated Post-only first/reuse pair on either chip with verified hash in this historical corpus. Compact DT Post-only example has same API and entry suffix and no per-launch ELF hash. Conditional prediction can distinguish histories; single forecast theorem applies only when history is withheld.'))
    for sku,z in result.items():
        v=z['exact_pair_set'];print(sku,v['per_protocol_feasible'],v['minimum_worst_protocol_wape_pct_at_max40'],v['required_sum_interval_us'],v['witness_us'])
if __name__=='__main__':run()
