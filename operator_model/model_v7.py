"""Bounded local latency models; inference never opens a trace or invokes CANN.

The original three families retain the frozen v6 physical models. New library
routes use nonnegative startup + critical-work proxies. This is local evidence,
not a claim that two calibration points recover a hardware throughput model.
"""
import math
from dataclasses import dataclass
from .model_v6 import Case as BaseCase

@dataclass(frozen=True)
class Case:
    op: str
    shape: tuple


def contract_attrs(op, shape, attrs):
    result={k:v for k,v in attrs.items() if k!='expert_counts'}
    if op=='addmm' and result.get('self_shape')==[shape[1]]:result['self_shape']=['N']
    return result

from .fast_query import Predictor as BasePredictor


def work(op, shape, attrs):
    s=tuple(shape)
    if not s or any(type(x) is not int or x<=0 for x in s):raise ValueError('positive integer shape required')
    ceil=lambda x,y:(x+y-1)//y
    if op=='rmsnorm':
        r,c=s;return ceil(r,64)*ceil(c,128)
    if op=='addmm':
        m,n,k=s;return ceil(ceil(m,16)*ceil(n,64),32)*ceil(k,16)
    if op=='lightning_indexer':
        b,q,l,g,d=s;return ceil(b*q,32)*ceil(l,512)*g*ceil(d,128)
    if op in ('mhc_pre','mhc_pre_sinkhorn','mhc_post','mhc'):
        t,n,d=s;waves=ceil(t,32 if op=='mhc_pre_sinkhorn' else 64)
        if op=='mhc_post':return waves*n*(n+1)*ceil(d,64)
        projection=n*d*(n*n+2*n)/128
        if op=='mhc_pre':return waves*projection
        return waves*(projection+attrs.get('num_iters',20)*n*n)
    if op=='mhc_sinkhorn':
        t,n=s;return ceil(t,64)*attrs.get('num_iters',20)*n*n
    if op in ('kda_recurrent','kda_chunk'):
        b,seq,hq,hv,k,v=s
        # These models are calibrated only at fixed B/H/K/V, with S varying.
        return seq
    if op in ('grouped_matmul_swiglu_quant_v2','gmm2'):
        m,k,f,e=s;counts=attrs.get('expert_counts')
        if not counts or len(counts)!=e or sum(counts)!=m or any(type(x)is not int or x<0 for x in counts):raise ValueError('valid local expert histogram required')
        tasks=sum(ceil(x,16) for x in counts if x)
        if op=='gmm2':return ceil(tasks*ceil(k,128),32)*ceil(f,64)
        return ceil(tasks*ceil(2*f,128),32)*ceil(k,64)
    raise ValueError('unknown local route '+op)


def feature_value(op,shape,attrs,feature='critical_work'):
    if feature=='activation_log2':
        if op not in ('mhc_sinkhorn','mhc_post'):raise ValueError('Invalid activation route')
        return math.log2(1+min(shape[0],64))
    if feature!='critical_work':raise ValueError('Unknown feature')
    return work(op,shape,attrs)


def fit_nonnegative(xs, ys):
    """Exact 2-variable NNLS via interior and boundary candidates, no SciPy."""
    if len(xs)!=len(ys) or any(not math.isfinite(v) or v<0 for v in (*xs,*ys)):raise ValueError('finite nonnegative paired samples required')
    n=len(xs)
    if n<2:raise ValueError('At least two calibration observations required')
    mx=sum(xs)/n;my=sum(ys)/n;variance=sum((x-mx)**2 for x in xs)
    candidates=[(my,0.)]
    if variance>0:
        b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/variance;a=my-b*mx
        if a>=0 and b>=0:candidates.append((a,b))
        candidates.append((0.,max(0.,sum(x*y for x,y in zip(xs,ys))/sum(x*x for x in xs))))
    a,b=min(candidates,key=lambda p:sum((p[0]+p[1]*x-y)**2 for x,y in zip(xs,ys)))
    return {'startup_us':a,'service_us_per_work':b,'work_identifiable':variance>0,'constant_us':my}


class Predictor:
    def __init__(self,model):
        if model['schema_version']!=7:raise ValueError('Expected v7')
        self.model=model;self.base=BasePredictor(model['base_v6'])

    def predict(self,case,attrs=None):
        attrs=attrs or {};op=case.op;shape=tuple(case.shape)
        if op in ('quant','matmul','fa'):
            out=self.base.predict(BaseCase(op,shape));return dict(out,model_version='v0.7',component_model_version='v0.6')
        if op=='moe_ffn':
            parts=[self.predict(Case(p,shape),attrs) for p in ('grouped_matmul_swiglu_quant_v2','gmm2')]
            valid=all(p['predicted_us'] is not None for p in parts)
            return {'op':op,'shape':list(shape),'attrs':attrs,'predicted_us':sum(p['predicted_us'] for p in parts) if valid else None,
                    'constant_us':sum(p['constant_us'] for p in parts) if valid else None,'components':parts,
                    'status':'local_composition' if valid else 'outside_calibrated_contract',
                    'scope':'serial kernel instruction sum; excludes routing/communication and launch gaps'}
        route=self.model['routes'].get(op)
        if route is None:return {'op':op,'shape':list(shape),'predicted_us':None,'status':'unqualified_route'}
        # Attribute equality is deliberate: another dtype/optional output is a new route.
        contract=contract_attrs(op,shape,attrs)
        limits=route['domain']
        in_range=len(shape)==len(limits) and all(lo<=v<=hi for v,(lo,hi) in zip(shape,limits))
        if contract!=route['attrs'] or not in_range:
            return {'op':op,'shape':list(shape),'attrs':attrs,'predicted_us':None,'status':'outside_calibrated_contract'}
        x=feature_value(op,shape,attrs,route.get('feature','critical_work'));p=route['parameters']
        return {'op':op,'shape':list(shape),'attrs':attrs,'predicted_us':p['startup_us']+p['service_us_per_work']*x,
                'constant_us':p['constant_us'],'work':x,'feature':route.get('feature','critical_work'),'work_identifiable':p['work_identifiable'],
                'status':'local_calibrated_proxy','scope':route['scope'],
                'parameters':p,'extrapolation':not(route['calibration_work_min']<=x<=route['calibration_work_max']),
                'limitation':'small-shape local calibration; tiling proxy, not independently identified ISA throughput'}

    def predict_us(self,case,attrs=None,use_cache=False):return self.predict(case,attrs)['predicted_us']

    def instruction_counts(self,case):
        if case.op in ('quant','matmul','fa'):return self.base.instruction_counts(BaseCase(case.op,case.shape))
        return {'estimated_count':None,'status':'not_calibrated; observed counts retained in dataset'}
