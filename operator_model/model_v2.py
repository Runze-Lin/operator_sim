"""Expanded shape/tiling regimes with pooled stage calibration.

No target-shape trace is consumed by predict(). v0.1 stays frozen separately.
"""
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path

from .model import align, ceildiv, fit_nonnegative_line, freeze, line, measured_stages, quant_partition, schedule


@dataclass(frozen=True)
class Case:
    op: str
    shape: tuple

    def __post_init__(self):
        shape=tuple(self.shape)
        if self.op=='fa' and len(shape)==3:shape=shape+(1,1)
        object.__setattr__(self,'shape',shape)
        if self.op not in ('quant','matmul','fa') or len(shape)!={'quant':2,'matmul':3,'fa':5}[self.op]:
            raise ValueError('Expected quant(R,C), matmul(M,N,K), fa(Q,KV,D[,B,H])')
        if any(type(d) is not int or not 1<=d<=4096 for d in shape):
            raise ValueError('Dimensions must be integers in [1,4096]')
        if self.op=='quant' and shape[1]%64:raise ValueError('Quant cols must be divisible by 64')
        if self.op=='fa' and (shape[2] not in (64,128) or shape[3]!=1 or shape[4]>32):
            raise ValueError('v0.2 FA: BF16 SBH, D64/128, B1, 1<=H<=32, no mask/dropout')


def plan(case):
    if case.op=='quant':
        r,c=case.shape;s=quant_partition(r,c)
        # BF16->FP8 block32 DB capacity from documented 950 host formula.
        max_groups=((253952-2048)//(256*6+8*2*5))*8
        col_groups=ceildiv(s['cols_per_core'],32)
        col_loops=ceildiv(col_groups,max_groups)
        groups_per_col=align(ceildiv(col_groups,col_loops),2)
        row_loops=ceildiv(s['rows_per_core'],max_groups//groups_per_col)
        rows_per_ub=ceildiv(s['rows_per_core'],row_loops)
        groups=rows_per_ub*groups_per_col
        main_loops=ceildiv(groups,8);scale_loops=ceildiv(groups,128)
        iterations=row_loops*col_loops
        elems=groups*32
        branch='ub_loop' if iterations>1 else ('slice512' if s['slice_cols']==512 else 'slice256')
        return {**s,'active_aic':0,'waves':1,'iterations':iterations,'branch':branch,
                'tile':[rows_per_ub,groups_per_col*32],
                'units':{'read':2*elems/1024,'write':(elems+groups)/1024,
                         'vector':main_loops+scale_loops,'cube':0.,'fix':0.},
                'opcode_features':[iterations,iterations*main_loops,iterations*scale_loops],
                'useful_work':r*c,'logical_bytes':r*c*(3+1/32),'work_per_core':s['rows_per_core']*s['cols_per_core']}
    if case.op=='matmul':
        m,n,k=case.shape;tm,tn=16,64
        # Balance elementary output tiles when the grid exceeds 32 AICs.
        # Source-informed hypothesis, checked with actual MMAD shapes/core counts.
        while ceildiv(m,tm)*ceildiv(n,tn)>32 and (tm<256 or tn<256):
            options=[]
            for a,b in ((tm*2,tn),(tm,tn*2)):
                if a>256 or b>256:continue
                count=ceildiv(m,a)*ceildiv(n,b)
                options.append((abs(count-32),ceildiv(m,a)*a*ceildiv(n,b)*b,a+b,a,b))
            _,_,_,tm,tn=min(options)
        tiles=ceildiv(m,tm)*ceildiv(n,tn);active=min(32,tiles);waves=ceildiv(tiles,active)
        em,en=min(m,tm),min(n,tn);ek=align(k,16)
        branch='gemv' if m==1 else ('k_tail' if k%16 else ('coarse_tile' if (tm,tn)!=(16,64) else 'dense'))
        return {'active_aic':active,'active_aiv':0,'waves':waves,'iterations':1,'branch':branch,
                'tile':[tm,tn,256],'units':{'read':2*(em*ek+en*ek)/1024,'cube':2*em*en*ek/131072,
                                         'fix':2*em*en/1024,'vector':0.,'write':0.},
                'opcode_features':[waves*ceildiv(k,256)],
                'useful_work':2*m*n*k,'logical_bytes':2*(m*k+k*n+m*n),'work_per_core':2*em*en*ek*waves}
    q,kv,d,b,h=case.shape
    qbase=64 if b*h*ceildiv(q,128)<32 else 128
    kvbase=256 if (d==64 and q%128==0 and kv%128==0) or kv>1024 else 128
    tq=16 if q==1 else min(q,qbase)
    tk=min(align(kv,16),kvbase)
    total=b*h*ceildiv(q,qbase);active=min(32,total);waves=ceildiv(total,32);iterations=ceildiv(kv,kvbase)
    return {'active_aic':active,'active_aiv':2*active,'waves':waves,'iterations':iterations,
            'branch':f'd{d}_kv{kvbase}','tile':[tq,tk,d],
            'q_tile_capacity':qbase,'kv_tile_capacity':kvbase,
            'units':{'read':2*(tq*d+2*tk*d)/1024,'cube':4*tq*tk*d/131072,
                     'vector':tq*tk/2/256,'fix':4*tq*tk/1024,'write':2*tq*d/1024},
            'opcode_features':[2*iterations*waves],
            'useful_work':4*q*kv*d*b*h,'logical_bytes':2*(2*q*d+2*kv*d)*b*h,
            'work_per_core':4*tq*tk*d*iterations*waves}


def body_time(op,p,stages):
    if op=='quant':
        return sum(stages.values())+(p['iterations']-1)*max(stages.values())
    return schedule(op,p,stages)


def solve(a,b):
    """Small pivoted Gaussian elimination; only used offline."""
    a=[list(row)+[rhs] for row,rhs in zip(a,b)];n=len(b)
    for j in range(n):
        pivot=max(range(j,n),key=lambda i:abs(a[i][j]))
        if abs(a[pivot][j])<1e-12:return None
        a[j],a[pivot]=a[pivot],a[j]
        div=a[j][j];a[j]=[v/div for v in a[j]]
        for i in range(n):
            if i!=j:
                factor=a[i][j];a[i]=[v-factor*w for v,w in zip(a[i],a[j])]
    return [row[-1] for row in a]


def nnls(xs,ys):
    """Enumerate active sets for <=3 nonnegative coefficients, scaled columns."""
    n=len(xs[0]);scales=[max(abs(row[j]) for row in xs) or 1 for j in range(n)]
    x=[[v/s for v,s in zip(row,scales)] for row in xs]
    best=[0.]*n;error=sum(y*y for y in ys)
    for size in range(1,n+1):
        for active in itertools.combinations(range(n),size):
            gram=[[sum(row[i]*row[j] for row in x)+(1e-9 if i==j else 0) for j in active] for i in active]
            rhs=[sum(row[i]*y for row,y in zip(x,ys)) for i in active]
            weights=solve(gram,rhs)
            if weights is None or min(weights)<-1e-9:continue
            candidate=[0.]*n
            for i,w in zip(active,weights):candidate[i]=max(0.,w)/scales[i]
            loss=sum((sum(a*b for a,b in zip(row,candidate))-y)**2 for row,y in zip(xs,ys))
            if loss<error:best,error=candidate,loss
    return best


def features(p,body):
    return [1.,body,math.log2(1+p['active_aic']+p['active_aiv'])]


def fit(records,manifest_sha256):
    if any(r['case']['split']!='calibration' for r in records):raise ValueError('Holdout leakage')
    data=[r for r in records if r['status']=='ok']
    # Old callers differ only in explicitly documented host-side B/H support.
    identity_keys=['soc_version','cann_version','package_manifest_sha256','platform_sha256','sim_config_sha256']
    if len({tuple(r['identity'][k] for k in identity_keys) for r in data})!=1:
        raise ValueError('Incompatible CA environments')
    model={'schema_version':2,'version':'0.2.0','soc_version':'Ascend950DT_9582',
           'manifest_sha256':manifest_sha256,'operators':{},'training_keys':[r['cache_key'] for r in data],
           'qualification':'pooled CA stage costs plus shrunk regime offsets; unverified regimes remain provisional'}
    for op in ('quant','matmul','fa'):
        rs=[r for r in data if r['case']['op']==op]
        if len(rs)<5:raise ValueError(f'{op}: too few calibration samples')
        ps=[plan(Case(op,r['case']['shape'])) for r in rs]
        costs={}
        for stage in ('read','vector','cube','fix','write'):
            xs=[p['units'][stage] for p in ps]
            ys=[measured_stages(r)[stage]/p['iterations']/p['waves'] for r,p in zip(rs,ps)]
            costs[stage]=fit_nonnegative_line(xs,ys) if any(xs) else {'intercept_us':0.,'slope_us_per_unit':0.}
        bodies=[body_time(op,p,{k:line(costs[k],u) for k,u in p['units'].items()}) for p in ps]
        targets=[r['measurement']['reference_us'] for r in rs]
        fs=[features(p,t) for p,t in zip(ps,bodies)];weights=nnls(fs,targets)
        groups=defaultdict(list)
        for r,p,f,y in zip(rs,ps,fs,targets):groups[p['branch']].append((r,y-sum(w*x for w,x in zip(weights,f))))
        branches={name:{'count':len(values),'offset_us':sum(v for _,v in values)/(len(values)+2),
                        'kernels':sorted({r['measurement']['kernel'] for r,_ in values})} for name,values in groups.items()}
        counts=[]
        for r in rs:
            cores=r['measurement']['cores']
            counts.append(max(sum(v for name,v in c['dynamic_opcodes'].items() if name.startswith('RV_')) if op=='quant' else c['dynamic_opcodes'].get('MMAD',0) for c in cores))
        opcode_weights=nnls([p['opcode_features'] for p in ps],counts)
        model['operators'][op]={'stage_costs':costs,'total_weights':weights,'branches':branches,
            'opcode_weights':opcode_weights,'constant_baseline_us':sum(targets)/len(targets),
            'shape_ranges':[[min(r['case']['shape'][i] for r in rs),max(r['case']['shape'][i] for r in rs)] for i in range(len(rs[0]['case']['shape']))],
            'effective_work_per_us':max(p['useful_work']/t for p,t in zip(ps,targets)),
            'effective_bytes_per_us':max(p['logical_bytes']/t for p,t in zip(ps,targets)),
            'mapping_checks':[{'id':r['case']['id'],'predicted':[p['active_aic'],p['active_aiv']],
                               'observed':[r['measurement']['active_aic'],r['measurement']['active_aiv']]} for r,p in zip(rs,ps)]}
    return model


class Predictor:
    def __init__(self,model,cache_size=4096):
        self.model=freeze(json.loads(Path(model).read_text()) if isinstance(model,(str,Path)) else model)
        if self.model['soc_version']!='Ascend950DT_9582':raise ValueError('Only 950DT is calibrated')
        self.cache=OrderedDict();self.cache_size=cache_size

    def predict(self,case,use_cache=True):
        if use_cache and case in self.cache:
            self.cache.move_to_end(case);return self.cache[case]
        p=plan(case);m=self.model['operators'][case.op]
        stages={k:line(m['stage_costs'][k],v) for k,v in p['units'].items()}
        body=body_time(case.op,p,stages);branch=m['branches'].get(p['branch'])
        base=sum(a*b for a,b in zip(features(p,body),m['total_weights']))
        prediction=max(.001,base+(branch['offset_us'] if branch else 0))
        inside=all(lo<=d<=hi for d,(lo,hi) in zip(case.shape,m['shape_ranges']))
        coverage='unseen_regime' if branch is None else 'provisional_regime' if branch['count']<3 else 'within_calibration_box' if inside else 'extrapolation'
        output=freeze({'op':case.op,'shape':case.shape,'predicted_us':prediction,'soc_version':self.model['soc_version'],
            'reference_scope':'cross-core instruction makespan','coverage':coverage,'plan':p,
            'stage_services_us':stages,'scheduled_body_us':body,'uncorrected_schedule_us':body,
            'constant_baseline_us':m['constant_baseline_us'],
            'empirical_roofline_us':max(p['useful_work']/m['effective_work_per_us'],p['logical_bytes']/m['effective_bytes_per_us']),
            'calibration_regime_count':branch['count'] if branch else 0,
            'expected_kernel_candidates':branch['kernels'] if branch else [],'kernel_selection_verified':False,
            'estimated_max_core_selected_opcode_count':sum(a*b for a,b in zip(p['opcode_features'],m['opcode_weights'])),
            'ideal_compute_efficiency':None,'efficiency_unavailable_reason':'Independent ISA issue/throughput model is not calibrated'})
        if use_cache and self.cache_size:
            self.cache[case]=output
            if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return output

    def predict_many(self,cases):return [self.predict(c) for c in cases]

    def instruction_counts(self,case):
        return {'family':'RV_* including load/store/control' if case.op=='quant' else 'MMAD',
                'estimated_max_core_count':self.predict(case)['estimated_max_core_selected_opcode_count']}
