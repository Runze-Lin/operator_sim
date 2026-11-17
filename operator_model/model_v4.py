"""Saturated small-quant control path; single-pass immutable predictions.

Extended dimensions are formula extrapolations, not claims of calibrated kernels.
Historical v0.3 code and model remain unchanged.
"""
from collections import OrderedDict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics

from .model import freeze, fit_nonnegative_line, line
from .model_v3 import Predictor as PreviousPredictor, quant_plan, quant_schedule, fit_quant


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
        if any(type(x) is not int or not 1<=x<=1048576 for x in shape):
            raise ValueError('Dimensions must be integers in [1,1048576]')
        if self.op=='quant' and (shape[1]%64 or shape[1]>65536):
            raise ValueError('Quant cols must be a multiple of 64, at most 65536')
        if self.op=='matmul' and max(shape)>4096:raise ValueError('Matmul domain remains <=4096')
        if self.op=='fa' and (shape[2] not in (64,128) or shape[3]>256 or shape[4]>32 or shape[0]>2048):
            raise ValueError('FA proxy: D64/128, B<=256, H<=32, Q<=2048, noncausal SBH')


def fit(records,analysis,base_model):
    # fit_quant performs trace hash/iteration checks; explicitly guard split and keys here.
    if any(r['status']!='ok' or r['case']['split']!='calibration' for r in records):
        raise ValueError('Only successful calibration data; holdout leakage')
    if len({r['cache_key'] for r in records})!=len(records):raise ValueError('Duplicate calibration inputs')
    keys=('soc_version','cann_version','package_manifest_sha256','platform_sha256','sim_config_sha256')
    if len({tuple(r['identity'][k] for k in keys) for r in records})!=1:raise ValueError('Incompatible environments')
    q=fit_quant(records,analysis)
    small=[r for r in records if r['case']['op']=='quant' and quant_plan(Case('quant',r['case']['shape']))['work_per_core']<=4096]
    xs=[min(4.,math.log2(1+quant_plan(Case('quant',r['case']['shape']))['active_aiv'])) for r in small]
    q['costs']['control_floor']=fit_nonnegative_line(xs,[r['measurement']['reference_us'] for r in small])
    q['small_control_count']=len(small)
    return {'schema_version':4,'version':'0.4.0','soc_version':'Ascend950DT_9582','base_v3_model':base_model,
            'training_keys':[r['cache_key'] for r in records],'quant':q,'control_log2_cap':4.,
            'constant_baselines':{op:statistics.mean(r['measurement']['reference_us'] for r in records if r['case']['op']==op) for op in ('quant','matmul','fa')},
            'qualification':'Saturated empirical control path; quant phases refit on 13 old traces; FA/matmul timing frozen v0.3; expanded shapes unvalidated extrapolation'}


def schedule_quant(p,model):
    # Saturation is an empirical small-workload fallback, not an ISA throughput.
    costs=model['quant']['costs']
    floor=line(costs['control_floor'],min(model['control_log2_cap'],math.log2(1+p['active_aiv'])))
    result=quant_schedule(p,{**costs,'control_floor':{'intercept_us':floor,'slope_us_per_unit':0.}})
    result['control_log2_cap']=model['control_log2_cap']
    return result


class Predictor:
    def __init__(self,model,cache_size=4096):
        self.model=freeze(json.loads(Path(model).read_text()) if isinstance(model,(str,Path)) else model)
        if self.model.get('schema_version')!=4 or self.model['soc_version']!='Ascend950DT_9582':raise ValueError('Expected v0.4 950DT model')
        if cache_size<0:raise ValueError('cache_size must be nonnegative')
        self.base=PreviousPredictor(self.model['base_v3_model'],cache_size=0)
        self.cache=OrderedDict();self.cache_size=cache_size

    def predict(self,case,use_cache=True):
        if use_cache and case in self.cache:
            self.cache.move_to_end(case);return self.cache[case]
        if case.op=='quant':
            p=quant_plan(case);s=schedule_quant(p,self.model);m=self.model['quant']
            output=freeze({'op':case.op,'shape':case.shape,'soc_version':self.model['soc_version'],
                'predicted_us':s['predicted_us'],'reference_scope':'cross-core instruction makespan',
                'coverage':'extrapolation' if max(case.shape)>4096 else 'provisional_saturated_control',
                'plan':p,'phase_schedule':s,'constant_baseline_us':self.model['constant_baselines']['quant'],
                'calibration_regime_count':m['multi_ub_calibration_count'] if p['iterations']>1 else m['small_control_count'],
                'expected_kernel_candidates':(), 'kernel_selection_verified':False,
                'estimated_max_core_selected_opcode_count':sum(a*b for a,b in zip(p['opcode_features'],m['opcode_weights'])),
                'ideal_compute_efficiency':None,'efficiency_unavailable_reason':'Independent ISA throughput is not calibrated'})
        else:
            # Already immutable: no duplicate planning, result traversal, or nested copying.
            old=self.base.predict(case,use_cache=False)
            output=type(old)({**old,'constant_baseline_us':self.model['constant_baselines'][case.op],
                'coverage':'extrapolation' if max(case.shape)>4096 or (case.op=='fa' and case.shape[3]!=1) else old['coverage']})
        if use_cache and self.cache_size:
            self.cache[case]=output
            if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return output

    def predict_us(self,case,use_cache=True):
        """Same model, scalar-only contract for callers that do not need diagnostics."""
        if use_cache:return self.predict(case)['predicted_us']
        if case.op=='quant':return schedule_quant(quant_plan(case),self.model)['predicted_us']
        return self.predict(case,use_cache=False)['predicted_us']

    def predict_many(self,cases):return [self.predict(c) for c in cases]

    def instruction_counts(self,case):
        return {'family':'RV_* including load/store/control' if case.op=='quant' else 'MMAD',
                'estimated_max_core_count':self.predict(case)['estimated_max_core_selected_opcode_count']}
