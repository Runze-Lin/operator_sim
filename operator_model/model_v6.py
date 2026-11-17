"""Guarded quant VF-envelope correction; direct scalar query for one UB tile.

A measured envelope minus SIMD union is a scheduling residual, not ISA service.
Other operators and multi-UB schedules retain their frozen v0.5 equations.
"""
from collections import OrderedDict
import json
import math
from pathlib import Path
from .model import freeze, line
from .model_v2 import plan as compact_plan
from .model_v3 import quant_plan
from .model_v4 import Case, schedule_quant
from .model_v5 import Predictor as V5


def applies(p):
    # A full row fragment emits >=32 scale bytes. This is a conservative
    # modelling guard, not proof of cache-line alignment or no write conflicts.
    return p['iterations']==1 and p['cols_per_core']>=1024


def correction(p,weights):
    return weights[0]+weights[1]*p['active_aiv']/64


class Predictor:
    def __init__(self,model,cache_size=4096):
        self.model=freeze(json.loads(Path(model).read_text()) if isinstance(model,(str,Path)) else model)
        if self.model.get('schema_version')!=6 or self.model['soc_version']!='Ascend950DT_9582':
            raise ValueError('Expected v0.6 950DT model')
        if cache_size<0:raise ValueError('cache_size must be nonnegative')
        self.base=V5(self.model['base_v5_model'],cache_size=0)
        self.quant_model=self.model['base_v5_model']['base_v4_model']
        self.cache=OrderedDict();self.cache_size=cache_size

    def predict(self,case,use_cache=True):
        if use_cache and case in self.cache:
            self.cache.move_to_end(case);return self.cache[case]
        if case.op!='quant':
            out=self.base.predict(case,use_cache=False)
        else:
            p=quant_plan(case);s=schedule_quant(p,self.quant_model);m=self.quant_model['quant']
            if applies(p):
                gap=correction(p,self.model['quant_vf_gap']['coefficients'])
                s.update(base_pipeline_us=s['pipeline_us'],nonbusy_vf_envelope_us=gap,
                    control_floor_applied=False,control_dominant=False,
                    pipeline_us=s['pipeline_us']+gap,predicted_us=s['pipeline_us']+gap,
                    vector_finish_us=[s['vector_finish_us'][0]+gap],write_finish_us=[s['write_finish_us'][0]+gap])
            out=freeze({'op':case.op,'shape':case.shape,'soc_version':self.model['soc_version'],
                'model_version':'0.6.0','predicted_us':s['predicted_us'],
                'reference_scope':'cross-core instruction makespan','plan':p,'phase_schedule':s,
                'coverage':'provisional_single_ub_vf_envelope' if applies(p) else 'unchanged_v04_quant_route',
                'constant_baseline_us':self.quant_model['constant_baselines']['quant'],
                'calibration_regime_count':self.model['quant_vf_gap']['calibration_count'] if applies(p) else m['multi_ub_calibration_count'] if p['iterations']>1 else m['small_control_count'],
                'expected_kernel_candidates':(),'kernel_selection_verified':False,
                'estimated_max_core_selected_opcode_count':sum(a*b for a,b in zip(p['opcode_features'],m['opcode_weights'])),
                'ideal_compute_efficiency':None,'efficiency_unavailable_reason':'Independent ISA throughput is not calibrated'})
        if use_cache and self.cache_size:
            self.cache[case]=out
            if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return out

    def predict_us(self,case,use_cache=True):
        if use_cache:return self.predict(case)['predicted_us']
        if case.op!='quant':return self.base.predict_us(case,use_cache=False)
        p=compact_plan(case)
        if p['iterations']!=1:
            return self.base.predict_us(case,use_cache=False)
        c=self.quant_model['quant']['costs'];a=math.log2(1+p['active_aiv'])
        # One tile: the recurrence collapses to a sum. No tile lists, diagnostic
        # arrays, recursive freezing or cache insertion are needed by this API.
        total=(line(c['prefix'],a)+line(c['first_read'],p['units']['read'])+
               line(c['vector'],p['units']['vector'])+c['write_drain_us']+c['suffix_us'])
        if applies(p):return total+correction(p,self.model['quant_vf_gap']['coefficients'])
        return max(total,line(c['control_floor'],min(self.quant_model['control_log2_cap'],a)))

    def predict_many(self,cases):return [self.predict(c) for c in cases]

    def instruction_counts(self,case):
        return {'family':'RV_* including load/store/control' if case.op=='quant' else 'MMAD',
                'estimated_max_core_count':self.predict(case)['estimated_max_core_selected_opcode_count']}
