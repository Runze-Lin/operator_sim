"""Scalar query implementation of frozen v0.6 equations, with O(1) scratch space.

This changes execution only, not parameters, guards, order or timing equations.
The archived v0.6 implementation remains available to replay its freeze hashes.
"""
import math
from .model import line,ceildiv
from .model_v2 import plan
from .model_v6 import Predictor as FrozenPredictor,applies,correction


class Predictor(FrozenPredictor):
    def predict_us(self,case,use_cache=True):
        if use_cache:return self.predict(case)['predicted_us']
        if case.op!='quant':return self.base.predict_us(case,use_cache=False)
        p=plan(case);cost=self.quant_model['quant']['costs']
        prefix=line(cost['prefix'],math.log2(1+p['active_aiv']))
        if p['iterations']==1:
            total=(prefix+line(cost['first_read'],p['units']['read'])+
                   line(cost['vector'],p['units']['vector'])+cost['write_drain_us']+cost['suffix_us'])
            if applies(p):return total+correction(p,self.model['quant_vf_gap']['coefficients'])
            return max(total,line(cost['control_floor'],min(self.quant_model['control_log2_cap'],math.log2(1+p['active_aiv']))))
        rows,cols=p['rows_per_core'],p['cols_per_core'];tr,tc=p['tile']
        # Retain only the last two input/output-slot owners; no UB tile list.
        rprev=vprev=wprev=volder=wolder=0.;i=0
        for row in range(0,rows,tr):
            r=min(tr,rows-row)
            for col in range(0,cols,tc):
                c=min(tc,cols-col);read_kib=2*r*c/1024;groups=r*ceildiv(c,32)
                latency=line(cost['first_read'],read_kib)
                ready=prefix+latency if i==0 else rprev+line(cost['read_interval'],read_kib)
                if i>=2:ready=max(ready,volder+latency)
                begin=max(ready,vprev if i else 0,wolder if i>=2 else 0)
                end=begin+line(cost['vector'],ceildiv(groups,8)+ceildiv(groups,128))
                written=max(end,wprev if i else 0)+cost['write_drain_us']
                volder,wolder=vprev,wprev;rprev,vprev,wprev=ready,end,written;i+=1
        total=wprev+cost['suffix_us']
        if applies(p):return total+correction(p,self.model['quant_vf_gap']['coefficients'])
        return max(total,line(cost['control_floor'],min(self.quant_model['control_log2_cap'],math.log2(1+p['active_aiv']))))
