"""Trace-decomposed quant schedule and structural FA instruction counts.

v0.1/v0.2 implementations and predictions remain immutable historical baselines.
"""
from collections import OrderedDict
import json
import math
from pathlib import Path
import statistics

from .model import ceildiv, fit_nonnegative_line, freeze, line
from .model_v2 import Case, Predictor as PreviousPredictor, plan as previous_plan, nnls


def quant_plan(case):
    p=previous_plan(case)
    rows,cols=p['rows_per_core'],p['cols_per_core']
    tr,tc=p['tile'];tiles=[]
    for row in range(0,rows,tr):
        for col in range(0,cols,tc):
            r,c=min(tr,rows-row),min(tc,cols-col)
            groups=r*ceildiv(c,32)
            tiles.append({'rows':r,'cols':c,'read_kib':2*r*c/1024,
                          'write_kib':(r*c+groups)/1024,'main_loops':ceildiv(groups,8),
                          'scale_loops':ceildiv(groups,128)})
    p['ub_tiles']=tiles;p['iterations']=len(tiles)
    p['opcode_features']=[len(tiles),sum(t['main_loops'] for t in tiles),sum(t['scale_loops'] for t in tiles)]
    return p


def quant_schedule(p,costs):
    """Max-plus double-buffer recurrence; O(UB tiles), no instruction replay.

    Read duration and read initiation interval are distinct. Input slots become
    reusable after vector consumption; output slots after their write completes.
    """
    prefix=line(costs['prefix'],math.log2(1+p['active_aiv']))
    reads=[];vectors=[];writes=[]
    for i,t in enumerate(p['ub_tiles']):
        latency=line(costs['first_read'],t['read_kib'])
        if i==0:ready=prefix+latency
        else:
            ready=reads[-1]+line(costs['read_interval'],t['read_kib'])
            if i>=2:ready=max(ready,vectors[i-2]+latency)
        begin=max(ready,vectors[-1] if i else 0,writes[i-2] if i>=2 else 0)
        end=begin+line(costs['vector'],t['main_loops']+t['scale_loops'])
        written=max(end,writes[-1] if i else 0)+costs['write_drain_us']
        reads.append(ready);vectors.append(end);writes.append(written)
    pipeline=writes[-1]+costs['suffix_us']
    control=line(costs['control_floor'],math.log2(1+p['active_aiv']))
    return {'pipeline_us':pipeline,'control_floor_us':control,'predicted_us':max(pipeline,control),
            'read_ready_us':reads,'vector_finish_us':vectors,'write_finish_us':writes,
            'prefix_us':prefix,'control_dominant':control>=pipeline}


def fit_quant(records,analysis):
    by_id={r['case']['id']:r for r in records if r['case']['op']=='quant'}
    rows=[a for a in analysis['quant'] if a['id'] in by_id]
    if len(rows)!=len(by_id):raise ValueError('Missing quant phase evidence')
    ps={a['id']:quant_plan(Case('quant',a['shape'])) for a in rows}
    for a in rows:
        r=by_id[a['id']]
        if a['trace_sha256']!=r['measurement']['trace_sha256']:raise ValueError('Stale phase evidence')
        if len(a['iterations'])!=ps[a['id']]['iterations']:raise ValueError('Iteration mismatch')
    small=[a for a in rows if ps[a['id']]['work_per_core']<=1024]
    multi=[a for a in rows if ps[a['id']]['iterations']>1]
    if len(small)<4 or not multi:raise ValueError('Need small controls and multi-UB evidence')
    head=fit_nonnegative_line([math.log2(1+ps[a['id']]['active_aiv']) for a in rows],[a['prefix_us'] for a in rows])
    read=fit_nonnegative_line([a['iterations'][0]['read']['bytes']/1024 for a in rows],[a['first_read_us'] for a in rows])
    # Busy intervals, not VF command queue spans, determine vector service work.
    vec=fit_nonnegative_line([ps[a['id']]['units']['vector'] for a in rows],
        [max(c['busy_union_us_by_pipe'].get('VECTOR_FAMILY',0) for c in by_id[a['id']]['measurement']['cores'])/ps[a['id']]['iterations'] for a in rows])
    intervals=[(it['read']['bytes']/1024,it['read']['end_us']-a['iterations'][i-1]['read']['end_us'])
               for a in multi for i,it in enumerate(a['iterations']) if i]
    interval=fit_nonnegative_line([x for x,y in intervals],[max(0,y) for x,y in intervals])
    floor=fit_nonnegative_line([math.log2(1+ps[a['id']]['active_aiv']) for a in small],[a['reference_us'] for a in small])
    costs={'prefix':head,'first_read':read,'vector':vec,'read_interval':interval,'control_floor':floor,
           'write_drain_us':statistics.median(i['write_drain_us'] for a in multi for i in a['iterations']),
           'suffix_us':statistics.median(a['suffix_us'] for a in rows)}
    ys=[max(sum(v for k,v in c['dynamic_opcodes'].items() if k.startswith('RV_')) for c in by_id[a['id']]['measurement']['cores']) for a in rows]
    counts=nnls([ps[a['id']]['opcode_features'] for a in rows],ys)
    return {'costs':costs,'opcode_weights':counts,'calibration_count':len(rows),'multi_ub_calibration_count':len(multi),
            'small_control_count':len(small),'assumptions':['DMA intervals include CA queueing; not hardware peak bandwidth',
            'control floor is a pooled small-workload latency path, not measured pure scalar execution',
            'post-first read interval and write drain have only two multi-UB calibration shapes']}


def fit(records,analysis,base_model):
    if any(r['case']['split']!='calibration' for r in records):raise ValueError('Holdout leakage')
    if any(r['status']!='ok' for r in records):raise ValueError('Only successful calibration records')
    if len({r['cache_key'] for r in records})!=len(records):raise ValueError('Duplicate calibration inputs')
    env_keys=('soc_version','cann_version','package_manifest_sha256','platform_sha256','sim_config_sha256')
    if len({tuple(r['identity'][k] for k in env_keys) for r in records})!=1:raise ValueError('Incompatible CA environments')
    return {'schema_version':3,'version':'0.3.0','soc_version':'Ascend950DT_9582','base_v2_model':base_model,
            'training_keys':[r['cache_key'] for r in records],'quant':fit_quant(records,analysis),
            'constant_baselines':{op:statistics.mean(r['measurement']['reference_us'] for r in records if r['case']['op']==op) for op in ('quant','matmul','fa')},
            'qualification':'trace-derived quant phases; FA Q64 non-Dn MMAD decomposition; v0.2 matmul/FA timing retained'}


class Predictor:
    def __init__(self,model,cache_size=4096):
        self.model=freeze(json.loads(Path(model).read_text()) if isinstance(model,(str,Path)) else model)
        if self.model.get('schema_version')!=3 or self.model['soc_version']!='Ascend950DT_9582':raise ValueError('Expected calibrated v0.3 950DT model')
        self.base=PreviousPredictor(self.model['base_v2_model'],cache_size=0)
        self.cache=OrderedDict();self.cache_size=cache_size

    def predict(self,case,use_cache=True):
        if use_cache and case in self.cache:
            self.cache.move_to_end(case);return self.cache[case]
        old=self.base.predict(case,use_cache=False);out=dict(old)
        out['previous_v02_us']=old['predicted_us'];out['previous_constant_baseline_us']=old['constant_baseline_us']
        out['constant_baseline_us']=self.model['constant_baselines'][case.op]
        if case.op=='quant':
            p=quant_plan(case);schedule=quant_schedule(p,self.model['quant']['costs'])
            out.update(predicted_us=schedule['predicted_us'],plan=p,phase_schedule=schedule,
                       estimated_max_core_selected_opcode_count=sum(a*b for a,b in zip(p['opcode_features'],self.model['quant']['opcode_weights'])),
                       coverage='provisional_phase_model',kernel_selection_verified=False)
            # Legacy stage fields have different definitions; never expose them as v0.3 phases.
            for k in ('stage_services_us','scheduled_body_us','uncorrected_schedule_us'):out.pop(k,None)
            out['calibration_regime_count']=self.model['quant']['multi_ub_calibration_count'] if p['iterations']>1 else self.model['quant']['small_control_count']
        elif case.op=='fa':
            q,kv,d,b,h=case.shape;p=out['plan']
            supported=p['q_tile_capacity']==64
            # QK MatmulN: N chunks of 128. PV MatmulK: K chunks of 128.
            count=2*ceildiv(kv,128)*p['waves'] if supported else None
            out.update(estimated_max_core_selected_opcode_count=count,
                       instruction_count_coverage='structural_q64_d64_d128' if supported else 'unvalidated_q128_route',
                       instruction_decomposition={'qk_per_q_tile':ceildiv(kv,128),'pv_per_q_tile':ceildiv(kv,128),'max_q_tiles_per_core':p['waves']} if supported else None)
        result=freeze(out)
        if use_cache and self.cache_size:
            self.cache[case]=result
            if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return result

    def predict_many(self,cases):return [self.predict(c) for c in cases]

    def instruction_counts(self,case):
        return {'family':'RV_* including load/store/control' if case.op=='quant' else 'MMAD',
                'estimated_max_core_count':self.predict(case)['estimated_max_core_selected_opcode_count']}
