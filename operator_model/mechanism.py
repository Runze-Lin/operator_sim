"""Hardware-separated resource/dependency model, without operator latency fits.

Public interface: build_graph(structural_plan), predict_graph(graph, profile),
predict_plan(structural_plan, profile). All objects are JSON-compatible.

Hardware profile schema::

    {"name": "explicit hypothesis or calibrated profile", "resources": {
      "vector_alu": {"pool": "aiv", "units": 64, "clock_ghz": 1.65,
        "work_per_cycle": 64, "latency_cycles": 1, "issue_cycles": 1,
        "work_unit": "scalar_ops", "evidence": "hypothesis", "source": "..."}}}

No numerical hardware defaults are provided. Resource names describe capabilities,
not operators. `pool` declares mutually exclusive resources (e.g. ALU/exp on AIV).
Stage work is TOTAL across its independent tasks and dependent serial steps.

Stage recurrence (all in us): service = work/(tasks*steps*rate_per_unit);
first = max(latency, service); step = first if state_serial else max(issue,service);
span = ceil(tasks/units)*(first + (steps-1)*step).
Stages use explicit predecessor completion plus resource-pool availability.
This is a deterministic aggregate schedule, NOT an instruction replay or a proven
latency bound: tile-boundary overlap, physical memory traffic and ISA expansion
must be supplied before declaring it a qualified predictor. Logical adapters
expose these omissions; candidate_us never silently becomes predicted_us.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping

# These are operation classes, not per-operator free coefficients.
RESOURCE_UNITS = {
    'vector_alu': 'scalar_ops', 'vector_exp': 'elements',
    'vector_div': 'elements', 'vector_rsqrt': 'elements',
    'vector_sigmoid': 'elements', 'vector_cast': 'elements',
    'cube_bf16': 'flops', 'cube_fp16': 'flops', 'cube_fp8': 'flops',
    'cube_hf32': 'flops', 'memory': 'bytes',
}
EVIDENCE = {'hypothesis', 'source', 'microbenchmark', 'trace_calibrated'}


def _number(x, name, positive=False):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(name + ' must be finite numeric')
    if x < 0 or (positive and x == 0):
        raise ValueError(name + ' has invalid sign')
    return x


def _integer(x, name, positive=True):
    if type(x) is not int or x < (1 if positive else 0):
        raise ValueError(name + ' must be an integer in range')
    return x


def validate_profile(profile):
    """Reject hidden per-op/per-shape knobs and dimensionally wrong resources."""
    if not isinstance(profile, Mapping) or set(profile)-{'name','resources','notes','ub_bytes'}:
        raise ValueError('Profile only accepts name/resources/notes/ub_bytes')
    if not profile.get('name') or not isinstance(profile.get('resources'), Mapping):
        raise ValueError('Named profile with resources is required')
    pools = {}
    for name, r in profile['resources'].items():
        if name not in RESOURCE_UNITS and not (name.startswith('isa:') and len(name)>4):
            raise ValueError('Unknown hardware resource: ' + name)
        keys = {'pool','units','clock_ghz','work_per_cycle','latency_cycles',
                'issue_cycles','work_unit','evidence','source'}
        if not isinstance(r, Mapping) or set(r) != keys:
            raise ValueError('Resource requires exactly: ' + ','.join(sorted(keys)))
        _integer(r['units'], 'units')
        for field in ('clock_ghz','work_per_cycle'):
            _number(r[field], field, True)
        for field in ('latency_cycles','issue_cycles'):
            _number(r[field], field)
        if r['work_unit'] != ('instructions' if name.startswith('isa:') else RESOURCE_UNITS[name]):
            raise ValueError('Wrong work unit for ' + name)
        if not isinstance(r['pool'], str) or not r['pool'] or not r['source'] or r['evidence'] not in EVIDENCE:
            raise ValueError('Resource pool and provenance required')
        if r['pool'] in pools and pools[r['pool']] != r['units']:
            raise ValueError('Shared pool resources must agree on physical unit count')
        pools[r['pool']] = r['units']
    if 'ub_bytes' in profile:
        _integer(profile['ub_bytes'], 'ub_bytes')
    return profile


def predict_graph(graph, profile):
    """Evaluate a topologically ordered aggregate DAG without shape lookup.

    Only a complete, qualified graph with independently calibrated resources
    returns predicted_us. Others return candidate_us plus explicit missing facts.
    Absence of a required resource returns candidate_us=None, never zero service.
    lower_bound_us excludes uncertain dispatch latency and is a logical-work
    resource floor, NOT a guaranteed bound if the caller supplies wrong work.
    """
    validate_profile(profile)
    if graph.get('status') == 'unsupported':
        return {'status':'unsupported', 'predicted_us':None, 'candidate_us':None, 'mechanism_estimate_us':None,
                'lower_bound_us':None, 'upper_bound_us':None,
                'missing_evidence':list(graph.get('missing_evidence',[])), 'stages':[]}
    resources = profile['resources']
    missing = list(graph.get('missing_evidence', []))
    end, available, stages = {}, {}, []
    pool_work = defaultdict(float)
    unknown_resources = set()
    used_resources = set()
    for stage in graph['stages']:
        sid = stage['id']
        if sid in end or not isinstance(sid, str):
            raise ValueError('Duplicate/invalid stage id')
        deps = stage.get('dependencies', [])
        if any(d not in end for d in deps):
            raise ValueError('DAG must be acyclic and topologically ordered')
        name = stage['resource']; work = _number(stage['work'], 'work')
        tasks = _integer(stage['parallel_tasks'], 'parallel_tasks')
        steps = _integer(stage.get('serial_steps',1), 'serial_steps')
        if type(stage.get('state_serial', False)) is not bool:
            raise ValueError('state_serial must be bool')
        if name not in resources:
            unknown_resources.add(name); end[sid] = None
            stages.append(dict(stage, start_us=None, end_us=None, duration_us=None))
            continue
        used_resources.add(name)
        r = resources[name]
        # GHz*1000 cycles/us; rate therefore has work-unit/us/unit dimensions.
        clock = r['clock_ghz']*1000
        rate = r['work_per_cycle']*clock
        pool = r['pool']
        active = min(tasks, r['units'])
        waves = (tasks+r['units']-1)//r['units']
        service = work/(tasks*steps*rate)
        first = max(r['latency_cycles']/clock, service) if work else 0.
        interval = first if stage.get('state_serial',False) else max(r['issue_cycles']/clock, service)
        span = waves*(first+(steps-1)*interval) if work else 0.
        pool_work[pool] += work/(r['units']*rate)
        predecessors = [end[d] for d in deps]
        if any(t is None for t in predecessors):
            start = finish = None
        else:
            start = max([available.get(pool,0.), *predecessors])
            finish = start+span
            available[pool] = finish
        end[sid] = finish
        stages.append(dict(stage, active_units=active, waves=waves,
                           service_per_step_us=service, interval_us=interval,
                           duration_us=span,start_us=start,end_us=finish))
    missing += ['hardware resource missing: '+r for r in sorted(unknown_resources)]
    hypothesis = [r for r in sorted(used_resources)
                  if resources[r]['evidence'] not in ('microbenchmark','trace_calibrated')]
    missing += ['hardware cost is not independently calibrated: '+r for r in hypothesis]
    if not graph.get('qualified_mapping',False):
        missing.append('physical tiling / ISA expansion / overlap is not fully qualified')
    candidate = max(end.values(), default=0.) if all(v is not None for v in end.values()) else None
    complete = not missing and candidate is not None
    return {'status':'qualified_resource_model' if complete else 'unqualified_mechanism_candidate',
            'profile':profile['name'], 'op':graph.get('op'),
            'predicted_us':candidate if complete else None, 'candidate_us':candidate,
            'mechanism_estimate_us':candidate,
            'lower_bound_us':max(pool_work.values()) if pool_work else None, 'upper_bound_us':None,
            'bound_scope':'resource floor of accounted logical work; missing work and traffic excluded',
            'evidence_level':graph.get('evidence_level','explicit_graph'),
            'missing_evidence':list(dict.fromkeys(missing)), 'stages':stages,
            'cost_scope':'aggregate device resources; excludes unmodeled launch/control and host overhead',
            'complexity':'O(number of macro stages); no token or trace replay'}


def build_graph(plan):
    """Adapt existing extension structure into a hardware-independent macro DAG.

    The adapter never consumes plan.predicted_us, reference_us, fitted coefficients
    or a target shape's trace. Each stage is a mathematical operation class.
    Existing 950 tiling fields are retained only as provenance, not re-used as
    another architecture's physical core count or UB capacity.
    """
    op = plan['op']; w = plan.get('work',{}); a = plan.get('attrs',{})
    shape = plan['shape']; dims = list(shape.values()) if isinstance(shape,Mapping) else list(shape)
    out = {'op':op,'shape':shape,'stages':[], 'qualified_mapping':False,
           'evidence_level':'source_logical_work', 'source_soc':plan.get('soc_version'),
           'source_tiling_status':plan.get('tiling',{}).get('status'),
           'missing_evidence':['logical work requires ISA expansion and physical traffic qualification',
                               'launch/control and synchronization costs not independently calibrated']}
    if plan.get('status') == 'unsupported':
        out.update(status='unsupported',missing_evidence=[plan.get('unsupported_reason',plan.get('reason','unsupported'))])
        return out
    out['missing_evidence'] += plan.get('adapter_missing_evidence',[])
    out['call_attrs'] = plan.get('call_attrs',a)
    stages = out['stages']

    def add(sid, resource, work, tasks=1, steps=1, serial=False, deps=None):
        if work is None:
            out['missing_evidence'].append(sid+' work is unknown'); return
        if work == 0:
            return
        predecessors = ([stages[-1]['id']] if stages else []) if deps is None else deps
        stages.append(dict(id=sid,resource=resource,work=work,parallel_tasks=max(1,tasks),
                           serial_steps=steps,state_serial=serial,dependencies=predecessors,
                           evidence='source_logical_work'))

    def vec(sid, count, tasks=1, resource='vector_alu',steps=1,serial=False):
        add(sid, resource, count, tasks,steps,serial)

    def memory():
        if 'logical_tensor_bytes' in w:
            add('logical_tensor_traffic','memory',w['logical_tensor_bytes'])
            out['missing_evidence'].append('unique logical tensor bytes are not physical DRAM/cache traffic')
        else:
            out['missing_evidence'].append('logical and physical tensor traffic are incomplete')

    memory()
    dtype = a.get('dtype','bf16')
    cube = {'bf16':'cube_bf16','fp16':'cube_fp16','fp32':'cube_hf32'}.get(dtype,'cube_fp8')
    if op in ('rmsnorm','rms_norm'):
        rows, cols = dims
        vec('square_reduce',rows*(2*cols-1),rows)
        vec('mean_epsilon',2*rows,rows)
        vec('rsqrt',w['rsqrt'],rows,'vector_rsqrt')
        vec('scale_output',2*rows*cols,rows)
    elif op == 'addmm':
        m,n,k = dims
        add('matmul',cube,w['matmul_flops'],m*n)
        vec('scale_combine',w['combine_additions']+w['nonunit_scale_multiplications'],m*n)
        out['missing_evidence'].append('matmul tile task count / add fusion / launch count unresolved')
    elif op == 'gmm2':
        m,k,f,e = dims
        counts=w.get('expert_counts',a.get('expert_counts',[]))
        for expert,tokens in enumerate(counts):
            if tokens:
                add('expert_'+str(expert)+'_gmm2','cube_fp8',2*tokens*f*k,tokens,
                    deps=['logical_tensor_traffic'] if stages and stages[0]['id']=='logical_tensor_traffic' else [])
        out['missing_evidence'].append('MX tensor traffic / expert tile scheduler unknown')
    elif op == 'lightning_indexer':
        b,q,length,heads,d = dims
        add('score_dot',cube,w['logical_dot_flops'],b*q*heads)
        vec('relu_head_reduce',w['relu_comparisons']+w['head_weight_multiplications']+w['head_reduction_additions'],b*q)
        add('topk','vector_alu',w.get('topk_comparisons'),b*q)
        out['missing_evidence'].append('TopK sort/merge algorithm and padded score tiles unknown')
    elif op == 'mhc_sinkhorn':
        t,n = dims; iters=a['num_iters']
        vec('softmax_prepare',w['softmax_max_comparisons']+w['softmax_subtractions'],t)
        vec('softmax_exp',w['exp'],t,'vector_exp')
        # Row/column passes remain serial within each independent token matrix.
        vec('normalization_reduce',w['reduction_additions']+w['epsilon_additions'],t,
            steps=2*iters,serial=True)
        vec('normalization_div',w['logical_divisions'],t,'vector_div',steps=2*iters,serial=True)
        out['missing_evidence'].append('aggregate reduction/divide stages do not capture per-pass interleaving')
    elif op == 'mhc_post':
        t,n,d = dims
        vec('residual_mix',w['residual_multiplications']+w['residual_reduction_additions'],t)
        vec('post_combine',w['post_multiplications']+w['combine_additions'],t)
        vec('output_cast',t*n*d,t,'vector_cast')
    elif op in ('mhc_pre','mhc_pre_sinkhorn'):
        t,n,d = dims
        vec('rms_prepare',sum(w[k] for k in ('rms_squares','rms_reduction_additions','rms_mean_scales','rms_eps_additions')),t)
        vec('rms_rsqrt',w['rms_rsqrt'],t,'vector_rsqrt')
        vec('gamma',w['gamma_multiplications'],t)
        add('projection','cube_hf32',w['projection_flops'],t)
        vec('mix_affine',sum(w[k] for k in ('mix_rms_multiplications','affine_multiplications','affine_additions')),t)
        vec('sigmoid',w['sigmoid'],t,'vector_sigmoid')
        vec('residual_weights',sum(w[k] for k in ('hpost_scale_multiplications','hc_epsilon_additions','hin_multiplications','hin_reduction_additions')),t)
        if op == 'mhc_pre_sinkhorn':
            sw=w['sinkhorn_logical_work']; iters=a['num_iters']
            vec('sinkhorn_exp',sw['exp'],t,'vector_exp')
            vec('sinkhorn_normalize',sw['reduction_additions']+sw['epsilon_additions']+sw['softmax_max_comparisons']+sw['softmax_subtractions'],t,steps=2*iters,serial=True)
            vec('sinkhorn_div',sw['logical_divisions'],t,'vector_div',steps=2*iters,serial=True)
    elif op.startswith('kda'):
        b,s,hq,hv,k,v = dims; tasks=b*hv
        if a['mode'] == 'recurrent':
            vec('state_recurrence',w['recurrence_scalar_flops'],tasks,steps=s,serial=True)
            out['missing_evidence'].append('gate/beta, qk normalization and per-token state handoff omitted')
        else:
            tasks = plan['tiling']['state_chains']
            head_chunks = plan['tiling']['parallel_chunk_head_tasks']
            for sid,key in [('qk','aqk_useful_dot_flops'),('kk','akk_useful_dot_flops'),('wu','post_wu_dense_equivalent_flops')]:
                add(sid,cube,w[key],head_chunks)
            add('triangular_solve','vector_alu',w['triangular_inverse_flops'],tasks)
            chunks = plan['tiling'].get('max_chunks_per_sequence', plan['tiling']['chunks_per_sequence'])
            add('state_update',cube,w['fwd_h_matmul_flops'],tasks,chunks,True)
            vec('state_decay',w['fwd_h_state_decay_multiplies'],tasks,steps=chunks,serial=True)
            add('finalize',cube,w['finalize_dense_equivalent_flops'],head_chunks)
            out['missing_evidence'].append('chunk gate/prepare/triangular solve work and fusion layout unknown')
    elif op in ('moe_ffn','groupedmatmulswiglugmm','grouped_matmul_swiglu_quant_v2'):
        m,k,f,e = dims; counts=w['expert_counts']; outdim=a.get('output_dim',k)
        # Preserve expert skew explicitly; independent experts are separate DAG
        # chains, rather than dividing total work by a mean-token expert.
        root_deps=[stages[-1]['id']] if stages else []
        for expert,tokens in enumerate(counts):
            if not tokens: continue
            tag='expert_'+str(expert)+'_'
            add(tag+'gmm1','cube_fp8',4*tokens*k*f,tokens,deps=root_deps)
            vec(tag+'swiglu_negate',tokens*f,tokens)
            vec(tag+'swiglu_exp',tokens*f,tokens,'vector_exp')
            vec(tag+'swiglu_add_one',tokens*f,tokens)
            vec(tag+'swiglu_div',tokens*f,tokens,'vector_div')
            vec(tag+'swiglu_multiply',tokens*f,tokens)
            vec(tag+'mx_cast',tokens*f,tokens,'vector_cast')
            if 'gmm2_useful_flops' in w:
                add(tag+'gmm2','cube_fp8',2*tokens*f*outdim,tokens)
        out['missing_evidence'].append('MX exponent reduction / scale traffic / expert tile scheduler unknown')
    elif op == 'mhc':
        out['missing_evidence'].append('compose component DAGs explicitly; external attention/MLP excluded')
        for component in plan.get('components',[]):
            child=build_graph(component); prefix=component['op']+'/'
            deps=[stages[-1]['id']] if stages else []
            for node in child['stages']:
                rewritten=dict(node,id=prefix+node['id'],dependencies=[prefix+d for d in node['dependencies']] or deps)
                stages.append(rewritten)
            out['missing_evidence']+=child['missing_evidence']
    else:
        out['missing_evidence'].append('No structural resource adapter for '+op)
    return out


def predict_plan(plan, profile):
    return predict_graph(build_graph(plan), profile)


def expand_instructions(graph, templates):
    """Expand source-defined work to ISA counts, without total-time fitting.

    templates is keyed by operation resource, e.g. vector_alu, never operator or
    shape. Each record has source/evidence and instructions_per_work_unit mapping
    e.g. {'isa:RV_ADD': 1/64}. A per-source implementation with different packing
    should build its own explicit graph instead of pretending this mapping is
    universal. Latency/II are still supplied separately by the hardware profile.
    """
    import copy
    result=copy.deepcopy(graph); rewritten=[]; replacements={}
    for stage in result['stages']:
        deps=[replacements.get(d,d) for d in stage.get('dependencies',[])]
        template=templates.get(stage['resource'])
        if template is None:
            rewritten.append(dict(stage,dependencies=deps));continue
        if set(template)!={'source','evidence','instructions_per_work_unit'} or not template['source']:
            raise ValueError('Source-backed instruction expansion required')
        if template['evidence'] not in ('source','trace_calibrated'):
            raise ValueError('Instruction expansion needs source/count evidence')
        entries=template['instructions_per_work_unit']
        if not entries:raise ValueError('Instruction expansion cannot be empty')
        for i,(opcode,factor) in enumerate(entries.items()):
            if not opcode.startswith('isa:'):raise ValueError('Expected isa: opcode resource')
            _number(factor,'instructions_per_work_unit',True)
            count=math.ceil(stage['work']/stage['parallel_tasks']/stage['serial_steps']*factor)*stage['parallel_tasks']*stage['serial_steps']
            sid=stage['id']+'/isa'+str(i)
            rewritten.append(dict(stage,id=sid,resource=opcode,work=count,dependencies=deps,
                                  evidence=template['evidence'],source=template['source']))
            deps=[sid]
        replacements[stage['id']]=sid
    result['stages']=rewritten
    return result


def extract_opcode_evidence(paths):
    """Read existing CA DBs once; no simulation and no operator-time regression.

    Returns observed latency distributions (ticks/cycles) and provenance per
    opcode/core-type. These include data length, queueing and dependency effects;
    they are NOT interchangeable with reciprocal throughput. Therefore issue
    and work_per_cycle remain None until independently measured/ISA-qualified.
    This evidence supports source-template instruction cost experiments without
    inventing a scalar FLOPs-to-instructions conversion or hiding missing II.
    """
    import hashlib
    from pathlib import Path
    import sqlite3
    distributions=defaultdict(lambda:defaultdict(int)); provenance=[]; digests=set()
    for value in paths:
        path=Path(value)
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
        if digest.hexdigest() in digests:
            raise ValueError('Duplicate CA DB evidence')
        digests.add(digest.hexdigest())
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
            if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('Corrupt CA DB')
            info=dict(db.execute('SELECT KernelInfoAttr,KernelInfoVal FROM KernelInfo'))
            if str(info.get('helper_version'))!='0.8.0' or int(info.get('contract_version',-1))!=7:
                raise ValueError('Unsupported CA helper contract')
            frequency=float(info['frequency_ghz']);_number(frequency,'frequency_ghz',True)
            count,start,end=db.execute('SELECT COUNT(*),MIN(ExecInstrTickStart),MAX(ExecInstrTickEnd) FROM ExecutedInstructions').fetchone()
            if not count or int(info.get('instrs_exec_num',-1))!=count or int(info.get('exec_time_ticks',-1))!=end-start:
                raise ValueError('Incomplete instruction population or makespan')
            sql='''SELECT ExecInstrName,CoreTypeId,ExecInstrTickEnd-ExecInstrTickStart,COUNT(*)
                   FROM ExecutedInstructions WHERE ExecInstrIsDurationExcluded=0
                   GROUP BY ExecInstrName,CoreTypeId,ExecInstrTickEnd-ExecInstrTickStart'''
            for opcode,core,duration,count in db.execute(sql):
                if duration<0:raise ValueError('Negative instruction duration')
                distributions[(opcode,core,frequency)][duration]+=count
        provenance.append({'path':str(path),'sha256':digest.hexdigest(),
                           'kernel':info.get('name'),'frequency_ghz':frequency})
    records=[]
    for (opcode,core,frequency),histogram in sorted(distributions.items()):
        total=sum(histogram.values());middle=(total-1)//2; cumulative=0;median=None
        for duration,count in sorted(histogram.items()):
            cumulative+=count
            if cumulative>middle:median=duration;break
        records.append({'resource':'isa:'+opcode,'core_type_id':core,'frequency_ghz':frequency,
                        'instruction_records':total,'duration_ticks_min':min(histogram),
                        'duration_ticks_median_lower':median,'duration_ticks_max':max(histogram),
                        'duration_ticks_mean':sum(d*n for d,n in histogram.items())/total,
                        'issue_cycles':None,'work_per_cycle':None,
                        'qualification':'observed opcode latency envelope; operand size and queue effects pooled; not throughput'})
    return {'schema_version':1,'source_dbs':provenance,'opcode_observations':records,
            'qualification':'shared resource evidence only; no operator makespan was used as regression target'}


def plan_from_case(op, shape, attrs=None):
    """Bridge actual collection-manifest attrs to the structural schema.

    `call_attrs` always preserves the complete caller contract. Unrepresented
    flags produce explicit adapter_missing_evidence rather than being silently
    discarded. Invalid/unsupported contracts return a plan with the reason.
    """
    from .extensions import plan as extension_plan
    import copy
    original=copy.deepcopy(attrs or {}); normalized=copy.deepcopy(original); gaps=[]
    for key in ('dtype','weight_dtype','output_dtype','gamma_dtype','weights_dtype','state_dtype','gate_dtype'):
        if isinstance(normalized.get(key),str):normalized[key]=normalized[key].lower()
    if op.startswith('kda'):
        for name in ('use_gate_in_kernel','output_final_state','export_gk'):
            if name in normalized and type(normalized[name]) is not bool:
                return {'op':op,'shape':list(shape),'attrs':normalized,'call_attrs':original,
                        'status':'unsupported','reason':name+' must be bool','work':{}}
        if normalized.get('chunk_size','absent') is None:
            normalized.pop('chunk_size')
        if 'gate_dtype' in normalized:
            gate_dtype=normalized.pop('gate_dtype')
            gaps.append('gate dtype '+gate_dtype+' preserved in call_attrs; gate instruction/traffic expansion unresolved')
        if 'use_gate_in_kernel' in normalized:
            flag=normalized.pop('use_gate_in_kernel')
            if flag:
                gaps.append('use_gate_in_kernel=True: raw gate activation parameters/algorithm are not expanded; core recurrence only')
            else:
                gaps.append('use_gate_in_kernel=False: caller gate is prepared; internal decay/exp instruction expansion still unresolved')
            # Core recurrence is common; deliberately do not invent safe/unsafe
            # raw-gate semantics from this single caller flag.
            normalized.setdefault('gate_mode','preactivated')
        if 'output_final_state' in normalized:
            flag=normalized.pop('output_final_state')
            gaps.append('output_final_state='+str(flag)+': final state storage traffic not yet expanded')
        if 'export_gk' in normalized:
            flag=normalized.pop('export_gk')
            gaps.append('export_gk='+str(flag)+': Gk export work/traffic is separate from generic export_intermediates')
    if op == 'gmm2':
        # Reuse only the validated shape/histogram parser; remove first-layer work.
        try:
            source=extension_plan('moe_ffn',shape,**normalized)
        except (ValueError,TypeError) as error:
            source={'status':'unsupported','reason':str(error),'shape':list(shape),'work':{}}
        if source.get('status')!='unsupported':
            source['work']={k:v for k,v in source['work'].items()
                            if k in ('valid_expert_token_rows','unused_capacity_rows','expert_counts',
                                     'active_experts','max_over_mean_expert_load','gmm2_useful_flops')}
            source['stage_dag']=['MX GMM2 [M,F] x [E,F,K] -> BF16 [M,K]']
            source['api']='aclnnGroupedMatmulV5'
            m,k,f,e=source['shape'].values(); groups=(f+63)//64
            tokens=source['work']['valid_expert_token_rows']; active=source['work']['active_experts']
            source['attrs']['output_dtype']='bf16'
            source['tensor_shapes']={'x':[m,f],'x_scale':[m,groups,2],
                'weight':[e,f,k],'weight_scale':[e,groups,k,2],'output':[m,k]}
            source['work']['logical_tensor_bytes']=tokens*f+active*f*k+tokens*groups*2+active*groups*k*2+2*tokens*k
            source['work']['traffic_scope']='valid-row input/output plus active expert weights/scales, no reload/padding/cache'
            source['kernel_package_path']='opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/grouped_matmul_apt/'
            # This collector uses K as final output dimension. Arbitrary output
            # widths require an explicit graph instead of changing axis meaning.
            if normalized.get('output_dim',list(source['shape'].values())[1]) != list(source['shape'].values())[1]:
                source.update(status='unsupported',reason='gmm2 case contract requires output_dim=K')
        source['op']='gmm2'
    else:
        try:
            source=extension_plan(op,shape,**normalized)
        except (ValueError,TypeError) as error:
            source={'op':op,'shape':list(shape),'attrs':normalized,'status':'unsupported',
                    'reason':'structural adapter rejected caller contract: '+str(error),'work':{}}
    source['call_attrs']=original
    source['adapter_missing_evidence']=gaps
    return source


def predict_case(op, shape, attrs, profile):
    """Return prediction with its exact structural plan/graph for freezing."""
    plan=plan_from_case(op,shape,attrs)
    graph=build_graph(plan)
    prediction=predict_graph(graph,profile)
    return dict(prediction,graph=graph,structural_plan=plan)
