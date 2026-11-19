"""950 KDA/MoE structural work plans; no calibrated timing or ISA counts.

Shapes: KDA (B,S,Hq,Hv,K,V); MoE (M,K,F,E), F is post-SwiGLU width.
Equivalent dictionaries use exactly those keys. Unsupported routes return a
reason, never a fallback latency. These are modelling plans, not ACLNN wrappers.
"""
from collections.abc import Mapping


OPS = ('kda', 'kda_recurrent', 'kda_chunk',
       'grouped_matmul_swiglu_quant_v2', 'groupedmatmulswiglugmm', 'moe_ffn')
_PACKAGE = 'opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/'


def _dims(shape, keys):
    if isinstance(shape, Mapping):
        if set(shape) != set(keys):
            raise ValueError('shape keys must be ' + ','.join(keys))
        values = [shape[k] for k in keys]
    else:
        values = list(shape)
    if len(values) != len(keys) or any(type(v) is not int or v <= 0 for v in values):
        raise ValueError('shape must contain positive integer dimensions: ' + ','.join(keys))
    return dict(zip(keys, values))


def _ceil(x, y):
    return (x + y - 1) // y


def _reject(out, reason):
    out.update(status='unsupported', reason=reason)
    return out


def plan(op, shape, **attrs):
    """Return JSON-ready useful-work metadata, with predicted_us always None.

    kda requires mode='recurrent' or 'chunk'; aliases fix it. Dense BSND only.
    MoE requires expert_counts or group_list + group_list_type (0 cumulative,
    1 counts). Routing histograms are post-dispatch, local to this rank.
    Unknown attributes fail explicitly so no silent dtype/layout assumption
    changes the route. See docs/extensions/kda-moe.md for model scope.
    """
    out = dict(op=op, shape=dict(shape) if isinstance(shape, Mapping) else list(shape),
               attrs=dict(attrs), soc_version='Ascend950DT_9582', cann_version='9.2.1',
               status='uncalibrated', predicted_us=None, ideal_compute_efficiency=None,
               ca_measurements=0, kernel_selection_verified=False,
               library_family='ops-transformer', work={}, tiling={}, stage_dag=[])
    if op not in OPS:
        return _reject(out, 'Unknown KDA/MoE operator')
    try:
        return _kda(out, shape, attrs) if op.startswith('kda') else _moe(out, shape, attrs)
    except (ValueError, TypeError) as error:
        return _reject(out, str(error))


def _kda(out, shape, attrs):
    allowed = {'mode', 'dtype', 'layout', 'chunk_size', 'gate_mode', 'state_dtype',
               'state_v_first', 'normalize_qk', 'export_intermediates'}
    alias_mode = {'kda_recurrent': 'recurrent', 'kda_chunk': 'chunk'}.get(out['op'])
    mode = attrs.get('mode', alias_mode)
    if mode == 'chunk':
        allowed.update(('cu_seqlens', 'dt_bias', 'initial_state'))
    if set(attrs) - allowed:
        return _reject(out, 'Unmodelled KDA attributes: ' + ','.join(sorted(set(attrs)-allowed)))
    if mode not in ('recurrent', 'chunk') or (alias_mode and alias_mode != mode):
        return _reject(out, 'Select an explicit, consistent recurrent or chunk mode')
    dims = _dims(shape, ('B', 'S', 'Hq', 'Hv', 'K', 'V'))
    b, s, hq, hv, k, v = dims.values()
    dtype = attrs.get('dtype', 'bf16')
    layout = attrs.get('layout', 'BSND')
    gate_mode = attrs.get('gate_mode', 'preactivated')
    state_dtype = attrs.get('state_dtype', 'fp32')
    if layout != 'BSND':
        return _reject(out, 'KDA work accounting currently requires rank-4 BSND inputs')
    if hv % hq:
        return _reject(out, 'Hv must be divisible by Hq')
    if gate_mode not in ('preactivated', 'raw_safe', 'raw_unsafe'):
        return _reject(out, 'Unknown gate_mode')
    for key in ('state_v_first', 'normalize_qk', 'export_intermediates'):
        if key in attrs and type(attrs[key]) is not bool:
            return _reject(out, key + ' must be bool')
    if mode == 'recurrent' and (dtype != 'bf16' or k != 128 or v not in (128, 256) or s > 8):
        return _reject(out, 'aclnnRecurrentKda requires BF16 q/k/v, K=128, V=128/256, S<=8; Q=9 needs a separately verified route')
    if state_dtype not in (('bf16', 'fp32') if mode == 'recurrent' else ('fp32',)):
        return _reject(out, 'Unsupported state dtype for selected KDA API')
    if mode == 'chunk' and (dtype not in ('bf16', 'fp16') or hv > 128 or
                            any(x < 16 or x > 256 or x % 16 for x in (k, v))):
        return _reject(out, 'Chunk KDA requires BF16/FP16, Hv<=128, K/V multiples of16 within16..256')
    if mode == 'recurrent' and ('chunk_size' in attrs or attrs.get('export_intermediates', False)):
        return _reject(out, 'Chunk size/intermediate exports are not recurrent-plan attributes')
    out['shape'] = dims
    out['attrs'] = dict(attrs, mode=mode, dtype=dtype, layout=layout,
                        gate_mode=gate_mode, state_dtype=state_dtype)
    out['output_shapes'] = {'output': [b, s, hv, v],
                            'state': [b, hv, v, k] if attrs.get('state_v_first', False) else [b, hv, k, v]}
    out['limitations'] = [
        'No CA timing, numerical execution, selected binary or dynamic instruction counts yet.',
        'Useful scalar arithmetic excludes gate/normalization, padding, copies and synchronization.',
        'Actual active cores, UB tiles and pipeline service times require host tiling and trace evidence.',
        'State-slot reuse and accepted speculative-token metadata are outside this dense plan.']
    out['work'] = {'gate_elements': b*s*hv*k, 'beta_elements': b*s*hv,
                   'state_elements': b*hv*k*v, 'state_tensor_bytes': b*hv*k*v*(4 if state_dtype == 'fp32' else 2)}
    if mode == 'recurrent':
        out.update(api='aclnnRecurrentKda', kernel_package_path=_PACKAGE+'recurrent_kda_apt/')
        out['work']['recurrence_scalar_flops'] = b*s*hv*(7*k*v+2*v)
        out['stage_dag'] = ['gate/beta preprocessing', 'for token: state decay -> prediction -> delta -> rank-one update -> output']
        out['tiling'] = {'status': 'source_schedule_only', 'logical_tasks': b*hv,
                         'task_mapping': 'task=blockIdx+j*GetBlockNum(); batch=task//Hv, head=task%Hv',
                         'serial_tokens_per_task': s, 'active_aic': None, 'active_aiv': None}
        return out
    c = attrs.get('chunk_size', 64)
    if type(c) is not int or c not in (64, 128):
        return _reject(out, 'chunk_size must be 64 or128')
    for flag in ('initial_state', 'dt_bias'):
        if flag in attrs and type(attrs[flag]) is not bool:
            return _reject(out, flag+' must be bool')
    offsets = attrs.get('cu_seqlens')
    length_classes = {s: b}
    if offsets is not None:
        if (b != 1 or not isinstance(offsets, (list, tuple)) or not 2 <= len(offsets) <= 1025
                or any(type(i) is not int for i in offsets) or offsets[0] != 0 or offsets[-1] != s):
            return _reject(out, 'cu_seqlens requires B=1, 1..1024 sequences and ordered integer offsets from 0 to S')
        length_classes = {}
        for start, end in zip(offsets, offsets[1:]):
            length = end-start
            if length < 0:
                return _reject(out, 'cu_seqlens must be nondecreasing')
            length_classes[length] = length_classes.get(length, 0)+1
    uniform = len(length_classes) == 1
    uniform_length = next(iter(length_classes)) if uniform else 0
    num_sequences = sum(length_classes.values())
    chains = num_sequences*hv
    chunk_count = square_sum = full_count = 0
    max_chunks = 0
    for length, count in length_classes.items():
        n, remainder = divmod(length, c)
        full_count += count*n
        chunk_count += count*(n+bool(remainder))
        max_chunks = max(max_chunks, n+bool(remainder))
        square_sum += count*(n*c*c+remainder*remainder)
    tokens = b*s
    pairs = (square_sum+tokens)//2
    strict_pairs = pairs-tokens
    out['output_shapes']['state'][0] = num_sequences
    out['work'].update(state_elements=chains*k*v, state_tensor_bytes=4*chains*k*v)
    out['attrs']['chunk_size'] = c
    out.update(api='aclnnChunkKdaFwd', kernel_package_path=_PACKAGE+'chunk_kda_fwd/')
    out['output_shapes'].update(aqk=[b,hv,s,c], akk=[b,hv,s,c])
    out['work'].update(
        aqk_useful_dot_flops=hv*2*k*pairs,
        akk_useful_dot_flops=hv*2*k*strict_pairs,
        post_wu_dense_equivalent_flops=hv*2*(k+v)*square_sum,
        fwd_h_matmul_flops=hv*4*tokens*k*v,
        fwd_h_state_decay_multiplies=hv*chunk_count*k*v,
        finalize_dense_equivalent_flops=hv*(2*tokens*k*v+2*v*square_sum),
        triangular_inverse_flops=None, total_executed_flops=None)
    out['stage_dag'] = ['Gate -> Prepare -> Post-WU -> FwdH (serial chunk state chain) -> Finalize']
    out['tiling'] = {'status': 'logical_partition_only', 'chunks_per_sequence': max_chunks if uniform else None,
                     'max_chunks_per_sequence': max_chunks, 'total_chunks': chunk_count,
                     'full_chunks_total': full_count,
                     'full_chunks': uniform_length//c if uniform else None,
                     'tail_tokens': uniform_length%c if uniform else None,
                     'state_chains': chains,
                     'parallel_chunk_head_tasks': hv*chunk_count, 'active_aic': None, 'active_aiv': None,
                     'single_kernel_internal_stages': True, 'actual_fusion_flags': None}
    out['limitations'].append('Stage names are internal dependencies, not five public launches; fast fusion depends on gate, dtype, tail and saved outputs.')
    return out


def _moe(out, shape, attrs):
    allowed = {'expert_counts', 'group_list', 'group_list_type', 'dtype', 'weight_dtype',
               'output_dtype', 'weight_transposed', 'layout', 'quant_mode', 'output_dim'}
    if set(attrs)-allowed:
        return _reject(out, 'Unmodelled MoE attributes: '+','.join(sorted(set(attrs)-allowed)))
    dims = _dims(shape, ('M', 'K', 'F', 'E'))
    m, k, f, e = dims.values()
    if e > 1024:
        return _reject(out, '950 V2 group count must be <=1024')
    dtype = attrs.get('dtype', 'fp8_e4m3fn')
    wdtype, odtype = attrs.get('weight_dtype', dtype), attrs.get('output_dtype', dtype)
    if any(x not in ('fp8_e4m3fn', 'fp8_e5m2') for x in (dtype, wdtype, odtype)):
        return _reject(out, 'Initial structural plan covers MXFP8 only; FP4/per-token have different pipelines')
    if attrs.get('quant_mode', 'mx') != 'mx' or attrs.get('layout', 'ND') != 'ND':
        return _reject(out, 'Initial route is ordinary V2, ND weights, MX scales; NZ/per-token need separate plans')
    if type(attrs.get('weight_transposed', False)) is not bool:
        return _reject(out, 'weight_transposed must be bool')
    if ('expert_counts' in attrs) == ('group_list' in attrs):
        return _reject(out, 'Supply exactly one post-dispatch local expert histogram: expert_counts or group_list')
    group_type = attrs.get('group_list_type', 1 if 'expert_counts' in attrs else None)
    if type(group_type) is not int or group_type not in (0, 1):
        return _reject(out, 'group_list_type must be explicit 0(cumulative) or1(counts)')
    if 'expert_counts' in attrs and group_type != 1:
        return _reject(out, 'expert_counts uses count semantics (group_list_type=1)')
    values = list(attrs.get('expert_counts', attrs.get('group_list')))
    if len(values) != e or any(type(x) is not int or x < 0 for x in values):
        return _reject(out, 'Expert histogram must contain E nonnegative integers')
    counts = values if group_type == 1 else [values[0]]+[b-a for a,b in zip(values, values[1:])]
    if any(x < 0 for x in counts) or sum(counts) > m:
        return _reject(out, 'Histogram must be monotone/nonnegative and total valid rows <=M')
    tokens, active = sum(counts), sum(x > 0 for x in counts)
    chain = out['op'] in ('groupedmatmulswiglugmm', 'moe_ffn')
    output_dim = attrs.get('output_dim', k)
    if type(output_dim) is not int or output_dim <= 0 or ('output_dim' in attrs and not chain):
        return _reject(out, 'output_dim is a positive GMM2 width and applies only to the MoE chain')
    out['shape'] = dims
    out['attrs'] = dict(attrs, dtype=dtype, weight_dtype=wdtype, output_dtype=odtype,
                        quant_mode='mx', layout='ND')
    out.update(api='aclnnGroupedMatmulSwigluQuantV2', kernel_package_path=_PACKAGE+'grouped_matmul_swiglu_quant_v2_apt/')
    out['work'] = {'valid_expert_token_rows': tokens, 'unused_capacity_rows': m-tokens,
                   'expert_counts': counts, 'active_experts': active,
                   'max_over_mean_expert_load': max(counts)/(tokens/e) if tokens else None,
                   'gmm1_useful_flops': 4*tokens*k*f, 'swiglu_elements': tokens*f,
                   'mx_output_logical_groups_32': tokens*_ceil(f, 32),
                   'mx_output_allocated_scale_elements': m*_ceil(f, 64)*2}
    out['tensor_shapes'] = {'x': [m,k], 'weight1': [e,2*f,k] if attrs.get('weight_transposed', False) else [e,k,2*f],
                            'weight1_scale': [e,2*f,_ceil(k,64),2] if attrs.get('weight_transposed', False) else [e,_ceil(k,64),2*f,2],
                            'x_scale': [m,_ceil(k,64),2], 'fusion_output': [m,f],
                            'output_scale': [m,_ceil(f,64),2]}
    out['tiling'] = {'status': 'unresolved_host_tiling', 'active_aic': None, 'active_aiv': None,
                     'reason': 'Expert histogram, TCubeTiling and ASWT tail scheduler determine tasks; not one core per expert'}
    out['stage_dag'] = ['GMM1 -> dequant -> SwiGLU -> MX quant (one fused kernel)']
    out['limitations'] = ['No CA calibration, numeric validation, runtime binary or host tiling observed.',
                          'Useful FLOPs are not padded MMAD counts; scalar SwiGLU counts are not instructions.',
                          'Post-dispatch histogram is required; batch/EP alone cannot determine local rows.',
                          'Input quantization, routing, dispatch/combine communication and TP collectives are not timed.',
                          'Unused capacity rows have no defined new output; zero experts do not imply zero launch latency.']
    if chain:
        out['work']['gmm2_useful_flops'] = 2*tokens*f*output_dim
        out['tensor_shapes']['weight2_logical'] = [e,f,output_dim]
        out['stage_dag'].append('GMM2 -> routed combine (separate operators, ABI integration pending)')
        out['gmm2_candidate_api'] = 'aclnnGroupedMatmulV5'
        out['limitations'].append('groupedmatmulswiglugmm is a composed FFN alias, not a confirmed single library kernel; GMM2 quant-scale ABI is not yet integrated.')
    return out
