"""Source-backed, uncalibrated LightningIndexer/mHC structural models.

Shapes: indexer=(B,Q,L,G,D), pre/post/chain=(T,N,D), sinkhorn=(T,N).
T may be B*S flattened from the public BSND API. Counts are logical scalar
work, never ISA instructions or measured cycles. See docs/extensions/indexer-mhc.md.
"""
from collections.abc import Mapping
import math

SOC = 'Ascend950DT_9582'
CANN = '9.2.1'
SOURCE_COMMIT = '14d022b1a4c3df263bf697534e3c25c7c992ff5d'
_PACKAGE = 'opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/'
_OPERATORS = {
    'lightning_indexer': ('aclnnLightningIndexer', 'lightning_indexer'),
    'mhc_pre': ('aclnnMhcPre', 'mhc_pre_apt'),
    'mhc_sinkhorn': ('aclnnMhcSinkhorn', 'mhc_sinkhorn_apt'),
    'mhc_pre_sinkhorn': ('aclnnMhcPreSinkhorn', 'mhc_pre_sinkhorn_apt'),
    'mhc_post': ('aclnnMhcPost', 'mhc_post_apt'),
    'mhc': (None, None),
}
SHAPE_AXES = {'lightning_indexer': ('B', 'Q', 'L', 'G', 'D'),
              'mhc_sinkhorn': ('T', 'N')}


def _shape(op, shape):
    axes = SHAPE_AXES.get(op, ('T', 'N', 'D'))
    if isinstance(shape, Mapping):
        if set(shape) != set(axes):
            raise ValueError(f'shape mapping must have exactly {axes}')
        values = tuple(shape[key] for key in axes)
    else:
        values = tuple(shape)
    if len(values) != len(axes) or any(type(v) is not int or v <= 0 for v in values):
        raise ValueError(f'expected positive integer dimensions {axes}')
    return values, axes


def _integer(value, name, low, high=None):
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(f'{name} must be an integer in [{low}, {high or "unbounded"}]')
    return value


def _bool(value, name):
    if type(value) is not bool:
        raise ValueError(f'{name} must be a bool')
    return value


def _epsilon(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def _attrs(attrs, defaults):
    extra = set(attrs) - set(defaults)
    if extra:
        raise ValueError(f'unknown attributes: {sorted(extra)}')
    return defaults | attrs


def _base(op, shape, axes, attrs):
    api, package = _OPERATORS[op]
    return {'op': op, 'shape': list(shape), 'shape_axes': list(axes), 'attrs': attrs,
            'soc_version': SOC, 'cann_version': CANN, 'status': 'uncalibrated',
            'predicted_us': None, 'ideal_compute_efficiency': None, 'ca_measurements': 0,
            'api': api, 'library_family': 'ops-transformer',
            'kernel_package_path': _PACKAGE + package + '/' if package else None,
            'kernel_selection_verified': False,
            'packaged_core_type': ('VectorCore' if op in ('mhc_post', 'mhc_sinkhorn')
                                   else 'MIX' if op != 'mhc' else None),
            'work': {}, 'tiling': {'status': 'unresolved_dispatch', 'active_aic': None,
                                 'active_aiv': None, 'tile_shape': None},
            'limitations': ['No CA measurement or numerical output validation for this extension.',
                            'Logical operation counts are not vector/Cube ISA instruction counts.',
                            'Unique tensor bytes exclude reloads, padding, workspace and cache behavior.',
                            'Source documentation is not proof of the selected packaged kernel.']}


def _unsupported(out, reason):
    out.update(status='unsupported', unsupported_reason=reason,
               unsupported_scope='initial_model_scope_or_documented_contract')
    return out


def plan(op, shape, **attrs):
    """Return a JSON-compatible structural plan; never invent latency.

    Unsupported layouts/dtypes/routes return status=unsupported. Malformed input
    (unknown attributes, invalid dimensions, invalid scalar types) raises ValueError.
    `mhc` is the *residual wrapper* around an external attention/MLP operation;
    it does not include that operation's compute or latency.
    """
    op = {'lightening_indexer': 'lightning_indexer', 'lightningindexer': 'lightning_indexer'}.get(op, op)
    if op not in _OPERATORS:
        raise ValueError(f'unknown operator: {op}')
    dims, axes = _shape(op, shape)
    if op == 'lightning_indexer':
        return _indexer(dims, axes, attrs)
    if op == 'mhc_sinkhorn':
        return _sinkhorn(dims, axes, attrs)
    if op == 'mhc':
        return _chain(dims, axes, attrs)
    return _mhc_tensor(op, dims, axes, attrs)


def _indexer(dims, axes, attrs):
    b, q, length, heads, d = dims
    a = _attrs(attrs, {'dtype': 'bf16', 'weights_dtype': None, 'layout': 'BSND',
                      'sparse_count': 512, 'sparse_mode': 3, 'return_values': False})
    a['weights_dtype'] = a['dtype'] if a['weights_dtype'] is None else a['weights_dtype']
    k = _integer(a['sparse_count'], 'sparse_count', 1)
    _integer(a['sparse_mode'], 'sparse_mode', 0)
    _bool(a['return_values'], 'return_values')
    out = _base('lightning_indexer', dims, axes, a)
    if a['layout'] != 'BSND':
        return _unsupported(out, 'TND and PA_BSND are library-supported, but their packing/block tables are not modeled yet.')
    if a['dtype'] not in ('bf16', 'fp16') or a['weights_dtype'] != a['dtype']:
        return _unsupported(out, '950 ordinary LightningIndexer requires matching BF16/FP16 query, key and weights; quantized APIs are separate.')
    if d != 128 or heads not in (8, 16, 24, 32, 64):
        return _unsupported(out, '950 contract requires D=128, G in {8,16,24,32,64}, key head count=1.')
    if not (k <= 2048 or k in (3072, 4096, 5120, 6144, 7168, 8192)):
        return _unsupported(out, 'sparse_count must be 1..2048 or one of 3072,4096,5120,6144,7168,8192.')
    if a['sparse_mode'] not in (0, 3):
        return _unsupported(out, 'Only sparse_mode=0 (dense) or 3 (right-down causal) is supported.')
    # Sum max(0, min(L, L-Q+i+1)); no loop over large Q/L.
    visible = q * length if a['sparse_mode'] == 0 else (
        q * (length - q) + q * (q + 1) // 2 if q <= length else length * (length + 1) // 2)
    pairs = b * visible
    out.update(
        input_shapes={'query': [b, q, heads, d], 'key': [b, length, 1, d], 'weights': [b, q, heads]},
        output_shapes={'sparse_indices': [b, q, 1, k],
                       'sparse_values': [b, q, 1, k] if a['return_values'] else None},
        stage_dag=['query @ key.T per head', 'ReLU', 'weight and sum heads', 'TopK indices/optional values'],
        work={'logical_valid_query_key_pairs': pairs, 'dense_query_key_pairs': b*q*length,
              'logical_dot_flops': 2*pairs*heads*d, 'dense_dot_flops': 2*b*q*length*heads*d,
              'relu_comparisons': pairs*heads, 'head_weight_multiplications': pairs*heads,
              'head_reduction_additions': pairs*(heads-1),
              'topk_candidate_elements': pairs, 'topk_lists': b*q,
              'topk_algorithm': None, 'topk_comparisons': None,
              'output_slots': b*q*k,
              'logical_tensor_bytes': 2*(b*q*heads*d+b*length*d+b*q*heads)+(4+2*a['return_values'])*b*q*k},
        api_fixed_attrs={'preTokens': 2**63-1, 'nextTokens': 2**63-1},
        execution_verified=False)
    out['limitations'] += [
        'Causal logical work may differ from executed/padded tiles; dense work is a reference, not a selected tiling.',
        'TopK does not reduce the preceding score work; sorting algorithm and merge route need trace/source validation.',
        'return_values=True is documented for training only; graph mode and padding values are not modeled.',
        'Model config supplies dimensions, but does not prove vLLM dispatch, cache layout or local TP head count.']
    return out


def _sinkhorn(dims, axes, attrs):
    t, n = dims
    a = _attrs(attrs, {'dtype': 'fp32', 'num_iters': 20, 'eps': 1e-6, 'return_intermediates': False})
    iters = _integer(a['num_iters'], 'num_iters', 1, 100)
    _epsilon(a['eps'], 'eps')
    aux = _bool(a['return_intermediates'], 'return_intermediates')
    out = _base('mhc_sinkhorn', dims, axes, a)
    if n not in (4, 6, 8) or a['dtype'] != 'fp32':
        return _unsupported(out, 'MhcSinkhorn supports FP32 with N in {4,6,8}.')
    aligned_n = (n + 7)//8*8
    norm_shape = [2*iters*t*n*aligned_n]
    sum_shape = [2*iters*t*aligned_n]
    out.update(
        input_shapes={'x': [t, n, n]},
        output_shapes={'output': [t, n, n], 'norm_out': norm_shape if aux else None,
                       'sum_out': sum_shape if aux else None},
        stage_dag=['row softmax + eps', 'column normalize', '(row normalize -> column normalize) * (I-1)'],
        work={'exp': t*n*n, 'softmax_max_comparisons': t*n*(n-1),
              'softmax_subtractions': t*n*n, 'normalization_passes_after_softmax': 2*iters-1,
              'logical_divisions': 2*iters*t*n*n, 'reduction_additions': 2*iters*t*n*(n-1),
              'epsilon_additions': t*n*n+(2*iters-1)*t*n,
              'logical_tensor_bytes': 8*t*n*n+(4*(math.prod(norm_shape)+math.prod(sum_shape)) if aux else 0)})
    return out


def _mhc_tensor(op, dims, axes, attrs):
    t, n, d = dims
    if op == 'mhc_post':
        a = _attrs(attrs, {'dtype': 'bf16', 'has_h_res': True})
        _bool(a['has_h_res'], 'has_h_res')
    else:
        defaults = {'dtype': 'bf16', 'norm_eps': 1e-6, 'hc_eps': 1e-6}
        defaults |= ({'num_iters': 20, 'need_backward': False} if op == 'mhc_pre_sinkhorn'
                     else {'alpha_count': 3, 'has_gamma': False, 'return_intermediates': False})
        a = _attrs(attrs, defaults)
        _epsilon(a['norm_eps'], 'norm_eps'); _epsilon(a['hc_eps'], 'hc_eps')
    out = _base(op, dims, axes, a)
    if a['dtype'] not in ('bf16', 'fp16'):
        return _unsupported(out, 'This plan requires BF16/FP16 x/hIn/hOut with FP32 projection/mixing tensors.')
    if op == 'mhc_post':
        # No 950 N=4/D%128 limit is invented from A2/A3 documentation.
        res = a['has_h_res']
        out.update(input_shapes={'x': [t,n,d], 'h_out': [t,d], 'h_post': [t,n],
                                 'h_res': [t,n,n] if res else None},
                   output_shapes={'out': [t,n,d]},
                   stage_dag=['FP32 residual mixing (or identity)', 'hOut * hPost and combine', 'cast to x dtype'],
                   work={'residual_multiplications': t*n*n*d if res else 0,
                         'residual_reduction_additions': t*n*(n-1)*d if res else 0,
                         'post_multiplications': t*n*d, 'combine_additions': t*n*d,
                         'logical_tensor_bytes': 4*t*n*d+2*t*d+4*t*n+(4*t*n*n if res else 0)},
                   execution_family_reference='AIV_VECTOR (installed 9.2.1 candidate metadata; selected dispatch unmeasured)')
        out['limitations'].append('9.1 source card requires hRes; current source docs allow 950 hRes=None. Version equivalence and DMA size boundaries are unverified.')
        return out
    fused = op == 'mhc_pre_sinkhorn'
    if fused:
        iters = _integer(a['num_iters'], 'num_iters', 1, 100)
        backward = _bool(a['need_backward'], 'need_backward')
        if (n, iters) != (4, 20) or d not in (4096, 7168):
            return _unsupported(out, '950 MhcPreSinkhorn documented scope: N=4, I=20, D in {4096,7168}.')
        if a['norm_eps'] != 1e-6 or a['hc_eps'] != 1e-6:
            return _unsupported(out, 'MhcPreSinkhorn documented eps values are 1e-6.')
        if backward:
            return _unsupported(out, 'Backward intermediates are library-supported but outside this initial inference model.')
        alpha, gamma, aux = 3, False, False
    else:
        alpha = _integer(a['alpha_count'], 'alpha_count', 2, 3)
        gamma = _bool(a['has_gamma'], 'has_gamma')
        aux = _bool(a['return_intermediates'], 'return_intermediates')
        if n not in (4,6,8) or d > 16384 or d % 16:
            return _unsupported(out, '950 MhcPre requires N in {4,6,8}, D<=16384 and D%16=0.')
    width = n*n+2*n if alpha == 3 else 2*n
    bytes_in = 2*t*n*d+4*(width*n*d+alpha+width+(n*d if gamma else 0))
    bytes_out = 2*t*d+4*t*n+(4*t*n*n if alpha == 3 else 0)
    if aux:
        bytes_out += 4*t*(1+width+n)
    out.update(input_shapes={'x': [t,n,d], 'phi': [width,n*d], 'alpha': [alpha],
                             'bias': [width], 'gamma': [n,d] if gamma else None},
               output_shapes={'h_in': [t,d], 'h_post': [t,n], 'h_res': [t,n,n] if alpha == 3 else None,
                              'inv_rms': [t] if aux else None, 'h_mix': [t,width] if aux else None,
                              'h_pre': [t,n] if aux else None},
               stage_dag=['RMS of flattened x', 'x [optional gamma] @ phi.T', 'scale, split, sigmoid/affine',
                          'weighted residual reduction to hIn']+(['Sinkhorn on hRes'] if fused else []),
               work={'projection_flops': 2*t*n*d*width, 'rms_squares': t*n*d,
                     'rms_reduction_additions': t*(n*d-1), 'rms_mean_scales': t,
                     'rms_eps_additions': t, 'rms_rsqrt': t,
                     'gamma_multiplications': t*n*d if gamma else 0,
                     'mix_rms_multiplications': t*width,
                     'affine_multiplications': t*width, 'affine_additions': t*width,
                     'sigmoid': 2*t*n, 'hpost_scale_multiplications': t*n,
                     'hc_epsilon_additions': t*n*(1+(alpha == 2)),
                     'hin_multiplications': t*n*d, 'hin_reduction_additions': t*(n-1)*d,
                     'logical_tensor_bytes': bytes_in+bytes_out})
    if fused:
        # The fused public API flattens N*N; standalone Pre returns [T,N,N].
        out['output_shapes']['h_res'] = [t,n*n]
        sink = _sinkhorn((t,n), ('T','N'), {'num_iters': iters, 'eps': a['hc_eps']})
        out['work']['sinkhorn_logical_work'] = {k:v for k,v in sink['work'].items() if k != 'logical_tensor_bytes'}
        out['composition'] = {'fused_stages': ['mhc_pre', 'mhc_sinkhorn'], 'aclnn_calls': 1,
                              'do_not_add_separate_pre_sinkhorn_latency': True}
        out['limitations'].extend([
            'HF32 projection mode and phi value range require numerical verification; no inference aux outputs are modeled.',
            'Fused pre-stage work uses the algebraically equivalent normalize-after-projection reference; actual rounding and instruction placement are unverified.'])
    else:
        out['h_res_semantics'] = 'pre-Sinkhorn logits' if alpha == 3 else 'absent'
    return out


def _chain(dims, axes, attrs):
    a = _attrs(attrs, {'dtype': 'bf16', 'implementation': 'fused', 'num_iters': 20})
    _integer(a['num_iters'], 'num_iters', 1, 100)
    out = _base('mhc', dims, axes, a)
    if a['implementation'] not in ('fused', 'split'):
        return _unsupported(out, 'implementation must be fused or split.')
    if a['implementation'] == 'fused':
        components = [plan('mhc_pre_sinkhorn', dims, dtype=a['dtype'], num_iters=a['num_iters'])]
    else:
        components = [plan('mhc_pre', dims, dtype=a['dtype']),
                      plan('mhc_sinkhorn', dims[:2], num_iters=a['num_iters'])]
    components.append(plan('mhc_post', dims, dtype=a['dtype']))
    out.update(components=components, aclnn_calls=len(components),
               stage_dag=[c['op'] for c in components[:-1]]+['external_attention_or_mlp', 'mhc_post'],
               external_block_included=False,
               work={'scope': 'Per-component counts only. Do not sum unique-byte lower bounds as physical chain traffic.'})
    unsupported = [c['op']+': '+c['unsupported_reason'] for c in components if c['status']=='unsupported']
    if unsupported:
        return _unsupported(out, '; '.join(unsupported))
    out['limitations'].append('One mHC wrapper only; a transformer layer may have both attention and MoE wrappers. Model config alone does not prove dispatch.')
    return out
