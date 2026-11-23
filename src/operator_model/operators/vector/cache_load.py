"""Work for a materialized inference cache load with a tail-axis encoding.

The input shape describes valid logical output rows. Packed tensors may be padded
on the last dimension. Neither storage capacity nor a DMA schedule is inferred.
"""
from collections.abc import Sequence
from math import prod

from operator_model.workload import _counts


def canonical_dtype(dtype):
    if not isinstance(dtype, str):
        raise ValueError('cache_load requires the input storage dtype')
    name = dtype.lower().removeprefix('torch.')
    if name in ('mxfp4', 'fp4', 'float4_e2m1fn_x2'):
        return 'mxfp4'
    if name in ('fp8_e4m3fn', 'float8_e4m3fn', 'fp8'):
        return 'fp8_e4m3fn'
    raise ValueError('cache_load requires packed E2M1 or E4M3FN cache storage')


def _shape(value, name):
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
            or any(type(d) is not int or d < 0 for d in value)):
        raise ValueError(f'{name} must be a sequence of nonnegative integer dimensions')
    return tuple(value)


def describe(shape, dtype, attrs):
    """Return (public work, structural projection) for quantized storage to BF16."""
    dtype = canonical_dtype(dtype)
    dims = _shape(shape, 'cache_load logical shape')
    if len(dims) < 2 or dims[-1] == 0:
        raise ValueError('cache_load requires [...,D] with a row dimension and a positive width')
    attrs = {} if attrs is None else attrs
    supported = {'storage_shape', 'scale_shape', 'scale_dtype', 'output_dtype', 'group_size',
                 'storage_precision', 'materialize', 'implementation'}
    if set(attrs) - supported:
        raise ValueError('Unsupported cache_load attributes: ' + str(sorted(set(attrs) - supported)))
    if 'storage_precision' in attrs and canonical_dtype(attrs['storage_precision']) != dtype:
        raise ValueError('storage_precision disagrees with cache_load dtype')
    if attrs.get('materialize', True) is not True:
        raise ValueError('cache_load requires a materialized BF16 output')
    if str(attrs.get('output_dtype', 'bf16')).lower().removeprefix('torch.') not in ('bf16', 'bfloat16'):
        raise ValueError('cache_load result dtype must be BF16')

    rows, width = prod(dims[:-1]), dims[-1]
    fp4 = dtype == 'mxfp4'
    impl = attrs.get('implementation')
    if 'implementation' in attrs and impl != 'aclnnAntiMxQuant':
        raise ValueError('The supported explicit cache_load implementation is aclnnAntiMxQuant')
    if impl and (not fp4 or width % 2 or not rows or len(dims) > 7):
        raise ValueError('aclnnAntiMxQuant cache model requires nonempty contiguous ND E2M1, even width and rank 2..7')
    group = attrs.get('group_size', 32)
    if type(group) is not int or group < 1 or (fp4 and group != 32):
        raise ValueError('cache_load expects positive group_size; E2M1 currently requires group_size=32')
    scale_dtype = 'e8m0' if fp4 else 'fp32'
    aliases = ('e8m0', 'float8_e8m0fnu') if fp4 else ('fp32', 'float32')
    if str(attrs.get('scale_dtype', scale_dtype)).lower().removeprefix('torch.') not in aliases:
        raise ValueError(f'cache_load {dtype} requires {scale_dtype} scales')

    value_shape = (*dims[:-1], (width+1)//2) if fp4 else dims
    scale_shape = (*dims[:-1], (width+63)//64, 2) if fp4 else dims[:-1]
    for label, expected in [('storage_shape', value_shape), ('scale_shape', scale_shape)]:
        if label in attrs and _shape(attrs[label], label) != expected:
            raise ValueError(f'{label} disagrees with the encoded operands; expected {expected}')
    value_bytes = prod(value_shape)  # Both packed x2 and single FP8 slots occupy one byte.
    scale_slots = prod(scale_shape)
    used_scales = rows*((width+31)//32) if fp4 else rows
    scale_bytes = scale_slots if fp4 else scale_slots*4
    values = rows*width
    # One mathematical scale application per valid output. The actual operation
    # precision/sequence is unspecified by the producer (E8M0 may use exponent logic).
    work = _counts(dtype, value_bytes+scale_bytes, 2*values, vec={'bf16' if impl else 'unspecified': values},
                   classes={'scale_application': values}, notes=[
                       'Scale applications count logical results, not issued SIMD instructions. '
                       'Decode, narrowing and unpacking are listed separately without adding ISA counts.',
                       'Logical reads count the full supplied scale tensor including its paired padding. '
                       'Physical off-chip traffic and on-chip temporary storage are unknown.'])
    if impl:
        work['notes'].append('AntiMxQuant casts each FP4 value and group scale to BF16, then multiplies in BF16.')
    work.update(cache_value_bytes=value_bytes, cache_scale_bytes=scale_bytes, valid_values=values,
                logical_scale_values=used_scales, scale_storage_slots=scale_slots,
                unused_scale_slots=scale_slots-used_scales, input_value_decodes=values,
                bf16_result_values=values, padded_value_lanes=rows*(width % 2) if fp4 else 0)
    projection = dict(method='materialized_cache_dequantization', qualification='work_only',
                      storage_precision='fp4' if fp4 else 'fp8', scale_dtype=scale_dtype,
                      logical_shape=list(dims), storage_shape=list(value_shape), scale_shape=list(scale_shape),
                      output_dtype='bf16', scale_group_size=group if fp4 else None,
                      valid_rows=rows, scale_mode='block32_paired' if fp4 else 'per_row',
                      latency_us=None)
    return work, projection


def price(shape, chip, attrs):
    """Price the named compatible implementation of an E2M1 cache load."""
    if not attrs or attrs.get('implementation') != 'aclnnAntiMxQuant':
        raise ValueError('Timed cache_load requires explicit implementation=aclnnAntiMxQuant')
    work, projection = describe(shape, 'mxfp4', attrs)
    from .cache_load_antimx import schedule
    us, plan, status = schedule(projection['valid_rows'], shape[-1], work, chip)
    projection.update(method='aclnnAntiMxQuant_E2M1_BF16', qualification='explicit_implementation',
                      timing_qualification=status, target_plan=plan, latency_us=us,
                      calibration_scope='Contiguous even-width E2M1 ND + paired block32 E8M0, BF16 output. '
                                        '950DT/PR at 1..128 folded rows with width 64/128/512/576; '
                                        'the software capture alone does not identify a kernel.')
    return us, work, projection
