"""System simulator interface. Chip files are read locally and require no CANN."""
from dataclasses import dataclass, replace
from math import ceil
from typing import Literal, Mapping, Optional, Sequence

from operator_model.chips import REFERENCE_NAME, load_chip
from operator_model.scheduling.reference import ALIASES, CUBE, FUSION, VECTOR, predict
from operator_model.workload import account_work, default_dtype


@dataclass(frozen=True)
class Estimate:
    """Device time in µs, algorithmic work, and provenance for one rank."""
    latency_us: Optional[float]
    status: Literal['estimated', 'extrapolated', 'unsupported', 'unavailable']
    reason: Optional[str] = None
    active_aiv: Optional[int] = None
    active_aic: Optional[int] = None
    chip_name: Optional[str] = None
    work: Optional[dict] = None
    projection: Optional[dict] = None
    rates: Optional[dict] = None
    kernel_count: Optional[int] = None
    dispatch_overhead_us: Optional[float] = None

    @property
    def scheduled_us(self) -> Optional[float]:
        """Serial call duration with the configured overhead outside the device window."""
        if self.latency_us is None or self.dispatch_overhead_us is None:
            return None
        return self.latency_us + self.dispatch_overhead_us


def _dtype_name(value):
    if not isinstance(value, str):
        raise ValueError('Expected a dtype string')
    name = value.lower()
    return {'bfloat16': 'bf16', 'float16': 'fp16', 'float32': 'fp32',
            'float8_e4m3fn': 'fp8_e4m3fn'}.get(name, name)


def _call_attrs(op, attrs, dtype):
    supplied = dict(attrs or {})
    declared = _dtype_name(supplied.pop('dtype')) if 'dtype' in supplied else None
    chosen = _dtype_name(dtype) if dtype is not None else declared or default_dtype(op)
    if op == 'cache_load':
        from operator_model.operators.vector.cache_load import canonical_dtype
        chosen = canonical_dtype(chosen)
        if declared is not None and canonical_dtype(declared) != chosen:
            raise ValueError('dtype conflicts with attrs["dtype"]')
        return supplied, chosen
    if op == 'matmul':
        from operator_model.operators.matmul.mx import canonical_dtype
        declared = canonical_dtype(declared)
        chosen = canonical_dtype(chosen)
    if declared is not None and declared != chosen:
        raise ValueError('dtype conflicts with attrs["dtype"]')
    if op == 'kda_chunk':
        supplied['dtype'] = chosen
        if 'beta_dtype' in supplied and not isinstance(supplied['beta_dtype'], str):
            raise ValueError('Expected a string beta_dtype')
    elif op == 'candidate_filtered_indexer' and supplied.get('implementation') == 'a5_qsli_projection':
        if chosen != 'mxfp4':
            raise ValueError('A5 QSLI requires packed MXFP4 Q/K and their E8M0 scales')
    elif op == 'matmul' and chosen in ('mxfp4', 'fp8_e4m3fn'):
        pass  # Validate these mixed-precision contracts at their own entry point.
    elif op == 'fa':
        if chosen not in ('bf16', 'fp16'):
            raise ValueError('Dense FA supports BF16 and the FP16 design scenario')
        implementation = supplied.pop('implementation', 'dense_sbh')
        if implementation == 'dense_nd_row':
            if supplied.get('vector_model', 'auto') not in ('auto', 'instruction_classes'):
                raise ValueError('The dense_nd_row design has an instruction-class VF program only')
            supplied['vector_model'] = 'instruction_classes'
        elif implementation != 'dense_sbh':
            raise ValueError('FA implementation must be dense_sbh or dense_nd_row')
    elif chosen != default_dtype(op):
        raise ValueError(f'{op} timing requires {default_dtype(op)} input, received {chosen}')
    return supplied, chosen


def _rates(work, latency_us, chip):
    """Useful work per predicted device window, in decimal units."""
    from operator_model.roofline import physical_roofline
    read, write = work['logical_read_bytes'], work['logical_write_bytes']
    logical = None if read is None or write is None else read + write
    bits = {'bf16': '16', 'fp16': '16', 'fp32': '32', 'fp8_e4m3fn': '8', 'mxfp4': '4'}
    def compute_us(counts, peaks):
        ideal_us = 0.
        for dtype, count in counts.items():
            peak = peaks.get(bits.get(dtype))
            if count and peak is None:
                return None
            if count:
                ideal_us += count / (1e6 * peak)
        return ideal_us

    cube = compute_us(work['cube_ops_by_dtype'], chip.cube_tops)
    vector = compute_us(work['vec_ops_by_dtype'], chip.vector_tops)
    hbm = None if logical is None else logical / (1000 * chip.hbm_bandwidth_gbs)
    ceilings = {'Cube': cube, 'Vector': vector, 'HBM': hbm}
    missing = [name for name, time_us in ceilings.items() if time_us is None]
    ideal = None if missing else max(ceilings.values())
    # One common efficiency multiplies all three peaks. Logical tensor traffic
    # can be served by L2, so the inferred ratio is deliberately uncapped.
    roofline = {
        'ideal_us': ideal,
        'utilization': None if ideal is None else ideal / latency_us,
        'bound': None if ideal is None else '/'.join(k for k, v in ceilings.items() if v == ideal),
        'reason': 'Missing peak or logical byte count: ' + ', '.join(missing) if missing else None,
        'cube_us': cube, 'vector_us': vector, 'hbm_us': hbm,
    }
    physical = physical_roofline(
        latency_us, chip.hbm_bandwidth_gbs,
        cube_us=cube if work.get('required_dense_cube_fma') else None,
        vector_us=0. if work.get('no_vector_engine') else None,
        physical_hbm_bytes=work.get('physical_hbm_bytes'))
    physical_terms = physical['terms_us']
    known_ideal = physical['ideal_us']
    return {
        'cube_nominal_peak_fraction': None if cube is None else cube / latency_us,
        'vector_nominal_peak_fraction': None if vector is None else vector / latency_us,
        'useful_cube_tops': work['cube_ops'] / latency_us / 1e6,
        'counted_vector_tops': work['vec_ops'] / latency_us / 1e6,
        'logical_io_gbs': None if logical is None else logical / latency_us / 1e3,
        'logical_roofline': {'reference_us': ideal, 'time_factor': roofline['utilization'],
                             'bound': roofline['bound'], 'reason': roofline['reason'],
                             'cube_us': cube, 'vector_us': vector, 'logical_io_us': hbm},
        'physical_roofline': physical,
        'roofline': {'scope': 'ca_window', 'ideal_us': physical['ideal_us'],
                     'utilization': physical['utilization'], 'missing': physical['missing'],
                     'cube_us': physical_terms['Cube'], 'vector_us': physical_terms['Vector'],
                     'hbm_us': physical_terms['HBM'],
                     'bound': None if known_ideal is None else '/'.join(
                         k for k,v in physical_terms.items() if v == known_ideal),
                     'reason': 'Missing same-window evidence: ' + ', '.join(physical['missing'])
                               if physical['missing'] else None},
    }


_RESOURCE_ONLY = {'mhc_sinkhorn', 'mhc_pre_sinkhorn', 'moe_gating_top_k', 'cast_bf16_fp32', 'attn_res_fwd', 'sigmoid_fp32'}
_RESOURCE_OPS = {'quant', 'matmul', 'fa', 'rmsnorm', 'addmm', 'mhc_post'} | _RESOURCE_ONLY


def _price_resource(op, shape, target, call, dtype):
    from operator_model.scheduling.cases import Case
    from operator_model.operators.quant.resources import resource_quant
    from operator_model.operators.matmul.resources import resource_matmul
    from operator_model.operators.matmul.dispatch import resource_matmul_dispatch
    from operator_model.operators.attention.resources import resource_fa

    work = None
    if op in ('mhc_sinkhorn', 'mhc_pre_sinkhorn'):
        if op == 'mhc_pre_sinkhorn':
            from operator_model.operators.mhc.pre_sinkhorn import fused as price
        elif call.get('resident_aiv') is not None:
            from operator_model.operators.mhc.sink_pipeline import price
        else:
            from operator_model.operators.mhc.sink_unknown import price
        us, aic, aiv, diag = price(shape, target, call)
        work = account_work(op, shape, {**call, 'dtype': dtype})
    elif op == 'attn_res_fwd':
        from operator_model.operators.attention.residual_timing import resource_attention
        us, work, diag = resource_attention(shape, target, call)
        aic, aiv = 0, diag['target_plan']['active_aiv']
    elif op == 'rmsnorm':
        from operator_model.operators.rmsnorm.pipeline import price_registered_rms
        us, aic, aiv, diag = price_registered_rms(shape, target, call)
    elif op == 'mhc_post':
        from operator_model.operators.mhc.history import price_post as resource_post
        us, aic, aiv, diag = resource_post(shape, target, call)
    elif op == 'addmm':
        if len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError('Addmm requires positive M,N,K')
        expected = {'self_shape': [shape[1]], 'alpha': 1, 'beta': 1, 'cube_math_type': 0}
        if any(k not in expected or v != expected[k] for k,v in call.items()):
            raise ValueError('Addmm requires contiguous BF16 ND, column bias, alpha=beta=1, cubeMathType=0')
        us, aic, aiv, diag = resource_matmul(shape, target, column_bias=True)
    elif op == 'sigmoid_fp32':
        from operator_model.operators.vector.instruction_classes import select_class_model
        classes = select_class_model(call)
        if call:
            raise ValueError('Sigmoid expects contiguous FP32 input without attributes')
        from operator_model.operators.vector.sigmoid import resource_sigmoid
        us, aic, aiv, diag = resource_sigmoid(shape, target, instruction_classes=classes)
    elif op == 'cast_bf16_fp32':
        from operator_model.operators.vector.instruction_classes import select_class_model
        classes = select_class_model(call)
        if call:
            raise ValueError('Cast expects a contiguous BF16 to FP32 conversion')
        from operator_model.operators.vector.cast import resource_cast
        us, aic, aiv, diag = resource_cast(shape, target, pressure_retention=True, instruction_classes=classes)
        # The kernel can issue its first two reads to different UB slots before
        # either retires. These are issue markers, separate from bandwidth time.
        stages = diag['stages_us']
        first = stages['arrival'] + stages['control_to_read']
        paired = diag['target_plan']['busy_core_waves'] >= 2
        spacing = target.details.get('cast_initial_read_pair_cycles', 91)
        second = first + spacing/(1000*target.frequency_ghz) if paired and spacing is not None else None
        diag['initial_read_issue_us'] = [first] if second is None else [first, second]
        diag['initial_read_pair_cycles'] = spacing if paired else None
        if second is not None:
            # The second tile must still pass through read, convert and write.
            # Local byte costs are in cycles/KiB. Expose only service beyond
            # the critical path already priced by the streaming model.
            plan = diag['target_plan']
            full = plan['ub_tile_elements']
            count = min(full, plan['busy_core_elements'] - full)
            dma_cycles = count * (2*diag['read_cycles_per_kib'] + 4*diag['write_cycles_per_kib']) / 1024
            convert = stages['vector'] * ceil(count/64) / ceil(full/64)
            finish = second + dma_cycles/(1000*target.frequency_ghz) + convert
            if finish > us:
                stages['exposed_initial_read'] = finish - us
                us = finish
    elif op == 'moe_gating_top_k':
        if call:
            raise ValueError('Router expects the fixed sigmoid/bias, one-group contract')
        from operator_model.operators.vector.router import resource_router
        us, aic, aiv, diag = resource_router(shape, target)
    elif op == 'fa':
        Case(op, shape)
        if set(call) - {'vector_model', 'fa_port_model'}:
            raise ValueError('Dense noncausal FA accepts vector_model and fa_port_model')
        us, aic, aiv, diag = resource_fa(shape, target, dtype=dtype, **call)
    else:
        if call:
            raise ValueError('This validated route has fixed attributes')
        rank = 2 if op == 'quant' else 3
        if len(shape) != rank or any(type(x) is not int or x <= 0 for x in shape):
            raise ValueError(f'{op} requires {rank} positive integer dimensions')
        if op == 'quant' and (shape[1] % 64 or shape[1] > 65536):
            raise ValueError('Quant cols must be a multiple of 64, at most 65536')
        price = resource_quant if op == 'quant' else resource_matmul_dispatch
        us, aic, aiv, diag = price(shape, target)
    if work is None:
        work = account_work(op, shape, {'dtype': dtype})
    if op in ('matmul', 'addmm') and aic:
        work['required_dense_cube_fma'] = True
        work['no_vector_engine'] = aiv == 0
    if op == 'matmul' and 'internal_work' in diag:
        work['implementation_work'] = diag['internal_work'].copy()
    projected = dict(method='target_resource_schedule', latency_us=us, active_aic=aic,
                     active_aiv=aiv, reference_latency_us=None, stages_us=diag['stages_us'],
                     target_plan=diag['target_plan'], timing_diagnostics=diag,
                     inherited_fields=list(target.inherited_fields),
                     notes=['950 resource defaults; 950 library tiling policy assumed.'])
    if us is None:
        return Estimate(None, 'unavailable', diag.get('qualification', 'No qualified timing for this branch'),
                        aiv, aic, target.name, work, projected)
    if op == 'matmul' and diag.get('internal_work'):
        return Estimate(us, 'extrapolated', diag['qualification'], aiv, aic, target.name,
                        work, projected, _rates(work, us, target))
    if op == 'attn_res_fwd' and diag.get('attn_sibling_projection'):
        return Estimate(us, 'extrapolated', diag['qualification'], aiv, aic, target.name,
                        work, projected, _rates(work, us, target))
    if op == 'mhc_pre_sinkhorn':
        verified = (diag['schedule'] == 'k_split_cv2_joint_sinkhorn'
                    and target in (load_chip(), load_chip('Ascend950PR_957b')))
        reason = ('Two-SKU validated K-split projection and joint Sinkhorn, T<128' if verified
                  else diag['qualification'])
        return Estimate(us, 'estimated' if verified else 'extrapolated', reason, aiv, aic,
                        target.name, work, projected, _rates(work, us, target))
    if op == 'mhc_sinkhorn' and 'history_timing_status' in diag:
        return Estimate(us, diag['history_timing_status'], diag['qualification'], aiv, aic,
                        target.name, work, projected, _rates(work, us, target))
    if op == 'rmsnorm' and call.get('resident_aiv') is not None:
        # Backward-compatible diagnostic for schedulers storing a DT high-water
        # mark. The target calculation itself consumes the target count directly.
        reference_active = ceil(shape[0] / ceil(shape[0] / 64))
        h = call['resident_aiv']
        mapped = min(reference_active, ceil(h * reference_active / aiv))
        if 0 < h < aiv and reference_active > 1:
            mapped = min(mapped, reference_active - 1)
        projected['resident_aiv_mapping'] = {'target':h,'reference':mapped}
    if op == 'addmm' and target == load_chip() and max(shape) <= 512:
        return Estimate(us, 'estimated', 'DT-calibrated MatMulV3 and fused bias transfer', aiv, aic,
                        target.name, work, projected, _rates(work, us, target))
    if op in ('cast_bf16_fp32','sigmoid_fp32') and classes:
        return Estimate(us, 'extrapolated', 'Experimental contiguous instruction-class schedule',
                        aiv, aic, target.name, work, projected, _rates(work,us,target))
    if op == 'sigmoid_fp32' and target == load_chip() and 12 <= shape[0]*shape[1] <= 3072*96:
        return Estimate(us, 'estimated', 'DT stage calibration range', aiv, aic, target.name,
                        work, projected, _rates(work, us, target))
    if op in ('moe_gating_top_k', 'cast_bf16_fp32') and target == load_chip():
        if op == 'cast_bf16_fp32':
            inside = shape[0]*shape[1] == 64*4096  # Single DT stage calibration size.
        else:
            inside = shape[0] <= 65 and 128 <= shape[1] <= 896 and 6 <= shape[2] <= 16
        return Estimate(us, 'estimated' if inside else 'extrapolated',
                        'DT-calibrated stage schedule' if inside else 'Outside DT timing coverage', aiv, aic,
                        target.name, work, projected, _rates(work, us, target))
    if op == 'fa' and (diag['vector_model'] == 'instruction_classes' or diag['fa_port_model'] == 'probe'):
        return Estimate(us, 'extrapolated', 'Experimental FA stage schedule; see per-stage assumptions',
                        aiv, aic, target.name, work, projected, _rates(work, us, target))
    if op == 'fa' and shape[1] > 131072:
        return Estimate(us, 'extrapolated', 'KV length exceeds the longest complete 950DT FA reference',
                        aiv, aic, target.name, work, projected, _rates(work, us, target))
    if op == 'fa' and target == load_chip():
        return Estimate(us, 'estimated', 'Dense BF16 FA resource prices calibrated on 950DT',
                        aiv, aic, target.name, work, projected, _rates(work, us, target))
    return Estimate(us,'extrapolated','Target resource schedule with inherited 950 phase defaults',
                    aiv,aic,target.name,work,projected,_rates(work,us,target))


_STRUCTURAL_OPS = {'candidate_filtered_indexer', 'cache_load', 'routed_grouped_matmul'}


# Qualified elapsed-time evidence, never used to choose a price or schedule.
_FP4_QUANT_GEOMETRIES = {
    (9, 128), (9, 288), (9, 7168), (576, 128),
    (2, 3584), (9, 512), (9, 1536), (9, 12288), (144, 288),
    (144, 384), (144, 7168), (576, 288), (576, 7168), (576, 12288),
}


def _native_alternative(us, work, projection, target, mode):
    """Price an explicitly selected library dispatch for a system graph node."""
    plan = projection['target_plan']
    status = projection['timing_qualification']
    reason = projection.get('reason') or 'Explicit library implementation; see measured scope for this contract'
    aiv, aic = plan.get('active_aiv'), plan.get('active_aic')
    if us is None or (mode == 'calibrated' and status != 'estimated'):
        return Estimate(None, 'unavailable', reason, aiv, aic, target.name, work, projection)
    return Estimate(us, status, reason, aiv, aic, target.name, work, projection, _rates(work, us, target))


def _evaluate(op, shape, attrs, soc, chip, dtype, timing_mode='auto'):
    try:
        if not isinstance(op, str) or ALIASES.get(op, op) not in VECTOR | CUBE | FUSION | _STRUCTURAL_OPS:
            raise ValueError(f'Unknown operator: {op}')
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
            raise ValueError('Expected an ordered shape of positive integers')
        if attrs is not None and not isinstance(attrs, Mapping):
            raise ValueError('Expected an operator attribute mapping')
        if chip is not None and soc != REFERENCE_NAME:
            raise ValueError('Specify either chip or soc')
        target = load_chip(chip if chip is not None else soc)
        op = ALIASES.get(op, op)
        result, raw = _price_device(op, shape, attrs, target, dtype, timing_mode)
        if result.latency_us is not None:
            # Counts of physical launches for the priced implementations.
            # FLA V2 is the only priced chunk KDA and has three main launches.
            count = _COMPOSITE_KERNEL_COUNTS.get(op, 1)
            result = replace(result, kernel_count=count,
                             dispatch_overhead_us=count * target.kernel_overhead_us)
        return result, raw
    except (ValueError, TypeError, OSError) as exc:
        return Estimate(None, 'unsupported', str(exc)), {}


_COMPOSITE_KERNEL_COUNTS = {'mhc': 3, 'moe_ffn': 2, 'kda_chunk': 3}


def _price_device(op, shape, attrs, target, dtype, timing_mode):
    reference = load_chip()
    raw = {}
    same = target == reference
    shape = tuple(shape)
    if op == 'fa' and len(shape) == 3:
        shape += (1, 1)
    call, chosen_dtype = _call_attrs(op, attrs, dtype)
    if timing_mode not in ('auto','resource','calibrated'):
        raise ValueError('timing_mode must be auto, resource or calibrated')
    if (op in ('cast_bf16_fp32', 'sigmoid_fp32') and timing_mode == 'calibrated'
            and call.get('vector_model') == 'instruction_classes'):
        return Estimate(None, 'unavailable', 'Instruction-class route has no dedicated calibration',
                        chip_name=target.name), raw
    if op == 'fa' and timing_mode == 'calibrated':
        if chosen_dtype == 'fp16' or call.get('vector_model') == 'instruction_classes' or call.get('fa_port_model') == 'probe':
            return Estimate(None, 'unavailable', 'Experimental FA has no dedicated calibrated route',
                            chip_name=target.name), raw
        if call.get('vector_model') in ('auto', 'calibrated'):
            call.pop('vector_model')
        if call.get('fa_port_model') == 'default':
            call.pop('fa_port_model')
    if op == 'candidate_filtered_indexer':
        from operator_model.operators.attention.candidate_indexer import PIPELINE, describe, price
        from operator_model.operators.attention import flash_sparse
        if call.get('implementation') == flash_sparse.IMPLEMENTATION:
            options = flash_sparse.options(call)
            us, work, plan = flash_sparse.price(shape, target, **options)
            return _native_alternative(us, work, plan, target, timing_mode), raw
        if call.get('implementation') == PIPELINE:
            us, work, plan = price(shape, target, call)
            return _native_alternative(us, work, plan, target, timing_mode), raw
        work, plan = describe(shape, call)
        return Estimate(None, 'unavailable', 'Missing a device implementation and CA timing for candidate scoring and selection',
                        chip_name=target.name, work=work, projection=plan), raw
    if op == 'cache_load' and call.get('implementation') == 'aclnnAntiMxQuant' and chosen_dtype == 'mxfp4':
        from operator_model.operators.vector.cache_load import price
        us, work, plan = price(shape, target, call)
        return _native_alternative(us, work, plan, target, timing_mode), raw
    if op == 'cache_load':
        from operator_model.operators.vector.cache_load import describe
        fp8_native = chosen_dtype == 'fp8_e4m3fn' and call.get('implementation') == 'aclnnAscendAntiQuant'
        attrs = {k:v for k,v in call.items() if k != 'implementation'} if fp8_native else call
        work, plan = describe(shape, chosen_dtype, attrs)
        if fp8_native:
            plan['implementation'] = 'aclnnAscendAntiQuant'
        reason = ('The FP8 per-row native route has measured timing evidence and no predictive schedule.'
                  if fp8_native else 'Cache load needs a selected implementation for device timing.')
        return Estimate(None, 'unavailable', reason,
                        chip_name=target.name, work=work, projection=plan), raw
    if op == 'routed_grouped_matmul' and call.get('implementation') == 'aclnnGroupedMatmulV5':
        from operator_model.operators.moe.routed_grouped_matmul import price
        us, work, plan = price(shape, chosen_dtype, call, target)
        if plan['timing_qualification'] == 'estimated' and (
                target not in (reference, load_chip('Ascend950PR_957b'))
                or shape[0] > 34 or max(call['expert_counts']) > 23):
            plan['timing_qualification'] = 'extrapolated'
            plan['reason'] = 'Resource configuration or expert row counts extend beyond the grouped development range.'
        return _native_alternative(us, work, plan, target, timing_mode), raw
    if op == 'routed_grouped_matmul':
        from operator_model.operators.moe.routed_grouped_matmul import describe
        work, plan = describe(shape, chosen_dtype, call)
        return Estimate(None, 'unavailable', 'Select the matching routed expert implementation to price device time',
                        chip_name=target.name, work=work, projection=plan), raw
    if op == 'quant_token_fp8':
        from operator_model.operators.quant.token_fp8 import price
        from operator_model.operators.vector.instruction_classes import select_class_model
        classes = select_class_model(call)
        us, work, plan = price(shape, target, call, instruction_classes=classes)
        return _native_alternative(us, work, plan, target, timing_mode), raw
    if op == 'quant_mxfp4':
        from operator_model.operators.quant.mxfp4 import contiguous_input, mx4_work, price
        if 'input_stride' in call and not contiguous_input(shape, call['input_stride']):
            return Estimate(None, 'unavailable', 'Strided BF16 input needs a gather or layout-conversion schedule',
                            chip_name=target.name, work=mx4_work(shape, call)), raw
        us, work, projection = price(shape, target, call)
        projection['target_plan']['active_aic'] = 0
        measured = (target in (reference, load_chip('Ascend950PR_957b')) and
                    shape in _FP4_QUANT_GEOMETRIES)
        projection['timing_qualification'] = 'estimated' if measured else 'extrapolated'
        projection['reason'] = ('Contiguous block32 FP4 wave and output queues calibrated on 950DT/950PR. '
                                + ('A qualified CA window covers this geometry.' if measured
                                   else 'Unmeasured geometry or target configuration.'))
        return _native_alternative(us, work, projection, target, timing_mode), raw
    if op == 'matmul' and chosen_dtype in ('mxfp4', 'fp8_e4m3fn'):
        from operator_model.operators.matmul.mx import price
        us, work, projection, reason = price(shape, target, chosen_dtype, call, timing_mode == 'calibrated')
        plan = projection['target_plan']
        return Estimate(us, projection['timing_qualification'] if us is not None else 'unavailable', reason,
                        plan.get('active_aiv'), plan.get('active_aic'), target.name, work, projection,
                        _rates(work, us, target) if us is not None else None), raw
    if op == 'lightning_indexer' and timing_mode in ('auto','resource'):
        from operator_model.operators.attention.indexer import applies, schedule
        if applies(shape,call,target):
            duration, plan, diag = schedule(shape,target)
            work = account_work(op,shape,{**call,'dtype':chosen_dtype})
            # Coverage is metadata only. None of these extents price an instruction.
            q, length = shape[1:3]
            in_archive = ((same and ((q <= 1 and length <= 8192) or (q <= 9 and length <= 4096)))
                          or (target == load_chip('Ascend950PR_957b') and
                              ((q <= 2 and length <= 3072) or (q <= 8 and length <= 1536))))
            reason = ('Single-wave BF16 query-group schedule within archived Q/KV bounds for this SKU'
                      if in_archive else 'Single-wave BF16 query-group schedule extrapolated from measured DT stages')
            projected = dict(method='query_group_kv_pipeline',target_plan=plan,latency_us=duration,
                             timing_diagnostics=diag,stages_us=diag['stages_us'],
                             inherited_fields=list(target.inherited_fields))
            return Estimate(duration,'estimated' if in_archive else 'extrapolated',reason,
                            plan['active_aiv'],plan['active_aic'],target.name,work,projected,
                            _rates(work,duration,target)), raw
    if op == 'mhc_sinkhorn' and 'resident_aiv' in call and call['resident_aiv'] is None:
        # An explicitly unknown history has the same semantics as omitting H.
        del call['resident_aiv']
    if (op in ('mhc_pre_sinkhorn', 'mhc_sinkhorn') or
            (op == 'attn_res_fwd' and timing_mode == 'calibrated' and call.get('optimize_prefill')
             and call.get('write_idx', -1) == -1 and not call.get('materialize'))):
        result = _price_resource(op, shape, target, call, chosen_dtype)
        if timing_mode == 'calibrated' and result.status == 'extrapolated':
            return Estimate(None, 'unavailable', result.reason, result.active_aiv, result.active_aic,
                            target.name, result.work, {**result.projection, 'latency_us': None}), raw
        return result, raw
    if op == 'mhc_post' and call.get('implementation') == 'vllm_hc_post':
        if timing_mode == 'calibrated':
            from operator_model.operators.mhc.post_layout import layout
            work, plan = layout(shape, target, call)
            projection = {'implementation': 'vllm_hc_post', 'qualification': 'structural', 'target_plan': plan}
            return Estimate(None, 'unavailable', 'The vLLM HcPost schedule is experimental',
                            plan['active_aiv'], 0, target.name, work, projection), raw
        from operator_model.operators.mhc.vllm_post import resource_hc_post
        us, work, diag = resource_hc_post(shape, target, call)
        diag['implementation'] = 'vllm_hc_post'
        if us is None:
            return Estimate(None, 'unavailable', ' '.join(diag['warnings']),
                            diag['active_aiv'], 0, target.name, work, diag), raw
        return Estimate(us, 'extrapolated', 'Explicit experimental arch35 vLLM HcPost resource model',
                        diag['active_aiv'], 0, target.name, work, diag, _rates(work, us, target)), raw
    if op == 'kda_recurrent' and len(shape) == 6 and (shape[0] != 1 or shape[3] != 1):
        from operator_model.operators.kda.recurrent_chains import estimate_recurrent_chains
        chain = estimate_recurrent_chains(shape, call, chip=target)
        us = chain['latency_us']
        plan = {k: chain[k] for k in ('active_aic', 'active_aiv', 'chains',
                                      'chains_per_busiest_aiv', 'tiling')}
        work = chain['work']
        reason = chain.get('reason') or chain.get('evidence')
        diag = dict(method='dense_recurrent_chains', target_plan=plan,
                    timing_diagnostics={k: v for k, v in chain.items() if k not in ('work', 'latency_us')},
                    inherited_fields=list(target.inherited_fields))
        if timing_mode == 'calibrated' or us is None:
            return Estimate(None, 'unavailable', reason or 'No calibrated multi-chain timing',
                            chain['active_aiv'], 0, target.name, work, diag), raw
        return Estimate(us, 'extrapolated', reason, chain['active_aiv'], 0,
                        target.name, work, dict(diag, latency_us=us), _rates(work, us, target)), raw
    if op == 'kda_chunk' and 'implementation' in call:
        from operator_model.operators.kda.fla import IMPLEMENTATION, resource
        if call['implementation'] != IMPLEMENTATION:
            raise ValueError('Unknown explicit chunk KDA implementation')
        if timing_mode == 'calibrated':
            raise ValueError('The FLA 26.9.1 V2 experimental route has no qualified calibration policy')
        us, aic, aiv, work, projected = resource(shape, target, call)
        return Estimate(us, 'extrapolated', 'Explicit experimental FLA 26.9.1 V2 resource model',
                        aiv, aic, target.name, work, projected, _rates(work, us, target)), raw
    # A batch contains multiple request contexts. The reference surrogate's
    # historical dimension limit does not bound the aggregate token axis.
    large_quant = op == 'quant' and len(shape) == 2 and type(shape[0]) is int and shape[0] > 1048576
    auto_resource = (op in _RESOURCE_ONLY or op in ('fa', 'matmul', 'addmm', 'mhc_post', 'rmsnorm') or large_quant
                     or (not same and op in _RESOURCE_OPS))
    use_resource = timing_mode == 'resource' or (timing_mode == 'auto' and auto_resource)
    if use_resource or (timing_mode == 'calibrated' and same and op in _RESOURCE_ONLY):
        if op not in _RESOURCE_OPS:
            raise ValueError(f'{op} has no target resource schedule yet')
        return _price_resource(op, shape, target, call, chosen_dtype), raw
    if timing_mode == 'calibrated' and not same and (
            op not in {'quant', 'matmul', 'fa'} or target != load_chip('Ascend950PR_957b')):
        raise ValueError('No calibration route for this operator and target configuration')
    # Residency is counted on the target. Translate only routes that use it.
    resident = None
    if not same and op in ('mhc_post', 'rmsnorm') and 'resident_aiv' in call:
        resident = call.pop('resident_aiv')
        if type(resident) is not int or not 0 <= resident <= target.vector_cores:
            raise ValueError('resident_aiv must be within the target Vector core count')
    raw = predict(op, shape, call)
    if raw.get('status') == 'unsupported':
        return Estimate(None, 'unsupported', raw.get('reason'), chip_name=target.name), raw
    work = account_work(op, shape, {**call, 'dtype': chosen_dtype})
    duration = raw['predicted_us']
    if duration is None:
        return Estimate(None, 'unavailable', raw.get('reason') or raw['status'],
                        chip_name=target.name, work=work), raw
    reason = raw.get('reason') if raw['status'] == 'extrapolation' else None
    if same and op == 'quant':
        cache_bytes = target.details.get('l2_bytes')
        logical_bytes = work['logical_read_bytes'] + work['logical_write_bytes']
        if cache_bytes and logical_bytes > cache_bytes:
            reason = 'Quant working set exceeds L2; this streaming regime lacks full instruction-window calibration'
    if op == 'fa' and shape[1] > 131072:
        reason = 'KV length exceeds the longest complete 950DT FA reference'
    if same:
        plan = raw.get('plan', {})
        if op in ('fa', 'matmul', 'quant'):
            from operator_model.profiles.tile_model import plan as reference_plan
            from operator_model.scheduling.cases import Case
            plan = reference_plan(Case(op, shape))
        aic = raw.get('predicted_active_aic', plan.get('active_aic'))
        aiv = raw.get('predicted_active_aiv', plan.get('active_aiv') or None)
        return Estimate(duration, 'extrapolated' if reason else 'estimated', reason,
                        aiv, aic, target.name, work, rates=_rates(work, duration, target)), raw

    from operator_model.scheduling.projection import project
    projected = project(op, shape, call, raw, work, reference, target)
    if resident is not None and projected['active_aiv']:
        target_active = projected['active_aiv']
        reference_active = raw['predicted_active_aiv']
        mapped = min(reference_active, ceil(resident * reference_active / target_active))
        if 0 < resident < target_active and reference_active > 1:
            mapped = min(reference_active - 1, mapped)
        call['resident_aiv'] = mapped
        raw = predict(op, shape, call)
        projected = project(op, shape, call, raw, work, reference, target)
        projected['resident_aiv_mapping'] = {'target': resident, 'reference': mapped}
    projected['inherited_fields'] = list(target.inherited_fields)
    why = 'Same implementation and tiling policy assumed from ' + reference.name
    if reason:
        why += '. ' + reason
    if projected['latency_us'] is None:
        return Estimate(None, 'unavailable', why, chip_name=target.name,
                        work=work, projection=projected), raw
    return Estimate(projected['latency_us'], 'extrapolated', why, projected['active_aiv'],
                    projected['active_aic'], target.name, work, projected, _rates(work, projected['latency_us'], target)), raw



def _memory_window(result, target, policy):
    if policy not in ('ca_window', 'cold_io'):
        raise ValueError('memory_policy must be ca_window or cold_io')
    if policy == 'ca_window' or result.latency_us is None:
        return result
    from operator_model.roofline import cold_io_window
    io = cold_io_window(result.latency_us, result.work, target)
    us = io['duration_us']
    rates = _rates(result.work, us, target)
    logical = rates['logical_roofline']
    rates.update(memory_window=io, roofline={'scope': 'cold_io', 'ideal_us': logical['reference_us'],
                 'utilization': logical['time_factor'], 'bound': logical['bound'], 'reason': logical['reason'],
                 'cube_us': logical['cube_us'], 'vector_us': logical['vector_us'], 'hbm_us': logical['logical_io_us']})
    status = 'extrapolated' if io['exposed_io_us'] > 0 else result.status
    return replace(result, status=status, latency_us=us, projection={**(result.projection or {}), 'memory_window':io},
                   rates=rates)


def estimate(op: str, shape: Sequence[int], attrs: Optional[Mapping] = None, *,
             soc: str = REFERENCE_NAME, chip=None, dtype: Optional[str] = None,
             timing_mode: str = 'auto', memory_policy: str = 'ca_window') -> Estimate:
    """Estimate one local dispatch. chip accepts a config, mapping, JSON file or SKU."""
    if memory_policy not in ('ca_window', 'cold_io'):
        raise ValueError('memory_policy must be ca_window or cold_io')
    result = _evaluate(op, shape, attrs, soc, chip, dtype, timing_mode)[0]
    if memory_policy == 'ca_window' or result.latency_us is None:
        return result
    return _memory_window(result, load_chip(chip if chip is not None else soc), memory_policy)
