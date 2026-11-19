"""System simulator interface. Chip files are read locally and require no CANN."""
from dataclasses import dataclass
from math import ceil
from typing import Literal, Mapping, Optional, Sequence

from .chips import REFERENCE_NAME, load_chip
from .predictor import ALIASES, CUBE, FUSION, VECTOR, predict
from .workload import account_work, default_dtype


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
    if declared is not None and declared != chosen:
        raise ValueError('dtype conflicts with attrs["dtype"]')
    if op == 'kda_chunk':
        supplied['dtype'] = chosen
        if 'beta_dtype' in supplied and not isinstance(supplied['beta_dtype'], str):
            raise ValueError('Expected a string beta_dtype')
    elif chosen != default_dtype(op):
        raise ValueError(f'{op} timing requires {default_dtype(op)} input, received {chosen}')
    return supplied, chosen


def _rates(work, latency_us, chip):
    """Useful work per predicted device window, in decimal units."""
    read, write = work['logical_read_bytes'], work['logical_write_bytes']
    logical = None if read is None or write is None else read + write
    bits = {'bf16': '16', 'fp16': '16', 'fp32': '32', 'fp8_e4m3fn': '8'}
    def peak_fraction(counts, peaks):
        ideal_us = 0.
        for dtype, count in counts.items():
            peak = peaks.get(bits.get(dtype))
            if count and peak is None:
                return None
            if count:
                ideal_us += count / (1e6 * peak)
        return ideal_us / latency_us

    return {
        'cube_nominal_peak_fraction': peak_fraction(work['cube_ops_by_dtype'], chip.cube_tops),
        'vector_nominal_peak_fraction': peak_fraction(work['vec_ops_by_dtype'], chip.vector_tops),
        'useful_cube_tops': work['cube_ops'] / latency_us / 1e6,
        'counted_vector_tops': work['vec_ops'] / latency_us / 1e6,
        'logical_io_gbs': None if logical is None else logical / latency_us / 1e3,
    }


_RESOURCE_OPS = {'quant', 'matmul', 'fa'}


def _price_resource(op, shape, target, call, dtype):
    from .model_v6 import Case
    from .quant_resource import resource_quant
    from .matmul_resource import resource_matmul
    from .fa_resource import resource_fa

    if call:
        raise ValueError('This validated route has fixed attributes')
    Case(op, shape)  # Apply the public shape contract before partitioning.
    price = {'quant': resource_quant, 'matmul': resource_matmul, 'fa': resource_fa}[op]
    us, aic, aiv, diag = price(shape, target)
    work = account_work(op, shape, {'dtype': dtype})
    projected = dict(method='target_resource_schedule', latency_us=us, active_aic=aic,
                     active_aiv=aiv, reference_latency_us=None, stages_us=diag['stages_us'],
                     target_plan=diag['target_plan'], timing_diagnostics=diag,
                     inherited_fields=list(target.inherited_fields),
                     notes=['950 resource defaults; 950 library tiling policy assumed.'])
    return Estimate(us,'extrapolated','Target resource schedule with inherited 950 phase defaults',
                    aiv,aic,target.name,work,projected,_rates(work,us,target))


def _evaluate(op, shape, attrs, soc, chip, dtype, timing_mode='auto'):
    raw = {}
    try:
        if not isinstance(op, str) or ALIASES.get(op, op) not in VECTOR | CUBE | FUSION:
            raise ValueError(f'Unknown operator: {op}')
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
            raise ValueError('Expected an ordered shape of positive integers')
        if attrs is not None and not isinstance(attrs, Mapping):
            raise ValueError('Expected an operator attribute mapping')
        if chip is not None and soc != REFERENCE_NAME:
            raise ValueError('Specify either chip or soc')
        reference = load_chip()
        target = load_chip(chip if chip is not None else soc)
        same = target == reference
        op = ALIASES.get(op, op)
        shape = tuple(shape)
        if op == 'fa' and len(shape) == 3:
            shape += (1, 1)
        call, chosen_dtype = _call_attrs(op, attrs, dtype)
        if timing_mode not in ('auto','resource','calibrated'):
            raise ValueError('timing_mode must be auto, resource or calibrated')
        if timing_mode == 'resource' or (timing_mode == 'auto' and not same and op in _RESOURCE_OPS):
            if op not in _RESOURCE_OPS:
                raise ValueError(f'{op} has no target resource schedule yet')
            return _price_resource(op, shape, target, call, chosen_dtype), raw
        if timing_mode == 'calibrated' and not same and (
                op not in _RESOURCE_OPS or target != load_chip('Ascend950PR_957b')):
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
        if op == 'fa' and shape[1] > 131072:
            reason = 'KV length exceeds the longest complete 950DT FA reference'
        if same:
            plan = raw.get('plan', {})
            if op in ('fa', 'matmul', 'quant'):
                from .model_v2 import plan as reference_plan
                from .model_v6 import Case
                plan = reference_plan(Case(op, shape))
            aic = raw.get('predicted_active_aic', plan.get('active_aic'))
            aiv = raw.get('predicted_active_aiv', plan.get('active_aiv') or None)
            return Estimate(duration, 'extrapolated' if reason else 'estimated', reason,
                            aiv, aic, target.name, work, rates=_rates(work, duration, target)), raw

        from .transfer import project
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
    except (ValueError, TypeError, OSError) as exc:
        return Estimate(None, 'unsupported', str(exc)), raw


def estimate(op: str, shape: Sequence[int], attrs: Optional[Mapping] = None, *,
             soc: str = REFERENCE_NAME, chip=None, dtype: Optional[str] = None,
             timing_mode: str = 'auto') -> Estimate:
    """Estimate one local dispatch. chip accepts a config, mapping, JSON file or SKU."""
    return _evaluate(op, shape, attrs, soc, chip, dtype, timing_mode)[0]
