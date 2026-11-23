"""Standalone aclnnDynamicQuantV2, symmetric BF16 rows -> E4M3FN and FP32 scales."""
from functools import lru_cache
from math import ceil, log2, prod

from operator_model.chips import load_chip
from operator_model.operators.quant.mxfp4 import contiguous_input
from operator_model.resources import load_parameters
from operator_model.scheduling.memory import stream_memory
from operator_model.workload import _counts


def work_for(shape, attrs):
    if len(shape)<2 or any(type(i) is not int or i < 1 for i in shape):
        raise ValueError('Expected positive row-major BF16 shape with at least two dimensions')
    if set(attrs)-{'input_stride'}:
        raise ValueError('quant_token_fp8 uses symmetric rowwise FP32 scale, no offset/smoothing')
    if 'input_stride' in attrs:
        contiguous_input(shape, attrs['input_stride'])
    rows, total = prod(shape[:-1]), prod(shape)
    classes = dict(absolute_value=total, maximum=total-rows, row_scale=rows, rescale=total, convert_to_e4m3fn=total)
    return _counts('bf16', 2*total, total+4*rows, vec={'fp32':sum(classes.values())}, classes=classes,
                   notes=['FP32 lane-equivalent row normalization including conversions; distinct from retired instructions.'])


@lru_cache(maxsize=1)
def coefficients():
    return load_parameters('token_quant_fp8')


def _ub_rows(width, capacity):
    aligned = 16*ceil(width/16)
    available = capacity-1024
    if 11*aligned > available:
        return 0, False
    if 22*aligned <= available:
        return min(255,available//(22*aligned)), True
    frame = 2*(2*aligned + 32*ceil(width/32))
    packed = (available-6144)//frame
    if packed>0:
        return min(255,packed), True
    return min(255,available//(11*aligned)), False


def price(shape,chip,attrs):
    work = work_for(shape,attrs)
    rows, width = prod(shape[:-1]), shape[-1]
    active = min(rows,chip.vector_cores)
    owner_rows = ceil(rows/active)
    capacity, doubled = _ub_rows(width,chip.details.get('ub_bytes') or 253952)
    plan = dict(active_aiv=active,active_aic=0,rows=rows,row_width=width,
                maximum_owner_rows=owner_rows,ub_rows=capacity,double_buffer=doubled)
    info = dict(method='aclnnDynamicQuantV2_rowwise',target_plan=plan,timing_qualification='unavailable')
    if not capacity or ('input_stride' in attrs and not contiguous_input(shape,attrs['input_stride'])):
        info.update(reason='Pricing requires a contiguous input with a full row fitting the 950-style UB schedule.')
        info['target_plan'] = dict(plan,active_aiv=None,active_aic=None)
        return None, work, info
    tile_rows = min(capacity,owner_rows)
    tiles = ceil(owner_rows/tile_rows)
    p = coefficients()
    clock = chip.frequency_ghz*1000
    ref = p['reference_ops_per_aiv_cycle']
    peak = chip.vector_tops['32']
    lane_scale = ref/(peak*1000/(chip.vector_cores*chip.frequency_ghz)) if peak else 1.
    rd = (chip.details.get('mte2_bandwidth_gbs') or p['read_B_per_cycle']*chip.frequency_ghz)/chip.frequency_ghz
    wr = (chip.details.get('mte3_bandwidth_gbs') or p['write_B_per_cycle']*chip.frequency_ghz)/chip.frequency_ghz
    def service(h):
        n = h*width
        read = p['read_setup_cycles']+2*n/rd
        vec = p['vector_setup_cycles']+(p['vector_cycles_per_element']*n)*lane_scale
        write = max(p['payload_setup_cycles']+n/wr,p['scale_retirement_cycles'] + p['scale_arrival_cycles']*log2(active) + 4*h/wr)
        return read,vec,write
    first = service(tile_rows)
    full_count, remainder = divmod(owner_rows,tile_rows)
    advance = max if doubled else sum
    steady = (full_count-1)*advance(first) + (advance(service(remainder)) if remainder else 0)
    core = sum(first)+steady
    memory = stream_memory(chip,active,work['logical_read_bytes'],work['logical_write_bytes'])
    shared = max(memory['external_floor_cycles'], memory['l2_floor_cycles'] or 0)
    stages=dict(first_payload_issue=p['issue_cycles'] + p['issue_arrival_cycles']*log2(active),read=first[0],vector_and_handoff=first[1],
                value_and_scale_writes=first[2],additional_tiles=steady,
                exposed_shared_memory=max(0.,shared-core),retire=p['end_cycles'])
    status = 'estimated' if (chip in (load_chip(),load_chip('Ascend950PR_957b')) and
                             len(shape)==2 and 9<=rows<=576 and 512<=width<=7168) else 'extrapolated'
    plan.update(tiles=tiles,tile_rows=tile_rows,memory=memory)
    info.update(timing_qualification=status,stages_us={k:v/clock for k,v in stages.items()},
                reason='Explicit CANN per-token standalone implementation; timing scope is contiguous full-load rows.')
    return sum(stages.values())/clock, work, info
