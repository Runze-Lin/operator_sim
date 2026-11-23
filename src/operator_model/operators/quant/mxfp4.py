"""Arch35 tail-axis block32 BF16 -> packed E2M1 and paired E8M0 scales."""
from functools import lru_cache
from collections.abc import Sequence
from math import log2, prod

from operator_model.operators.quant.repeat import NEG, compose, repeat
from operator_model.profiles.stage_model import ceildiv
from operator_model.resources import load_parameters
from operator_model.scheduling.memory import stream_memory
from operator_model.scheduling.tiling import quant_plan
from operator_model.workload import _counts


def _geometry(shape, attrs):
    if not 1 <= len(shape) <= 7 or any(type(x) is not int or x < 1 for x in shape):
        raise ValueError('Expected contiguous BF16, rank 1 to 7, with positive integer dimensions')
    if shape[-1] < 32 or shape[-1] % 2:
        raise ValueError('Block32 packed E2M1 route requires an even last dimension of at least 32')
    fixed = {'group_size': 32, 'axis': -1, 'round_mode': 'round', 'scale_alg': 0}
    for name, value in attrs.items():
        if name == 'input_stride':
            contiguous_input(shape, value)
            continue
        if name not in fixed or type(value) is not type(fixed[name]) or value != fixed[name]:
            raise ValueError('quant_mxfp4 requires last-axis group_size=32, round_mode=round, scale_alg=0')
    return prod(shape[:-1]), shape[-1]


def contiguous_input(shape, stride):
    """Element strides of a dense row-major input; singleton strides are immaterial."""
    if (not isinstance(stride, Sequence) or isinstance(stride, (str, bytes)) or len(stride) != len(shape)
            or any(type(x) is not int or x < 0 for x in stride)):
        raise ValueError('input_stride requires one nonnegative integer element stride per axis')
    expected = 1
    for size, step in zip(reversed(shape), reversed(stride)):
        if size > 1 and step != expected:
            return False
        expected *= size
    return True


def mx4_work(shape, attrs):
    rows, cols = _geometry(shape, attrs)
    n = rows*cols
    groups = rows*ceildiv(cols, 32)
    scales = 2*rows*ceildiv(cols, 64)
    classes = {'exponent_extract': n, 'max_comparison': n-groups,
               'rescale_multiply': n, 'convert_to_e2m1': n}
    work = _counts('bf16', 2*n, n//2+scales, vec={'bf16': sum(classes.values())},
                   classes=classes, notes=['Useful 16-bit-source lane equivalents; scale construction and packing are counted separately in the source-loop schedule.'])
    work.update(groups=groups, packed_value_bytes=n//2, scale_bytes=scales,
                packed_shape=[*shape[:-1], cols//2], scale_shape=[*shape[:-1], ceildiv(cols, 64), 2])
    return work


@lru_cache(maxsize=1)
def _transport():
    return load_parameters('quant_resources')['parameters']


def price(shape, chip, attrs):
    work = mx4_work(shape, attrs)
    if 'input_stride' in attrs and not contiguous_input(shape, attrs['input_stride']):
        raise ValueError('Packed FP4 quantization timing requires a contiguous BF16 input')
    rows, cols = prod(shape[:-1]), shape[-1]
    plan = quant_plan((rows, cols), chip.vector_cores, chip.details.get('ub_bytes') or 253952, output_bits=4)
    h, w = plan['rows_per_core'], plan['cols_per_core']
    tile_h, tile_w = plan['tile']
    clock = 1000 * chip.frequency_ghz
    transport = _transport()

    def cycles(key):
        return chip.details.get('quant_'+key) or transport[key]

    def byte_rate(port, default):
        value = chip.details.get(port+'_bandwidth_gbs')
        return value/chip.frequency_ghz if value else transport[default]

    read_rate = byte_rate('mte2', 'read_bytes_per_cycle')
    issue_rate = byte_rate('mte2', 'read_issue_bytes_per_cycle')
    write_rate = byte_rate('mte3', 'write_bytes_per_cycle')
    # Source Reg calls are not decoded instructions. Assume one call per cycle
    # on a 950DT-class datapath; scale this scenario by per-AIV peak per clock.
    peak = chip.vector_tops['16']
    vf_issue = (peak/(chip.vector_cores*chip.frequency_ghz))/(54/(64*1.65)) if peak else 1.

    def service(height, width):
        padded_groups = height*2*ceildiv(width, 64)
        x_iters, scale_iters = ceildiv(padded_groups, 8), ceildiv(padded_groups, 128)
        source_calls = (6+9)*x_iters + 16*scale_iters
        read_bytes = 2*height*width
        write_bytes = height*width//2+padded_groups
        read = cycles('read_latency_cycles')+read_bytes/read_rate
        vector = cycles('vf_dependency_cycles')+source_calls/vf_issue
        # Value and scale descriptors issue in order. Only exposed retirement
        # and their combined byte service are priced on the single write port.
        write = cycles('write_latency_cycles')+transport['scale_tail_cycles']+write_bytes/write_rate
        return dict(read_cycles=read, read_issue_cycles=read_bytes/issue_rate,
                    vector_cycles=vector, write_cycles=write, vector_source_calls=source_calls,
                    x_loop_iterations=x_iters, scale_loop_iterations=scale_iters)

    def transition(height, width):
        tile = service(height, width)
        r, ii = tile['read_cycles'], tile['read_issue_cycles']
        v, wout = tile['vector_cycles'], tile['write_cycles']
        return ((ii, NEG, NEG, ii, NEG),
                (r+v, v, NEG, r+v, v),
                (r+v+wout, v+wout, wout, r+v+wout, v+wout),
                (NEG, 0., NEG, NEG, NEG), (NEG, NEG, 0., NEG, NEG))

    def stripe(height):
        full, tail = divmod(w, tile_w)
        timeline = repeat(transition(height, tile_w), full)
        return compose(transition(height, tail), timeline) if tail else timeline

    full, tail = divmod(h, tile_h)
    timeline = repeat(stripe(tile_h), full)
    if tail:
        timeline = compose(stripe(tail), timeline)
    core_cycles = timeline[2][0]
    active = plan['active_aiv']
    memory = stream_memory(chip, active, work['logical_read_bytes'], work['logical_write_bytes'])
    shared = max(memory['external_floor_cycles'], memory['l2_floor_cycles'] or 0)
    launch = cycles('launch_cycles')
    if plan['iterations'] > 1:
        launch = chip.details.get('quant_prefetch_launch_cycles') or launch
    spread = cycles('arrival_log2_cycles')*log2(active)
    stages = {'first_issue': launch/clock, 'critical_core_and_arrival': (spread+core_cycles)/clock,
              'exposed_shared_transfer': max(0., shared-spread-core_cycles)/clock,
              'drain': cycles('end_cycles')/clock}
    diag = dict(method='arch35_mxfp4_block32_source_schedule', target_plan=plan, stages_us=stages,
                first_tile=service(min(h, tile_h), min(w, tile_w)), source_calls_per_cycle=vf_issue,
                critical_core_cycles=core_cycles, arrival_spread_cycles=spread,
                mixed_stream=memory, inherited_fields=list(chip.inherited_fields))
    return sum(stages.values()), work, diag
