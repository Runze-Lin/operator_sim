"""FP32 contiguous Sigmoid. Price an ATVOSS elementwise grid on target resources."""
from math import log2

from .workload import _counts


def _ceil(n, d):
    return (n + d - 1) // d


def _elements(shape):
    if len(shape) != 2 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError('sigmoid_fp32 expects positive integer [tokens, local_heads]')
    return shape[0] * shape[1]


def sigmoid_work(shape):
    n = _elements(shape)
    return _counts('fp32', 4*n, 4*n, vec={'fp32': 4*n},
                   classes={name: n for name in ('negate', 'exp', 'add', 'divide')},
                   notes=['Each transcendental/divide counts once per element. Dense FMA peak is not its instruction rate.'])


def resource_sigmoid(shape, chip):
    from .rms_resource import READ_PER_KIB, WRITE_PER_KIB
    n = _elements(shape)
    potential = min(chip.vector_cores, _ceil(n, 1024))  # Minimum 4 KiB input per block.
    block = 512 * _ceil(n, 512*potential)
    active = _ceil(n, block)
    local = min(n, block)
    cap = ((chip.details.get('ub_bytes') or 253952) // 16 // 64) * 64
    if cap < 64:
        raise ValueError('Sigmoid needs at least one FP32 register group in each of four UB buffers')
    tiles = _ceil(local, cap)
    f = chip.frequency_ghz
    read_port = chip.details.get('mte2_bandwidth_gbs') or 1024*f / READ_PER_KIB
    write_port = chip.details.get('mte3_bandwidth_gbs') or 1024*f / WRITE_PER_KIB
    shared = min(chip.hbm_bandwidth_gbs, chip.details.get('l2_bandwidth_gbs') or float('inf'))
    fp32 = chip.vector_tops['32']
    simd_factor = (27/(64*1.65))/(fp32/(chip.vector_cores*f)) if fp32 else 1.
    # Single 950DT AIV: 3012 cycles to the first tensor read; 64-AIV
    # terminal core: 4170 cycles from the earliest core. This includes arrival.
    front = 3012 + 193 * log2(active)

    def service(amount):
        data = 4 * amount
        local_read = data * f / read_port
        local_write = data * f / write_port
        # Fair-share lower bounds. Physical write miss/eviction counts are unknown.
        shared_one = data * active * f / shared
        read = 1159 + max(local_read, shared_one)
        groups = _ceil(amount, 64)
        vector = (79 + 4*(groups-1)) * simd_factor
        write = 159 + max(local_write, shared_one)
        return read, vector, write, max(local_read, local_write, 2*shared_one, vector,
                                       (read+vector)/2, (vector+write)/2)

    first_count = min(cap, local)
    read, vector, write, interval = service(first_count)
    last_count = local - (tiles-1)*cap
    lr, lv, lw, _ = service(last_count)
    # Keep the three execution engines ordered if a short last tile catches up
    # to its predecessor. Descriptor latency may overlap across the two buffers.
    last_read_at = front + (tiles-1)*interval
    last_vec_at = last_read_at + lr
    last_write_at = last_vec_at + lv + 17
    if tiles > 1:
        prior = last_read_at - interval
        last_vec_at = max(last_vec_at, prior + read + vector)
        last_write_at = max(last_vec_at + lv + 17, prior + read + vector + 17 + write)
    end = last_write_at + lw + 10
    factor = 1000*f
    stages = {'front':front/factor, 'first_read':read/factor, 'first_vector':vector/factor,
              'first_write_and_flags':(17+write+10)/factor,
              'exposed_following_tiles':max(0.,end-front-read-vector-17-write-10)/factor}
    plan = dict(elements=n, active_aic=0, active_aiv=active, block_elements=block,
                critical_core_elements=local, ub_tile_elements=cap, critical_core_tiles=tiles,
                last_tile_elements=last_count, vector_groups=_ceil(local,64))
    # O(1) timeline representation, including a separately scheduled tail.
    schedule = dict(first_read_at=front/factor, first_read=read/factor, first_vector=vector/factor,
                    first_write=write/factor, write_dependency=17/factor, interval=interval/factor,
                    last_read_at=last_read_at/factor, last_vector_at=last_vec_at/factor,
                    last_write_at=last_write_at/factor, last_read=lr/factor, last_vector=lv/factor,
                    last_write=lw/factor)
    return end/factor, 0, active, dict(target_plan=plan, stages_us=stages, schedule_us=schedule,
                                       port_read_gbs=read_port, port_write_gbs=write_port, shared_gbs=shared,
                                       special_function_rate_scale=simd_factor)
