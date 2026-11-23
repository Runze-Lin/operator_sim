"""Single-launch scaled Matmul resource schedules, independent of target CA."""
from functools import lru_cache
from math import log2

from operator_model.chips import load_chip
from operator_model.resources import load_parameters

from operator_model.operators.matmul.resources import _costs
from operator_model.workload import _counts

FP4 = 'mxfp4'
FP8 = 'fp8_e4m3fn'
@lru_cache(maxsize=1)
def _mx_prices():
    return load_parameters('mx_matmul_phases')


def _div(n, d):
    return (n + d - 1) // d


def _align(n, d):
    return _div(n, d) * d


def canonical_dtype(name):
    aliases = {'float4_e2m1': FP4, 'fp4_e2m1': FP4, 'mxfp8': FP8, 'float8_e4m3fn': FP8}
    return aliases.get(name, name)


def contract(shape, dtype, attrs):
    if len(shape) != 3 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError('Dense MX Matmul expects positive integer M,N,K')
    dtype = canonical_dtype(dtype)
    weight_dtype = attrs.get('weight_dtype', FP4 if dtype == FP4 else None)
    if not isinstance(weight_dtype, str) or canonical_dtype(weight_dtype.lower()) != FP4:
        raise ValueError('Dense MX Matmul expects explicit weight_dtype=mxfp4 for an FP8 activation')
    if dtype not in (FP4, FP8):
        raise ValueError('Dense MX Matmul expects MXFP4 E2M1 or FP8 E4M3FN activation')
    # The represented layout is one concrete V5 contract. Reject other input
    # semantics even when the public ACLNN interface has additional branches.
    expected = {'transpose_a': False, 'transpose_b': True, 'layout': 'nd',
                'output_dtype': 'bf16', 'accumulator_dtype': 'fp32',
                'scale_dtype': 'e8m0', 'group_size': 32, 'bias': False}
    for key, value in attrs.items():
        if key == 'weight_dtype':
            continue
        if isinstance(value, str):
            value = {'bfloat16': 'bf16', 'float32': 'fp32'}.get(value.lower(), value.lower())
        if key not in expected or type(value) is not type(expected[key]) or value != expected[key]:
            raise ValueError(f'Unsupported MX Matmul attribute {key}={value!r}, expected {expected.get(key)!r}')
    k = shape[2]
    if dtype == FP4 and (k % 2 or k <= 2):
        raise ValueError('Native packed MXFP4 ND input requires even K greater than 2')
    if dtype == FP8 and k % 8:
        raise ValueError('A8W4 FP4 weight prologue requires K divisible by 8')
    return dtype


def _payload(rows, depth, bits):
    return rows * _div(depth * bits, 8)


def mx_work(shape, dtype, attrs):
    dtype = contract(shape, dtype, attrs)
    m, n, k = shape
    bits = 4 if dtype == FP4 else 8
    scales = 2 * _div(k, 64)
    operands = {'activation': _payload(m, k, bits), 'weight': _payload(n, k, 4),
                'activation_scale': m * scales, 'weight_scale': n * scales}
    work = _counts(dtype, sum(operands.values()), 2*m*n, {dtype: 2*m*n*k},
                   notes=['Both E8M0 scale tensors use pairs per 64 K values. '
                          'Mixed-route weight expansion is implementation work.'])
    work['operand_bytes'] = operands
    return work


def _footprint(bm, bn, bk, mixed):
    # Reserve two aligned operand pages locally and one FP32 accumulator tile.
    a = _payload(bm, bk, 8 if mixed else 4)
    b = _payload(bn, bk, 8 if mixed else 4)
    scales = (bm + bn) * 2 * _div(bk, 64)
    return {'l1_bytes': 2*(a+b+scales), 'l0a_bytes': 2*a, 'l0b_bytes': 2*b,
            'l0c_bytes': 4*bm*bn, 'ub_bytes': 2*(b + _payload(bn, bk, 4)) if mixed else 0}


def _native_output(m, n, budget):
    # DAV_3510 default: adjust the initial output rectangle to use a single
    # wave when possible, then rebalance when an edge is at least twice as long.
    bm, bn = min(256, _align(m, 16)), min(256, _align(n, 16))
    mc, nc = _div(m, bm), _div(n, bn)
    if mc*nc >= budget:
        return bm, bn
    def rect(mc, nc):
        return _align(_div(m, mc), 16), _align(_div(n, nc), 16)
    if _div(m, 16) <= _div(n, 16):
        nc = budget // mc
    else:
        mc = budget // nc
    bm, bn = rect(mc, nc)
    mc, nc = _div(m, bm), _div(n, bn)
    while bn >= 2*bm and nc < budget//2 and bn > 16:
        nc *= 2
        bm, bn = rect(budget//nc, nc)
        mc, nc = _div(m, bm), _div(n, bn)
    while bm >= 2*bn and mc < budget//2 and bm > 16:
        mc *= 2
        bm, bn = rect(mc, budget//mc)
        mc, nc = _div(m, bm), _div(n, bn)
    return bm, bn


def _mixed_output(m, n, budget):
    # Bounded approximation to the V4 solver. Try dyadic subgrids and prefer
    # fuller grids, balanced rectangles and reuse of the wider FP8 activation.
    def powers(length):
        value = 1
        while value <= min(budget, _div(length, 16)):
            yield value
            value *= 2
    candidates = []
    for mc in powers(m):
        for nc in powers(n):
            if mc*nc <= budget:
                bm, bn = _align(_div(m, mc), 16), _align(_div(n, nc), 16)
                active = _div(m, bm)*_div(n, bn)
                candidates.append(((-active, max(bm, bn)/min(bm, bn), nc), bm, bn))
    _, bm, bn = min(candidates)
    return min(256, bm), min(256, bn)


def _tile(shape, chip, mixed, core_budget):
    m, n, k = shape
    bm, bn = (_mixed_output if mixed else _native_output)(m, n, core_budget)
    bm, bn = min(bm, 256), min(bn, 256)
    # Large K fragments come directly from the double-buffered L0 A/B budget.
    # Cap page depth at 2048 as a conservative scenario for unobserved K loops.
    bk = min(2048, _align(k, 64))
    while True:
        footprint = _footprint(bm, bn, bk, mixed)
        if all(size <= (chip.details.get(key) or 0) for key, size in footprint.items() if size):
            return (bm, bn, bk), footprint
        if bk > 64:
            bk = max(64, (bk//2)//64*64)
        elif max(bm, bn) > 16:
            if bn >= bm:
                bn = _align(_div(bn, 2), 16)
            else:
                bm = _align(_div(bm, 2), 16)
        else:
            return None, footprint


def price(shape, chip, dtype, attrs, calibrated_only=False):
    work = mx_work(shape, dtype, attrs)
    m, n, k = shape
    mixed = work['dtype'] == FP8
    bits = 8 if mixed else 4
    name = 'a8w4_inline_dequant' if mixed else 'a4w4_native'
    launched = min(2, max(1, chip.vector_cores // chip.cube_cores)) if mixed else 0
    budget = min(chip.cube_cores, chip.vector_cores // launched) if mixed else chip.cube_cores
    tile, footprint = _tile(shape, chip, mixed, budget)
    plan = {'route': name, 'layout': 'ND A[M,K], transposed physical ND W[N,K], ND Y[M,N]',
            'weight_dtype': FP4, 'cube_dtype': work['dtype'], 'accumulator_dtype': 'fp32',
            'output_dtype': 'bf16', 'scale_shapes': [[m, _div(k, 64), 2], [n, _div(k, 64), 2]],
            'scale_dtype': 'e8m0', 'group_k': 32, 'aiv_per_aic': launched,
            'implementation_kernel_count': 1, 'required_buffer_bytes': footprint,
            'weight_expansion': 'AIV GM/UB packed FP4 to FP8, UB to L1' if mixed else 'native scaled Cube'}
    diag = {'method': 'mx_matmul_page_pipeline', 'target_plan': plan, 'latency_us': None,
            'inherited_fields': list(chip.inherited_fields), 'timing_qualification': 'extrapolated'}
    if tile is None:
        return None, work, diag, 'No 16x16 output tile with a 64-value K page fits the target local buffers'
    bm, bn, bk = tile
    mt, nt = _div(m, bm), _div(n, bn)
    tiles = mt * nt
    active = min(tiles, budget)
    waves = _div(tiles, active)
    pages = _div(k, bk)
    # Observed small K fragments go to veccore0; the paired veccore1 executes
    # control. Outside single-page coverage allow producers to alternate pages.
    producers = min(launched, pages)
    sku = next((name for name in ('Ascend950DT_9582', 'Ascend950PR_957b') if chip == load_chip(name)), None)
    covered = bool(sku and 1 <= m <= 2048 and (n, k) == (128, 256))
    diag['calibration_scope'] = {'covered': covered, 'development_m': [1, 256],
                                 'm_range': [1, 2048], 'n': 128, 'k': 256,
                                 'hardware_equivalent_sku': sku, 'method': 'MX phase calibration'}
    plan.update(weight_producer_aiv=active*producers, launched_aiv=active*launched, tile=list(tile), tile_grid=[mt, nt], active_aic=active, active_aiv=active*launched,
                output_tiles=tiles, busiest_core_tiles=waves, pages_per_tile=pages,
                busiest_core_pages=waves*pages, last_wave_tiles=(tiles-1)%active+1)
    peak = chip.cube_tops[str(bits)]
    vector_peak = chip.vector_tops['8']
    if peak is None or (mixed and vector_peak is None):
        return None, work, diag, 'Missing target Cube peak or mixed-route 8-bit Vector proxy peak'
    diag['timing_qualification'] = 'estimated' if covered else 'extrapolated'
    reason = ('MX phase model in the two-SKU evaluation range' if covered else
              'MX phase schedule outside the two-SKU calibration scope')
    if calibrated_only and not covered:
        return None, work, diag, reason

    ghz = chip.frequency_ghz
    defaults = _costs()
    phases = _mx_prices()[name]
    prices = phases.get(sku, phases['pooled'])
    unpack_ops = phases.get('unpack_peak_equivalents_per_weight', 0)
    details = chip.details
    def clocks(key):
        return details.get('matmul_' + key) or defaults[key]
    def port(engine, key):
        rate = details.get(engine + '_bandwidth_gbs')
        return rate/ghz if rate else defaults[key]
    mte2 = port('mte2', 'mte2_bytes_per_cycle')
    mte1 = port('mte1', 'mte1_bytes_per_cycle')
    mte3 = port('mte3', 'write_bytes_per_cycle')
    hbm = chip.hbm_bandwidth_gbs/ghz * (details.get('matmul_hbm_efficiency') or 1)
    l2 = (details.get('l2_bandwidth_gbs') or 100*chip.cube_cores*ghz)/ghz
    l2 *= details.get('matmul_l2_efficiency') or 1
    cube = peak*1000/(ghz*chip.cube_cores)
    unpack = vector_peak*1000/(ghz*chip.vector_cores*unpack_ops) if mixed else None

    # Traffic on shared links. When the live K slice does not fit L2, the
    # upper traffic scenario repeats inputs for each consumer output tile.
    ab = work['operand_bytes']
    l2_reads = nt*(ab['activation']+ab['activation_scale']) + mt*(ab['weight']+ab['weight_scale'])
    first_k = min(bk, k)
    live = (_payload(m, first_k, bits) + _payload(n, first_k, 4)
            + 2*_div(first_k, 64)*(m+n))
    l2_size = details.get('l2_bytes') or 0
    hbm_reads = work['logical_read_bytes'] if live <= l2_size else l2_reads
    write = work['logical_write_bytes']
    retention = details.get('matmul_output_retained_fraction')
    retention = 1.0 if retention is None else retention
    hbm_write = write - min(write, max(0, l2_size-live))*retention
    def axis(length, size):
        full, tail = divmod(length, size)
        return [(extent, count) for extent, count in ((size, full), (tail, 1)) if count and extent]
    histogram = [(h, w, rows*cols) for h, rows in axis(m, bm) for w, cols in axis(n, bn)]
    physical_ops = sum(2*_align(h, 16)*_align(w, 16)*_align(k, 64)*count for h,w,count in histogram)
    work['implementation_work'] = {
        'padded_cube_ops_candidate': physical_ops, 'logical_k_pages': tiles*pages,
        'weight_values_expanded': mt*n*k if mixed else 0,
        'ub_to_l1_expanded_bytes': mt*n*k if mixed else 0,
        'unpack_peak_equivalents_per_weight': unpack_ops if mixed else 0,
        'dynamic_instruction_counts': None}
    h, w = min(bm, m), min(bn, n)
    def page(depth):
        scales = 2*_div(depth, 64)
        a = _payload(h, depth, bits) + h*scales
        b = _payload(w, depth, 4) + w*scales
        expanded = w*depth if mixed else 0
        # Independent AIC and AIV readers in mixed mode. Include shared
        # cache/HBM contention as well as the individual core-port limits.
        local = max(a, b/producers) if mixed else a+b
        rd = max(local/mte2, active*(a+b)/l2, active*(a+b)*hbm_reads/l2_reads/hbm)
        convert = expanded/(producers*unpack) if mixed else 0
        handoff = expanded/(producers*mte3) if mixed else 0
        ld = (a + b + (expanded-_payload(w, depth, 4) if mixed else 0))/mte1
        compute = 2*_align(h, 16)*_align(w, 16)*_align(depth, 64)/cube
        return {'read': rd, 'unpack': convert, 'handoff': handoff, 'load': ld, 'cube': compute}

    first, last = page(first_k), page((k-1)%bk+1)
    # Two pages can be in flight. The read response and Cube completion
    # latency are paid on the dependent edge; steady issue obeys the ports.
    read_latency = details.get('matmul_read_latency_cycles') or prices['read_latency_cycles']
    load_latency = details.get('matmul_l0_latency_cycles') or phases['l0_latency_cycles']
    cube_latency = details.get('matmul_mmad_latency_cycles') or phases['cube_latency_cycles']
    ii = max(clocks('read_ii_cycles'), first['read'], (read_latency+first['read'])/2,
             first['unpack'] + phases.get('unpack_setup_cycles', 0), first['handoff'], first['load'], clocks('mmad_ii_cycles')+first['cube'])
    out_service = max(2*h*w/mte3, 2*h*w*active/l2, 2*h*w*active*hbm_write/write/hbm)
    spread = prices['spread_cycles'] * (active-1 if mixed else log2(active))
    stages = {'launch': details.get('matmul_launch_cycles') or prices['launch_cycles'],
              'core_arrival_spread': spread,
              'read_fill': read_latency+last['read'], 'inline_unpack': last['unpack'] + phases.get('unpack_setup_cycles', 0),
              'ub_to_l1': last['handoff'] + phases.get('handoff_latency_cycles', 0), 'l1_to_l0': load_latency+last['load'],
              'cube_tail': cube_latency+last['cube'], 'write_tail': (details.get('matmul_drain_cycles') or prices['drain_cycles'])+out_service,
              'page_intervals': (pages-1)*ii}
    # The busiest core processes ceil(tiles/cores) rectangles. Use a full
    # rectangle envelope for its waves; shared-link totals below use exact tails.
    stages['additional_waves'] = (waves-1)*max(pages*ii, out_service)
    bus_floor = max((hbm_reads+hbm_write)/hbm, (l2_reads+write)/l2)
    stages['exposed_shared_traffic'] = max(0, bus_floor-(sum(stages.values())-stages['launch']))
    stage_us = {key: value/(1000*ghz) for key, value in stages.items()}
    us = sum(stage_us.values())
    diag.update(latency_us=us, stages_us=stage_us, stage_semantics='additive critical-path budgets',
                page_service_cycles={'first': first, 'last': last, 'initiation_interval': ii},
                traffic_bytes={'l2_read': l2_reads, 'hbm_cold_read_scenario': hbm_reads,
                               'l2_write': write, 'hbm_exposed_write_scenario': hbm_write},
                prices={'cube_ops_per_cycle_per_aic': cube, 'weights_per_cycle_per_aiv': unpack,
                        'hbm_bytes_per_cycle_chip': hbm, 'l2_bytes_per_cycle_chip': l2,
                        'mte2_bytes_per_cycle_core': mte2, 'mte1_bytes_per_cycle_core': mte1,
                        'mte3_bytes_per_cycle_core': mte3, 'read_latency_cycles': read_latency,
                        'read_issue_cycles': clocks('read_ii_cycles'), 'mmad_latency_cycles': cube_latency,
                        'mmad_issue_cycles': clocks('mmad_ii_cycles')},
                price_sources={'peaks': 'target chip cube_tops, vector_tops, core counts and frequency',
                               'vector': 'weight prologue service in 950 INT8 peak equivalents per value; not a dynamic instruction count',
                               'ports_and_phase_cycles': 'target detail overrides, MX DT/PR development phase prices and inherited BF16 MTE bytes/cycle/steady issue',
                               'tiling': 'V3 source-based output rebalance; V4 candidate dyadic subgrid; K from double-buffer capacities'})
    return us, work, diag, reason
