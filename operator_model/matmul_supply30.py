"""Capacity-triggered read/write service for the conditional B-resident MatMulV3 route."""
from functools import lru_cache
import json
from math import ceil, floor, isfinite
from pathlib import Path

from .chips import REFERENCE_NAME, load_chip
from .matmul_full_load import estimate_matmul_full_load


@lru_cache(maxsize=1)
def _prices():
    source = Path(__file__).resolve().parents[1] / 'results/v0.30/matmul/prices.json'
    return json.loads(source.read_text())


def _mixed_efficiency(table, fraction):
    # Traffic histograms use nearest-decile write share. Each source contributes
    # one per-bin mean after accumulating at least 128 occupied channel-periods.
    center = min(10, floor(fraction * 10 + 0.5))
    idx = min(table, key=lambda k: abs(int(k) - center))
    return table[idx], int(idx)


def estimate_matmul_supply(shape, *, chip=REFERENCE_NAME, dtype='bf16', layout='ND_ND',
                           mixed_efficiency=None, input_mode='calibrated_ingress', read_interface_ratio=None):
    """Predict device µs. Peak BW is physical; logical cache supply is separately bounded.

    input_mode='cold' removes credit for overlapping initial read supply.
    A provided efficiency (0,1] replaces the independent DT DDR observation.
    A provided read_interface_ratio >=1 overrides the SKU's empirical read-side
    service factor. An unknown architecture normally starts with ratio one.
    """
    if input_mode not in ('cold', 'calibrated_ingress'):
        raise ValueError('input_mode must be cold or calibrated_ingress')
    if mixed_efficiency is not None and (type(mixed_efficiency) not in (int, float)
                                         or not isfinite(mixed_efficiency) or not 0 < mixed_efficiency <= 1):
        raise ValueError('mixed_efficiency must be in (0,1]')
    if read_interface_ratio is not None and (type(read_interface_ratio) not in (int, float)
                                              or not isfinite(read_interface_ratio) or read_interface_ratio < 1):
        raise ValueError('read_interface_ratio must be finite and at least one')
    target = load_chip(chip)
    v29 = estimate_matmul_full_load(shape, chip=target, dtype=dtype, layout=layout)
    if v29['latency_us'] is None:
        return v29
    active, waves = v29['active_aic'], v29['waves']
    bm, bn, bk = v29['tile']
    nt = shape[0] // bm
    w, cyc = v29['work'], v29['cycles']
    a, c = w['per_tile_a_bytes'], w['per_tile_store_bytes']
    a_page = 2 * bm * v29['a_page_k']
    weights = w['logical_b_bytes']
    size_l2 = target.details.get('l2_bytes') or load_chip().details['l2_bytes']
    # B is shared in L2 across AICs. Reserve two in-flight A pages per active
    # AIC for ping/pong. The remaining C budget is an optimistic write-back cache.
    dirty_capacity = max(0, size_l2 - weights - active * 2 * a_page)
    held_tiles = min(nt, dirty_capacity // c)
    hbm = target.hbm_bandwidth_gbs / target.frequency_ghz
    l2 = v29['rates']['l2_bytes_per_chip_cycle']
    controls = _prices()
    sku_read = controls['logical_ingress_by_calibrated_sku'].get(target.name)
    factor = read_interface_ratio
    if factor is None:
        factor = max(1.0, sku_read['minimum_logical_gbs'] / sku_read['reference_nominal_gbs']) if sku_read else 1.0
    if input_mode == 'cold':
        factor = 1.0
    observations, direction_bin = _mixed_efficiency(controls['saturated_write_fraction_deciles'], c/(a+c))
    eta = mixed_efficiency if mixed_efficiency is not None else target.details.get('matmul_hbm_efficiency')
    eta = eta if eta is not None else observations['median_eta']
    # A logical ingress faster than the off-chip port requires read hits. The
    # initial population is bounded by L2 and cannot be credited again per wave.
    cache_credit = min(size_l2, max(0, held_tiles*a*(1-1/factor)))
    warm_fraction = cache_credit / (held_tiles*a) if held_tiles else 0.0
    # Partition only at first, dirty-capacity boundary and last wave. All other
    # waves in the corresponding interval have identical demand and occupancy.
    boundary_wave = held_tiles // active
    breaks = sorted({i for i in (0, 1, boundary_wave, boundary_wave+1, waves-1, waves) if 0 <= i <= waves})
    segments = []
    total = cyc['first_cube'] + cyc['response_after_last_issue_interval'] + cyc['final_write']
    for begin, end in zip(breaks, breaks[1:]):
        width = min(active, nt-begin*active)
        reads_only_tiles = min(width, max(0, held_tiles-begin*active))
        spilling_tiles = width - reads_only_tiles
        fraction = reads_only_tiles / width
        read_prefix = reads_only_tiles * a
        read_mixed = spilling_tiles * a
        # The first A page for every active AIC was included in first_cube.
        if begin == 0:
            read_prefix -= reads_only_tiles * a_page
            read_mixed -= spilling_tiles * a_page
        read_cycles = read_prefix * (1-warm_fraction) / hbm
        mixed_cycles = (read_mixed+spilling_tiles*c) / (hbm*eta)
        # Both phases send A and C across the L2 interface. The first A pages
        # were charged in startup. The L2-to-AIC logical ingress also caps hit delivery.
        l2_cycles = (read_prefix+read_mixed+width*c) / l2
        logical_cycles = read_prefix/(hbm*factor) if read_prefix else 0
        duration = max(cyc['per_aic_step'], max(logical_cycles, read_cycles)+mixed_cycles, l2_cycles)
        phases = {'core':cyc['per_aic_step'], 'memory_supply':max(logical_cycles,read_cycles)+mixed_cycles, 'l2':l2_cycles}
        segments.append(dict(first_wave=begin,exclusive_end_wave=end,active_aic=width,output_tiles_per_wave=width,
                             cached_store_tiles_per_wave=reads_only_tiles,read_only_part=fraction,
                             read_cycles=read_cycles,mixed_cycles=mixed_cycles,l2_cycles=l2_cycles,
                             cycles_per_wave=duration,limiter=max(phases,key=phases.get)))
        total += (end-begin)*duration
    return dict(status='conditional_resource',latency_us=total/(1000*target.frequency_ghz),
                target_extrapolated=v29['target_extrapolated'],shape=list(shape),dtype=dtype,tile=list(v29['tile']),
                waves=waves,active_aic=active,mmad_total=v29['mmad_total'],cube_ops=v29['cube_ops'],work=w,
                original_us=v29['latency_us'],cycles=total,first_cube_cycles=cyc['first_cube'],
                terminal_cycles=cyc['response_after_last_issue_interval']+cyc['final_write'],segments=segments,
                pricing=dict(l2_bytes=size_l2,dirty_capacity_bytes=dirty_capacity,read_pages_reserved=2*active,
                             output_tiles_before_spill=held_tiles,logical_ingress_ratio=factor,
                             maximum_initial_read_credit_bytes=cache_credit,input_mode=input_mode,
                             hbm_bytes_per_cycle=hbm,mixed_efficiency=eta,mixed_write_fraction=c/(a+c),
                             service_write_decile=direction_bin,external_efficiency_range=[observations['min_eta'],observations['max_eta']],
                             source='explicit' if mixed_efficiency is not None else 'target config' if target.details.get('matmul_hbm_efficiency') else 'independent saturated 950DT Cast/quant DDR periods',
                             ingress_source='caller' if read_interface_ratio is not None else 'bounded independent logical GM windows' if sku_read else 'nominal cold'),
                scope='Aligned B-in-L1 route only. Output residency approximated by a fully associative dirty buffer with first-victim stores; physical traffic and initial cache contents are not directly measured. No future-chip timing validation.')
