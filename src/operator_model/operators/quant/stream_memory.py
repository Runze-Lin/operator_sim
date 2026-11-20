"""Streaming cold-input/dirty-output scenario for DynamicMxQuant; no target traces required."""
from math import ceil, isfinite

from operator_model.operators.quant.resources import resource_quant


def scenario(shape, chip, base_us, *, mixed_bus_efficiency, wire_to_configured_bandwidth=1.0):
    """Estimate a sequential write-back stream with a paired read/data+scale implementation.

    base_us is the normal kernel prediction. mixed_bus_efficiency is a payload
    fraction of the target memory pin rate; wire_to_configured_bandwidth translates
    the chip config's advertised GB/s to that pin rate. Results assume cold input
    and proportional retention of clean/dirty lines outside active prefetch slots.
    """
    if not (isfinite(base_us) and base_us > 0 and 0 < mixed_bus_efficiency <= 1 and
            isfinite(wire_to_configured_bandwidth) and wire_to_configured_bandwidth > 0):
        raise ValueError('A positive base time, physical link rate and efficiency in (0,1] are required')
    resource, _, active, diag = resource_quant(shape, chip)
    plan = diag['target_plan']
    h, w = plan['rows_per_core'], plan['cols_per_core']
    tr, tc = plan['tile']
    r = diag['assumed_offchip_read_bytes']
    wout = diag['logical_write_bytes']
    capacity = chip.details.get('l2_bytes')
    if capacity is None:
        raise ValueError('The streaming scenario needs an L2 capacity')
    # Input pages still in the two-slot working set occupy L2 in the same
    # assumption as the old resource model. Account for them only once.
    live = min(r, min(2, plan['iterations']) * active * diag['first_tile_cycles']['read_bytes'])
    available = max(0., capacity-live)
    retained_fraction = min(1., available / (r-live+wout))
    retained = wout*retained_fraction
    common_fraction = 1-retained_fraction
    # First phase reads into a fitting clean/dirty footprint. As older
    # lines leave L2, mixed input reads and dirty data+scale writes share pins.
    read_prefix, read_mixed, write_mixed = r*retained_fraction, r*common_fraction, wout*common_fraction
    peak = chip.hbm_bandwidth_gbs * 1000 * wire_to_configured_bandwidth
    launch = diag['first_launch_cycles']/(1000*chip.frequency_ghz) + diag['stages_us']['drain']
    prefix_us = min(base_us, launch + read_prefix/peak)
    mixed_us = (read_mixed+write_mixed)/(peak*mixed_bus_efficiency)
    # The read prefix can be shorter with pre-existing residency. Capping
    # its estimate at the original full kernel time also joins the two
    # cache regimes continuously even if the base has a warm read path.
    # The local work grows with critical tiles, never with chip-wide volume.
    # Four tile classes (rectangular body, row/column tails, corner) suffice.
    f = chip.frequency_ghz * 1000
    loops = diag['first_tile_cycles']['loops']
    per_loop = diag['effective_vec_ops_per_loop'] / diag['vec_ops_per_cycle_per_core']
    fixed_vf = diag['first_tile_cycles']['compute_cycles'] - loops*per_loop
    full_r, tail_r = divmod(h, tr)
    full_c, tail_c = divmod(w, tc)
    vector_cycles = 0.
    for nrows, height in ((full_r,tr), (int(bool(tail_r)),tail_r)):
        for ncols, width in ((full_c,tc),(int(bool(tail_c)),tail_c)):
            groups = height*ceil(width/32)
            vector_cycles += nrows*ncols*(fixed_vf+(ceil(groups/8)+ceil(groups/128))*per_loop)
    local_us = dict(MTE2=2*h*w/(diag['read_bytes_per_cycle_per_core']*f),
                    SIMD=vector_cycles/f, MTE3=h*(w+2*ceil(w/64))/(diag['write_bytes_per_cycle_per_core']*f))
    # These are lower bounds on interleaved service. Preserve the base estimate
    # when the whole working set fits; do not attach a bus launch to a warm call.
    candidate = max(base_us, prefix_us+mixed_us, *local_us.values())
    return dict(latency_us=candidate, base_us=base_us, old_resource_us=resource, active_aiv=active,
                critical_local_us=local_us, plan=plan, live_bytes=live, logical_read=r, logical_write=wout,
                dirty_retained_bytes=retained, read_prefix_bytes=read_prefix, read_mixed_bytes=read_mixed,
                write_mixed_bytes=write_mixed, prefix_us=prefix_us, mixed_us=mixed_us,
                mixed_bus_efficiency=mixed_bus_efficiency, wire_to_configured_bandwidth=wire_to_configured_bandwidth)
