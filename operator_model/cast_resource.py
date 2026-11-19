"""Contiguous BF16-to-FP32 Cast in the 950 ops-math package."""
from .workload import _counts


# Median full-tile DMA descriptor response divided by its modeled port
# service on a v0.20 950DT [262144, 32] critical AIV. These dimensionless
# factors price slot residence, separately from the port throughput below.
# Source: results/cast-capacity/two-slot-freeze.json.
_READ_RESPONSE_FACTOR = 1.4394182462466096
_WRITE_RESPONSE_FACTOR = 1.9750504967790543


def _elements(shape):
    if len(shape)!=2 or any(type(x) is not int or x<=0 for x in shape):
        raise ValueError('cast_bf16_fp32 expects positive integer [tokens, channels]')
    return shape[0]*shape[1]


def cast_work(shape):
    n=_elements(shape)
    return _counts('bf16',2*n,4*n,vec={'fp32':n},classes={'bf16_to_fp32_conversion':n},
                   notes=['One exact widening conversion per element. This lane-operation count describes conversions, not dense FP32 multiply-adds.'])


def resource_cast(shape, chip):
    n = _elements(shape)
    # 2048-element minimum grid grain inferred from a 16384-element launch of
    # eight active AIVs. Above saturation, share 32-byte aligned input blocks.
    grain = 2048
    active = min(chip.vector_cores,(n+grain-1)//grain)
    blocks = (n+15)//16
    local = min(n,16*((blocks+active-1)//active))
    cycles_per_us = 1000*chip.frequency_ghz
    # Terminal AIV at [64,4096] ran one 8192-byte read and 16384-byte write.
    # The service-byte slopes are borrowed from the 950DT BF16/FP32 DMA
    # schedule; first readiness, dependencies and the release offset use Cast.
    from .rms_resource import READ_PER_KIB, WRITE_PER_KIB
    def byte_price(port, reference):
        rate = chip.details.get(port)
        return 1024*chip.frequency_ghz / rate if rate else reference
    rprice = byte_price('mte2_bandwidth_gbs',READ_PER_KIB)
    wprice = byte_price('mte3_bandwidth_gbs',WRITE_PER_KIB)
    read_fixed = 965 - 8*READ_PER_KIB
    write_fixed = 455 - 16*WRITE_PER_KIB
    # Two input buffers and two widened output buffers share UB.
    cap = ((chip.details.get('ub_bytes') or 253952)//(2*(2+4))//64)*64
    if cap < 64:
        raise ValueError('UB too small for one 64-lane Cast vector')
    first = min(cap,local)
    waves = (local+cap-1)//cap
    target = (chip.vector_tops['32'] or 0)*1000/(chip.vector_cores*chip.frequency_ghz)
    vec_price = (27000/(64*1.65))/target if target else 1.
    arrival = (1035/63)*(active-1)
    # The 113 busy Vector cycles covered 64 groups of 64 elements in the
    # reference trace. Dense peak pricing assumes similar target issue width.
    groups = (first+63)//64
    simd = 113*groups/64*vec_price
    stages = {'arrival':arrival,'control_to_read':1367.,
              'read':read_fixed+2*first/1024*rprice,
              'vector':simd,'dependency_to_write':838.,
              'write':write_fixed+4*first/1024*wprice}
    remaining = local-first
    stages['steady_tiles'] = max(2*remaining/1024*rprice,4*remaining/1024*wprice,
                                 (113/64)*((remaining+63)//64)*vec_price)
    input_slot_ii = output_slot_ii = None
    if remaining:
        # The next tile can reuse an input slot once compute consumed it,
        # and an output slot once its write retired. Each pair of slots can
        # sustain one tile per half-residence; the slower pool limits issue.
        input_slot_ii = (_READ_RESPONSE_FACTOR*stages['read'] + simd)/2
        output_slot_ii = (_WRITE_RESPONSE_FACTOR*stages['write'] + simd)/2
        repeated_tile_equivalents = remaining/cap
        stages['steady_tiles'] = max(stages['steady_tiles'],
                                     repeated_tile_equivalents*max(input_slot_ii, output_slot_ii))
    from .vector_memory import stream_memory
    memory = stream_memory(chip, active, 2*n, 4*n)
    shared = 1753 + max(memory['external_floor_cycles'], memory['l2_floor_cycles'] or 0)
    stages['shared_io'] = max(0.,shared-sum(stages.values()))
    times = {k:v/cycles_per_us for k,v in stages.items()}
    plan = dict(active_aic=0,active_aiv=active,elements=n,busy_core_elements=local,
                minimum_elements_per_core=grain,busy_core_waves=waves,ub_tile_elements=cap)
    return sum(times.values()),0,active,dict(target_plan=plan,stages_us=times,
                                            input_slot_ii_cycles=input_slot_ii,
                                            output_slot_ii_cycles=output_slot_ii,
                                            read_cycles_per_kib=rprice,write_cycles_per_kib=wprice,
                                            standalone_first_launch=True, **memory)
