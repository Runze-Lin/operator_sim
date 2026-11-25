"""Resource schedule for arch35 resident AttnRes inference, with aliased outputs.

Phase clocks use eleven reference-device windows of kernel 40020.
No timing on a target being queried is required.
"""
from math import ceil
from operator_model.operators.rmsnorm.resources import WRITE_PER_KIB

# One-off dispatch, per-phase instruction/issue clocks, 64-FP32-element-group
# prices and DMA response intercepts respectively. All measured on the source
# branch. The two removed GM writes are deliberately absent from the schedule.
DISPATCH = 4022.0
WEIGHT_FIXED, WEIGHT_GROUP = 1784.8336276083464, 0.985805613642992
PREFIX_FIXED, PREFIX_GROUP = 1577.8180577849118, 2.1045455815402594
SCORE_FIXED, SCORE_PER_ROW, SCORE_GROUP_ROW = 657.8523974343806, 901.3041720108591, 8.911023796571937
NORM_FIXED, NORM_GROUP = 446.2947030497593, 6.200963081861957
GAMMA_RESPONSE, WRITE_RESPONSE, DRAIN = 637.8267698060969, 284.98174, 10.0
REFERENCE_FP32_PER_AIV_CYCLE = 27_000 / (64 * 1.65)
REFERENCE_MTE2_BYTES_PER_CYCLE = 48.3629559884


def resident_inference_price(shape, chip, work, *, vf=None):
    """Charge a resident row bank and the one-entry output queue in closed form."""
    _, hidden, _, valid = shape
    fp32, bf16 = chip.vector_tops['32'], chip.vector_tops['16']
    if not fp32 or not bf16:
        return None
    clock = 1000 * chip.frequency_ghz
    fp32_per_cycle = fp32 * 1000 / (chip.vector_cores * chip.frequency_ghz)
    bf16_per_cycle = bf16 * 1000 / (chip.vector_cores * chip.frequency_ghz)
    vector_ratio = REFERENCE_FP32_PER_AIV_CYCLE / fp32_per_cycle
    read_bw = chip.details.get('mte2_bandwidth_gbs')
    write_bw = chip.details.get('mte3_bandwidth_gbs')
    read_bytes_per_cycle = read_bw / chip.frequency_ghz if read_bw else REFERENCE_MTE2_BYTES_PER_CYCLE
    write_bytes_per_cycle = write_bw / chip.frequency_ghz if write_bw else 1024 / WRITE_PER_KIB
    row_bytes = 2 * hidden
    read = row_bytes / read_bytes_per_cycle
    store = WRITE_RESPONSE + row_bytes / write_bytes_per_cycle
    gamma = GAMMA_RESPONSE + read
    groups = ceil(hidden / 64)
    bank = valid + 1
    group_work = groups * vector_ratio
    rounds = work['tokens_per_core']
    cache = work['cached_output_norm']

    weights = WEIGHT_FIXED + WEIGHT_GROUP * group_work + 2 * read
    prefix = PREFIX_FIXED + PREFIX_GROUP * group_work + 2 * read
    score = SCORE_FIXED + SCORE_PER_ROW * bank + SCORE_GROUP_ROW * group_work * bank + valid * read
    norm = NORM_FIXED + NORM_GROUP * group_work
    if vf:
        weights = WEIGHT_FIXED + vf('att_weights',1,hidden) + 2*read
        prefix = PREFIX_FIXED + vf('att_prefix',1,hidden) + 2*read
        score = SCORE_FIXED + SCORE_PER_ROW*bank + vf('att_score',1,hidden,bank) + valid*read
        norm = NORM_FIXED + sum(vf(k,1,hidden) for k in ('rms_reduce','rms_stat','rms_norm'))
    # 40021 reads output gamma into its own UB row in Init. 40020 can
    # issue the one-row gamma read while scoring, as seen in the physical traces.
    init_gamma = gamma + hidden / bf16_per_cycle if cache else 0.
    norm_ready = score if cache else max(score, gamma)
    # Each token writes prefixOut and later Y via the same one-slot outQue.
    # Prefix store can run while the kernel scores/mixes resident rows; the
    # preceding token's Y can retire while the next prefix starts.
    post_prefix = max(store, norm_ready + norm)
    interval = max(prefix, store) + post_prefix
    serial = DISPATCH + weights + init_gamma + prefix + post_prefix + (rounds - 1) * interval + store + DRAIN
    cold_input = work['cold_input_hbm_floor_us']
    total_bytes = work['dma_read_bytes'] + work['dma_write_bytes']
    l2_bw = chip.details.get('l2_bandwidth_gbs')
    l2 = total_bytes / (1000 * l2_bw) if l2_bw else 0.
    memory = max(cold_input, l2)
    stages = dict(dispatch=DISPATCH/clock, setup_weights=(weights+init_gamma)/clock,
                  first_row=(prefix+post_prefix)/clock, following_rows=(rounds-1)*interval/clock,
                  final_store_drain=(store+DRAIN)/clock, exposed_shared=max(0., memory-serial/clock))
    return dict(latency_us=sum(stages.values()), stages_us=stages, attn_sibling_projection=True,
                model='resident_aliased_inference', qualification='Source schedule and replayed sibling-stage prices; no direct independent timing validation for this contract.',
                shared_service_us={'cold_distinct_inputs':cold_input, 'all_dma_l2':l2 if l2_bw else None,
                                   'cold_reads_and_written_outputs_scenario':(work['distinct_input_bytes']+work['dma_write_bytes'])/(1000*chip.hbm_bandwidth_gbs)},
                per_core_cycles=dict(weight_setup=weights, gamma_response=gamma, gamma_setup=init_gamma, prefix=prefix,
                                     score_softmax_mix=score, output_norm=norm, store=store, tail=DRAIN,
                                     interval=interval, prefix_store_through_norm=post_prefix),
                ports_bytes_per_cycle=dict(mte2=read_bytes_per_cycle,mte3=write_bytes_per_cycle),
                fp32_group_ratio=vector_ratio, physical_status='arch35 scheduling hypothesis, all 970 data extrapolated')
