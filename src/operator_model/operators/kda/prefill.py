"""Kimi-K3 prefill dispatch for the vLLM Ascend / FLA A5 26.9.1 graph."""
from itertools import accumulate

from operator_model.backend import estimate
from operator_model.chips import REFERENCE_NAME, load_chip
from operator_model.operators.kda.fla import IMPLEMENTATION
from operator_model.operators.kda.fused_work import kimi_v1_work


def plan_kimi_kda_prefill(query_lengths, *, attention_tp, chip=REFERENCE_NAME,
                          initialized_sequences=None):
    """Resolve one rank's actual prefill chunk call from this step's token counts.

    Zero-length scheduler slots are compacted. The pinned Kimi-K3 has 96 equal
    QK/V heads, D=128, raw BF16 gate and preprocessed FP32 beta. Valid cache
    indices and a dense FP32 VK payload are assumed. `initialized_sequences`
    counts true gather flags among nonempty requests. Only the V2 main kernel
    subtotal is currently priced. full_dispatch_us includes graph dependencies.
    v1_work describes useful Finalize work on an explicit-length four-phase
    V1 route. Its ideal Cube floor has a narrower scope than device latency.
    """
    if type(attention_tp) is not int or attention_tp < 1 or 96 % attention_tp:
        raise ValueError('attention_tp must divide the 96 Kimi-K3 KDA heads')
    lengths = tuple(query_lengths)
    if not lengths or any(type(n) is not int or n < 0 for n in lengths):
        raise ValueError('query_lengths must contain nonnegative integer scheduler lengths')
    active = tuple(n for n in lengths if n)
    batch = len(active)
    if initialized_sequences is not None and (type(initialized_sequences) is not int or
            not 0 <= initialized_sequences <= batch):
        raise ValueError('initialized_sequences must count nonempty rows with valid prior state')
    target = load_chip(chip)
    heads = 96 // attention_tp
    tokens = sum(active)
    chunks = sum((n + 63) // 64 for n in active)
    work = chunks * heads
    shape = (1, tokens, heads, heads, 128, 128)
    state_row = heads * 128**2 * 4
    cached = batch * state_row
    min_reads = 0 if initialized_sequences is None else initialized_sequences * state_row
    max_reads = cached if initialized_sequences is None else min_reads
    hd = tokens * heads * 128
    normalized = 4 * hd  # two BF16 tensors (Q and K)
    # Distinct logical payloads at the FLA call: Q/K/V/g, beta, A_log,
    # dt_bias, initial state, followed by O/Aqk/Akk and a distinct final state.
    visible = 10 * hd + (4 + 4 * 64) * tokens * heads + 4 * heads * 129 + 2 * cached
    plan = dict(chip=target.name, selector='vllm_ascend_kimi_k3_fla_a5_26.9.1',
                chip_policy="assume the same 950 FLA implementation and host selection rule on future chips",
                local_heads=heads, query_lengths=lengths, nonempty_requests=batch,
                shape=shape, chunks=chunks, work_items=work, aclnn=None,
                main_kernels=(), main_kernels_us=None, full_dispatch_us=None, v1_work=None,
                status='unavailable', reason=None,
                graph_io=dict(qk_l2norm_read_bytes=normalized, qk_l2norm_write_bytes=normalized,
                              state_gather_read_bytes_range=(min_reads, max_reads),
                              state_gather_write_bytes=cached, state_scatter_read_bytes=cached,
                              state_scatter_write_bytes=cached),
                main_stages_us={}, minimum_payload_bytes=None, capacity_bytes=round(target.hbm_capacity_gb*1e9),
                scope='one prefill dispatch, after convolution and beta preprocessing; excludes decoder and speculative recurrent calls')
    if not batch:
        plan.update(status='empty', reason='scheduler supplied no prefill tokens',
                    full_dispatch_us=0.0, minimum_payload_bytes=0)
        return plan
    if tokens > 2**31-1 or work > 2**32-1 or batch > 1024:
        plan.update(status='unsupported', reason='packed token, work index or 26.9.1 sequence-count limit exceeded')
        return plan
    if work < 4096:
        # 950 host uses a four-phase physical split for canonical varlen with
        # at least two chunks. These launches all belong to the same V1 family.
        stages = ('gate_prepare', 'post_wu', 'fwd_h', 'finalize') if chunks > 1 else ('full',)
        plan.update(aclnn='aclnnChunkKdaFwd', main_kernels=tuple('ChunkKdaFwd:'+s for s in stages),
                    minimum_payload_bytes=visible,
                    v1_work=kimi_v1_work(active, heads, target, split_phases=chunks > 1),
                    reason='selected fused V1 has no qualified cost for this dispatch')
        return plan
    # At the Prepare/FwdH boundary gk (FP32), W/U/Kg/QgScaled (BF16)
    # coexist. At FwdH/Finalize, QgScaled, vNew, and all BF16 chunk H do.
    after_prepare = 12 * hd
    after_fwdh = 4 * hd + 2 * work * 128**2
    lower = visible + max(after_prepare, after_fwdh)
    plan.update(aclnn='aclnnChunkKdaFwdV2', main_kernels=('chunk_kda_fwd_prepare','chunk_fwd_h','chunk_kda_fwd_finalize'),
                minimum_payload_bytes=lower,
                memory_bytes=dict(public_live=visible, intermediate_after_prepare=after_prepare,
                                  intermediate_after_fwdh=after_fwdh))
    if lower > plan['capacity_bytes']:
        plan.update(status='infeasible', reason='minimum simultaneous tensor payload exceeds the chip HBM capacity')
        return plan
    if min(active) != max(active):
        plan.update(reason='ragged packed route identified; frozen resource scheduler covers equal nonempty lengths')
        return plan
    offsets = tuple(accumulate(active, initial=0))
    estimate_main = estimate('kda_chunk', shape, {'implementation': IMPLEMENTATION, 'cu_seqlens': offsets}, chip=target)
    if estimate_main.latency_us is None:
        plan.update(status=estimate_main.status, reason=estimate_main.reason)
        return plan
    plan.update(status='extrapolated', main_kernels_us=estimate_main.latency_us,
                main_stages_us=estimate_main.projection['stages_us'],
                reason='three serial V2 compute kernels; graph helpers, possible materialization and launch gaps have no qualified times')
    return plan
