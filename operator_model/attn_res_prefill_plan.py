"""Ideal-service schedule for the cached-gamma fused Kimi prefill branch."""
from .attention_residual import attention_residual_work
from .chips import load_chip
from .rms_resource import READ_PER_KIB, WRITE_PER_KIB


def prefill_service_plan(shape, chip, *, addend=True, write_idx=-1):
    """Report conditional service at peak; waits, nonlinear unit and launch are unpriced.

    Only the arch35 resident inference path with a cached normalization weight is
    described. This diagnostic leaves the public device latency untouched.
    """
    target = load_chip(chip)
    tokens, hidden, slots, valid = shape
    work = attention_residual_work(tokens, hidden, slots, valid, chip=target,
                                  addend=addend, write_idx=write_idx, optimize_prefill=True)
    if work['tiling_key'] != 40021 or not valid:
        raise ValueError('Requires mixed resident prefill with cached output norm')
    fp32 = target.vector_tops['32']
    fp16 = target.vector_tops['16']
    if not fp32 or not fp16:
        raise ValueError('Missing FP32 or BF16 vector peak')
    rows = work['tokens_per_core']
    bank = valid + 1
    row_bytes = 2 * hidden
    fp32_rate = fp32 * 1e6 / target.vector_cores
    bf16_rate = fp16 * 1e6 / target.vector_cores
    rd = target.details.get('mte2_bandwidth_gbs') or 1024 * target.frequency_ghz / READ_PER_KIB
    wr = target.details.get('mte3_bandwidth_gbs') or 1024 * target.frequency_ghz / WRITE_PER_KIB
    read_row = row_bytes / (1000 * rd)
    write_row = row_bytes / (1000 * wr)
    # PREP fetches score weights and gamma on each payload AIV, once.
    setup = 3 * read_row + hidden / fp32_rate
    prefix = (1 + int(addend)) * read_row + int(addend) * hidden / bf16_rate
    score = valid * read_row + bank * (5 * hidden) / fp32_rate
    softmax = (4 * bank - 2) / fp32_rate
    output = (2 * bank * hidden + 4 * hidden + 1) / fp32_rate
    saved_writes = int(addend) + int(write_idx >= 0)
    # Store descriptors are serial on MTE3, with separate prefix/output
    # buffers. Candidate assumes overlap with score; next row waits for output.
    branches = max(saved_writes * write_row, score + softmax + output)
    tile_us = prefix + branches + write_row
    vector32 = (hidden + rows * (bank * (7 * hidden) + 4 * hidden + 1 + 4 * bank - 2)) / fp32_rate
    vector16 = rows * int(addend) * hidden / bf16_rate
    resources = dict(mte2=(3 + rows * (1 + int(addend) + valid)) * read_row,
                     mte3=rows * (1 + saved_writes) * write_row,
                     fp32=vector32, bf16=vector16, cold_input_hbm=work['cold_input_hbm_floor_us'])
    if target.details.get('l2_bandwidth_gbs'):
        resources['l2'] = ((work['dma_read_bytes'] + work['dma_write_bytes']) /
                           (1000 * target.details['l2_bandwidth_gbs']))
    return dict(model='resident_cached_gamma_ideal_service', target=target.name,
                tiling_key=40021, active_aiv=work['payload_aiv'], launch_aiv=work['launch_aiv'],
                critical_rows=rows, logical_work=work, component_floor_us=resources,
                ports_gbs_per_core=dict(MTE2=rd, MTE3=wr), phases_us=dict(
                    setup=setup, prefix=prefix, scores=score, softmax=softmax,
                    weighted_norm=output, prefix_or_bank_store=saved_writes*write_row,
                    output_store=write_row, token_interval=tile_us),
                conditional_schedule_us=max(max(resources.values()), setup + rows*tile_us),
                qualified_latency_us=None,
                omissions=['scalar launch and DMA command/response', 'rsqrt, reciprocal and exp service',
                           'reduction tree latency, VF control, bank conflicts and cross-core contention'])
