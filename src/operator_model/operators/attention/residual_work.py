"""Inference work and arch35 tiling of vLLM Ascend's Kimi AttnResFwd."""
from operator_model.chips import load_chip


def attention_residual_work(tokens, hidden, slots, valid, *, chip=None, addend=False,
                            output_norm=True, write_idx=-1, materialize=False,
                            mix=True, optimize_prefill=False):
    """Return source-derived work and cold-input/FP32 peak floors, in microseconds.

    Counts refer to a contiguous BF16 prefix and per-token BF16 bank rows.
    Tiling and buffer guards follow vllm-ascend c09d3e20. Physical instruction
    overhead, synchronization, DMA response and exp/rsqrt services are unpriced.
    """
    if any(type(n) is not int for n in (tokens,hidden,slots,valid,write_idx)):
        raise ValueError('tokens, hidden, slots, valid and write_idx must be integers')
    if tokens<0 or hidden<=0 or slots<=0 or not 0<=valid<=slots:
        raise ValueError('invalid prefix or bank dimensions')
    if write_idx != -1 and not valid<=write_idx<slots:
        raise ValueError('write_idx must identify an unused bank slot')
    if not all(type(x) is bool for x in (addend,output_norm,materialize,mix,optimize_prefill)):
        raise ValueError('operation flags must be boolean')
    target=load_chip() if chip is None else load_chip(chip)
    ub=target.details.get('ub_bytes')
    if ub is None:
        raise ValueError('target UB capacity is required for arch35 branch selection')
    ceildiv=lambda x,y: (x+y-1)//y
    launch=min(tokens,target.vector_cores)
    rounds=ceildiv(tokens,launch) if launch else 0
    active=ceildiv(tokens,rounds) if rounds else 0
    bank_rows=valid+1
    # The upstream fusedChain reserves the fusedPrefix staging row even when
    # there is no addend. Tokens with no valid history take a mix bypass.
    max_resident=max(0,ub-22*hidden-32768)//(2*hidden+4)
    resident=bank_rows<=max_resident
    b16=16*ceildiv(hidden,16)
    f32=8*ceildiv(hidden,8)
    meta=8*ceildiv(bank_rows,8)
    footprint=(2*(2+bank_rows) if resident else 6)*b16 + 2*b16 + 12*f32 + 40*meta+32+32768
    cached_gamma=optimize_prefill and output_norm and rounds>1 and footprint+2*b16<=ub
    key=(40020 if resident else 40010)+int(cached_gamma)
    mixing=mix and valid>0
    block_reads=valid*tokens*(1 if resident else 2) if mixing else 0
    weight_reads=2*active if mixing else 0
    gamma_reads=(active if cached_gamma else tokens) if output_norm else 0
    row_bytes=2*hidden
    dma_read=row_bytes*(tokens*(1+int(addend))+block_reads+weight_reads+gamma_reads)
    writes=row_bytes*tokens*(1+int(addend)+int(write_idx>=0)+int(materialize))
    unique_input=row_bytes*(tokens*(1+int(addend)+(valid if mixing else 0))
                            + (int(mixing)*2+int(output_norm) if tokens else 0))
    # Useful scalar equivalent operations. Keep nonlinear reductions separate.
    normalized_rows=(tokens*bank_rows if mixing else 0)+tokens*int(output_norm)
    dot_rows=tokens*bank_rows if mixing else 0
    # Squares, reductions, multiplication by 1/H and epsilon addition.
    rms_ops=normalized_rows*(2*hidden+1)
    normalized_dot=dot_rows*(3*hidden-1)
    weighted=dot_rows*2*hidden       # source MulAddDst into zero-filled vector
    softmax_basic=tokens*(3*bank_rows-2) if mixing else 0  # max, shift, add
    softmax_scale=tokens*bank_rows if mixing else 0
    fp32_ops=(rms_ops+normalized_dot+weighted+softmax_basic+softmax_scale
              +2*hidden*tokens*int(output_norm)+active*hidden*int(mixing))
    useful_add=hidden*tokens*int(addend)
    aggregate_peak=target.vector_tops['32']
    effective_peak=aggregate_peak*active/target.vector_cores if aggregate_peak and active else None
    arithmetic_floor=fp32_ops/(effective_peak*1e6) if effective_peak else (0. if not tokens else None)
    return dict(qualification='source-derived work; device latency unpriced', latency_us=None,
                implementation='vllm-ascend AttnResFwd arch35 BF16 inference',
                launch_aiv=launch, payload_aiv=active, tokens_per_core=rounds,
                tiling_key=key if tokens else None, resident=resident,
                max_resident_rows=max_resident,cached_output_norm=cached_gamma,
                dma_read_bytes=dma_read,dma_write_bytes=writes,
                distinct_input_bytes=unique_input,fp32_basic_ops=fp32_ops,
                prefix_add_elements=useful_add,exp_elements=dot_rows,rsqrt_elements=normalized_rows,
                reciprocal_elements=tokens if mixing else 0,cube_ops=0,
                cold_input_hbm_floor_us=unique_input/(target.hbm_bandwidth_gbs*1000),
                fp32_peak_floor_us=arithmetic_floor)
