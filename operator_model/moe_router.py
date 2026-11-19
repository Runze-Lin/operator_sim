"""One-group FP32 sigmoid MoE router, calibrated on Ascend 950 DT."""
from math import ceil

from .workload import _counts


def _validate(shape):
    if len(shape) != 3 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError('moe_gating_top_k expects positive integer [tokens,experts,k]')
    _, experts, k = shape
    if not 2 < experts <= 2048 or k > experts:
        raise ValueError('Require 2 < experts <= 2048 and k <= experts')


def router_work(shape):
    _validate(shape)
    t, e, k = shape
    classes = {'sigmoid_neg_exp_add_reciprocal':4*t*e, 'bias_add':t*e,
               'selected_sum':t*(k-1), 'eps_add':t, 'scale_normalize':t*k}
    return _counts('fp32',4*(t*e+e),8*t*k,vec={'fp32':sum(classes.values())},classes=classes,
                   notes=['Scores include sigmoid(x), selection with FP32 bias and normalization of selected original scores. Comparisons, index work, alignment and cache traffic are outside the nominal FLOP count.'])


def resource_router(shape, chip):
    """Return the all-core window and symbolic pipeline of this single launch."""
    _validate(shape)
    t, e, k = shape
    rows_per_core = ceil(t / chip.vector_cores)
    cores = ceil(t / rows_per_core)
    clock = 1000 * chip.frequency_ghz
    peak = chip.vector_tops['32']
    if peak is None:
        raise ValueError('FP32 Vector peak is required for MoE routing')
    peak_per_core = peak * 1e6 / chip.vector_cores
    # Per-core MTE throughput reaches a share of aggregate HBM or its issue cap.
    dma_bytes_us = min(32 * clock, chip.hbm_bandwidth_gbs * 1000 / cores)
    read_bytes = 4 * e * (1 + rows_per_core)
    write_bytes = 8 * k * rows_per_core
    # Two complete single-row traces provide the read-ready -> first store
    # interval, measured in cycles at 128 and 896 experts (2 and 14 vectors).
    blocks = ceil(e / 64)
    vec_equivalent_ops = (139.33333333333334 + 208.33333333333334 * blocks) * (27e6 / (64 * 1650))
    stages = {'control_and_tiling':4201 / clock,
              'read_ready':max(1243 / clock, read_bytes / dma_bytes_us),
              'rank_and_normalize':rows_per_core * vec_equivalent_ops / peak_per_core,
              'drain_weights_ids':max(311.5 / clock,write_bytes / dma_bytes_us)}
    plan = {'active_aic':0,'active_aiv':cores,'rows_per_busy_core':rows_per_core,
            'expert_blocks_per_row':blocks,'rank_candidates_per_row':e,'selected_per_row':k}
    return sum(stages.values()),0,cores,{'target_plan':plan,'stages_us':stages,
                                         'dma_bytes_per_us':dma_bytes_us,
                                         'vec_peak_equivalent_ops_per_row':vec_equivalent_ops}
