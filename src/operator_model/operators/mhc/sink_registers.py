"""N4 FP32 Sinkhorn, same installed 950 implementation, corrected register work."""
from math import ceil

from operator_model.chips import load_chip
from operator_model.operators.mhc.sink_resources import compute_scale, parameters, sink as single_register_price


def registers_per_helper(local_matrices, register_bytes=256, ub_block_bytes=32):
    """Each vector pass handles one UB block per matrix (CANN Calc* loopSize)."""
    if register_bytes < ub_block_bytes or register_bytes % ub_block_bytes:
        raise ValueError('An integral number of UB blocks must fit a vector register')
    return ceil(local_matrices / (register_bytes // ub_block_bytes))


def price(shape, chip, attrs):
    """Correct the per-iteration SIMD work while retaining calibrated stage prices.

    The 256-byte register and 32-byte UB block describe the assumed implementation.
    Kernel-ISA changes require a separate contract and issue-cost calibration.
    """
    base, aic, aiv, diag = single_register_price(shape, chip, attrs)
    plan = diag['target_plan']
    logical_chunks = plan['packed_chunks']
    groups = registers_per_helper(plan['rows_per_core'])
    plan['register_passes'] = groups
    if logical_chunks == groups:
        return base, aic, aiv, diag
    fixed, per_iteration = parameters()['phases']['mhc_sinkhorn']['SIMD']
    ref = load_chip()
    stages = diag['stages_us']
    stages['SIMD'] = (fixed * ref.frequency_ghz / chip.frequency_ghz
                      + per_iteration * groups * plan['iterations'] * compute_scale(ref, chip, 'vector', '32'))
    return sum(stages.values()), aic, aiv, diag
