"""Opt-in one-output-rectangle Stream-K estimate for contiguous BF16 MatMulV3."""
from math import ceil, log2

from .chips import load_chip
from .matmul_resource import _costs, resource_matmul


def estimate_matmul_splitk(shape, *, chip='Ascend950DT_9582', dtype='bf16',
                          layout='ND_ND', deterministic_level=0):
    """Compare ordinary tiling with one K partition per AIC on the same output.

    shape orders logical M,N,K of A[M,K] @ B[K,N]. Operands have no bias,
    transpose, striding or broadcasting. The code uses only the chip config.
    The host may also choose other modes on shapes with several output tiles.
    """
    if (not isinstance(shape, (tuple, list)) or len(shape) != 3
            or any(type(x) is not int or x < 1 for x in shape)):
        raise ValueError('Expected three positive integer dimensions [M,N,K]')
    if dtype != 'bf16' or layout != 'ND_ND':
        raise ValueError('Candidate requires contiguous untransposed BF16 ND_ND')
    if type(deterministic_level) is not int or not 0 <= deterministic_level <= 3:
        raise ValueError('deterministic_level must be one of 0, 1, 2, 3')
    target = load_chip(chip)
    m, n, k = shape
    ordinary, aic, aiv, diagnostics = resource_matmul(shape, target)
    tiles = diagnostics['target_plan']['output_tiles']
    cores = target.cube_cores
    threshold = max(8192, 256 * cores) / 2  # host byte threshold / sizeof(BF16)
    base = dict(route='ordinary', latency_us=ordinary, active_aic=aic, active_aiv=aiv,
                ordinary_us=ordinary, threshold_aligned_k=threshold, output_tiles=tiles,
                cube_ops=2*m*n*k, input_bytes=2*k*(m+n), final_output_bytes=2*m*n,
                evidence='experimental; no timing qualification on a new chip')
    if tiles != 1:
        return {**base, 'reason': 'Several ordinary output rectangles; other provider Stream-K routes are unmodeled'}
    if ceil(k/256)*256 < threshold:
        return {**base, 'reason': 'Aligned K below the DAV_3510 host threshold'}
    if deterministic_level > 1:
        return {**base, 'reason': 'The host disables split K at this determinism level'}
    if not target.vector_tops.get('32'):
        return {**base, 'reason': 'Missing FP32 Vector throughput; split candidate cannot be priced'}
    splits = min(cores, ceil(k/64))
    local_k = 64 * ceil(ceil(k/splits)/64)
    _, _, _, partial = resource_matmul([m, n, local_k], target)
    stages = partial['stages_us']
    pipeline = sum(stages[stage] for stage in
                   ('read_fill', 'l0_lead', 'page_intervals', 'last_mmad', 'write_tail'))
    prices = _costs()
    clock = target.frequency_ghz
    launch_cycles = target.details.get('matmul_launch_cycles', prices['launch_cycles'])
    launch = (launch_cycles + prices['arrival_log2_cycles']*log2(splits)) / (1000*clock)
    cold = base['input_bytes'] / (1000*target.hbm_bandwidth_gbs)
    merge_bytes = 2*m*n*(4*splits-1)
    vec_ops_cycle = target.vector_tops['32']*1000 / (target.vector_cores*clock)
    merge_issue = (1000 + 60*splits*256/vec_ops_cycle) / (1000*clock)
    bandwidth = target.details.get('l2_bandwidth_gbs') or target.hbm_bandwidth_gbs
    merge = merge_issue + merge_bytes/(1000*bandwidth)
    total = launch + max(pipeline, cold) + merge
    if total >= ordinary:
        return {**base, 'reason': 'Ordinary route is cheaper at the supplied target rates',
                'split_candidate_us': total}
    return {**base, 'route': 'one_output_split_k', 'latency_us': total,
            'active_aic': splits, 'active_aiv': min(target.vector_cores, ceil(splits*target.vector_cores/cores)),
            'k_splits': splits, 'max_aligned_depth': local_k, 'partial_fixup_bytes': merge_bytes,
            'logical_reduction_adds': m*n*(splits-1),
            'launch_us': launch, 'partial_pipeline_us': pipeline, 'cold_floor_us': cold,
            'merge_us': merge, 'merge_issue_us': merge_issue,
            'reason': 'One output rectangle and profitable split K on the supplied target resources',
            'mix_qualification': ('Assumes a compatible 1:2 mixed executable'
                                  if target.vector_cores == 2*cores else
                                  'Requires a future mixed executable for the new target AIC:AIV ratio')}
