"""One-output-rectangle Stream-K selection for contiguous BF16 MatMulV3."""
from math import ceil, log2

from operator_model.chips import load_chip
from operator_model.operators.matmul.resources import _costs, resource_matmul


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
    return _select(shape, target, resource_matmul(shape, target), deterministic_level)


def _select(shape, target, baseline, deterministic_level):
    m, n, k = shape
    ordinary, aic, aiv, diagnostics = baseline
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
            'partial_tile': partial['target_plan']['tile'],
            'mmads_per_partial_budget': partial['target_plan']['mmads_total'],
            'partial_read_page_k': partial['target_plan']['read_page_k'],
            'launch_us': launch, 'partial_pipeline_us': pipeline, 'cold_floor_us': cold,
            'merge_us': merge, 'merge_issue_us': merge_issue,
            'reason': 'One output rectangle and profitable split K on the supplied target resources',
            'mix_qualification': ('Assumes a compatible 1:2 mixed executable'
                                  if target.vector_cores == 2*cores else
                                  'Requires a future mixed executable for the new target AIC:AIV ratio')}

def resource_matmul_dispatch(shape, target):
    """Return the selected schedule and physically distinct internal work."""
    baseline = resource_matmul(shape, target)
    result = _select(shape, target, baseline, 0)
    if result['route'] == 'ordinary':
        us, aic, aiv, diag = baseline
        return us, aic, aiv, {**diag, 'streamk_selection': {
            'route': 'ordinary', 'reason': result['reason'],
            'threshold_aligned_k': result['threshold_aligned_k']}}
    m, n, _ = shape
    s = result['k_splits']
    plan = dict(active_aic=result['active_aic'], active_aiv=result['active_aiv'], branch=result['route'],
                output_tiles=1, waves=1, tile=result['partial_tile'], k_splits=s,
                max_aligned_depth=result['max_aligned_depth'], read_page_k=result['partial_read_page_k'],
                mmads_total=s*result['mmads_per_partial_budget'], iterations=result['mmads_per_partial_budget'])
    internal = dict(fp32_partial_write_bytes=4*m*n*s, fp32_partial_read_bytes=4*m*n*s,
                    fp32_reduction_adds=result['logical_reduction_adds'],
                    priced_incremental_fixup_bytes=result['partial_fixup_bytes'])
    stages = dict(launch=result['launch_us'], partial_pipeline=result['partial_pipeline_us'],
                  exposed_hbm_read=max(0., result['cold_floor_us']-result['partial_pipeline_us']),
                  merge_issue=result['merge_issue_us'], merge_transfer=result['merge_us']-result['merge_issue_us'])
    diag = dict(model='target_resource_matmul_single_output_split_k', target_plan=plan,
                streamk_selection=result, stages_us=stages, internal_work=internal,
                logical_cube_ops=result['cube_ops'], logical_read_bytes=result['input_bytes'],
                write_bytes=result['final_output_bytes'], mmads_total=plan['mmads_total'],
                qualification='One-output long-K resource estimate; limited 950 geometry coverage. '+result['mix_qualification'])
    return result['latency_us'], result['active_aic'], result['active_aiv'], diag
