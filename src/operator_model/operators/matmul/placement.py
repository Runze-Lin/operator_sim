"""BF16 contiguous MatMulV3 placement hypothesis for the CANN 9.2.1 950 implementation."""
from operator_model.chips import REFERENCE_NAME, load_chip
from operator_model.profiles.stage_model import align, ceildiv
from operator_model.operators.matmul.multi_split import estimate_matmul_retile


def matmul_placement(shape, *, chip=REFERENCE_NAME):
    """Predict the 950 library's half-core split-K boundary and price an available schedule.

    The phase-one workspace signal describes allocation size, not HBM traffic.
    No CANN code, target trace or measured target time is queried. The physical
    plan applies to contiguous untransposed BF16, no bias, default determinism.
    """
    target = load_chip(chip)
    previous = estimate_matmul_retile(shape, chip=target)  # validates shape and prices physical pipelines
    m, n, k = shape
    # Two BF16 L0 buffers must each hold at least 64 positions of K.
    max_m = (target.details.get('l0a_bytes') or 65536) // (2*2*64*16) * 16
    max_n = (target.details.get('l0b_bytes') or 65536) // (2*2*64*16) * 16
    if min(max_m,max_n) < 16:
        raise ValueError('Double-buffered L0 does not fit one BF16 16x64 operand')
    rows, cols = ceildiv(m,max_m), ceildiv(n,max_n)
    groups = rows*cols
    bm, bn = align(ceildiv(m,rows),16), align(ceildiv(n,cols),16)
    cores = target.cube_cores
    threshold = max(8192,256*cores)//2
    admitted = (groups <= cores//2 and align(k,256) >= threshold and
                4*bm*bn <= (target.details.get('l0c_bytes') or 262144) and bool(target.vector_tops.get('32')))
    data = dict(route='split_k' if admitted else 'ordinary', host_grid=[rows,cols],
                host_groups=groups, max_groups_for_split=cores//2, threshold_aligned_k=threshold,
                aligned_k=align(k,256), latency_us=previous['ordinary_us'], price_source='ordinary_pipeline')
    if not admitted:
        return data
    splits = cores//groups
    depth = ceildiv(k,splits); tail = k-(splits-1)*depth
    if tail <= 0:
        # With many more cores than K positions the simple equal-segment formula
        # cannot recover a valid last segment. Do not advertise a timing here.
        return {**data,'latency_us':None,'price_source':'unresolved_partition'}
    if (previous['route'] in ('one_output_split_k','retiled_output_split_k')
            and previous['output_tiles'] == groups and previous['k_splits'] == splits):
        price = previous['latency_us']
        price_source = previous['route']
    else:
        price = previous.get('retile_candidate_us', previous.get('split_candidate_us'))
        price_source = 'partial_pipeline_and_merge' if price is not None else 'unresolved_partial_pipeline'
    return {**data,'splits_per_output':splits,'max_segment_k':depth,'tail_segment_k':tail,
            'productive_aic':groups*splits,'base_m':bm,'base_n':bn,
            'fp32_partial_bytes_each_direction':4*m*n*splits,'fp32_reduction_adds':m*n*(splits-1),
            'latency_us':price,'price_source':price_source}
