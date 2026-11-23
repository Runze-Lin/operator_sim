"""Logical work for one routed grouped MXFP4 projection producing materialized BF16."""
from collections.abc import Mapping, Sequence

from operator_model.workload import _counts


def describe(shape, dtype, attrs):
    """Describe (packed routed rows, output channels, reduction width) with per-expert counts."""
    if dtype != 'mxfp4':
        raise ValueError('routed_grouped_matmul requires MXFP4 E2M1 activations and weights')
    if (not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)) or len(shape) != 3
            or any(type(d) is not int for d in shape)):
        raise ValueError('Expected integer shape (valid_rows, N, K)')
    rows, n, k = shape
    if rows < 0 or n <= 0 or k <= 0 or k % 2:
        raise ValueError('Require nonnegative valid_rows, positive N and positive even packed K')
    if not isinstance(attrs, Mapping):
        raise ValueError('Provide rank-local expert_counts for routed_grouped_matmul')
    if 'implementation' in attrs and attrs['implementation'] != 'aclnnGroupedMatmulV5':
        raise ValueError('Only the transposed native aclnnGroupedMatmulV5 implementation is recognized')
    if set(attrs) - {'expert_counts', 'dispatch_capacity_rows', 'implementation'}:
        raise ValueError('Only expert_counts and optional dispatch_capacity_rows are supported')
    counts = attrs.get('expert_counts')
    if (not isinstance(counts, Sequence) or isinstance(counts, (str, bytes)) or len(counts) == 0
            or any(type(m) is not int or m < 0 for m in counts) or sum(counts) != rows):
        raise ValueError('expert_counts must be nonnegative local counts summing to valid_rows')
    capacity = attrs.get('dispatch_capacity_rows')
    if 'dispatch_capacity_rows' in attrs and (type(capacity) is not int or capacity < rows):
        raise ValueError('dispatch_capacity_rows must be an integer at least as large as valid_rows')

    experts = len(counts)
    nonempty = sum(m > 0 for m in counts)
    packed_row = k // 2
    scale_groups = (k + 31) // 32
    scale_slots = 2 * ((scale_groups + 1) // 2)
    # Weight storage is [expert,N,packed_K], presented to GMM as a [expert,K,N] view.
    # Each participating expert supplies one shared B matrix to all of its routed rows.
    activation_bytes = rows * packed_row
    activation_scale_bytes = rows * scale_slots
    weight_bytes = nonempty * n * packed_row
    weight_scale_bytes = nonempty * n * scale_slots
    read = activation_bytes + activation_scale_bytes + weight_bytes + weight_scale_bytes
    work = _counts('mxfp4', read, 2 * rows * n, cube={'mxfp4': 2 * rows * n * k}, notes=[
        'A routing histogram is a host-side attribute in this capture; no group-list tensor '
        'or device-side routing read is assumed.',
        'Reads count one set of encoded weights and scales per nonempty expert, including '
        'paired scale storage slots. Cross-tile reloads, cache hits and physical HBM traffic are unknown.',
        'No separate semantic vector primitive is present. Actual vector instructions, scale '
        'application hardware and device schedule are unobserved.',
    ])
    work.update(
        scope='Useful MXFP4 FMA work and unique encoded tensor bytes for a rank-local '
              'projection; includes paired scale slots, excludes unused dispatch capacity.',
        activation_bytes=activation_bytes,
        activation_scale_bytes=activation_scale_bytes,
        active_weight_bytes=weight_bytes,
        active_weight_scale_bytes=weight_scale_bytes,
        all_expert_weight_and_scale_storage_bytes=experts * n * (packed_row + scale_slots),
        valid_rows=rows,
        nonempty_experts=nonempty,
        unused_scale_slots=(rows + nonempty * n) * (scale_slots - scale_groups),
    )
    projection = {
        'method': 'routed_grouped_matmul_mxfp4',
        'qualification': 'work_only',
        'expert_counts': list(counts),
        'valid_rows': rows,
        'local_experts': experts,
        'dispatch_capacity_rows': capacity,
        'unused_dispatch_capacity_rows': None if capacity is None else capacity - rows,
        'weight_view_shape': [experts, k, n],
        'weight_scale_shape': [experts, scale_slots // 2, n, 2],
        'activation_scale_shape': [rows, scale_slots // 2, 2],
        'group_size': 32,
        'scale_dtype': 'e8m0',
        'accumulator_dtype': 'fp32',
        'output_dtype': 'bf16',
        'latency_us': None,
    }
    return work, projection


def price(shape, dtype, attrs, chip):
    from operator_model.operators.moe.routed_schedule import schedule
    work, projection = describe(shape, dtype, attrs)
    us, plan = schedule(*shape, projection['expert_counts'], work, chip)
    covered = (us is not None and chip.name in ('Ascend950DT_9582', 'Ascend950PR_957b')
               and 2 <= len(projection['expert_counts']) <= 4 and shape[1] <= 512 and shape[2] <= 1024)
    projection.update(implementation='aclnnGroupedMatmulV5', target_plan=plan, latency_us=us,
                      qualification='native_resource_schedule' if us is not None else 'work_only',
                      timing_qualification=('estimated' if covered else 'extrapolated') if us is not None else 'unavailable')
    return us,work,projection
