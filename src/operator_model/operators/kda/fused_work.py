"""Work in the explicit-length Finalize stage of FLA 26.9.1 on Ascend 950."""
from collections import Counter

from operator_model.chips import REFERENCE_NAME, load_chip


def kimi_v1_work(lengths, heads, chip, *, split_phases):
    """Describe the selected V1 route without assigning a time to faulty launches.

    The calling selector supplies nonempty lengths and a positive local head
    count. At most 4,095 block/head tasks can be on this V1 route.
    """
    work = {
        'implementation': 'fla_a5_26.9.1_chunk_kda_fwd',
        'target_extrapolated': chip not in (load_chip(REFERENCE_NAME), load_chip('Ascend950PR_957b')),
        'timing_qualified': False,
        'qualification_reason': (
            'Official 950 CA observed repeated M-to-MTE1 SET in V1 despite correct whole outputs. '
            'No qualified full latency is available for this implementation.'
        ),
        'finalize': None,
    }
    if not split_phases:
        work['work_scope'] = 'One FULL main kernel. The separate Finalize-phase schedule does not apply.'
        return work

    # Chunk indices follow packed sequence order. Within each chunk, task IDs
    # enumerate heads. The official BF16 varlen scheduler strides by AIC count.
    rows_per_task = Counter()
    per_core = [0] * chip.cube_cores
    task = 0
    state_ops = local_ops = 0
    for length in lengths:
        for start in range(0, length, 64):
            rows = min(64, length - start)
            state_product = 2 * rows * 128 * 128  # Qg[t,K] @ H[K,V]
            local_product = 2 * rows * rows * 128  # Aqk[t,t] @ vNew[t,V]
            state_ops += heads * state_product
            local_ops += heads * local_product
            rows_per_task[rows] += heads
            for owner_task in range(task, task + heads):
                per_core[owner_task % chip.cube_cores] += state_product + local_product
            task += heads

    critical_ops = max(per_core)
    peak = chip.cube_tops.get('16')
    floor = critical_ops * chip.cube_cores / (peak * 1e6) if peak else None
    work['work_scope'] = 'Logical Cube products of the fourth physical V1 phase on the canonical packed 950 route.'
    work['finalize'] = {
        'logical_cube_ops': state_ops + local_ops,
        'logical_state_product_ops': state_ops,
        'logical_local_product_ops': local_ops,
        'valid_rows_per_task_histogram': dict(sorted(rows_per_task.items())),
        'work_items': task,
        'aic_with_output_tasks': min(task, chip.cube_cores),
        'logical_ops_per_aic': tuple(per_core),
        'critical_owner_logical_ops': critical_ops,
        'chip_bf16_cube_tops': peak,
        'cube_useful_work_floor_us': floor,
        'floor_scope': 'Useful Cube work at symmetric per-AIC BF16 peak. Excludes tiling overhead, transfers, waits, other phases and helpers.',
    }
    return work
