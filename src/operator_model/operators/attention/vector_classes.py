"""Analytical VF program for one busy AIV; no per-register simulation at query time."""
from functools import lru_cache

from operator_model.resources import load_parameters


_CLASSES = ('arithmetic', 'reduce', 'exp', 'divide', 'convert', 'load', 'store')
_CAPACITIES = ('issue_width', 'alu_lanes', 'ub_ports', 'ub_store_ports', 'writeback_ports', 'store_fence_cycles', 'sum_reduce_extra_cycles')
VF_CHIP_FIELDS = {f'fa_vf_{name}_{unit}' for name in _CLASSES
                  for unit in ('ii_cycles', 'latency_cycles')}
VF_CHIP_FIELDS.update(f'fa_vf_fp16_convert_{unit}' for unit in ('ii_cycles', 'latency_cycles'))
VF_CHIP_FIELDS.update(f'fa_vf_{name}' for name in _CAPACITIES)


@lru_cache(maxsize=1)
def _reference():
    return load_parameters('fa_vector_classes')


def _phase(counts, chain, ii, latency, capacity):
    """Service lower bounds and the longest producer chain for this template fence."""
    # Reductions and stores are single-issue in this template. Other ops take
    # one issue slot; all arithmetic classes share an assumed pair of lanes.
    exclusive = counts.get('reduce', 0) + counts.get('store', 0)
    regular = sum(counts.values()) - exclusive
    reads = counts.get('load', 0)
    writes = counts.get('store', 0)
    alu_service = sum(count * ii[k] for k, count in counts.items()
                      if k not in ('load', 'store'))
    bounds = {
        'dependency': sum(count * latency[k] for k, count in chain.items()),
        'issue': exclusive + regular / capacity['issue_width'],
        'alu': alu_service / capacity['alu_lanes'],
        'ub': (reads * ii['load'] + writes * ii['store']) / capacity['ub_ports'],
        'ub_store': writes * ii['store'] / capacity['ub_store_ports'],
        'writeback': (regular + exclusive - writes) / capacity['writeback_ports'],
    }
    return dict(counts=counts, chain=chain, bounds_cycles=bounds, cycles=max(bounds.values()))


def vector_class_tile(rows, n, d, *, first, last, dtype, chip):
    """Return occupied vector cycles, and when this page's probability tile can leave UB.

    n and d are effective page and head widths. Counts include masked tail registers.
    We assume equally available FP32 execution classes on another chip until explicit
    per-class initiation intervals or widths are supplied. GHz converts the resulting
    cycles outside this function; AIV count has already determined the busy-row count.
    """
    if min(rows, n, d) < 1 or dtype not in ('bf16', 'fp16'):
        raise ValueError('Instruction-class FA requires positive dimensions and bf16 or fp16')
    profile = _reference()
    def value(key, default):
        setting = chip.details.get('fa_vf_' + key)
        return setting if setting is not None else default

    ii = {k: value(k + '_ii_cycles', v['ii_cycles']) for k, v in profile['classes'].items()}
    latency = {k: value(k + '_latency_cycles', v['latency_cycles'])
               for k, v in profile['classes'].items()}
    if dtype == 'fp16':
        ii['convert'] = value('fp16_convert_ii_cycles', ii['convert'])
        latency['convert'] = value('fp16_convert_latency_cycles', latency['convert'])
    capacity = {k: value(k, v) for k, v in profile['capacities'].items()}

    # Each row combines X score registers into one max vector (tree), then
    # horizontally reduces that vector. Scaled scores are staged in UB.
    x = (n + 63) // 64
    y = (d + 63) // 64
    # Half-width output stores pack two converted FP32 vectors per transaction.
    p_stores = (n + 127) // 128
    y_stores = (d + 127) // 128
    update = int(not first)
    depth = (x - 1).bit_length()
    # Distinct PCs scale in two ways: score loops walk every row, whereas
    # the online state loop operates on one packed vector of row statistics.
    packs = (rows + 63) // 64

    def phase(per_row, chain, packed=None):
        counts = {k: rows * v for k, v in per_row.items() if v}
        for k, v in (packed or {}).items():
            if v:
                counts[k] = counts.get(k, 0) + packs * v
        return _phase(counts, chain, ii, latency, capacity)

    row_max = phase(
        dict(load=x, arithmetic=x + x - 1 + x // 4 * 2, reduce=1, store=x + 1),
        dict(load=1, arithmetic=1 + depth, reduce=1, store=1),
        dict(load=2 * update, arithmetic=update, store=update),
    )
    # SMEM_BAR protects the row maxima before the probability loop reloads
    # them. At 256 keys we observe one barrier in the first page and two in
    # each update. Short-page kernels use the same barrier pair on update.
    row_max['fence_cycles'] = (1 + update) * capacity['store_fence_cycles']
    row_max['cycles'] += row_max['fence_cycles']
    # Probability and row sum consume the same FP32 score registers. Sum
    # intermediates stay in registers until the horizontal reduction.
    probability = phase(
        dict(load=x, exp=x, convert=x, arithmetic=x - 1 + x // 2,
             reduce=1, store=p_stores + 1),
        dict(load=1, exp=1, convert=1, store=1),
    )
    probability_ready = probability['cycles']
    reduction_chain = capacity['sum_reduce_extra_cycles'] + latency['load'] + latency['exp'] + depth * latency['arithmetic'] + latency['reduce'] + latency['store']
    probability['bounds_cycles']['row_sum_chain'] = reduction_chain
    probability['cycles'] = max(probability['bounds_cycles'].values())
    statistics = phase({}, {}, dict(load=4 * update, exp=update,
                                    arithmetic=2 * update, store=3 * update))
    if update:
        # Packed alpha is stored and reloaded by the state update in the
        # same VF, but has only one exponential per register of rows.
        rescale_chain = (latency['load'] + latency['exp'] + latency['store']
                         + latency['load'] + 2 * latency['arithmetic'] + latency['store'])
        statistics['bounds_cycles']['rescale_dependency'] = rescale_chain
        statistics['cycles'] = max(statistics['bounds_cycles'].values())
    output = phase(
        dict(load=y + update * (y + 1) + int(last), arithmetic=2 * y * update,
             divide=y * last, convert=y * last, store=y_stores if last else y),
        dict(load=1, arithmetic=2 * update, divide=int(last), convert=int(last), store=1),
    )
    p_ready = row_max['cycles'] + probability_ready
    softmax_done = row_max['cycles'] + probability['cycles'] + statistics['cycles']
    return {
        'vec1': softmax_done, 'p_ready': p_ready,
        'vec2': output['cycles'], 'template': profile['profile'], 'dtype': dtype,
        'phases': dict(row_max=row_max, probability=probability, statistics=statistics, output=output),
        'units': 'cycles per busy AIV; counts in vector instructions',
        'assumptions': profile['scope'],
    }
