"""Analytical VF program for one busy AIV; no per-register simulation at query time."""
from functools import lru_cache

from operator_model.resources import load_parameters


_CLASSES = ('arithmetic', 'reduce', 'exp', 'divide', 'convert', 'load', 'store')
_CAPACITIES = ('issue_width', 'alu_lanes', 'ub_ports', 'ub_store_ports', 'writeback_ports')
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
    def phase(per_row, chain):
        counts = {k: rows * v for k, v in per_row.items() if v}
        return _phase(counts, chain, ii, latency, capacity)

    row_max = phase(
        dict(load=x + update, arithmetic=x + (x - 1) + update, reduce=1, store=x),
        dict(load=1, arithmetic=1 + depth + update, reduce=1),
    )
    row_max['bounds_cycles']['scaled_score_store_dependency'] = (
        latency['load'] + latency['arithmetic'] + latency['store'])
    row_max['cycles'] = max(row_max['bounds_cycles'].values())
    # The first store family stages FP32 probabilities for the row sum; the
    # second publishes 16-bit probabilities to the PV input transfer.
    probability = phase(
        dict(load=x, exp=x, convert=x, store=x + p_stores),
        dict(load=1, exp=1, convert=1, store=1),
    )
    # Update state l = sum(p) + exp(old_max-new_max) * old_l. Row-local
    # statistics use separate scalar-sized vector transactions in this design.
    statistics = phase(
        dict(load=x + 2 * update, reduce=1, arithmetic=x - 1 + 2 * update,
             exp=update, store=2 + update),
        dict(load=1, reduce=1, arithmetic=depth + update, store=1),
    )
    if update:
        rescale_chain = latency['load'] + latency['exp'] + 2 * latency['arithmetic'] + latency['store']
        statistics['bounds_cycles']['rescale_dependency'] = rescale_chain
        statistics['cycles'] = max(statistics['cycles'], rescale_chain)
    # O_new = O_old * alpha + PV. On the last page divide by the final
    # normalizer and cast for output; intermediate accumulators stay FP32.
    output = phase(
        dict(load=y + update * (y + 1) + int(last), arithmetic=2 * y * update,
             divide=y * last, convert=y * last, store=y_stores if last else y),
        dict(load=1, arithmetic=2 * update, divide=int(last), convert=int(last), store=1),
    )
    p_ready = row_max['cycles'] + probability['cycles']
    return {
        'vec1': p_ready + statistics['cycles'], 'p_ready': p_ready,
        'vec2': output['cycles'], 'template': profile['profile'], 'dtype': dtype,
        'phases': dict(row_max=row_max, probability=probability, statistics=statistics, output=output),
        'units': 'cycles per busy AIV; counts in vector instructions',
        'assumptions': profile['scope'],
    }
