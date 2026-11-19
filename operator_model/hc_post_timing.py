"""Experimental device timing for the pinned vLLM Ascend HcPost BF16 kernel."""
from collections import deque
from functools import lru_cache
import json
from pathlib import Path

from .chips import load_chip
from .hc_post_layout import layout


@lru_cache(maxsize=1)
def _parameters():
    path = Path(__file__).resolve().parents[1] / 'configs/vllm-hc-post-arch35.json'
    return json.loads(path.read_text(encoding='utf-8'))


def _dma_prices(chip, prices):
    lines = prices['dma']
    def scale(port, reference):
        target_gbs = chip.details.get(port)
        if target_gbs is None:
            return 1.
        ref_bytes_per_tick = 1. / lines[reference]['ticks_per_unit']
        return ref_bytes_per_tick / (target_gbs / chip.frequency_ghz)
    read_scale = scale('mte2_bandwidth_gbs', 'first_res')
    write_scale = scale('mte3_bandwidth_gbs', 'write')
    return {
        key: (line['setup'], line['ticks_per_unit'] * (write_scale if key == 'write' else read_scale))
        for key, line in lines.items()
    }


def _repeated_rows(history):
    """A translation of all live timestamps preserves every max-plus dependency."""
    for period in range(1, min(4, (len(history) - 1) // 2) + 1):
        current, prior, older = history[-1], history[-1-period], history[-1-2*period]
        differences = [now - before for now, before in zip(current, prior)]
        differences += [now - before for now, before in zip(prior, older)]
        if max(differences) - min(differences) < 1e-7:
            return period, differences[-1]
    return 0, 0.


def _schedule(rows, widths, rank, prices, dma, vec_scale):
    def transfer(kind, size):
        fixed, per_byte = dma[kind]
        return fixed + per_byte * size

    first_coeff = prices['first_coeff'] + rank * prices['issue_fanout_per_aiv']
    first_admission = prices['first_vf_admission'] + rank * prices['admission_fanout_per_aiv']
    body = prices['arith_body']
    coeff_service = transfer('x', 64)
    cast_ticks = prices['cast_body'] * vec_scale
    history = deque(maxlen=9)
    # Only these six clocks survive a row boundary with both slots occupied.
    last_x = coeff = cast_end = write = older_write = write_end = 0.
    row = issued = skipped = 0
    first_vf = first_write = None
    while row < rows:
        if row == 0:
            coeff = first_coeff
        else:
            coeff = max(coeff + prices['next_row_coeff_issue'],
                        last_x + prices['next_row_coeff_issue'] - prices['payload_after_coeff'])
        coeff_end = coeff + prices['coeff_issue_gap'] + coeff_service
        for i, width in enumerate(widths):
            if i == 0:
                earliest = coeff + prices['payload_after_coeff']
                if len(widths) > 1:
                    earliest += prices['split_descriptor_extra']
            else:
                earliest = last_x + prices['second_fragment_issue']
                if width < 4096:
                    earliest += prices['tail_descriptor_extra']
            reuse = older_write + prices['triple_slot_release']
            if issued >= 2 and reuse >= earliest:
                last_x = reuse
                gap = prices['triple_slot_release']
            else:
                last_x = earliest
                gap = prices['initial_read_issue_gap'] + (prices['full_read_issue_extra'] if width == 4096 else 0)
            x_end = last_x + transfer('x' if issued else 'first_x', 2 * width)
            residual_end = last_x + gap + transfer('res' if issued else 'first_res', 8 * width)
            vf = max(x_end, residual_end, coeff_end) + prices['input_ready']
            if issued:
                vf = max(vf, cast_end - min(cast_ticks, prices['next_head_overlap']))
            groups = (width + 63) // 64
            arithmetic = (body['setup'] + body['ticks_per_unit'] * groups) * vec_scale
            admission = prices['warm_dispatch'] if issued else first_admission
            # A target with a much faster vector datapath cannot overlap more
            # descriptor time than the producing stage actually occupies.
            cast_start = max(vf, vf + admission + arithmetic - prices['cast_head_overlap'])
            cast_end = cast_start + cast_ticks
            older_write, write = write, max(write_end + prices['triple_slot_release'], cast_end + prices['cast_to_write'])
            write_end = write + transfer('write', 8 * width)
            if not issued:
                first_vf, first_write = vf, write
            issued += 1
        row += 1
        if issued < 2 or row == rows:
            continue
        state = (last_x, coeff, cast_end, write, older_write, write_end)
        history.append(state)
        period, ticks = _repeated_rows(history)
        periods_left = (rows - row) // period if period else 0
        if periods_left:
            elapsed = periods_left * ticks
            last_x, coeff, cast_end, write, older_write, write_end = (v + elapsed for v in state)
            advanced = periods_left * period
            row += advanced
            issued += advanced * len(widths)
            skipped += advanced
            history.clear()
    return dict(core_ticks=write_end + prices['return_after_write'], critical_rows=rows,
                critical_cohort_rank=rank, skipped_steady_rows=skipped, first_coeff_issue_tick=first_coeff,
                first_vf_issue_tick=first_vf, first_write_issue_tick=first_write,
                final_write_issue_tick=write, final_write_retire_tick=write_end)


def resource_hc_post(shape, chip, attrs):
    """Return (latency_us or None, logical work, projection) for an explicit HcPost dispatch."""
    attrs = dict(attrs or {})
    if attrs.pop('dtype', 'bf16') != 'bf16':
        raise ValueError('The measured vLLM HcPost variant requires BF16 activations and FP32 coefficients')
    work, plan = layout(shape, chip, attrs)
    params = _parameters()
    tokens, width = shape[0] * shape[1], shape[-1]
    warnings = []
    diag = dict(method='experimental_vllm_hc_post_resource_schedule', qualification='experimental',
                timing_evidence='frozen 950DT/950PR stage model; target-specific timing requires independent evidence',
                target_plan=plan, active_aic=0, active_aiv=plan['active_aiv'],
                prediction_freeze_sha256=params['freeze_sha256'], warnings=warnings)
    ub = chip.details.get('ub_bytes')
    short_tail = 0 < width % 4096 < 64
    if width % 16 or short_tail or ub is None or ub < params['min_ub_bytes'] or chip.vector_tops['32'] is None:
        if short_tail:
            warnings.append('A final feature fragment with fewer than 64 elements is outside the valid timing domain: both 950 endpoints failed the BF16 golden for a 16-element tail.')
        if width % 16:
            warnings.append('Dense HcPost timing requires D divisible by 16 for the measured GM/UB layout.')
        if ub is None or ub < params['min_ub_bytes']:
            warnings.append(f'The fixed tile requires at least {params["min_ub_bytes"]} available UB bytes per AIV.')
        if chip.vector_tops['32'] is None:
            warnings.append('FP32 Vector throughput is required to price MulAdd and CAST.')
        diag.update(qualification='unavailable', latency_us=None, stages_us={})
        return None, work, diag

    prices = params['prices']
    active = plan['active_aiv']
    vec_scale = prices['vector_reference_ops32_cycle'] / (
        chip.vector_tops['32'] * 1000 / chip.vector_cores / chip.frequency_ghz)
    dma = _dma_prices(chip, prices)
    full, tail = divmod(width, 4096)
    widths = [4096] * full + ([tail] if tail else [])
    rows, extra = divmod(tokens, active)
    cohorts = [(rows, active - 1)]
    if extra:
        cohorts.append((rows + 1, extra - 1))
    critical = max((_schedule(n, widths, rank, prices, dma, vec_scale) for n, rank in cohorts),
                   key=lambda result: result['core_ticks'])
    read, write = work['logical_read_bytes'], work['logical_write_bytes']
    freq = chip.frequency_ghz
    capacity = chip.details.get('l2_bytes')
    if capacity is None:
        capacity = 0
        warnings.append('Unknown L2 capacity: the cold external-memory bound assumes no retained output.')
    dram_ticks = (read + max(0, write - capacity)) * freq / chip.hbm_bandwidth_gbs
    l2_bw = chip.details.get('l2_bandwidth_gbs')
    l2_ticks = (read + write) * freq / l2_bw if l2_bw else 0.
    core_us = critical['core_ticks'] / (freq * 1000)
    latency = max(critical['core_ticks'], dram_ticks, l2_ticks) / (freq * 1000)
    if plan['max_rows_per_core'] > 2 or width > 16448:
        warnings.append('Beyond the development geometry (up to two rows per AIV or D=16448); long steady-state timing is an extrapolation.')
    if read + write > chip.hbm_capacity_gb * 1e9:
        warnings.append('The unchunked input and output tensors exceed configured external-memory capacity.')
    target_has_ca = chip.name in ('Ascend950DT_9582', 'Ascend950PR_957b') and chip == load_chip(chip.name)
    diag['target_has_archived_ca'] = target_has_ca
    if not target_has_ca:
        warnings.append('No target CA evidence. Assumes the same compiled tile geometry, issue queues, and local DMA setup costs.')
    if 'details.ub_bytes' in chip.inherited_fields:
        warnings.append('Per-AIV UB capacity is inherited from 950DT.')
    explicit_mte = [p for p in ('mte2_bandwidth_gbs', 'mte3_bandwidth_gbs') if chip.details.get(p)]
    if explicit_mte:
        warnings.append('Explicit per-AIV MTE GB/s scales byte prices against effective 950 slopes; fixed latency and issue intervals are inherited.')
    k = len(widths)
    diag.update(latency_us=latency, stages_us={'critical_core': core_us, 'exposed_shared_limit': latency - core_us},
                pipeline={**critical, 'vf_commands': 2 * tokens * k, 'cast_output_slots': 16384 * tokens * k,
                          'descriptor_counts': dict(meta=active, post=tokens, comb=tokens, x=tokens*k,
                                                    res=tokens*k, write=tokens*k),
                          'offchip_cold_bound_ticks': dram_ticks, 'l2_peak_bound_ticks': l2_ticks,
                          'clock_ghz': freq, 'mte_overrides': explicit_mte, 'min_ub_bytes': params['min_ub_bytes']})
    return latency, work, diag
