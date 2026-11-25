"""Proposed native block-scaled A8W8 program; both operands carry E8M0 scales."""

from math import ceil


def price(shape, chip, work, calibrated_only):
    m, n, k = shape
    bm = min(128, 16 * ceil(m / 16))
    core_budget = chip.cube_cores
    mtiles = ceil(m / bm)
    bn = min(256, max(16, 16 * ceil(n / max(1, core_budget // mtiles) / 16)))
    details = chip.details
    cap_c = details.get("l0c_bytes") or 262144
    while bm * bn * 4 > cap_c and max(bm, bn) > 16:
        if bm > bn:
            bm = max(16, bm // 32 * 16)
        else:
            bn = max(16, bn // 32 * 16)
    # Each pair of operand pages in L1 includes the paired E8M0 exponents.
    cap_a, cap_b = (details.get(name) or 65536 for name in ("l0a_bytes", "l0b_bytes"))
    cap_l1 = details.get("l1_bytes") or 524288
    bk = min(
        512, 64 * ceil(k / 64), 64 * (min(cap_a // (2 * bm), cap_b // (2 * bn)) // 64)
    )
    while bk >= 64 and 2 * (bm + bn) * (bk + 2 * ceil(bk / 64)) > cap_l1:
        bk -= 64
    plan = dict(
        route="a8w8_native",
        tile_mnk=(bm, bn, bk),
        group_k=32,
        scale_dtype="e8m0",
        weight_dtype="mxfp8",
        output_dtype="bf16",
        active_aiv=0,
    )
    diag = dict(
        method="native_mxfp8_cube_pipeline",
        target_plan=plan,
        timing_qualification="extrapolated",
        vector_model="not_applicable",
        implementation="native scaled Cube with FP32 accumulation and BF16 FIXP",
    )
    if calibrated_only or bk < 64 or bm * bn * 4 > cap_c or not chip.cube_tops.get("8"):
        return (
            None,
            work,
            diag,
            "Missing calibrated A8W8 price, FP8 peak or feasible double buffer",
        )
    jobs = ceil(m / bm) * ceil(n / bn)
    active = min(jobs, core_budget)
    waves = ceil(jobs / active)
    pages = ceil(k / bk)
    freq = chip.frequency_ghz
    core_rate = chip.cube_tops["8"] * 1000 / (core_budget * freq)
    port = min(128.0, (details.get("mte2_bandwidth_gbs") or 128 * freq) / freq)
    cube = waves * (2 * bm * bn * (32 * ceil(k / 32)) / core_rate + 32 * pages)
    panel = waves * (min(m, bm) + min(n, bn)) * (k + 2 * ceil(k / 64)) / port
    hbm = (
        (work["logical_read_bytes"] + work["logical_write_bytes"])
        * freq
        / min(chip.hbm_bandwidth_gbs, active * port * freq)
    )
    # The installed MX program posts two sideband-scale descriptors before the
    # value panels. First-fragment readiness is a latency; do not multiply it
    # by pages or waves. The closed-form interval below never expands the DAG.
    first_k = min(k, bk)
    first_bytes = (min(m, bm) + min(n, bn)) * (first_k + 2 * ceil(first_k / 64))
    first_service = first_bytes / port
    first_compute = 2 * bm * bn * (32 * ceil(first_k / 32)) / core_rate + 32
    later = max(0, panel - first_service, cube - first_compute)
    # Launch spread, scalar setup, DMA response beyond payload, L1->L0,
    # MMAD response beyond service and retirement: measured in 950 core cycles.
    # Spread cannot be identified beyond the observed launch fanout.
    launch_spread = 525 * min(1, (active - 1) / 15)
    scalar_setup = 4964
    first_dma = 959 + first_service
    operand_ready = 56
    first_mmad = 73 + first_compute
    drain = 586
    local = first_dma + operand_ready + first_mmad + later
    cycles = launch_spread + scalar_setup + max(local, hbm) + drain
    us = cycles / (1000 * freq)
    plan.update(
        active_aic=active,
        waves=waves,
        k_pages=pages,
        output_tiles=jobs,
        scale_shapes=((m, ceil(k / 64), 2), (n, ceil(k / 64), 2)),
        cube_ops_per_cycle=core_rate,
        l1_double_buffer_bytes=2 * (bm + bn) * (bk + 2 * ceil(bk / 64)),
        cycles=dict(
            core_start_spread=launch_spread,
            scalar_setup=scalar_setup,
            first_dma=first_dma,
            operand_ready=operand_ready,
            first_mmad=first_mmad,
            remaining_overlap=later,
            drain=drain,
            cube=cube,
            panel_mte2=panel,
            hbm=hbm,
        ),
    )
    diag["latency_us"] = us
    return (
        us,
        work,
        diag,
        "Native E4M3FN × E4M3FN, paired E8M0 per K32; shape-dependent policy extrapolation",
    )
