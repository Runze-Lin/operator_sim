"""BF16 full-row AddRmsNorm: contiguous row ownership and partial rstd write drain."""

from dataclasses import replace
from math import ceil


def full_rows(estimate, chip, rows):
    plan = dict(estimate.work["schedule"])
    rows_per_core = ceil(rows / chip.vector_cores)
    active = ceil(rows / rows_per_core)
    # Owners of adjacent 4-byte rstd slices may contend for a shared write unit.
    service_bytes = chip.details.get("add_rms_rstd_service_bytes") or 128
    contenders = min(active, ceil(service_bytes / (4 * rows_per_core)))
    interval = chip.details.get("add_rms_rstd_owner_cycles") or 110
    tail = (contenders - 1) * interval
    native = chip.sources.get("timing_template", chip.name)
    issue = chip.details.get("add_rms_fullrow_setup_cycles") or (
        4764 if native == "Ascend950PR_957b" else 5162
    )
    clock = chip.frequency_ghz * 1000
    exposed = plan["core_pipeline_cycles"] + tail
    cycles = issue + max(exposed, plan["shared_payload_floor_cycles"])
    plan.update(
        setup_cycles=issue,
        active_aiv=active,
        busy_rows=rows_per_core,
        rstd_owners=contenders,
        rstd_owner_cycles=interval,
        rstd_service_bytes=service_bytes,
        rstd_drain_cycles=tail,
        control_anchor="Single-AIV BF16 complete windows; DT setup inherited for unmeasured SKU",
        evidence="Whole-row tiling key 1000; shared partial rstd write service inferred from MTE3 descriptors",
    )
    return replace(
        estimate, latency_us=cycles / clock, work={**estimate.work, "schedule": plan}
    )
