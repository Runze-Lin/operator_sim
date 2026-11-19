"""Shared external traffic of a contiguous, cold, out-of-place streaming pass."""


def stream_memory(chip, active, read_bytes, write_bytes):
    """A proportional mixture of clean and dirty lines survives in isolated L2.

    Full-chip issue competes with the external interface. 30 B/cycle/AIV is
    the measured *mixed-stream* request limit, rounded from the ratio of DT
    controller traffic to the payload explained by a full-tile issue interval
    (see results/v0.21/vector/method.md). Do not add this interval to DMA latency.
    """
    total = read_bytes + write_bytes
    resident = min(total, chip.details.get('l2_bytes') or 0)
    retained_write = resident * (write_bytes / total)
    spilled = write_bytes - retained_write
    # Explicit local port speeds scale the inherited AIV issue ceiling if
    # present. Slow ports also constrain the operator's per-core schedule.
    from .rms_resource import READ_PER_KIB, WRITE_PER_KIB
    f = chip.frequency_ghz
    rd = chip.details.get('mte2_bandwidth_gbs') or 1024*f / READ_PER_KIB
    wr = chip.details.get('mte3_bandwidth_gbs') or 1024*f / WRITE_PER_KIB
    issue = active * 30 * f * min(rd/(1024*f/READ_PER_KIB), wr/(1024*f/WRITE_PER_KIB))
    shared = min(chip.hbm_bandwidth_gbs, issue)
    external = read_bytes + spilled
    l2 = chip.details.get('l2_bandwidth_gbs')
    return dict(l2_input_reserve_bytes=resident-retained_write,
                assumed_retained_write_bytes=retained_write,
                modeled_external_write_bytes=spilled, modeled_external_bytes=external,
                external_issue_gbs=issue, effective_external_gbs=shared,
                external_floor_cycles=external*f/shared,
                l2_floor_cycles=total*f/l2 if l2 else None)
