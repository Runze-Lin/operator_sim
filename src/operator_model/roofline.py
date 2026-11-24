"""Physical time bounds for a single device window. All rates are decimal."""
from math import isfinite


def cold_io_window(schedule_us, work, chip):
    """Cold reads and materialized writes sharing the target HBM interface.

    The byte counts are a deployment assumption, not CA controller counters.
    The scheduler already prices DMA, so only an exposed tail extends its time.
    """
    read, write = work['logical_read_bytes'], work['logical_write_bytes']
    if not isfinite(schedule_us) or schedule_us < 0:
        raise ValueError('cold_io requires a finite nonnegative device window')
    if any(value is None or not isfinite(value) or value < 0 for value in (read,write)):
        raise ValueError('cold_io requires explicit nonnegative read and write byte counts')
    service = (read + write) / (1000 * chip.hbm_bandwidth_gbs)
    end = max(schedule_us, service)
    return dict(policy='cold_io', assumed_hbm_read_bytes=read, assumed_hbm_write_bytes=write,
                hbm_service_us=service, pipeline_us=schedule_us, exposed_io_us=end-schedule_us,
                duration_us=end, hbm_fraction=service/end if end else None)


def physical_roofline(duration_us, bandwidth_gbs, *, cube_us=None, vector_us=None,
                      physical_hbm_bytes=None):
    """Compare time with necessary work priced at the corresponding peak.

    Pass 0 for a known inactive engine and None for missing evidence. Memory
    bytes must cross the HBM interface in this very window. Cube and Vector
    times must reflect required instructions at matching engine peaks.
    """
    for name, value in (('duration_us', duration_us), ('bandwidth_gbs', bandwidth_gbs)):
        if not isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    for value in (cube_us, vector_us, physical_hbm_bytes):
        if value is not None and (not isfinite(value) or value < 0):
            raise ValueError('Physical work must be finite and nonnegative')
    hbm_us = None if physical_hbm_bytes is None else physical_hbm_bytes / (1000 * bandwidth_gbs)
    terms = {'Cube': cube_us, 'Vector': vector_us, 'HBM': hbm_us}
    missing = [engine for engine, time in terms.items() if time is None]
    known = max((time for time in terms.values() if time is not None), default=None)
    floor = None if missing else known
    return {
        'ideal_us': floor, 'utilization': None if floor is None else floor / duration_us,
        'known_lower_bound_us': known, 'missing': missing, 'terms_us': terms,
        'below_known_bound': None if known is None else duration_us < known * (1 - 1e-12),
    }
