"""Physical time bounds for a single device window. All rates are decimal."""
from math import isfinite


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
