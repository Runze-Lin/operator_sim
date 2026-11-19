"""Reconstruct the frozen research point from the public resource history band."""
from operator_model.rms_dispatch import price_rms
from operator_model.rms_history import estimate_rms_history_band
from development.validation.rms_unknown23.candidate import DECISION_TOLERANCE


def forecast(shape, chip):
    band = estimate_rms_history_band(shape, chip=chip)
    old = price_rms(shape, chip, {})[0]
    lower, upper = band.lower_us, band.upper_us
    lo = (1 - DECISION_TOLERANCE) * upper
    hi = (1 + DECISION_TOLERANCE) * lower
    center = max(lo, min(hi, old)) if lo <= hi else 2 * lower * upper / (lower + upper)
    return dict(active_aiv=band.active_aiv, previous_us=old, conditional_us_by_H=band.by_resident_aiv_us,
                lower_us=lower, upper_us=upper, candidate_us=center)
