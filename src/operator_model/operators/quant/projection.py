"""Calibrated critical-core timeline for the PR single-tile control route."""
from math import log2

from operator_model.chips import load_chip
from operator_model.profiles.quant_dispatch import applies
from operator_model.scheduling.reference import reference_prices, _profile


def project_quant(shape, factors, reference, target, target_plan):
    """Return the phase sum for an identified calibrated SKU and route, else None."""
    if reference != load_chip() or target != load_chip('Ascend950PR_957b'):
        return None
    p = target_plan
    reference = reference_prices()
    log_cores = log2(1 + p['active_aiv'])
    if p['iterations'] != 1 or applies(p) or log_cores < reference.control_log2_cap:
        return None

    # These endpoints were observed on this target. Applying target/reference
    # hardware ratios a second time would double-charge the memory difference.
    coeff = _profile('quant_calibrated_phases')
    slopes = reference.quant_costs
    phases = {
        'before_data_read': coeff['prefix_intercept_us'] + slopes['prefix']['slope_us_per_unit'] * log_cores,
        'data_read': coeff['read_intercept_us'] + slopes['first_read']['slope_us_per_unit'] * p['units']['read'],
        'read_to_simd': coeff['read_to_simd_us'],
        'simd_to_write': coeff['vector_intercept_us'] + slopes['vector']['slope_us_per_unit'] * p['units']['vector'],
        'write_window': coeff['write_window_us'],
        'end': coeff['end_us'],
    }
    return sum(phases.values()), {'method': 'PR single-UB saturated control route', 'phases_us': phases}
