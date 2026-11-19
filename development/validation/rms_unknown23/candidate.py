"""Forecast a no-history RMSNorm call with the v0.22 physical lane/sector states as constraints."""
from operator_model.rms_dispatch import price_rms
from operator_model.rms_lane_schedule import schedule


DECISION_TOLERANCE = 0.20


def forecast(shape,chip):
    """Keep the existing point where it covers every state; otherwise enforce a relative-error budget."""
    old,_,active,diag=price_rms(shape,chip,{})
    conditional=diag['conditional_us']
    times=[conditional['first']]
    if diag['target_plan']['iterations']==1:
        for high_water in range(1,active+1):
            phases,_=schedule(shape,chip,high_water,diag)
            times.append(sum(phases.values()))
    else:
        times.extend([conditional['expand']]*(active-1))
        times.append(conditional['reuse'])
    lower,upper=min(times),max(times)
    lo=(1-DECISION_TOLERANCE)*upper
    hi=(1+DECISION_TOLERANCE)*lower
    minimax=2*lower*upper/(lower+upper)
    prediction=max(lo,min(hi,old)) if lo<=hi else minimax
    return dict(candidate_us=prediction,previous_us=old,active_aiv=active,
                conditional_us_by_H=times,extreme_H=[times.index(lower),times.index(upper)],
                lower_us=lower,upper_us=upper,model_relative_radius=(upper-lower)/(upper+lower),
                all_state_tolerance_intersection=[lo,hi] if lo<=hi else None,
                policy='feasible-state projection' if lo<=hi else 'relative minimax')
