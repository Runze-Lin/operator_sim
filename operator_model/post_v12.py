"""Critical-core Post model. An unknown dispatch history is not assumed hot."""
from functools import lru_cache
import json
from pathlib import Path

from .post_v11 import cohort
from .vector_v9 import schedule
from .wide_v10 import occupancy


@lru_cache(maxsize=1)
def parameters():
    return json.loads((Path(__file__).resolve().parents[1] / 'results/v0.12/post/profile.json').read_text())


def predict(op, shape, attrs=None):
    return estimate(parameters(), op, shape, attrs)


def estimate(profile, op, shape, attrs=None):
    attrs = dict(attrs or {})
    resident = attrs.pop('resident_aiv', None)
    try:
        if op != 'mhc_post' or any(profile['contract'].get(k, object()) != v for k, v in attrs.items()):
            raise ValueError('Unsupported route or tensor attributes')
        if resident is not None and (type(resident) is not int or not 0 <= resident <= 64):
            raise ValueError('resident_aiv must be an integer between 0 and 64')
        route, active, rows, units = schedule(op, shape, attrs)
    except (TypeError, ValueError, KeyError) as exc:
        return dict(predicted_us=None, status='unsupported', reason=str(exc))
    family = cohort(route, shape[0], active)
    counts = profile['context_counts'][family]
    stages = {s: {q: coeff[s] + coeff['service'] * units[q] + coeff['per_extra_row'] * (rows - 1)
                  for q, coeff in profile['phases'][route].items()} for s in counts}
    totals = {s: sum(q.values()) for s, q in stages.items()}
    low, high = min(totals, key=totals.get), max(totals, key=totals.get)
    if resident is None and family == 'channel_split':
        # The harmonic center minimizes the larger relative error at the two endpoints.
        lo, hi = totals[low], totals[high]
        w = lo / (lo + hi)
        chosen = 'relative_minimax'
        result = {q: (1 - w) * stages[low][q] + w * stages[high][q] for q in stages[low]}
    else:
        if resident is not None:
            chosen = occupancy(active, resident)
        else:
            cumulative = 0
            for chosen in sorted(totals, key=totals.get):
                cumulative += counts[chosen]
                if cumulative * 2 >= sum(counts.values()):
                    break
        result = stages[chosen]
    return dict(predicted_us=sum(result.values()), stages_us=result, route=route,
                predicted_active_aiv=active, rows_per_core=rows, context=chosen,
                context_predictions_us=totals, status='950DT critical-AIV estimate')
