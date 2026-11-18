"""Critical-AIV estimate for the BF16, four-stream mHC Post implementation."""
from functools import lru_cache
import json
from pathlib import Path

from .vector_v9 import schedule
from .wide_v10 import occupancy


@lru_cache(maxsize=1)
def parameters():
    path = Path(__file__).resolve().parents[1] / 'results/v0.11/post/profile.json'
    return json.loads(path.read_text())


def cohort(route, tokens, active):
    if route == 'post_token':
        return 'token'
    return 'channel_split' if active > tokens else 'channel_unsplit'


def predict(op, shape, attrs=None):
    profile = parameters()
    attrs = dict(attrs or {})
    resident = attrs.pop('resident_aiv', None)
    try:
        if op != 'mhc_post' or any(profile['contract'].get(key, object()) != value
                                   for key, value in attrs.items()):
            raise ValueError('Unsupported route or tensor attributes')
        if resident is not None and (type(resident) is not int or not 0 <= resident <= 64):
            raise ValueError('resident_aiv must be an integer between 0 and 64')
        route, active, rows, units = schedule(op, shape, attrs)
    except (TypeError, ValueError, KeyError) as exc:
        return dict(predicted_us=None, status='unsupported', reason=str(exc))

    prior = profile['context_counts'][cohort(route, shape[0], active)]
    totals = {}
    stages = {}
    for state in prior:
        times = {queue: coeff[state] + coeff['service'] * units[queue]
                 + coeff['per_extra_row'] * (rows - 1)
                 for queue, coeff in profile['phases'][route].items()}
        stages[state] = times
        totals[state] = sum(times.values())
    if resident is not None:
        chosen = occupancy(active, resident)
    else:
        # Conditional median minimizes absolute time error with unknown stream history.
        cumulative = 0
        for chosen in sorted(totals, key=totals.get):
            cumulative += prior[chosen]
            if 2 * cumulative >= sum(prior.values()):
                break
    return dict(predicted_us=totals[chosen], stages_us=stages[chosen], route=route,
                predicted_active_aiv=active, rows_per_core=rows, context=chosen,
                context_predictions_us=totals, status='950DT critical-AIV estimate')
