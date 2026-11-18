"""Predict critical AIV time with a separate first-use/expansion/resident cost."""
from functools import lru_cache
import json
from pathlib import Path
from .vector_v9 import schedule


@lru_cache(maxsize=1)
def parameters():
    return json.loads((Path(__file__).resolve().parents[1] / 'results/v0.10/wide/profile.json').read_text())


def occupancy(active, resident):
    return 'first' if resident == 0 else ('expand' if active > resident else 'reuse')


def predict(op, shape, attrs=None):
    """Default: marginalize unobserved dispatch state. No call-order side effects."""
    p = parameters()
    attrs = dict(attrs or {})
    resident = attrs.pop('resident_aiv', None)
    try:
        if resident is not None and (type(resident) is not int or not 0 <= resident <= 64):
            raise ValueError('resident_aiv must be an integer between 0 and 64')
        if op not in p['contracts'] or any(p['contracts'][op].get(k, object()) != v for k, v in attrs.items()):
            raise ValueError('Unsupported route or tensor attributes')
        route, active, rows, units = schedule(op, shape, attrs)
        weights = {key: 0.0 for key in ('first','expand','reuse')}
        if resident is None:
            for high,weight in p['occupancy_prior'][op]:
                weights[occupancy(active,high)] += weight
        else:
            weights[occupancy(active,resident)] = 1.0
        stages = {}
        for queue,coeff in p['phases'][route].items():
            stages[queue] = sum(weights[k]*coeff[k] for k in weights) + coeff['service']*units[queue]
            if queue == 'MTE3':
                stages[queue] += coeff.get('per_extra_row',0.0)*(rows-1)
        return dict(predicted_us=sum(stages.values()),stages_us=stages,predicted_active_aiv=active,
                    rows_per_core=rows,route=route,context_weights=weights,status='950DT critical-AIV estimate')
    except (TypeError, KeyError, ValueError) as exc:
        return dict(predicted_us=None,status='unsupported',reason=str(exc))
