"""Critical-core queue overlap for contiguous BF16 Addmm with column bias."""
from operator_model.resources import load_parameters
from functools import lru_cache
from math import log2

from operator_model.profiles.tile_model import plan as tile_plan
from operator_model.scheduling.cases import Case
from operator_model.profiles.stage_model import line

def bias_plan(shape, attrs):
    case = Case('matmul', tuple(shape))
    _, n, _ = case.shape
    defaults = {'self_shape': [n], 'alpha': 1, 'beta': 1, 'dtype': 'bf16', 'cube_math_type': 0}
    if any(key not in defaults or value != defaults[key] for key, value in (attrs or {}).items()):
        raise ValueError('Addmm requires contiguous BF16 ND, self=[N], alpha=beta=1, cubeMathType=0')
    geometry = tile_plan(case)
    geometry['units']['read'] += 2 * min(n, geometry['tile'][1]) / 1024
    return geometry


@lru_cache(maxsize=1)
def profile():
    return load_parameters('addmm_queues')


def predict(op, shape, attrs=None):
    if op != 'addmm':
        raise ValueError('Expected Addmm')
    plan = bias_plan(shape, attrs)
    hw = profile()
    services = {q:line(hw['queue_costs'][q],plan['units'][u]) for q,u in
                [('CUBE','cube'),('MTE2','read'),('MTE1','read'),('FIXP','fix')]}
    longest = max(services.values())
    exposed = hw['exposed_fraction']*(sum(services.values())-longest)
    arrival = hw['arrival_us_per_log_core']*log2(plan['active_aic'])
    total = hw['control_us']+arrival+plan['waves']*longest+exposed
    return dict(predicted_us=total,plan=plan,stage_services_us=services,arrival_us=arrival,
                control_us=hw['control_us'],exposed_us=exposed,status='A5_bias_fused_cube')
