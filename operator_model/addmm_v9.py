"""Critical-core queue overlap for contiguous BF16 Addmm with column bias."""
from functools import lru_cache
import json
from math import log2
from pathlib import Path

from .cube_v8 import predict as baseline
from .model import line


@lru_cache(maxsize=1)
def profile():
    return json.loads((Path(__file__).resolve().parents[1]/'results/v0.9/addmm/profile.json').read_text())


def predict(op, shape, attrs=None):
    if op != 'addmm':
        raise ValueError('Expected Addmm')
    plan = baseline(op,shape,attrs)['plan']
    hw = profile()
    services = {q:line(hw['queue_costs'][q],plan['units'][u]) for q,u in
                [('CUBE','cube'),('MTE2','read'),('MTE1','read'),('FIXP','fix')]}
    longest = max(services.values())
    exposed = hw['exposed_fraction']*(sum(services.values())-longest)
    arrival = hw['arrival_us_per_log_core']*log2(plan['active_aic'])
    total = hw['control_us']+arrival+plan['waves']*longest+exposed
    return dict(predicted_us=total,plan=plan,stage_services_us=services,arrival_us=arrival,
                control_us=hw['control_us'],exposed_us=exposed,status='A5_bias_fused_cube')
