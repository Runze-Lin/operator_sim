"""BF16 Matmul and Addmm with a contiguous output-column bias on Ascend 950DT."""
import json
from functools import lru_cache
from pathlib import Path
from .model import line
from .model_v2 import plan, body_time
from .model_v6 import Case
from .fast_query import Predictor

ROOT = Path(__file__).resolve().parents[1]

@lru_cache(maxsize=1)
def parameters():
    base = Predictor(str(ROOT / 'results/v0.6/model.json'))
    stages = json.loads((ROOT / 'results/v0.2/model.json').read_text())['operators']['matmul']['stage_costs']
    return base, stages


def predict(op, shape, attrs=None):
    attrs = dict(attrs or {})
    case = Case('matmul', tuple(shape))
    base, costs = parameters()
    if op == 'matmul' and not attrs:
        return base.predict(case, use_cache=False)
    if op != 'addmm':
        raise ValueError('Expected BF16 matmul or addmm')
    m,n,k = case.shape
    defaults = {'self_shape':[n], 'alpha':1, 'beta':1, 'dtype':'bf16', 'cube_math_type':0}
    if any(key not in defaults or value != defaults[key] for key,value in attrs.items()):
        raise ValueError('Addmm requires contiguous BF16 ND, self=[N], alpha=beta=1, cubeMathType=0')
    p = plan(case)
    # Include one BF16 bias load per output-column tile. Source/reference traces
    # support the single-AIC-family path at three tiny shapes; dispatch is tested below.
    p['units']['read'] += 2*min(n,p['tile'][1])/1024
    stages = {resource:line(costs[resource], amount) for resource,amount in p['units'].items()}
    body = body_time('matmul',p,stages)
    # In the measured single-tile path, scalar control is partially covered by
    # read/compute/write. The non-overlapping control is measured from the union
    # of ALL service queues (offline script), before testing the new shapes.
    uncovered = control_us()
    return {'op':op,'shape':list(shape),'predicted_us':uncovered+body,'plan':p,
            'stage_services_us':stages,'control_us':uncovered,'status':'A5 BF16 bias-fused Cube candidate'}


@lru_cache(maxsize=1)
def control_us():
    profile=json.loads((ROOT/'results/v0.8/cube/control.json').read_text())
    return profile['median_uncovered_us']
