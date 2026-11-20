"""KDA chunk resource graph for the dense contracts attempted on 950DT."""
from operator_model.resources import load_parameters
from functools import lru_cache
from operator_model.scheduling.graph import predict_case


@lru_cache(maxsize=1)
def _profile():
    return load_parameters('instruction_graph_reference')


def predict(op, shape, attrs=None):
    if op != 'kda_chunk':
        raise ValueError('Expected kda_chunk')
    original = dict(attrs or {})
    normalized = dict(original)
    safe = normalized.pop('safe_gate', False)
    if type(safe) is not bool:
        raise ValueError('safe_gate must be boolean')
    lower = normalized.pop('lower_bound', -5.)
    if safe and not (-5. <= lower < 0):
        raise ValueError('Expected safe gate lower bound in [-5,0)')
    dtype = normalized.pop('beta_dtype', 'fp32').lower()
    if dtype not in ('bf16','fp32'):
        raise ValueError('Unsupported beta dtype')
    if normalized.get('use_gate_in_kernel', False):
        normalized['gate_mode'] = 'raw_safe' if safe else 'raw_unsafe'
    result = predict_case(op, shape, normalized, _profile())
    result['call_attrs'] = original
    result['ca_timing_status'] = 'Fused V1 has synchronization errors. The separate FLA V2 implementation has one measured development window and no timing model.'
    return result
