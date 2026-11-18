"""Stable adapter for a system simulator's per-device operator cost callback."""
from dataclasses import dataclass
from typing import Literal, Mapping, Optional, Sequence

from .predictor import ALIASES, CUBE, FUSION, VECTOR, predict


@dataclass(frozen=True)
class Estimate:
    """Device time in microseconds. Missing timing always has latency_us=None."""

    latency_us: Optional[float]
    status: Literal['estimated', 'extrapolated', 'unsupported', 'unavailable']
    reason: Optional[str] = None
    active_aiv: Optional[int] = None


def estimate(op: str, shape: Sequence[int], attrs: Optional[Mapping] = None,
             *, soc: str = 'Ascend950DT_9582') -> Estimate:
    """Price one dispatched operator on one rank from local shape and attributes."""
    if soc != 'Ascend950DT_9582':
        return Estimate(None, 'unsupported', f'No timing profile for {soc}')
    if not isinstance(op, str):
        return Estimate(None, 'unsupported', 'Expected an operator name')
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        return Estimate(None, 'unsupported', 'Expected an ordered shape of positive integers')
    if attrs is not None and not isinstance(attrs, Mapping):
        return Estimate(None, 'unsupported', 'Expected an operator attribute mapping')
    op = ALIASES.get(op, op)
    if op not in VECTOR | CUBE | FUSION:
        return Estimate(None, 'unsupported', f'Unknown operator: {op}')
    if op == 'kda_chunk' and attrs and 'beta_dtype' in attrs and not isinstance(attrs['beta_dtype'], str):
        return Estimate(None, 'unsupported', 'Expected a string beta_dtype')
    result = predict(op, shape, attrs)
    duration = result['predicted_us']
    reason = result.get('reason')
    if duration is None:
        if result['status'] == 'unsupported':
            return Estimate(None, 'unsupported', reason)
        return Estimate(None, 'unavailable', reason or result['status'])
    if result['status'] == 'extrapolation':
        return Estimate(duration, 'extrapolated', reason)
    if op == 'fa' and shape[1] > 131072:
        return Estimate(duration, 'extrapolated', 'KV length exceeds the longest complete 950DT FA reference')
    return Estimate(duration, 'estimated', active_aiv=result.get('predicted_active_aiv'))
