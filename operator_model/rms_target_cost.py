"""Optional per-chip first-dispatch costs for the RMSNorm resource schedule.

A missing calibration inherits the reference model. Parameters express pipeline
cycles, independent of chip names or input shape. Only the standalone kernel's
first-call state has paired evidence.
"""
from dataclasses import dataclass
from math import isfinite

from .rms_resource import FIRST_PAYLOAD_RELEASE, FIRST_SCALAR, resource_rms


@dataclass(frozen=True)
class RmsFirstDispatch:
    scalar_cycles: float
    first_payload_cycles: float

    def __post_init__(self):
        for value in (self.scalar_cycles,self.first_payload_cycles):
            if not isinstance(value,(int,float)) or not isfinite(value) or value<=0:
                raise ValueError('First dispatch costs must be positive finite cycle counts')


def resource_rms_target(shape, chip, attrs=None, first_dispatch=None):
    original=resource_rms(shape,chip,attrs)
    if first_dispatch is None:
        return original
    _,aic,aiv,d=original
    clock=1000*chip.frequency_ghz
    first=dict(d['conditional_stages_us']['first'])
    first['SCALAR']=first_dispatch.scalar_cycles/clock
    before_shared=sum(v for k,v in first.items() if k!='shared_io')
    bound=(first_dispatch.first_payload_cycles+d['shared_io_floor_cycles'])/clock
    first['shared_io']=max(0.,bound-before_shared)
    phases={**d['conditional_stages_us'],'first':first}
    totals={**d['conditional_us'],'first':sum(first.values())}
    state=d['residency']
    d={**d,'conditional_stages_us':phases,'conditional_us':totals,'stages_us':phases[state],
       'first_dispatch_cycles':{'scalar':first_dispatch.scalar_cycles,
                                'payload':first_dispatch.first_payload_cycles}}
    return totals[state],aic,aiv,d


def configured_rms(shape, chip, attrs=None):
    scalar=chip.details.get('rms_first_scalar_cycles')
    payload=chip.details.get('rms_first_payload_cycles')
    prices=None
    if scalar is not None or payload is not None:
        prices=RmsFirstDispatch(
            FIRST_SCALAR if scalar is None else scalar,
            FIRST_PAYLOAD_RELEASE if payload is None else payload)
    return resource_rms_target(shape,chip,attrs,prices)
