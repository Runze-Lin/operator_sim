"""Conditional timing for independent dense aclnnRecurrentKda state chains.

The single-chain model supplies per-target pipeline costs. The A5 kernel and
host tiler determine which of those chains run sequentially on the same core.
The bound on hidden prefetch is a sensitivity calculation, not a CA interval.
"""
from math import isfinite

from .chips import REFERENCE_NAME, load_chip
from .workload import account_work


def _ceil(x, y):
    return (x + y - 1) // y


def _round16(x):
    return _ceil(x, 16) * 16


def _profile(heads, k, v, ub_bytes):
    """Reference host EvaluateBufferProfile for A5, K-first FP32 state."""
    ah, ak, av = map(_round16, (heads, k, v))
    fixed = 8 * (8*ak + 2*av) + 8*4*ah + 64 + 2048
    working = fixed + 8*(4*av + 12*ak + 4*ah)
    profiles = []
    for states, outputs in ((1, 1), (1, 2), (2, 2)):
        coeff = 4*ak*(2 + 1 + states) + 2*outputs + (4*ak + 8)
        room = (ub_bytes-working)//coeff//16*16
        if room < 8:
            continue
        repeat = _ceil(v, room)
        step = _round16(_ceil(v, repeat))
        profiles.append(((repeat, -states-outputs, -step), dict(v_step=step, stripes=repeat,
                        state_output_slots=states, attention_output_slots=outputs,
                        ub_fixed_bytes=fixed, ub_working_bytes=working)))
    if not profiles:
        raise ValueError('State stripe cannot fit the target UB')
    return min(profiles, key=lambda x: x[0])[1]


def estimate_recurrent_chains(shape, attrs=None, *, chip=REFERENCE_NAME, hbm_efficiency=1.0,
                              hbm_read_fraction=1.0, hbm_write_fraction=1.0):
    """Return JSON-ready geometry, work, and conditional time in microseconds.

    A5 dense nonpaged BSND, equal local Q and V heads, preactivated gate,
    BF16 QKV, FP32 K-first initial/final state. Only a two-64-stripe layout
    has a diagnostic single-chain template. Other UB profiles retain geometry
    and work without publishing a numerical prediction. Raises ValueError on
    a different operator contract. External read/write fractions and effective
    HBM bandwidth are scenario inputs, not trace observations.
    """
    from .backend import estimate
    from collections.abc import Mapping
    if attrs is not None and not isinstance(attrs, Mapping):
        raise ValueError('Expected an attribute mapping')
    if not isinstance(shape,(list,tuple)) or len(shape)!=6 or any(type(x) is not int or x<=0 for x in shape):
        raise ValueError('Expected positive integer [B,S,Hq,Hv,K,V]')
    b,s,hq,hv,k,v=shape
    if hq!=hv or s>8 or k!=128 or v!=128:
        raise ValueError('Chain timing requires equal Q/V heads, S<=8, K=V=128')
    contract=dict(layout='BSND',dtype='bf16',gate_mode='preactivated',state_dtype='fp32',
                  state_v_first=False, normalize_qk=False, gate_dtype='fp32', use_gate_in_kernel=False,
                  output_final_state=True)
    for key,val in (attrs or {}).items():
        if key not in contract or not isinstance(val,type(contract[key])) or (val.lower() if key in ('dtype','gate_dtype','state_dtype') else val)!=contract[key]:
            raise ValueError(f'Unsupported dense RecurrentKda attribute: {key}={val!r}')
    if type(hbm_efficiency) not in (float,int) or not isfinite(hbm_efficiency) or not 0<hbm_efficiency<=1:
        raise ValueError('Expected hbm_efficiency in (0,1]')
    for value in (hbm_read_fraction,hbm_write_fraction):
        if type(value) not in (float,int) or not isfinite(value) or not 0<=value<=1:
            raise ValueError('HBM read/write fractions must be finite in [0,1]')
    t=load_chip(chip); chains=b*hv; cores=min(chains,t.vector_cores); rounds=_ceil(chains,cores)
    work=account_work('kda_recurrent',shape,{'dtype':'bf16'})
    hw=t.frequency_ghz*1000
    # Single chain S=1 controls separate the startup metadata from the first
    # beta command and separate the final state DMA retirement from END.
    # Before first beta includes initial task preparation, so offer a second
    # cutoff (metadata retired) instead of pretending this is a measured loop II.
    meta_us,beta_us,drain_us,prefetch_us=(cycles/hw for cycles in (1218,5650,10,887))
    state=chains*k*v*4
    # Unique compulsory bytes in the stated cold-read + final-drain scenario;
    # not every queue descriptor, nor every intermediate state written to GM.
    # Dense gate is FP32 and requires K values per token/head.
    read_bytes=state+chains*s*(2*(2*k+v)+4*k+4)
    write_bytes=state+chains*s*2*v
    cold_bytes=read_bytes+write_bytes
    shared=(read_bytes*hbm_read_fraction+write_bytes*hbm_write_fraction)/(t.hbm_bandwidth_gbs*hbm_efficiency*1000)
    info=dict(chip=t.name,shape=list(shape),status='conditional_extrapolation',latency_us=None,
              active_aiv=cores,active_aic=0,chains=chains,chains_per_busiest_aiv=rounds,
              cores_with_full_rounds=chains-(rounds-1)*cores,source='A5 round-robin dense BSND',
              tiling=None,work=work,logical_state_store_descriptor_bytes=state*s,
              unique_state_bytes=state,cold_external_bytes=cold_bytes,hbm_efficiency=hbm_efficiency,
              external_service_floor_us=shared,cold_io_floor_us=cold_bytes/(t.hbm_bandwidth_gbs*hbm_efficiency*1000),
              external_read_fraction=hbm_read_fraction,external_write_fraction=hbm_write_fraction,
              unique_read_bytes=read_bytes,unique_write_bytes=write_bytes,calibrated_multichain=False)
    ub=t.details.get('ub_bytes')
    if type(ub) is not int or ub<=0:
        return dict(info,status='unpriced_tiling',reason='No known target/inherited UB size in bytes')
    try:
        geom=_profile(hv,k,v,ub)
    except ValueError as exc:
        return dict(info,status='unpriced_tiling',reason=str(exc))
    info['tiling']=geom
    if (geom['v_step'],geom['stripes'],geom['state_output_slots'],geom['attention_output_slots'])!=(64,2,2,2):
        return dict(info,status='unpriced_tiling',reason='Single-chain template does not price this target UB stripe/queue profile')
    single=estimate('kda_recurrent',(1,s,1,1,k,v),chip=t)
    if single.latency_us is None:
        return dict(info,status='unpriced_single_chain',reason=single.reason)
    one=single.latency_us
    if not isfinite(one) or one<=beta_us+drain_us:
        return dict(info,status='incompatible_stage_prices',reason='Repriced single-chain window is smaller than its calibrated prefix plus drain')
    def makespan(prefix,save):
        body=one-prefix-drain_us
        local=body*rounds-max(0,rounds-1)*min(save,body)
        return max(prefix+local+drain_us,shared)
    middle=makespan(beta_us,0)
    return dict(info,latency_us=middle,single_chain_us=one,once_prefix_us=beta_us,once_drain_us=drain_us,
                per_chain_body_us=one-beta_us-drain_us,prefetch_descriptor_us=prefetch_us,
                model_sensitivity_us=[makespan(beta_us,prefetch_us),makespan(meta_us,0)],
                repeated_cold_windows_us=rounds*one,
                bottleneck='shared cold IO floor' if shared>beta_us+rounds*(one-beta_us-drain_us)+drain_us else 'critical AIV chain loop',
                evidence='Single-chain DT pipeline and end-point controls only; multi-AIV contention and cross-chain handshake not fitted.')
