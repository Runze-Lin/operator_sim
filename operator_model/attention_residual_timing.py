"""Public result adapter for the fused Kimi attention residual."""
from functools import lru_cache
import json
from math import ceil
from pathlib import Path

from .attention_residual import attention_residual_work
from .workload import _counts


@lru_cache(maxsize=1)
def _prices():
    path=Path(__file__).resolve().parents[1]/'results/v0.19/workloads/attn-res-prices.json'
    return json.loads(path.read_text())['stage_prices']


def resource_attention(shape, chip, attrs):
    attrs=dict(attrs or {})
    dtype=attrs.pop('dtype','bf16')
    if dtype!='bf16' or len(shape)!=4:
        raise ValueError('attn_res_fwd requires (tokens, hidden, slots, valid) in BF16')
    allowed={'addend','output_norm','write_idx','materialize','mix','optimize_prefill'}
    if set(attrs)-allowed:
        raise ValueError('Unknown attn_res_fwd attributes')
    d=attention_residual_work(*shape,chip=chip,**attrs)
    classes={k:d[k] for k in ('fp32_basic_ops','prefix_add_elements','exp_elements','rsqrt_elements','reciprocal_elements')}
    vec={'fp32':sum(classes.values())-d['prefix_add_elements'],'bf16':d['prefix_add_elements']}
    work=_counts(dtype,d['distinct_input_bytes'],d['dma_write_bytes'],vec=vec,classes=classes,
                 notes=('Logical inputs count distinct tensor elements. DMA reads include bank reloads and per-core weight traffic.',
                        'Arithmetic counts include useful reductions and point operations. Nonlinear functions are counted per element.'))
    plan={'active_aic':0,'active_aiv':d['payload_aiv'],'launch_aiv':d['launch_aiv'],
          'tile':[1,shape[1]],'iterations':d['tokens_per_core'],'waves':1 if shape[0] else 0,
          'tiling_key':d['tiling_key'],'branch':('resident' if d['resident'] else 'reload')}
    diagnostics=dict(model='attn_res_fwd_arch35', target_plan=plan, stages_us={}, **d)
    covered=(d['resident'] and d['tokens_per_core']==1 and shape[3]>0 and
             attrs.get('addend',False) and attrs.get('output_norm',True) and
             attrs.get('materialize',False) and attrs.get('mix',True) and
             attrs.get('write_idx',-1)>=0 and chip.vector_tops['32'])
    if not covered:
        return None,work,diagnostics

    tokens,h,_,n=shape
    lanes=ceil(h/64)
    read_rate=(chip.details.get('mte2_bandwidth_gbs') or 48.3629559884*chip.frequency_ghz)/chip.frequency_ghz
    reference_peak=27000/(64*1.65)
    target_peak=chip.vector_tops['32']*1000/(chip.vector_cores*chip.frequency_ghz)
    phases=(('dispatch',1,0,0),('score_weights',1,2,lanes),('prefix_add_write',1,2,lanes),
            ('bank_scores',n,n,lanes*n),('weighted_norm_write',1,1,lanes*(n+1)))
    clock=chip.frequency_ghz*1000
    times={}
    for name,repeats,reads,chunks in phases:
        prices=_prices()[name]
        cycles=(prices['fixed_cycles']*repeats+prices['cycles_per_vector_chunk']*chunks*reference_peak/target_peak
                +reads*2*h/read_rate)
        times[name]=cycles/clock
    transfer=d['cold_input_hbm_floor_us']
    if chip.details.get('l2_bandwidth_gbs'):
        transfer=max(transfer,(d['dma_read_bytes']+d['dma_write_bytes'])/(chip.details['l2_bandwidth_gbs']*1000))
    if chip.details.get('mte3_bandwidth_gbs'):
        transfer=max(transfer,d['dma_write_bytes']/(tokens*chip.details['mte3_bandwidth_gbs']*1000))
    times['exposed_transfer_floor']=max(0.,transfer-sum(times.values()))
    latency=sum(times.values())
    diagnostics.update(latency_us=latency, qualification='DT stage calibration plus two-SKU tail proxy; single-token resident mix, add, write, materialize and output norm',
                       stages_us=times)
    return latency,work,diagnostics
