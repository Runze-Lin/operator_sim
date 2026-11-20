"""Reference-chip schedules for the public system backend."""
from operator_model.resources import load_parameters
from functools import lru_cache
from operator_model.operators.fused import templates as fusion_templates
from operator_model.operators.vector import instructions as vector_instructions
from operator_model.operators.quant.repeat import price as quant_time
from operator_model.profiles.scalar_query import ReferencePricing
from operator_model.scheduling.cases import Case

VECTOR = {'quant', 'rmsnorm', 'mhc_sinkhorn', 'mhc_post', 'kda_recurrent', 'moe_gating_top_k', 'cast_bf16_fp32', 'attn_res_fwd', 'sigmoid_fp32'}
CUBE = {'matmul', 'addmm', 'gmm2'}
FUSION = {'fa', 'lightning_indexer', 'mhc_pre', 'mhc_pre_sinkhorn', 'mhc',
          'kda_chunk', 'grouped_matmul_swiglu_quant_v2', 'moe_ffn'}
ALIASES = {'rms_norm':'rmsnorm', 'lightening_indexer':'lightning_indexer',
           'lightningindexer':'lightning_indexer', 'groupedmatmulswiglugmm':'moe_ffn'}

@lru_cache(maxsize=4)
def _profile(path):
    return load_parameters(path)

@lru_cache(maxsize=1)
def reference_prices():
    return ReferencePricing(_profile('reference_pricing'))

def predict(op, shape, attrs=None):
    """Estimate net device instruction time in microseconds, using shape and caller attributes."""
    op = ALIASES.get(op, op)
    if op not in VECTOR | CUBE | FUSION:
        raise ValueError('Unknown operator: '+op)
    family = 'Vector' if op in VECTOR else 'Cube' if op in CUBE else 'CV'
    attrs = dict(attrs or {})
    shape = tuple(shape)
    try:
        # The attention-residual contract also accepts an empty history or token batch.
        if op != 'attn_res_fwd' and any(type(x) is not int or x<=0 for x in shape):
            raise ValueError('Expected positive integer dimensions')
        if op in ('quant','matmul','fa'):
            if attrs:
                raise ValueError('This validated route has fixed attributes')
            case = Case(op,shape)
            us = quant_time(reference_prices(),case) if op=='quant' else reference_prices().predict_us(case)
            result = {'predicted_us':us}
        elif op == 'attn_res_fwd':
            from operator_model.chips import load_chip
            from operator_model.operators.attention.residual_timing import resource_attention
            duration, _, diag = resource_attention(shape, load_chip(), attrs)
            result = dict(predicted_us=duration, predicted_active_aiv=diag['target_plan']['active_aiv'],
                          status='unqualified_ca_timing' if duration is None else 'A5_estimate')
        elif op == 'sigmoid_fp32':
            if attrs:
                raise ValueError('Sigmoid requires the current contiguous FP32 contract')
            from operator_model.chips import load_chip
            from operator_model.operators.vector.sigmoid import resource_sigmoid
            duration, _, active, diag = resource_sigmoid(shape, load_chip())
            result = dict(predicted_us=duration, predicted_active_aiv=active, stages_us=diag['stages_us'])
        elif op == 'cast_bf16_fp32':
            if attrs:
                raise ValueError('Cast requires the current fixed BF16 to FP32 contract')
            from operator_model.chips import load_chip
            from operator_model.operators.vector.cast import resource_cast
            duration, _, active, diag = resource_cast(shape, load_chip(), pressure_retention=True)
            result = dict(predicted_us=duration, predicted_active_aiv=active, stages_us=diag['stages_us'])
        elif op == 'moe_gating_top_k':
            if attrs:
                raise ValueError('Router requires the current fixed sigmoid/bias contract')
            from operator_model.chips import load_chip
            from operator_model.operators.vector.router import resource_router
            duration, _, active, diag = resource_router(shape, load_chip())
            result = dict(predicted_us=duration, predicted_active_aiv=active, stages_us=diag['stages_us'])
        elif op == 'addmm':
            from operator_model.operators.matmul.addmm import predict as estimate_addmm
            result = estimate_addmm(op, shape, attrs)
        elif op == 'rmsnorm':
            from operator_model.operators.rmsnorm.writeback import predict as estimate_rms
            result = estimate_rms(op, shape, attrs)
        elif op == 'mhc_post':
            from operator_model.operators.mhc.overlap import predict as estimate_post
            result = estimate_post(op, shape, attrs)
        elif op == 'mhc_sinkhorn':
            from operator_model.operators.vector.partition import predict as estimate_vector
            result = estimate_vector(op, shape, attrs)
        elif op in ('mhc_pre', 'mhc'):
            from operator_model.operators.mhc.pre_dispatch import predict as estimate_mhc
            if len(shape) == 3 and shape[0] > 256 and shape[2] <= 16384:
                result = fusion_templates.predict(op, shape, attrs, _profile('fusion_overlap'))
                result['status'] = 'extrapolation'
                result['reason'] = 'Above the calibrated 256-token split mHC partition; using the previous overlap estimate'
            else:
                result = estimate_mhc(op, shape, attrs)
        elif op in VECTOR:
            result = vector_instructions.predict(op,shape,attrs,_profile('vector_instructions'))
        elif op == 'kda_chunk':
            from operator_model.operators.kda.chunk import predict as chunk_structure
            result = chunk_structure(op,shape,attrs)
            if result.get('status')!='unsupported':
                result.update(status='unqualified_ca_timing',
                              reason='No calibrated KDA chunk timing: fused V1 synchronization errors; direct FLA V2 has one development measurement.')
        else:
            result = fusion_templates.predict(op,shape,attrs,_profile('fusion_overlap'))
    except (TypeError, ValueError) as error:
        result = {'predicted_us':None,'status':'unsupported','reason':str(error)}
    return {'op':op,'family':family,'shape':shape,'soc_version':'Ascend950DT_9582',
            'scope':'device instruction window; sum of windows for a serial chain',
            'status':'A5_estimate', **result}
