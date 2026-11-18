"""System simulator entry point. Load the frozen parameters once per process."""
import json
from functools import lru_cache
from pathlib import Path
from . import cube_v8, fusion_v8, vector_v8
from .quant_repeat import price as quant_time
from .fast_query import Predictor
from .model_v6 import Case

ROOT = Path(__file__).resolve().parents[1]
VECTOR = {'quant', 'rmsnorm', 'mhc_sinkhorn', 'mhc_post', 'kda_recurrent'}
CUBE = {'matmul', 'addmm', 'gmm2'}
FUSION = {'fa', 'lightning_indexer', 'mhc_pre', 'mhc_pre_sinkhorn', 'mhc',
          'kda_chunk', 'grouped_matmul_swiglu_quant_v2', 'moe_ffn'}
ALIASES = {'rms_norm':'rmsnorm', 'lightening_indexer':'lightning_indexer',
           'lightningindexer':'lightning_indexer', 'groupedmatmulswiglugmm':'moe_ffn'}

@lru_cache(maxsize=4)
def _profile(path):
    return json.loads((ROOT/path).read_text())

@lru_cache(maxsize=1)
def _legacy():
    return Predictor(_profile('results/v0.6/model.json'), cache_size=0)

def _estimate(op, shape, attrs, revision):
    """Estimate net device instruction time in microseconds, using shape and caller attributes."""
    op = ALIASES.get(op, op)
    if op not in VECTOR | CUBE | FUSION:
        raise ValueError('Unknown operator: '+op)
    family = 'Vector' if op in VECTOR else 'Cube' if op in CUBE else 'CV'
    attrs = dict(attrs or {})
    shape = tuple(shape)
    try:
        if any(type(x) is not int or x<=0 for x in shape):
            raise ValueError('Expected positive integer dimensions')
        if op in ('quant','matmul','fa'):
            if attrs:
                raise ValueError('This validated route has fixed attributes')
            case = Case(op,shape)
            us = quant_time(_legacy(),case) if op=='quant' else _legacy().predict_us(case,use_cache=False)
            result = {'predicted_us':us}
        elif revision >= 9 and op == 'addmm':
            from .addmm_v9 import predict as estimate_addmm
            result = estimate_addmm(op, shape, attrs)
        elif revision >= 10 and op in ('rmsnorm', 'mhc_post'):
            from .wide_v10 import predict as estimate_wide
            result = estimate_wide(op, shape, attrs)
        elif revision >= 9 and op in ('rmsnorm', 'mhc_sinkhorn', 'mhc_post'):
            from .vector_v9 import predict as estimate_vector
            result = estimate_vector(op, shape, attrs)
        elif revision >= 9 and op in ('mhc_pre', 'mhc'):
            from .mhc_v9 import predict as estimate_mhc
            if len(shape) == 3 and shape[0] > 256 and shape[2] <= 16384:
                result = fusion_v8.predict(op, shape, attrs, _profile('results/v0.8/fusion/model.json'))
                result['status'] = 'extrapolation'
                result['reason'] = 'Above the calibrated 256-token split mHC partition; using the previous overlap estimate'
            else:
                result = estimate_mhc(op, shape, attrs)
        elif op == 'addmm':
            result = cube_v8.predict(op,shape,attrs)
        elif op in VECTOR:
            result = vector_v8.predict(op,shape,attrs,_profile('results/v0.8/vector/profile.json'))
        elif op == 'kda_chunk':
            from .cube_v8_chunk import predict as chunk_structure
            result = chunk_structure(op,shape,attrs)
            if result.get('status')!='unsupported':
                result.update(status='unqualified_ca_timing',
                              reason='ChunkKdaFwd synchronization error in the CA reference')
        else:
            result = fusion_v8.predict(op,shape,attrs,_profile('results/v0.8/fusion/model.json'))
    except (TypeError, ValueError) as error:
        result = {'predicted_us':None,'status':'unsupported','reason':str(error)}
    return {'op':op,'family':family,'shape':shape,'soc_version':'Ascend950DT_9582',
            'scope':'device instruction window; sum of windows for a serial chain',
            'status':'A5_estimate', **result}

def predict(op, shape, attrs=None):
    """Predict the current model's net device time, in microseconds."""
    return _estimate(op, shape, attrs, revision=10)


def predict_v8(op, shape, attrs=None):
    """Replay the v0.8 predictor for same-shape version comparisons."""
    return _estimate(op, shape, attrs, revision=8)

def predict_v9(op, shape, attrs=None):
    """Replay the previous critical-core models for calibration comparisons."""
    return _estimate(op, shape, attrs, revision=9)
