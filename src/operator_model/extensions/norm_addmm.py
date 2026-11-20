"""Uncalibrated RMSNorm/Addmm structural plans for Ascend950DT_9582.

This module describes logical work and a limited source-derived RMSNorm route.
It does not turn v0.4 Matmul timing into an unvalidated Addmm latency.
"""
import math
from operator_model.profiles.stage_model import ceildiv, align
from operator_model.profiles.tile_model import plan as matmul_plan
from operator_model.scheduling.cases import Case

_BYTES = {'bf16': 2, 'fp16': 2, 'fp32': 4}
_ROOT = 'opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_nn/'


def _dimensions(shape, count):
    values = tuple(shape)
    if len(values) != count or any(type(x) is not int or x <= 0 for x in values):
        raise ValueError(f'Expected {count} positive integer dimensions')
    return values


def _base(op, shape, attrs, api):
    return {'op': op, 'shape': list(shape), 'attrs': attrs,
            'soc_version': 'Ascend950DT_9582', 'status': 'uncalibrated',
            'predicted_us': None, 'ideal_compute_efficiency': None,
            'api': api, 'library_family': 'ops-nn',
            'kernel_selection_verified': False, 'calibration_measurements': 0,
            'invocation_validation_report': 'docs/extensions/smoke-950dt.json'}


def plan(op, shape, **attrs):
    """Plan rmsnorm(R,C) or addmm(M,N,K); all latency fields remain None.

    RMSNorm: contiguous last-axis normalization; gamma has C elements.
    Addmm: dense ND operands; self_shape may be scalar (), (N,), (1,N),
    (M,1), or (M,N), with ordinary right-aligned broadcasting.
    """
    if op in ('rmsnorm', 'rms_norm'):
        return _rmsnorm(shape, attrs)
    if op == 'addmm':
        return _addmm(shape, attrs)
    raise ValueError('Expected rmsnorm or addmm')


def _rmsnorm(shape, attrs):
    if set(attrs)-{'dtype','gamma_dtype','epsilon'}:raise ValueError('Unknown RMSNorm attributes')
    rows, cols = _dimensions(shape, 2)
    dtype = attrs.get('dtype','bf16'); gamma_dtype = attrs.get('gamma_dtype',dtype)
    if dtype not in _BYTES or gamma_dtype not in (dtype,'fp32'):
        raise ValueError('RMSNorm supports matching BF16/FP16/FP32 or FP32 gamma in this plan')
    epsilon = attrs.get('epsilon',1e-6)
    if not isinstance(epsilon,(int,float)) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be finite and positive')
    xb, gb = _BYTES[dtype], _BYTES[gamma_dtype]
    out = _base('rmsnorm', (rows,cols), {'dtype':dtype,'gamma_dtype':gamma_dtype,'epsilon':epsilon}, 'aclnnRmsNorm')
    row_factor = ceildiv(rows,64); active = ceildiv(rows,row_factor)
    ca = align(cols,32//xb); vl = 64
    fold = 1 << ((ca-1).bit_length()-1) if ca > 1 else 1
    first_reduce_bytes = align(ceildiv(fold,vl),32)
    rows_ub = max(0,(253952-1024-2048-ca*gb)//(ca*xb*4+4*3+first_reduce_bytes))
    trans_possible = cols <= 32//xb and row_factor > vl*8
    normal_possible = ca <= vl*vl*4 and rows_ub >= 1 and not trans_possible
    if normal_possible:
        tile_rows = min(rows_ub,row_factor)
        tiling = {'status':'source_candidate_unverified','route':'REGBASE_NORMAL',
                  'source_tiling_key':5000,'active_aic':0,'active_aiv':active,
                  'max_rows_per_core':row_factor,'last_core_rows':rows-(active-1)*row_factor,
                  'tile_rows':tile_rows,'aligned_cols':ca,'max_ub_iterations':ceildiv(row_factor,tile_rows),
                  'column_fold_point':fold,'ub_bytes':253952,'host_reserved_bytes':1024,'arch35_reserved_bytes':2048,'vector_fp32_lanes':vl,
                  'input_output_double_buffer':True}
    else:
        tiling = {'status':'unresolved_dispatch','route_candidates':
                  ['REGBASE_TRANSPOSE'] if trans_possible else ['REGBASE_SPLIT','REGBASE_SPLIT_AR'],
                  'active_aic':None,'active_aiv':None,
                  'reason':'Short-column transpose or split reduction is outside the initial full-row tiling port'}
    out.update(tiling=tiling,
        work={'rows':rows,'reduction_length':cols,'multiplications':rows*(3*cols+1),
              'additions':rows*cols,'rsqrt':rows,'flops_excluding_rsqrt':rows*(4*cols+1),
              'logical_tensor_bytes':2*rows*cols*xb+cols*gb+4*rows,
              'traffic_scope':'x read + y write + one logical gamma read + FP32 rstd write; cache/reloads/padding excluded'},
        output_shapes={'y':[rows,cols],'rstd':[rows,1]},
        kernel_package_path=_ROOT+'rms_norm/',
        stage_dag=['read x/gamma','square + row reduction','scale + epsilon + rsqrt','multiply x and gamma','write y/rstd'],
        limitations=['No latency calibration; tiny invocation evidence is separate from generic dispatch qualification.',
            'Tiling is a port of one fixed-source branch, not proof that packaged CANN dispatches identically.',
            'Counts are mathematical scalar operations, not vector ISA instruction counts.',
            'Logical bytes are not per-core or HBM traffic.'])
    return out


def _addmm(shape, attrs):
    if set(attrs)-{'dtype','self_shape','alpha','beta','cube_math_type'}:raise ValueError('Unknown Addmm attributes')
    m,n,k = _dimensions(shape,3); dtype = attrs.get('dtype','bf16')
    if dtype not in _BYTES:raise ValueError('Expected BF16/FP16/FP32')
    self_shape=tuple(attrs.get('self_shape',(n,)))
    if len(self_shape)>2 or any(type(x) is not int or x<=0 for x in self_shape):raise ValueError('Invalid self shape')
    padded=(1,)*(2-len(self_shape))+self_shape
    if any(a not in (1,b) for a,b in zip(padded,(m,n))):raise ValueError('self does not broadcast to [M,N]')
    alpha,beta=attrs.get('alpha',1.),attrs.get('beta',1.)
    if any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in (alpha,beta)):
        raise ValueError('alpha and beta must be finite')
    cube_math_type=attrs.get('cube_math_type',0)
    if type(cube_math_type) is not int or cube_math_type not in (0,1,2,3,4):
        raise ValueError('cube_math_type must be an integer in 0..4')
    attrs={'dtype':dtype,'self_shape':list(self_shape),'alpha':alpha,'beta':beta,'cube_math_type':cube_math_type}
    fmm_candidate=(self_shape==(m,n) and abs(alpha-1)<=2**-23 and abs(beta-1)<=2**-23
                   and (dtype!='fp32' or cube_math_type==3))
    out=_base('addmm',(m,n,k),attrs,'aclnnAddmm')
    hypothesis=None
    if max(m,n,k)<=4096 and dtype=='bf16':
        hypothesis=matmul_plan(Case('matmul',(m,n,k)))
    out.update(work={'matmul_flops':2*m*n*k,'combine_additions':m*n,
        'nonunit_scale_multiplications':m*n*(int(alpha!=1)+int(beta!=1)),
        'logical_tensor_bytes':_BYTES[dtype]*(m*k+k*n+math.prod(self_shape)+m*n),
        'traffic_scope':'distinct A/B/self reads and output write; intermediates, cache and repeated tiles excluded',
        'zero_coefficient_policy':'Dense algebraic upper-work plan; alpha/beta=0 dispatch simplifications are not modeled'},
        tiling={'status':'unresolved_dispatch','active_aic':None,'active_aiv':None,
            'matmul_only_candidate':hypothesis,'candidate_is_actual_addmm_mapping':False,
            'fused_add_source_predicates_match':fmm_candidate,
            'source_dispatch_precedence':['empty and zero coefficients','NeedToConvertBias','inplace FP32 GemmV3',
                                          'DAV_3510 FusedMatMul(add)','Matmul + Muls/Add/Axpy fallback'],
            'source_match_is_actual_packaged_dispatch':False},
        stage_dag=['mat1 @ mat2','scale by alpha; broadcast self and scale by beta','combine and write output'],
        kernel_package_path=None,
        kernel_candidates=['MatMulV3 or fused matmul family, with optional separate scale/add/copy kernels'],
        limitations=['Generic dispatch is unqualified; the tiny invocation record only identifies its own shape.',
            'Fusion and launch count depend on attributes, dtype, layouts and broadcasting.',
            'A single-kernel CA parser may reject this API if it dispatches multiple kernels; preserve the entire call chain.',
            'Do not reuse Matmul latency as total Addmm latency.',
            'No CA calibration or vector/Cube utilization claim.'])
    return out
