"""Price one rank-local sim-platform OpFeatures without importing the system simulator.

Explicit lowerings represent a choice of implementation for a captured operation.
A missing time lets the caller continue through its existing backend chain.
"""
from dataclasses import dataclass
from math import isfinite, prod
from typing import Optional

from operator_model.backend import estimate
from operator_model.operators.quant.mxfp4 import contiguous_input


ADAPTER_VERSION = "1"
BF16 = "bfloat16"
F32 = "float32"
FP4 = "float4_e2m1fn_x2"
FP8 = "float8_e4m3fn"
E8 = "float8_e8m0fnu"


@dataclass(frozen=True)
class NodeEstimate:
    contract: Optional[str]
    status: str
    latency_us: Optional[float] = None
    reason: Optional[str] = None
    work: Optional[dict] = None


class _Mismatch(ValueError):
    """The captured layout or semantics differ from the priced implementation."""


def _require(condition, message):
    if not condition:
        raise _Mismatch(message)


def _fields(pairs):
    _require(isinstance(pairs, tuple) and all(isinstance(pair, tuple) and len(pair) == 2
             and isinstance(pair[0], str) for pair in pairs), "Expected named immutable capture attributes")
    return dict(pairs)


def _dtype(tensor):
    return tensor.dtype.removeprefix("torch.")


def _arity(features, inputs, outputs):
    _require((len(features.inputs), len(features.outputs)) == (inputs, outputs),
             "Input/output count differs from the standalone kernel")


def _layout(tensor, dtype, physical, *, strides=None):
    """Validate actual storage in elements (packed slots for FP4)."""
    physical = tuple(physical)
    _require(_dtype(tensor) == dtype, f"Expected {dtype}, received {tensor.dtype}")
    _require(tensor.shape == physical, "Tensor dimensions differ from the lowering")
    _require(tensor.layout in ("strided", "torch.strided"), "Expected a strided ND tensor")
    if strides is None:
        valid = len(tensor.stride) == len(physical) and all(i >= 0 for i in tensor.stride)
        valid = valid and contiguous_input(physical, tensor.stride)
    else:
        valid = len(tensor.stride) == len(physical) and all(
            size == 1 or actual == wanted
            for size, actual, wanted in zip(physical, tensor.stride, strides))
    _require(valid, "Tensor stride differs from the priced contiguous or packed layout")


def _packed(features, side, index, logical, axis=-1, strides=None):
    # The live API passes packed storage dimensions in TensorFeatures. The
    # optional logical extents are a separate attribute in convert_node_features.
    logical = tuple(logical)
    shapes = dict(features.attributes).get(side[:-1] + "_logical_shapes")
    if shapes is not None:
        _require(isinstance(shapes, tuple) and len(shapes) == len(getattr(features, side)),
                 "Invalid logical shape metadata")
        _require(shapes[index] in (None, logical), "Packed logical dimensions disagree with the local operation")
    physical = list(logical)
    _require(physical[axis] % 2 == 0, "Packed FP4 axis must be even")
    physical[axis] //= 2
    _layout(getattr(features, side)[index], FP4, physical, strides=strides)


def _shape(value, rank=None):
    _require(isinstance(value, tuple) and value and all(type(x) is int and x > 0 for x in value),
             "Missing positive rank-local tensor dimensions")
    _require(rank is None or len(value) == rank, "Tensor rank differs from the lowering")
    return value


def _quant(features, sem):
    _arity(features, 1, 2)
    x, = features.inputs
    y, scale = features.outputs
    shape = _shape(x.shape)
    _require(sem.get("logical_shape") == shape and sem.get("padding") == (0, 0),
             "Expected the unpadded local quantization shape")
    _layout(x, BF16, shape)
    if features.raw_name == "npu::npu_dynamic_mx_quant":
        _require(sem.get("logical_dtype") == "fp4_e2m1" and sem.get("group_size") == 32
                 and sem.get("round_mode") == "round" and sem.get("scale_alg", 0) == 0
                 and sem.get("axis", -1) in (-1, len(shape)-1)
                 and sem.get("scale_dtype") == "float8_e8m0fnu",
                 "Expected last-axis block32 MXFP4, paired E8M0 scale and OCP rounding")
        _packed(features, "outputs", 0, shape)
        _layout(scale, E8, (*shape[:-1], (shape[-1] + 63)//64, 2))
        return "quant_mxfp4", shape, "bf16", {}, "quant_mxfp4"
    _require(sem.get("granularity") == "per_token" and sem.get("logical_dtype") == "fp8",
             "Expected per-token FP8 quantization")
    _layout(y, FP8, shape)
    _layout(scale, F32, shape[:-1])
    return "quant_token_fp8", shape, "bf16", {}, "quant_token_fp8/aclnnDynamicQuantV2"


def _cache(features, sem):
    _arity(features, 2, 1)
    _, scale = features.inputs
    y, = features.outputs
    _require(sem.get("storage_precision") == "fp4" and sem.get("group_size") == 32,
             "Only the materialized block32 MXFP4 cache has an AntiMxQuant time")
    shape = _shape(sem.get("logical_shape"))
    _require(len(shape) >= 2 and sem.get("conversion") == "unpack and dequantize valid rows to BF16",
             "Expected standalone dequantization of the valid rows")
    _packed(features, "inputs", 0, shape)
    _layout(scale, E8, (*shape[:-1], (shape[-1]+63)//64, 2))
    _layout(y, BF16, shape)
    return "cache_load", shape, "mxfp4", {"implementation": "aclnnAntiMxQuant"}, "cache_load/aclnnAntiMxQuant"


def _mx_matmul(features, sem):
    grouped = features.raw_name == "npu::npu_grouped_matmul"
    _arity(features, 4, 1)
    _, _, sw, sx = features.inputs
    y, = features.outputs
    _require(sem.get("activation_scale_layout") == "mx_e8m0" and sem.get("weight_precision") == "fp4"
             and sem.get("activation_precision") == "fp4" and sem.get("group_size") == 32
             and sem.get("padding") == (0, 0) and sem.get("accumulator_dtype") == "float32",
             "Expected A4W4 with two block32 MX scales, zero padding and FP32 accumulation")
    tensor_semantics = _fields(dict(features.attributes).get("tensor_semantics", ()))
    records = tensor_semantics.get("inputs", ())
    _require(isinstance(records, tuple), "Expected indexed MX input roles")
    roles = {record.get("index"): record.get("role") for record in map(_fields, records)}
    expected_roles = ("activation", "weight", "weight_scale", "activation_scale")
    _require(roles == dict(enumerate(expected_roles)), "MX activation, weight and scale roles are required in schema order")
    m, k = _shape(sem.get("logical_input_shape"), 2)
    weights = _shape(sem.get("logical_weight_shape"), 3 if grouped else 2)
    batch, n = weights[:-2], weights[-2]
    _require(weights[-1] == k and k % 2 == 0, "Reduction dimension differs or is unpackable")
    g = (k+63)//64
    _packed(features, "inputs", 0, (m, k))
    # Logical W[K,N] is a transposed view on contiguous packed [N,K/2];
    # its [G,N,2] scale view has the corresponding [N,G,2] backing order.
    prefix_w = (n*(k//2),) if grouped else ()
    prefix_s = (n*g*2,) if grouped else ()
    _packed(features, "inputs", 1, (*batch, k, n), axis=-2, strides=(*prefix_w, 1, k//2))
    _layout(sw, E8, (*batch, g, n, 2), strides=(*prefix_s, 2, 2*g, 1))
    _layout(sx, E8, (m, g, 2))
    _layout(y, BF16, (m,n))
    if grouped:
        counts = sem.get("group_counts")
        _require(isinstance(counts, tuple) and len(counts) == batch[0] and
                 all(type(v) is int and v >= 0 for v in counts) and sum(counts) == m and
                 sem.get("gemm_grouping") != "output_head_groups", "Missing a valid local expert histogram")
        attrs = {"expert_counts": counts, "implementation": "aclnnGroupedMatmulV5"}
        return "routed_grouped_matmul", (m,n,k), "mxfp4", attrs, "routed_grouped_matmul/aclnnGroupedMatmulV5"
    return "matmul", (m,n,k), "mxfp4", {}, "matmul/A4W4"


def _unary(features, sem):
    _arity(features, 1, 1)
    x, = features.inputs
    y, = features.outputs
    raw = features.raw_name
    cast = raw == "aten._to_copy.default"
    shape = _shape(x.shape)
    _require(x.device == y.device, "Device transfer belongs to the system scheduler")
    if cast:
        # Inference capture may tag every tensor as meta while retaining an
        # explicit CPU/device conversion in the aten kwargs.
        options = dict(features.attributes).get("kwargs", ("mapping", ()))
        _require(isinstance(options, tuple) and len(options) == 2 and options[0] == "mapping",
                 "Unrecognized aten copy keyword encoding")
        keywords = _fields(options[1])
        _require(set(keywords) <= {"dtype", "non_blocking"} and
                 keywords.get("dtype", "torch.float32") in ("torch.float32", "float32"),
                 "Expected a local BF16 to FP32 conversion without transfer or layout change")
    _layout(x, BF16 if cast else F32, shape)
    _layout(y, F32, shape)
    op = "cast_bf16_fp32" if cast else "sigmoid_fp32"
    return op, (prod(shape[:-1]),shape[-1]), "bf16" if cast else "fp32", {}, op


def _rms(features, sem):
    _arity(features, 2, 2)
    x, gamma = features.inputs
    y, rstd = features.outputs
    shape = _shape(x.shape)
    _require(not sem.get("gemma_weight_offset") and not sem.get("gated")
             and sem.get("normalized_shape", shape[-1:]) == shape[-1:]
             and sem.get("reduction_dtype", F32) == F32,
             "Expected ordinary weighted RMSNorm over the last axis")
    epsilon = sem.get("epsilon")
    _require(type(epsilon) in (int, float) and isfinite(epsilon) and epsilon > 0,
             "Missing a finite positive RMSNorm epsilon")
    # The captured scalar changes values, while the same reduction, add and
    # rsqrt instruction schedule applies throughout the positive epsilon range.
    for t in (x, y):
        _layout(t, BF16, shape)
    _layout(gamma, BF16, shape[-1:])
    _layout(rstd, F32, (*shape[:-1], 1))
    # No stream residency inferred from ordering or layer names.
    return "rmsnorm", (prod(shape[:-1]), shape[-1]), "bf16", {}, "rmsnorm"


def _hc_post(features, sem):
    _arity(features, 4, 1)
    x, residual, post, comb = features.inputs
    b,s,d = _shape(x.shape, 3)
    _require(sem.get("hc_mult") == 4, "vLLM HcPost requires four residual branches")
    _layout(x, BF16, (b,s,d))
    _layout(residual, BF16, (b,s,4,d))
    _layout(post, F32, (b,s,4))
    _layout(comb, F32, (b,s,4,4))
    _layout(features.outputs[0], BF16, residual.shape)
    return "mhc_post", (b,s,4,d), "bf16", {"implementation":"vllm_hc_post"}, "mhc_post/vllm_hc_post"


def _mm(features, sem):
    _arity(features, 2, 1)
    x,w = features.inputs
    m,k = _shape(x.shape, 2)
    k2,n = _shape(w.shape, 2)
    _require(k == k2, "Matmul reduction dimensions differ")
    _layout(x, BF16, (m,k))
    _layout(w, BF16, (k,n))
    _layout(features.outputs[0], BF16, (m,n))
    return "matmul", (m,n,k), "bf16", {}, "matmul/BF16"


_LOWERINGS = {
    "npu::npu_dynamic_mx_quant": _quant,
    "inference::dynamic_mx_quant_fp8_reference": _quant,
    "inference::cache_load": _cache,
    "npu::npu_quant_matmul": _mx_matmul,
    "npu::npu_grouped_matmul": _mx_matmul,
    "aten.sigmoid.default": _unary,
    "aten._to_copy.default": _unary,
    "npu::npu_rms_norm": _rms,
    "_C_ascend::npu_hc_post": _hc_post,
    "aten.mm.default": _mm,
}


def estimate_features(features, *, chip):
    """Use one published lowering with latency in microseconds, or give the fallback reason.

    This is a prediction of the selected implementation. Capture does not identify an ELF.
    Pass sim-platform OpFeatures (immutable tuples for shapes, strides and attributes).
    """
    lower = _LOWERINGS.get(features.raw_name)
    if lower is None:
        return NodeEstimate(None, "unsupported", reason="No matched standalone operator-model contract")
    if features.category in {"comm", "zero", "control", "h2d", "d2h"}:
        return NodeEstimate(None, "unsupported", reason="Node is managed by the system scheduler")
    devices = {tensor.device for tensor in features.inputs + features.outputs}
    if (len(devices) != 1 or
            any(device.split(":")[0] not in ("meta", "npu", "privateuseone") for device in devices)):
        return NodeEstimate(None, "unsupported", reason="Expected local Ascend or captured meta tensors")
    try:
        sem = _fields(dict(features.attributes).get("semantic", ()))
        op, shape, dtype, attrs, contract = lower(features, sem)
    except _Mismatch as error:
        return NodeEstimate(None, "unsupported", reason=str(error))
    priced = estimate(op, shape, attrs, chip=chip, dtype=dtype)
    return NodeEstimate(contract, priced.status, priced.latency_us, priced.reason, priced.work)
