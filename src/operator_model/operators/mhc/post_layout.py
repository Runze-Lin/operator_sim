"""Source tiling for vLLM Ascend HcPost on DAV_3510, BF16 + FP32 coefficients."""
from operator_model.workload import _counts


def hc_post_work(shape, attrs):
    if attrs != {"implementation": "vllm_hc_post"}:
        raise ValueError("HcPost expects BF16 activations and FP32 post/comb using the pinned vLLM binding")
    if len(shape) != 4 or any(type(n) is not int or n <= 0 for n in shape) or shape[2] != 4:
        raise ValueError("vLLM HcPost expects [batch, sequence, 4, hidden]")
    batch, sequence, streams, width = shape
    tokens = batch * sequence
    outputs = tokens * streams * width
    multiplies = outputs * (streams + 1)
    adds = outputs * streams
    return _counts("bf16", 2*tokens*width*(1+streams) + 4*tokens*(streams+streams**2),
                   2*outputs, vec={"fp32": multiplies+adds},
                   classes={"multiply": multiplies, "add": adds},
                   notes=["FP32 useful arithmetic; BF16 conversions and physical DMA traffic are separate."])


def layout(shape, chip, attrs):
    work = hc_post_work(shape, attrs)
    batch, sequence, _, width = shape
    tokens = batch * sequence
    active = min(tokens, chip.vector_cores)
    rows, extra = divmod(tokens, active)
    full, tail = divmod(width, 4096)
    chunks = full + int(tail > 0)
    plan = dict(active_aic=0, active_aiv=active, min_rows_per_core=rows, max_rows_per_core=rows+bool(extra),
                larger_cores=extra, d_tile=4096, full_d_tiles=full, tail_elements=tail, d_tiles_per_row=chunks,
                x_residual_reads=2*tokens*chunks, coeff_reads=2*tokens, output_writes=tokens*chunks)
    return work, plan
