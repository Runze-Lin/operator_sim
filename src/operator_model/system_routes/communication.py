"""Rank-local collective schedules under an explicitly parameterized flat interconnect."""
from dataclasses import dataclass
from math import ceil, isfinite, prod

from operator_model.system_adapter import NodeEstimate

NAMES = frozenset({"collective::all_reduce", "collective::all_gather",
                   "npu::npu_moe_distribute_dispatch_v2", "npu::npu_moe_distribute_combine_v2"})
BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "int64": 8, "int32": 4,
         "bool": 1, "float8_e4m3fn": 1, "float8_e8m0fnu": 1, "float4_e2m1fn_x2": 1}


@dataclass(frozen=True)
class FabricScenario:
    """Per-rank full-duplex injection in decimal GB/s and startup per serial exchange in µs."""
    injection_gbs: float = 200.
    step_us: float = 1.

    def __post_init__(self):
        if not all(type(x) in (float, int) and isfinite(x) and x > 0
                   for x in (self.injection_gbs, self.step_us)):
            raise ValueError("Fabric injection and step latency must be positive and finite")


def _bytes(tensor):
    return prod(tensor.shape) * BYTES[tensor.dtype.removeprefix("torch.")]


def _quote(chip, fabric, name, group, rounds, sent, received, reads, writes, ops, dtype, tiles, extra):
    active = min(chip.vector_cores, max(1, tiles))
    rate = chip.vector_tops["32" if dtype == "float32" else "16"]
    if ops and not rate:
        raise ValueError("Missing Vector rate for the reduction")
    payload = max(sent, received)
    link = payload / (1000 * fabric.injection_gbs)
    memory = (reads+writes) / (1000 * chip.hbm_bandwidth_gbs * active/chip.vector_cores)
    vector = ops / (1e6 * rate * active/chip.vector_cores) if ops else 0.
    pipeline = max(link, memory, vector)
    duration = (rounds*fabric.step_us + pipeline + chip.kernel_overhead_us) if (rounds or pipeline) else 0.
    schedule = dict(algorithm=name, group_ranks=len(group), serial_steps=rounds, remote_send_bytes=sent,
                    remote_recv_bytes=received, link_service_us=link, startup_us=rounds*fabric.step_us,
                    local_memory_us=memory, local_vec_us=vector, participating_aiv=active if tiles else 0,
                    fabric_gbs=fabric.injection_gbs, fabric_step_us=fabric.step_us,
                    evidence="rank-local communication scenario", **extra)
    work = dict(cube_ops=0, cube_ops_by_dtype={}, vec_ops=ops, vec_ops_by_dtype={dtype: ops} if ops else {},
                logical_read_bytes=reads, logical_write_bytes=writes, communication_access=payload,
                communication_us=rounds*fabric.step_us+link, schedule=schedule)
    return NodeEstimate(name,"extrapolated",duration, "Flat full-duplex scenario; rank-local payload and startup.", work)


def _ring(f, chip, fabric):
    ctx = dict(f.communication)
    facts = getattr(f, "collective", None)
    if facts:
        group = facts.ordered_group
    else:
        rank = ctx["rank"]
        group, = (g for g in ctx["rank_groups"] if rank in g)
    p = len(group)
    source, = f.inputs
    size = _bytes(source)
    gather = f.raw_name == "collective::all_gather"
    if not gather and (ctx.get("reduce_op") or getattr(facts,"reduction",None) or "").upper() != "SUM":
        raise ValueError("Only SUM all-reduce has a reduction schedule")
    if sum(map(_bytes,f.outputs)) != size*(p if gather else 1):
        raise ValueError("Collective output extent disagrees with rank group")
    rounds = (p-1)*(1 if gather else 2)
    transfer = size*(p-1) if gather else ceil(2*size*(p-1)/p)
    ops = 0 if gather else ceil(prod(source.shape)*(p-1)/p)
    tile = max(32, chip.details["ub_bytes"]//4//32*32)
    # Segment parallelism across UB buffers; ring partitions the reduction tensor.
    tiles = ceil(size/tile) if gather else ceil(size/p/tile)*p
    return _quote(chip,fabric,"ring_allgather" if gather else "ring_reduce_scatter_allgather",
                  group,rounds,transfer,transfer,transfer+(ops*BYTES[source.dtype.removeprefix("torch.")] if ops else 0),
                  transfer,ops,source.dtype.removeprefix("torch."),tiles,dict(segment_bytes=tile))


def _moe(f, chip, fabric):
    s = dict(dict(f.attributes)["semantic"])
    phase = s["phase"]
    combine = f.raw_name.endswith("combine_v2")
    if (phase != ("combine" if combine else "dispatch") or
        (combine and (s["codec"] != "fp8_token" or not s["weighted_finalize_inside_boundary"])) or
        (not combine and s["codec"] != "mxfp4")):
        raise ValueError("Unexpected transport precision or finalize boundary")
    group = s["group"]
    self_index = s["ep_rank_id"]
    if not 0 <= self_index < len(group):
        raise ValueError("Missing EP group index")
    def vector(key):
        v=s[key]
        if len(v)!=len(group) or any(type(x) is not int or x<0 for x in v):
            raise ValueError("Incomplete peer vector: "+key)
        return v
    outgoing = [a+b for a,b in zip(vector("send_payload_bytes"),vector("send_scale_bytes"))]
    incoming = [a+b for a,b in zip(vector("recv_payload_bytes"),vector("recv_scale_bytes"))]
    rows_out, rows_in = sum(vector("send_splits")), sum(vector("recv_splits"))
    if s["valid_rows"] != (rows_out if combine else rows_in):
        raise ValueError("Valid expert row count disagrees with traffic")
    peers = sum(1 for i,(a,b) in enumerate(zip(outgoing,incoming)) if i!=self_index and (a or b))
    sent, received = sum(outgoing)-outgoing[self_index], sum(incoming)-incoming[self_index]
    # AIVs distribute row slices across peers. Peer notifications may run concurrently.
    width = f.outputs[0].shape[-1] if combine else f.inputs[0].shape[-1]*2
    elements = rows_in*width if combine else 0
    ops = 3*elements if combine else 0  # dequantize, multiply gate and accumulate
    reads = sum(outgoing) + (sum(incoming) if combine else 0) + (rows_in if combine else rows_out)*8
    writes = _bytes(f.outputs[0]) if combine else sum(incoming)
    tile_bytes = max(32,chip.details["ub_bytes"]//4//32*32)
    tiles = max(rows_in,rows_out)*max(1,ceil(width*2/tile_bytes))
    return _quote(chip,fabric,"ep_weighted_combine" if combine else "ep_valid_row_dispatch",
                  group,ceil(peers/chip.vector_cores),sent,received,reads,writes,ops,"float32",tiles,
                  dict(active_peers=peers,self_group_index=self_index,valid_send_rows=rows_out,
                       valid_recv_rows=rows_in,reserved_rows=s["capacity_rows"],ub_tile_bytes=tile_bytes))


def price(features, chip, fabric):
    try:
        if features.raw_name.startswith("collective::"):
            return _ring(features,chip,fabric)
        return _moe(features,chip,fabric)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return NodeEstimate(None,"unsupported",reason=f"Collective facts mismatch: {exc}")
