"""Per-stage VF service for proposed system programs; categories count register issues.

The scope carries the public selector across nested schedules (including native
lowerings), without global mutable state or changes to the external graph schema.
"""

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from math import ceil

MODES = ("auto", "calibrated", "instruction_classes")
_mode = ContextVar("system_vector_model", default="calibrated")
_log = ContextVar("system_vf_receipt", default=None)


def validate(mode):
    if mode not in MODES:
        raise ValueError("vector_model must be auto, calibrated or instruction_classes")
    return mode


@contextmanager
def selection(mode):
    token = _mode.set(validate(mode))
    records = []
    log = _log.set(records)
    try:
        yield records
    finally:
        _mode.reset(token)
        _log.reset(log)


def selected(known):
    mode = _mode.get()
    return ("calibrated" if known else "instruction_classes") if mode == "auto" else mode


def native_attrs(attrs):
    return dict(attrs or {}, vector_model=_mode.get())


def stage(chip, label, old_cycles, elements, *, bits=32, known=False, **lane_visits):
    """Each named lane_visits value is number of element-instructions on the busy AIV.

    Group-round each issue class to register width; load/store are *UB* port
    issues, not duplicate GM bytes. Inputs may be fractional for balanced stripes.
    """
    mode = selected(known)
    if elements <= 0:
        return old_cycles
    if mode == "calibrated":
        if (log := _log.get()) is not None:
            log.append(dict(stage=label, method=mode, legacy_cycles=old_cycles))
        return old_cycles
    cfg = chip.details
    lanes = max(1, int((cfg.get("vector_register_bytes") or 256) * 8 / bits))
    ins = {k: ceil(v / lanes) for k, v in lane_visits.items() if v > 0}
    widths = cfg.get("vf_issue_width") or 2
    ii = {
        k: (
            cfg.get("vf_" + k + "_ii_cycles")
            or {"exp": 4, "divide": 4, "rsqrt": 4, "log": 4, "reduce": 2}.get(k, 1)
        )
        for k in ins
    }
    costs = {k: v * ii[k] for k, v in ins.items()}
    bounds = {
        **costs,
        "issue": sum(ins.values()) / widths,
        "ub_ports": (costs.get("load", 0) + costs.get("store", 0)) / (cfg.get("vf_ub_ports") or 2),
        "alu_ports": sum(costs[k] for k in costs if k not in ("load", "store", "predicate"))
        / (cfg.get("vf_alu_lanes") or 2),
    }
    cycles = max(bounds.values(), default=0.0)
    if (records := _log.get()) is not None:
        records.append(
            dict(
                stage=label,
                method=mode,
                lanes=lanes,
                counts=ins,
                legacy_cycles=old_cycles,
                class_cycles=cycles,
                limiting_resource=max(bounds, key=bounds.get),
            )
        )
    return cycles if mode == "instruction_classes" else old_cycles


# Proposed register programs for full panels. Entries describe instructions per
# element pass (NOT shares of total time); variable-width reductions follow below.
_PROGS = {
    "sigmoid_times_value": "load load arithmetic exp arithmetic divide arithmetic store",
    "sigmoid.default": "load arithmetic exp arithmetic divide store",
    "swiglu": "load load arithmetic exp arithmetic divide arithmetic arithmetic store",
    "swiglu_clipped": "load load compare compare compare arithmetic exp arithmetic divide arithmetic arithmetic store",
    "situ_gate_and_linear_branch": "load load load exp exp arithmetic arithmetic divide divide arithmetic arithmetic store",
    "grouped_rms_sigmoid": "load load load arithmetic reduce arithmetic divide arithmetic exp arithmetic divide arithmetic store",
    "weighted_rms_rows": "load load arithmetic reduce arithmetic divide arithmetic arithmetic store",
    "norm/l2_last_axis": "load arithmetic reduce divide arithmetic store",
    "rope_pair": "load load load permute arithmetic arithmetic arithmetic store",
    "rope_interleaved_pair": "load load load permute arithmetic arithmetic arithmetic store",
    "mx4_cache_unpack": "load load permute convert arithmetic store",
    "fp8_cache_dequant": "load load convert arithmetic store",
    "engram_unpack_scales": "load load convert arithmetic store",
    "engram_sharded_row_gather": "load permute store",
    "selected_row_gather": "load permute store",
    "quantize_last_axis": "load arithmetic reduce divide convert store",
    "mx4_query_and_head_weights": "load arithmetic reduce divide convert permute store",
    "quantize_scatter_current_rows": "load arithmetic reduce divide convert permute store",
    "indexer/key_layernorm_mx4_insert": "load load load reduce arithmetic arithmetic reduce divide arithmetic arithmetic arithmetic reduce divide convert permute store",
    "layernorm_mx4_scatter_current_rows": "load reduce arithmetic arithmetic reduce divide arithmetic arithmetic reduce divide convert permute store",
    "engram_norm_dot_sigmoid_mix": "load load arithmetic reduce divide arithmetic arithmetic reduce exp arithmetic divide arithmetic arithmetic store",
    "sub.Tensor": "load arithmetic store",
    "neg.default": "load arithmetic store",
    "add.Tensor": "load load arithmetic store",
    "mul.Tensor": "load load arithmetic store",
    "pow.Tensor_Scalar": "load arithmetic store",
    "rsqrt.default": "load divide store",
    "lt.Scalar": "load compare store",
    "ge.Scalar": "load compare store",
    "bitwise_or.Tensor": "load load logical store",
    "masked_fill.Scalar": "load load select store",
    "masked_fill_.Scalar": "load load select store",
    "_to_copy.default": "load convert store",
}


def panel(chip, p, size, old_cycles):
    if not p.ops or size <= 0:
        return old_cycles
    tag = p.algorithm
    n = size
    ops = size * p.ops / p.elements
    if tag in _PROGS:
        visits = {k: v * n for k, v in Counter(_PROGS[tag].split()).items()}
    elif tag == "strided_group_reduce":
        visits = dict(load=ops + n, reduce=ops + n, store=n)
    elif tag == "score_full_row_then_hash_gather":
        visits = dict(
            load=n, exp=n, divide=2 * n, arithmetic=3 * n, permute=max(0, (ops - 7 * n) / 2), store=n
        )
    elif tag == "router_score_topk_merge":
        visits = dict(load=n, exp=n, arithmetic=4 * n, divide=n, compare=max(n, ops - 6 * n), store=n)
    elif tag.startswith("prefix_bank_"):
        visits = {k: v * size / p.elements for k, v in p.class_visits.items()}
    elif tag == "mhc_weighted_residual_mix":
        visits = dict(load=ops, arithmetic=ops / 2, store=n)
    elif tag == "ngram_hash_modulo":
        visits = dict(load=n, arithmetic=max(n, ops - 2 * n), divide=n, logical=n, store=n)
    elif tag == "kda/causal_depthwise_conv_prefill":
        taps = max(1, (ops / n - 4) / 2)
        visits = dict(load=2 * taps * n, arithmetic=(taps + 2) * n, exp=n, divide=n, store=n)
    else:
        raise ValueError("No system Vector issue program for " + tag)
    visits.setdefault("predicate", n)
    return stage(
        chip,
        tag,
        old_cycles,
        n,
        bits=p.register_bits or (16 if p.precision == "bf16" else 32),
        known=p.known_precision,
        **visits,
    )
