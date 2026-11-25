"""Vector and state programs for the rank-local text inference catalogue."""

from math import ceil, prod

from .vector import Tiles, _size, _schedule
from operator_model.system_adapter import NodeEstimate

NAMES = frozenset(
    {
        "aten.arange.default",
        "aten.arange.start_step",
        "aten._softmax.default",
        "aten.pow.Scalar",
        "aten.sin.default",
        "aten.cos.default",
        "aten.div.Tensor",
        "aten.silu.default",
        "npu::npu_add_rms_norm",
        "inference::indexer_layer_norm",
        "inference::hc_gate_mix",
        "inference::hc_combine",
        "inference::ple_gate_conv",
        "inference::ngram_lookup",
        "inference::ngram_hash",
        "inference::prepare_ngram_context",
        "inference::mla_cache_write",
        "inference::recurrent_state_commit",
    }
)


def _d(t):
    return t.dtype.removeprefix("torch.")


def _check(ok, why):
    if not ok:
        raise ValueError(why)


def state_rows(f, sem):
    """Existing resident state is an in-place destination, not a dense input read."""
    xs, ys = f.inputs, f.outputs
    if (
        len(ys) == 2
        and sem.get("base_state_preserved")
        and sem.get("verify_state_strategy")
        == "materialized per-token reference snapshots"
    ):
        state, snapshots = ys
        _check(
            len(xs) >= 2
            and state.shape == xs[-1].shape
            and snapshots.shape[0] == state.shape[0]
            and snapshots.shape[2:] == state.shape[1:],
            "Invalid query-prefix state geometry",
        )
        read = sum(_size(t, distinct=True) for t in xs)
        write = _size(state) + _size(snapshots)
        return Tiles("materialize_prefix_states", write, 512, read, write, 0)
    _check(
        len(ys) == 1 and len(xs) >= 2 and xs[-1].shape == ys[0].shape,
        "Expected one in-place state destination",
    )
    x = xs[0]
    src = xs[:-1]
    read = sum(_size(t, distinct=True) for t in src)
    payload = [t for t in src if _d(t) in {"bfloat16", "float16", "float32"}]
    _check(
        payload and _d(x) in {"bfloat16", "float16", "float32"},
        "Expected floating state rows",
    )
    width = ys[0].shape[-1]
    # K and V can be separate sources, or two sections of a single MLA row.
    # For compression the state may be a short ring. In that case only the
    # incoming rows that fit in the ring survive the invocation.
    n = min(
        prod(ys[0].shape),
        sum(prod(t.shape) for t in payload if t is x or t.shape[0] == x.shape[0]),
    )
    size = _size(ys[0]) // max(1, prod(ys[0].shape))
    return Tiles("scatter_live_state_rows", n, width, read, n * size, 0)


def _plan(f, sem):
    xs, ys, name = f.inputs, f.outputs, f.raw_name
    y = ys[0]
    if name in {"aten.arange.default", "aten.arange.start_step"}:
        n = prod(y.shape)
        _check(
            not xs and len(y.shape) == 1 and _d(y) in {"int64", "int32", "float32"},
            "Expected device range",
        )
        return Tiles(
            "affine_index_sequence",
            n,
            256,
            0,
            _size(y),
            2 * n,
            register_bits=8 * _size(y) // max(1, n),
            class_visits=dict(arithmetic=2 * n, store=n),
        )
    x = xs[0]
    n, width = prod(y.shape), y.shape[-1] if y.shape else 1
    read = sum(_size(t, distinct=True) for t in xs)
    write = sum(_size(t) for t in ys)
    if name == "inference::mla_cache_write":
        return state_rows(f, sem)
    if name == "inference::recurrent_state_commit":
        accepted = sem.get("accepted_tokens")
        _check(
            accepted is not None and len(xs) >= len(ys),
            "Expected accepted prefix selection",
        )
        if isinstance(accepted, int):
            fraction = float(accepted > 0)
        else:
            _check(len(accepted) > 0, "Empty batch")
            fraction = sum(a > 0 for a in accepted) / len(accepted)
        copied = round(write * fraction)
        return Tiles("copy_accepted_state_slices", copied, 256, copied, copied, 0)
    if name == "inference::ngram_lookup":
        _check(
            len(xs) in (2, 3)
            and len(x.shape) == 2
            and len(y.shape) == 2
            and x.shape[0] == y.shape[0],
            "Expected hashed IDs and a packed token-row output",
        )
        _check(
            sem.get("table_memory_space") == "device",
            "Host table requires a host-side timing model",
        )
        # The hash IDs pick a vocabulary shard; all ranks produce a full masked row.
        tp = int(sem["tp_size"])
        _check(tp > 0, "Expected positive vocabulary sharding")
        read = _size(x) + ceil(write / tp)
        visits = dict(
            load=n / tp + prod(x.shape), compare=prod(x.shape), select=n, store=n
        )
        return Tiles(
            "masked_sharded_ngram_gather",
            n,
            width,
            read,
            write,
            n + prod(x.shape),
            class_visits=visits,
            work_source="uniform hashed IDs across TP vocabulary shards; includes masked output fill",
        )
    if name == "inference::ngram_hash":
        k = int(sem["ngram_size"])
        _check(k >= 2 and _d(y) == "int64", "Expected integer n-gram hash")
        return Tiles(
            "ngram_hash_modulo",
            n,
            width,
            read,
            write,
            (3 * k + 2) * n,
            register_bits=64,
        )
    if name == "inference::prepare_ngram_context":
        _check(
            len(xs) == 4 and len(y.shape) == 2,
            "Expected context, offsets and query-start table",
        )
        touched = _size(y) + _size(xs[-1])
        visits = dict(load=n, arithmetic=n, store=n)
        return Tiles(
            "prepare_request_context",
            n,
            width,
            touched + _size(xs[2]),
            touched,
            n,
            class_visits=visits,
        )
    if name == "inference::ple_gate_conv":
        streams, taps, dilation = (
            int(sem[k]) for k in ("streams", "kernel_size", "dilation")
        )
        _check(
            len(x.shape) == 3
            and x.shape[1] == streams
            and xs[5].shape[-1] == taps
            and dilation > 0
            and taps > 0
            and y.shape == x.shape,
            "Expected stream-wise causal PLE convolution",
        )
        # Current input history is retained in a dilated ring. Verification may
        # also expose a second output containing the materialized prefix states.
        history = min(_size(xs[6]), x.shape[0] * streams * width * (taps - 1) * 2)
        read = sum(_size(t, distinct=True) for t in xs[:6]) + history
        write += min(_size(xs[6]), _size(x)) if len(ys) == 1 else 0
        visits = dict(
            load=(6 + taps) * n,
            reduce=3 * n,
            arithmetic=(8 + 2 * taps) * n,
            exp=n,
            divide=n + 3 * x.shape[0],
            store=n,
        )
        return Tiles(
            "ple_norm_gate_dilated_depthwise",
            n,
            width,
            read,
            write,
            (13 + 2 * taps) * n,
            class_visits=visits,
        )
    _check(
        _d(x) in {"float32", "bfloat16", "float16"} and _d(y) == _d(x),
        "Expected floating activations",
    )
    if name == "npu::npu_add_rms_norm":
        _check(
            len(xs) == 3
            and len(ys) == 3
            and x.shape == xs[1].shape == y.shape == ys[2].shape
            and xs[2].shape == (width,)
            and ys[1].shape == (*y.shape[:-1], 1)
            and _d(ys[1]) == "float32",
            "Invalid AddRmsNorm shape, gamma or rstd",
        )
        rows = prod(y.shape[:-1])
        visits = dict(
            load=3 * n,
            arithmetic=(4 + int(bool(sem.get("gemma_weight_offset")))) * n + 2 * rows,
            reduce=n,
            rsqrt=rows,
            store=2 * n + rows,
        )
        return Tiles(
            "add_rms_normalize_rows",
            n,
            width,
            read,
            write,
            5 * n + 3 * rows + int(bool(sem.get("gemma_weight_offset"))) * n,
            setup="aclnnAddRmsNorm",
            class_visits=visits,
        )
    if name == "inference::indexer_layer_norm":
        _check(
            len(xs) == 3
            and xs[1].shape == xs[2].shape == (width,)
            and x.shape == y.shape,
            "LayerNorm needs last-axis affine weights and bias",
        )
        rows = prod(y.shape[:-1])
        visits = dict(load=3 * n, arithmetic=6 * n, reduce=2 * n, rsqrt=rows, store=n)
        return Tiles(
            "indexer_affine_layer_norm",
            n,
            width,
            read,
            write,
            8 * n + 3 * rows,
            class_visits=visits,
        )
    if name == "aten._softmax.default":
        axis = int(dict(f.attributes)["dim"])
        _check(
            axis in (-1, len(y.shape) - 1) and x.shape == y.shape,
            "Expected last-axis softmax",
        )
        visits = dict(
            load=n, reduce=2 * n, arithmetic=2 * n, exp=n, divide=n // width, store=n
        )
        return Tiles(
            "stable_row_softmax",
            n,
            width,
            read,
            write,
            5 * n + n // width,
            class_visits=visits,
        )
    if name in {
        "aten.pow.Scalar",
        "aten.cos.default",
        "aten.sin.default",
        "aten.div.Tensor",
    }:
        # Positive scalar base pow uses a precomputed log(base), then mul + exp.
        # Trig assumes range reduction plus a five-coefficient Horner polynomial.
        if name == "aten.pow.Scalar":
            _check(
                float(dict(f.attributes)["self"]) > 0, "Expected positive scalar base"
            )
            visits = dict(arithmetic=n, exp=n)
        elif name == "aten.div.Tensor":
            visits = dict(divide=n)
        else:
            visits = dict(arithmetic=12 * n, convert=2 * n, select=n)
        return Tiles(
            name,
            n,
            512,
            read,
            write,
            sum(visits.values()),
            class_visits=dict(visits, load=n * len(xs), store=n),
        )
    if name == "aten.silu.default":
        _check(x.shape == y.shape and len(xs) == 1, "Expected unary SiLU")
        visits = dict(load=n, arithmetic=3 * n, exp=n, divide=n, store=n)
        return Tiles("silu", n, 1024, read, write, 5 * n, class_visits=visits)
    streams = int(sem["streams"])
    _check(streams > 0, "Invalid residual stream count")
    if name == "inference::hc_gate_mix":
        _check(
            len(xs) == 2 and prod(x.shape) == prod(xs[1].shape) == n * streams,
            "Expected per-stream gate",
        )
        visits = dict(
            load=2 * n * streams,
            arithmetic=3 * n * streams,
            exp=n * streams,
            divide=n * streams,
            reduce=n * streams,
            store=n,
        )
        return Tiles(
            "gated_stream_reduce",
            n,
            width,
            read,
            write,
            6 * n * streams,
            class_visits=visits,
        )
    _check(
        name == "inference::hc_combine" and x.shape == y.shape and len(xs) >= 3,
        "Expected residual stream injection",
    )
    return Tiles(
        "weighted_stream_injection",
        n,
        width,
        read,
        write,
        2 * n,
        class_visits=dict(load=3 * n, arithmetic=2 * n, store=n),
    )


def price(f, chip):
    try:
        return _schedule(_plan(f, dict(dict(f.attributes).get("semantic", ()))), chip)
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        return NodeEstimate(None, "unsupported", reason=f"{f.raw_name}: {exc}")
