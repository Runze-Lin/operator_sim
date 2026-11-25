"""LayerNormV4 scalar-last-axis and optional Slice windows for rank-local BF16."""

from math import ceil, log2
from operator_model.system_adapter import NodeEstimate
from operator_model.workload import _counts
from . import vf


def price(f, chip):
    x, g, b = f.inputs
    (y,) = f.outputs
    if (
        len(x.shape) != 2
        or x.shape != y.shape
        or g.shape != b.shape
        or g.shape != (x.shape[-1],)
        or not all(t.dtype.removeprefix("torch.") == "bfloat16" for t in (x, g, b, y))
    ):
        raise ValueError(
            "Expected BF16 last-axis affine LayerNorm with BF16 affine operands"
        )
    rows, width = x.shape
    if min(rows, width) < 1 or x.stride[-1] != 1 or x.stride[0] < width:
        raise ValueError(
            "Expected positive non-overlapping rows with contiguous channels"
        )
    materialize = rows > 1 and x.stride[0] != width
    n = rows * width
    row_bytes = 2 * width
    clock = chip.frequency_ghz
    ticks = 1000 * clock
    rd = (chip.details.get("mte2_bandwidth_gbs") or 48.36 * clock) / clock
    wr = (chip.details.get("mte3_bandwidth_gbs") or 32 * clock) / clock
    hbm = chip.hbm_bandwidth_gbs / clock
    active = min(chip.vector_cores, rows)
    busy = ceil(rows / active)
    slice_cores = 0
    slice_cycles = 0
    if materialize:
        grain = max(1, 1024 // row_bytes)
        slice_cores = min(chip.vector_cores, ceil(rows / grain))
        local = ceil(rows / slice_cores) * row_bytes
        # Tiling read, source transfer, then contiguous output; head/tail descriptor response.
        slice_cycles = (
            3720 + 22 * slice_cores + max(local / rd + local / wr, 4 * n / hbm)
        )
    # Four MTE2 reads (tiling, x, gamma, beta); previous Slice changes x residency.
    # Parameter fanout increases arrival spread on the busiest core.
    start = 2740 if materialize else 3670
    fetch = start + 140 * log2(active) + row_bytes * (busy + 2) / rd
    tile_rows = max(
        1, ((chip.details.get("ub_bytes") or 253952) - 8192) // (16 * width)
    )
    blocks = ceil(busy / tile_rows)
    lanes = chip.vector_tops["32"] * 1000 / (2 * chip.vector_cores * clock)
    elements = busy * width
    simd = vf.stage(
        chip,
        "layernorm/affine_reduce",
        (8 * elements + 3 * busy) / lanes,
        elements,
        known=False,
        load=3 * elements,
        arithmetic=6 * elements,
        reduce=2 * elements,
        rsqrt=busy,
        store=elements,
    )
    reduce = blocks * (90 + 8 * ceil(log2(width))) + simd
    # The native ELF emits two FP32 statistics streams even if the caller only exposes Y.
    # Each writer owns 'busy' rows. Different cores updating one 128-byte line may wait.
    writers = min(active, ceil(128 / (4 * busy)))
    write = 780 + 134 * (writers - 1) + busy * (row_bytes + 8) / wr
    norm_cycles = max(fetch + reduce + write, (4 * n + 4 * width + 8 * rows) / hbm)
    us = (slice_cycles + norm_cycles) / ticks + (
        1 + int(materialize)
    ) * chip.kernel_overhead_us
    work = _counts("bf16", 2 * n + 4 * width, 2 * n, {}, {"fp32": 8 * n + 3 * rows})
    work["schedule"] = dict(
        algorithm="strided Slice followed by affine LayerNormV4",
        active_aiv=ceil(rows / busy),
        available_row_slots=active,
        materialize=materialize,
        slice_aiv=ceil(rows / ceil(rows / slice_cores)) if slice_cores else 0,
        slice_us=slice_cycles / ticks,
        normalize_us=norm_cycles / ticks,
        rows_per_busy_aiv=busy,
        ub_row_tiles=blocks,
        statistics_bytes=8 * rows,
        stat_line_writers=writers,
        first_reads_cycles=fetch,
        reduction_cycles=reduce,
        write_cycles=write,
        internal_slice_bytes=4 * n if materialize else 0,
    )
    return NodeEstimate(
        "indexer_affine_layer_norm",
        "extrapolated",
        us,
        "950 LayerNormV4 and Slice full-window development traces; layout and resource projection.",
        work,
    )
