"""Explicit FLA 26.9.1 Chunk KDA V2 experimental resource schedule."""

from collections import Counter
from functools import lru_cache, reduce
from operator import add
import json
from pathlib import Path

from .chips import REFERENCE_NAME, load_chip
from . import kda_fwdh_schedule

IMPLEMENTATION = "fla_26_9_1_v2"
FAMILIES = ("chunk_kda_fwd_prepare", "chunk_fwd_h", "chunk_kda_fwd_finalize")


def div(a, b):
    return (a + b - 1) // b


def _sum_cycles(values):
    # Match the frozen left-to-right cycle accumulation on all supported Pythons.
    return reduce(add, values, 0.0)


@lru_cache(maxsize=1)
def prices():
    path = Path(__file__).resolve().parents[1] / "configs/kda-fla-v2-26.9.1.json"
    return json.loads(path.read_bytes())


def vf_weight(i, n, family):
    # V1 does a growing prefix on each active 16-row band, then pads fixed banks.
    if family == FAMILIES[0] and i == 1:
        return n + sum(min(n, j * 16) for j in range(1, div(n, 16) + 1))
    return n


def vf_estimate(family, i, n, chip):
    vf = prices()["vfs"][family][i]
    base = vf["s2_rvec_counts"]
    full = vf["s64_rvec_counts"]
    weight = lambda t: vf_weight(i, t, family)
    fraction = (weight(n) - weight(2)) / (weight(64) - weight(2))
    counts = {
        q: max(0, round(v + (full.get(q, 0) - v) * fraction)) for q, v in base.items()
    }
    dt = load_chip(REFERENCE_NAME)
    per_aiv = lambda c: c.vector_tops["32"] / (c.frequency_ghz * c.vector_cores)
    # Four RV execution records per clock is an issue-width proxy. Critical VF
    # dependency/dispatch cost stays on the control clock; price extra work separately.
    ratio = per_aiv(dt) / per_aiv(chip)
    base_cycles = vf["s2_vf_cycles"]
    observed_service = min(base_cycles, base["RVECEX"] / 4)
    cycles = (base_cycles - observed_service) + max(
        0, observed_service + (counts["RVECEX"] - base["RVECEX"]) / 4
    ) * ratio
    return cycles, counts


class Tile:
    def __init__(self, chip, family, active):
        self.chip = chip
        self.family = family
        self.active = active
        self.legs = []
        self.bytes = Counter()
        self.issues = Counter()
        self.mm = Counter()
        self.vf = Counter()

    def dma(self, route, *sizes):
        rates = prices()["local_bytes_per_cycle"]
        lat = prices()["descriptor_latency_cycles"]
        self.bytes[route] += sum(sizes)
        self.issues[route] += len(sizes)
        d = self.chip.details
        ghz = self.chip.frequency_ghz
        local = (
            "mte2"
            if route.startswith("OUT")
            else "mte1"
            if route.startswith("L1")
            else "mte3"
        )
        bandwidth = (d.get(local + "_bandwidth_gbs") or rates[route] * ghz) / ghz
        if "OUT" in route:
            # Cache-payload pipe shares aggregate fabric; off-chip spill is a separate floor.
            l2 = (d.get("l2_bandwidth_gbs") or 100 * self.chip.cube_cores * ghz) / ghz
            bandwidth = min(bandwidth, l2 / self.active)
        return lat[route] + sum(sizes) / bandwidth + max(0, len(sizes) - 1) * 2

    def calc(self, shapes):
        clocks = 0
        cfg = self.chip
        peak = cfg.cube_tops["16"] * 1000 / (cfg.frequency_ghz * cfg.cube_cores)
        for m, n, k, dtype in shapes:
            key = f"{m}x{n}x{k}:{dtype}"
            self.mm[key] += 1
            work = 2 * (16 * div(m, 16)) * (16 * div(n, 16)) * (16 * div(k, 16))
            # Release C4/C5 use ordinary FP32, SetHF32Mode(false). The frozen
            # candidate assumes half the BF16 rate; it is not a calibrated price.
            factor = 1.0
            if dtype == "fp32":
                fp32 = cfg.cube_tops["32"]
                factor = fp32 / cfg.cube_tops["16"] if fp32 else 0.5
            clocks += prices()["mmad_latency_cycles"] + work / (peak * factor)
        return clocks

    def vec(self, i, n):
        t, counts = vf_estimate(self.family, i, n, self.chip)
        self.vf.update(counts)
        return t

    def leg(self, name, *steps):
        self.legs.append((name, _sum_cycles(steps)))


def prepare(n, chip, a):
    j = div(n, 16)
    m = 32 if n <= 32 else 64
    v = 256 * n
    z = Tile(chip, FAMILIES[0], a)
    z.leg(
        "V0 gate and prefix",
        z.dma("OUT->UB", v, v, v, 4 * n, 512, 4),
        z.vec(0, n),
        z.dma("UB->OUT", v, v, 2 * v),
    )
    z.leg("V1 score packing", z.vec(1, n), z.dma("UB->OUT", 32768, 40960))
    read = z.dma("OUT->L1", 16384, 16384, 4096, 8192, 12288, 16384)
    c2 = []
    for t in range(1, j + 1):
        ld = max(z.dma("L1->L0A", 4096, 4096), z.dma("L1->L0B", 4096 * t))
        c2.append((ld, z.calc([(32, 16 * t, 128, "bf16")]), z.dma("L0C->UB", 2048 * t)))
    z.leg("C2 triangular bands", read, pipeline(c2))
    z.leg(
        "V3 invert two 32-row blocks",
        z.vec(2, n),
        z.dma("UB->OUT", 128 * n, 4096, 4096, 4096, 8192, 128 * n),
    )
    rows = [8192] + ([4096] * 3 if n > 32 else [])
    c4 = z.dma("OUT->L1", *rows)
    if n > 32:
        for i in range(2):
            c4 += max(z.dma("L1->L0A", 4096), z.dma("L1->L0B", 4096))
            c4 += z.calc([(32, 32, 32, "fp32")])
            if i == 0:
                c4 += z.dma("L0C->OUT", 4096) + z.dma("OUT->L1", 4096)
            else:
                c4 += max(z.dma("L0C->L1", 4096), z.dma("L0C->OUT", 128 * (n - 32)))
    vm = (
        z.dma("OUT->UB", v, v, 2 * v, v)
        + z.vec(3, n)
        + z.dma("UB->OUT", v, v, 256 * m, 256 * m)
    )
    z.leg("C4/C5 quadrant and V6 RHS", max(c4, vm))
    z.leg(
        "C7 W and U",
        z.dma("OUT->L1", 256 * m, 256 * m),
        max(z.dma("L1->L0A", 2 * m * m), z.dma("L1->L0B", 256 * m, 256 * m)),
        z.calc([(m, 128, m, "bf16")] * 2),
        z.dma("L0C->OUT", 2 * v, 2 * v),
    )
    return z


def recurrence(n, chip, a, first, last):
    z = Tile(chip, FAMILIES[1], a)
    v = 256 * n
    p = 16 * div(n, 16)
    if first:
        z.leg(
            "FP32 initial state to BF16 H",
            z.dma("OUT->UB", 65536),
            z.vec(0, n),
            z.dma("UB->OUT", 32768),
        )
    z.leg(
        "S0 W times H",
        z.dma("OUT->L1", v, 32768),
        max(z.dma("L1->L0A", 256 * p), z.dma("L1->L0B", 32768)),
        z.calc([(p, 128, 128, "bf16")]),
        z.dma("L0C->UB", 2 * v),
    )
    z.leg(
        "S1 corrected value", z.dma("OUT->UB", v), z.vec(1, n), z.dma("UB->OUT", v, v)
    )
    c = z.dma("OUT->L1", v, v) + max(
        z.dma("L1->L0A", 256 * p), z.dma("L1->L0B", 256 * p)
    )
    c += z.calc([(128, 128, p, "bf16")]) + z.dma("L0C->UB", 65536)
    # Stage3 state and gate reads can overlap the S2 K^T times right product.
    state = z.dma("OUT->UB", *([65536, 512] if first else [512]))
    z.leg("S2 outer product", max(c, state))
    z.leg(
        "S3 update recurrent state",
        z.vec(2, n),
        z.dma("UB->OUT", 65536 if last else 32768),
    )
    return z


def final(n, chip, a, mover):
    p = 16 * div(n, 16)
    v = 256 * n
    z = Tile(chip, FAMILIES[2], a)
    if mover:
        # AIV MTE2 gathers padded NZ bands. Eight separate MTE3 moves pack
        # a 128-column Q/H/V tile into L1, one move per 16-column stripe.
        z.leg(
            "AIV Q/H to UB and L1",
            z.dma("OUT->UB", v, 32768),
            z.dma("UB->L1", *([32 * n] * 8 + [4096] * 8)),
        )
        z.leg(
            "AIV Aqk/V to UB and L1",
            z.dma("OUT->UB", 2 * n * p, v),
            z.dma("UB->L1", *([32 * n] * (p // 16) + [32 * n] * 8)),
        )
    else:
        z.leg("Q, H, Aqk and value", z.dma("OUT->L1", v, 32768, 2 * n * n, v))
    pairs = [
        (
            max(z.dma("L1->L0A", 256 * p), z.dma("L1->L0B", 32768)),
            z.calc([(n, 128, 128, "bf16")]),
            0,
        ),
        (
            max(z.dma("L1->L0A", 2 * p * p), z.dma("L1->L0B", 256 * p)),
            z.calc([(n, 128, n, "bf16")]),
            0,
        ),
    ]
    z.leg("QH plus AqkV", pipeline(pairs))
    z.leg("output", z.dma("L0C->OUT", 2 * v))
    return z


def pipeline(jobs):
    ends = [0, 0, 0]
    for job in jobs:
        for i, t in enumerate(job):
            ends[i] = max(ends[i], ends[i - 1] if i else 0) + t
    return ends[-1]


def weighted(hist, tiles, attr):
    out = Counter()
    for key, count in hist.items():
        for k, v in getattr(tiles[key], attr).items():
            out[k] += count * v
    return dict(sorted(out.items()))


def schedule(shape, chip):
    if len(shape) != 6 or any(type(x) is not int or x < 1 for x in shape):
        raise ValueError("Expected positive integer B,S,Hq,Hv,K,V")
    b, s, hq, h, k, v = shape
    if h != hq or (k, v) != (128, 128):
        raise ValueError(
            "FLA V2 experimental timing covers equal Kimi heads, K=V=128 only"
        )
    if b * s > 2**31 - 1 or b * h * div(s, 64) > 2**32 - 1:
        raise ValueError(
            "Packed sequence or tiling work exceeds the signed token or unsigned work index"
        )
    if not chip.cube_tops["16"] or not chip.vector_tops["32"] or chip.vector_cores < 2:
        raise ValueError("BF16 Cube, FP32 Vector and a two-AIV group are required")
    chunks = div(s, 64)
    num = b * chunks
    last = s - 64 * (chunks - 1)
    mixed_capacity = min(chip.cube_cores, chip.vector_cores // 2)
    stages = []
    for i, f in enumerate(FAMILIES):
        cap = chip.cube_cores if i == 2 else mixed_capacity
        group = h if num >= cap else 1
        if i == 1:
            act = div(b * h, div(b * h, cap))
        else:
            act = min(cap, num * h // group)
        mover = i == 2 and group == h and num // act >= 8
        if mover:
            # 950 arch35 host permits 2 AIV to gather+pack into L1. A custom 1:1
            # SKU needs a core mapping, explicit rather than silently scheduling two AIVs.
            cap = mixed_capacity
            if cap < act:
                group = h if num >= cap else 1
                act = min(cap, num * h // group)
        histograms = []
        for core in range(act):
            if i == 1:
                stride = div(b * h, act)
                heads = min(stride, b * h - core * stride)
                seq = Counter()
                if chunks == 1:
                    seq[(last, True, True)] = heads
                else:
                    seq[(64, True, False)] = heads
                    if chunks > 2:
                        seq[(64, False, False)] = (chunks - 2) * heads
                    seq[(last, False, True)] = heads
                histograms.append(seq)
            else:
                parts = num * h // group
                begin = parts * core // act * group
                end = parts * (core + 1) // act * group

                def tails(x):
                    period = chunks * h
                    return (x // period) * h + max(0, x % period - (chunks - 1) * h)

                tail = tails(end) - tails(begin)
                hist = Counter()
                if tail:
                    hist[last] += tail
                if end - begin > tail:
                    hist[64] += end - begin - tail
                histograms.append(hist)
        total = Counter()
        for hist in histograms:
            total.update(hist)
        phase = {}
        for key in total:
            if i == 0:
                phase[key] = prepare(key, chip, act)
            elif i == 1:
                phase[key] = recurrence(key[0], chip, act, key[1], key[2])
            else:
                phase[key] = final(key, chip, act, mover)
        # Retired small-case front end up to first data-bearing MTE2 command; metadata
        # <=48 bytes is included here. Charge once per concurrently launched core group.
        startup = prices()["first_payload_cycles"][f]
        prefetch = {}
        if i == 1 and b * h == act:
            # Each active group owns one identical state chain. Input DMA for
            # the next chunk can advance independently of the recurrent H edge.
            critical_cycles, timeline = kda_fwdh_schedule.schedule_chain(
                s, startup, Tile(chip, f, act)
            )
            critical = 0
            prefetch = dict(
                prefetch_domain="one sequence × head per active mixed workgroup",
                prefetch_timeline_cycles=timeline,
                prefetch_sample_chunk_indices=list(range(min(chunks - 1, 3))) + [chunks - 1],
                prefetch_time_origin="first instruction of the FwdH kernel",
                representative_phase_scope="standalone tile costs, before descriptor overlap",
            )
        else:
            cycles = [
                startup
                + _sum_cycles(
                    count * _sum_cycles(t for _, t in phase[key].legs)
                    for key, count in hist.items()
                )
                for hist in histograms
            ]
            critical = max(range(act), key=cycles.__getitem__)
            critical_cycles = cycles[critical]
            if i == 1:
                prefetch = dict(
                    prefetch_domain="multiple head chains per workgroup; serialized v0.23 fallback",
                    prefetch_timeline_cycles=[], prefetch_sample_chunk_indices=[],
                )
        dma = weighted(total, phase, "bytes")
        desc = weighted(total, phase, "issues")
        if i == 0:
            dma["OUT->UB"] += 96 * act
            desc["OUT->UB"] += 2 * act
        mm = weighted(total, phase, "mm")
        vec = weighted(total, phase, "vf")
        # Scenario pressure: only charge spill beyond aggregate L2. Local rates
        # already charge all workspace traffic once at the shared on-chip boundary.
        external_read_spill = max(
            0,
            sum(v for k, v in dma.items() if k.startswith("OUT"))
            - (chip.details.get("l2_bytes") or 0),
        )
        # OUT descriptors include workspace reads from previous stages. The cold
        # miss model is a scenario; churn does not prove those bytes left the chip.
        floor = external_read_spill / (chip.hbm_bandwidth_gbs * 1000)
        est = max(critical_cycles / (chip.frequency_ghz * 1000), floor)
        physical_ops = Counter()
        for key, count in mm.items():
            dims, fmt = key.split(":")
            m, n, kdim = map(int, dims.split("x"))
            physical_ops[fmt] += 2 * m * n * kdim * count
        head_tiles = sum(total.values())
        # Source-verifiable subsets of executed lane arithmetic. Instructions,
        # scalar-equivalent ops and DMA bytes deliberately have separate fields.
        if i == 0:
            lane_ops = {
                "two_triangular_inverses_mul_sub": head_tiles
                * 2
                * (31 * 32 // 2)
                * 2
                * 32
            }
        elif i == 1:
            lane_ops = {
                "corrected_value_subtract": b * s * h * v,
                "state_scale_plus_add": head_tiles * 2 * k * v,
            }
        else:
            lane_ops = {}  # Both finalize implementations use MMAD and data movers.
        stages.append(
            dict(
                head_chunks=head_tiles,
                counted_vector_lane_ops_by_class=lane_ops,
                counted_vector_lane_scope="known subset; padded inverse arithmetic included in Prepare",
                issued_cube_tops=sum(physical_ops.values()) / (est * 1e6),
                partition_unit="sequence × head"
                if i == 1
                else "chunk × local head partition",
                head_chunk_waves_lower_bound=div(head_tiles, act),
                issued_cube_ops_by_dtype=dict(physical_ops),
                family=f,
                active_aic=act,
                launched_aiv=act * 2 if i < 2 or mover else 0,
                aiv_mover=mover,
                issued_mmad_ops=sum(physical_ops.values()),
                timing_evidence="uncalibrated AIV-mover extrapolation"
                if mover
                else "resource candidate",
                chip_group_assumption="two physical AIV per mixed workgroup",
                mmads=mm,
                dma_bytes=dma,
                dma_descriptors=desc,
                estimated_rvec_subqueue_records=vec,
                first_payload_cycles=startup,
                max_core_head_chunks=max(sum(x.values()) for x in histograms),
                device_us=est,
                hbm_spill_scenario_us=floor,
                critical_core=critical,
                critical_core_head_chunks={
                    str(k): v for k, v in histograms[critical].items()
                },
                representative_tile_phase_cycles={
                    str(k): dict(v.legs) for k, v in phase.items()
                },
                **prefetch,
            )
        )
    times = [s["device_us"] for s in stages]
    return dict(
        shape=list(shape),
        chip=chip.name,
        dtype="bf16",
        stages=stages,
        serialized_device_us=_sum_cycles(times)
        if all(x is not None for x in times)
        else None,
        scope="three main V2 kernels only, no auxiliary launches or inter-launch intervals",
        uncalibrated=[
            "cross-head overlap",
            "FP32 MMAD throughput",
            "HBM spill and cache reuse",
            "AIV input-mover finalization",
            "970 physical mapping",
        ],
    )


def _normalize_contract(shape, attrs):
    """Return the logical uniform batch used by the varlen release kernel."""
    from math import isclose, sqrt

    if len(shape) != 6 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError("KDA requires positive B,S,Hq,Hv,K,V")
    attrs = dict(attrs)
    offsets = attrs.pop("cu_seqlens", None)
    defaults = {
        "implementation": IMPLEMENTATION,
        "dtype": "bf16",
        "layout": "BSND",
        "mode": "chunk",
        "chunk_size": 64,
        "gate_mode": "raw_safe",
        "gate_dtype": "bf16",
        "beta_dtype": "fp32",
        "state_dtype": "fp32",
        "state_v_first": True,
        "initial_state": True,
        "output_final_state": True,
        "dt_bias": True,
        "use_gate_in_kernel": True,
        "safe_gate": True,
        "normalize_qk": False,
        "use_qk_l2norm_in_kernel": False,
        "use_beta_sigmoid_in_kernel": False,
        "allow_neg_eigval": False,
        "use_exp2": True,
        "export_intermediates": False,
        "export_gk": False,
        "lower_bound": -5.0,
        "epsilon": 1e-6,
        "scale": 1 / sqrt(128),
    }
    for key, value in attrs.items():
        expected = defaults.get(key)
        if expected is None:
            raise ValueError(f"Unsupported FLA 26.9.1 V2 attribute {key}")
        if key in ("beta_dtype", "gate_dtype", "state_dtype") and isinstance(
            value, str
        ):
            value = {"bfloat16": "bf16", "float32": "fp32"}.get(
                value.lower(), value.lower()
            )
        if type(expected) is float:
            matches = type(value) in (int, float) and isclose(
                value, expected, rel_tol=1e-8
            )
        else:
            matches = type(value) is type(expected) and value == expected
        if not matches:
            raise ValueError(f"FLA 26.9.1 V2 contract requires {key}={expected!r}")
    b, s, hq, h, k, v = shape
    if offsets is not None:
        if (
            not isinstance(offsets, (tuple, list))
            or len(offsets) < 2
            or b != 1
            or any(type(n) is not int for n in offsets)
            or offsets[0] != 0
            or offsets[-1] != s
        ):
            raise ValueError(
                "Packed cu_seqlens requires B=1 and offsets from zero to total S"
            )
        seq = offsets[1]
        if seq <= 0 or any(q - p != seq for p, q in zip(offsets, offsets[1:])):
            raise ValueError(
                "FLA V2 experimental timing requires uniform nonempty packed sequence lengths"
            )
        b, s = len(offsets) - 1, seq
    return b, s, hq, h, k, v


def resource(shape, chip, attrs):
    """Package-facing resource estimate and one-call logical tensor accounting."""
    from .workload import _counts

    logical = _normalize_contract(shape, attrs)
    result = schedule(logical, chip)
    b, s, _, h, k, v = logical
    chunks, tail = divmod(s, 64)
    squares = b * h * (chunks * 64**2 + tail**2)
    rows = b * s * h
    chains = b * h
    state = 4 * chains * k * v
    read_bytes = 2 * rows * (3 * k + v) + 4 * rows + 4 * h * (k + 1) + state
    write_bytes = 2 * rows * v + 4 * rows * 64 + state
    # Per-head QK/KK triangular dots together cover n*n entries. W/U and
    # AqkV use dense-equivalent algebra before the kernel's tile padding.
    dots = 2 * k * squares
    wy = 2 * (k + v) * squares
    recurrent = 4 * rows * k * v
    finalize = 2 * rows * k * v + 2 * v * squares
    classes = {
        "corrected_value_subtract": rows * v,
        "recurrent_state_scale_add": 2 * b * h * div(s, 64) * k * v,
    }
    work = _counts(
        "bf16",
        read_bytes,
        write_bytes,
        {"bf16": dots + wy + recurrent + finalize},
        {"fp32": sum(classes.values())},
        classes=classes,
        notes=[
            "Visible outputs: BF16 attention, Aqk and Akk, plus FP32 VK. Metadata and internal workspaces excluded.",
            "Cube work is pre-padding dot/dense-equivalent algebra, omitting triangular inverse work.",
            "Vector work counts state/value updates only. Per-stage RVEC records are instruction proxies.",
        ],
    )
    work["stage_issued_cube_ops"] = {
        s["family"]: s["issued_cube_ops_by_dtype"] for s in result["stages"]
    }
    work["stage_dma_bytes"] = {s["family"]: s["dma_bytes"] for s in result["stages"]}
    plan = dict(
        api="aclnnChunkKdaFwdV2",
        implementation=IMPLEMENTATION,
        shape=list(shape),
        logical_sequence_shape=list(logical),
        layout="packed BSND",
        source="flash-linear-attention-npu A5 26.9.1",
        chunk_size=64,
        stages=result["stages"],
    )
    active_aic = max(s["active_aic"] for s in result["stages"])
    active_aiv = max(s["launched_aiv"] for s in result["stages"])
    projection = dict(
        method="experimental_fla_v2_resource_schedule",
        qualification="experimental",
        prediction_freeze_sha256=kda_fwdh_schedule.FREEZE_SHA256
        if result["stages"][1]["prefetch_timeline_cycles"]
        else prices()["freeze_sha256"],
        resource_price_freeze_sha256=prices()["freeze_sha256"],
        latency_us=result["serialized_device_us"],
        active_aic=active_aic,
        active_aiv=active_aiv,
        stages_us={s["family"]: s["device_us"] for s in result["stages"]},
        target_plan=plan,
        inherited_fields=list(chip.inherited_fields),
        scope=result["scope"],
        unvalidated=result["uncalibrated"],
    )
    return result["serialized_device_us"], active_aic, active_aiv, work, projection
