"""Alternative SIMD issue budgets for native A5 kernels. Counts are designs, not ISA traces."""

from dataclasses import replace
from math import ceil, log2


SUPPORTED_OPS = frozenset(
    {
        "quant",
        "quant_mxfp4",
        "rmsnorm",
        "mhc_pre",
        "mhc_post",
        "mhc",
        "mhc_sinkhorn",
        "mhc_pre_sinkhorn",
        "attn_res_fwd",
        "moe_gating_top_k",
        "cache_load",
    }
)


class Pricer:
    """Price critical-AIV register groups. Record one example of each body, not each replay."""

    def __init__(self, chip):
        self.chip = chip
        self.samples = {}
        self.cache = {}
        self.clock = chip.frequency_ghz * 1000

    def __call__(self, name, rows, width, n=1):
        key = (name, rows, width, n)
        if key in self.cache:
            return self.cache[key]
        d = self.chip.details
        lanes = int(d.get("vector_register_bytes") or 256) // 4
        g = rows * ceil(width / lanes)  # each row has its own predicate tail
        tree = rows * ceil(log2(max(1, ceil(width / lanes))))
        if name in ("mx4", "mx8"):
            b = rows * 2 * ceil(width / 64)  # E8M0 uses paired group32 slots
            x, s = ceil(b / 8), ceil(b / 128)
            c = dict(
                predicate=x + s,
                load=2 * x + s,
                arithmetic=3 * x + 4 * s,
                reduce=x,
                permute=x + 3 * s,
                convert=2 * x + 4 * s,
                store=x + s,
            )
            if name == "mx4":
                c["permute"] += x  # pack two output nibbles
        elif name == "antimx":
            b = rows * 2 * ceil(width / 64)
            x, s = ceil(b / 16), ceil(b / 256)
            c = dict(
                predicate=x + s,
                load=2 * x + s,
                permute=2 * x + s,
                convert=4 * x + s,
                arithmetic=4 * x,
                store=4 * x,
            )
        elif name == "rms_reduce":
            c = dict(predicate=g, load=g, convert=g, arithmetic=2 * g, reduce=g + tree, store=rows)
        elif name == "rms_stat":
            c = dict(load=rows, arithmetic=2 * rows, rsqrt=rows, store=rows)
        elif name == "rms_norm":
            c = dict(predicate=g, load=2 * g, convert=3 * g, arithmetic=2 * g, store=g)
        elif name == "post_mix":
            c = dict(
                predicate=g, load=(n + 1) * g, convert=(n + 1) * g, arithmetic=n * (n + 1) * g, store=n * g
            )
        elif name == "cast_out":
            c = dict(load=g, convert=g, store=g)
        elif name == "sink_normalize":
            passes = ceil(rows / (lanes // 8 or 1))
            c = dict(
                load=4 * passes,
                permute=3 * passes,
                arithmetic=4 * passes,
                divide=4 * passes,
                store=4 * passes,
            )
        elif name == "pre_cast":
            c = dict(load=g, convert=g, arithmetic=g, reduce=g + tree, store=g)
        elif name == "pre_mix":
            c = dict(load=(n + 1) * g, convert=n * g, arithmetic=2 * n * g, rsqrt=rows, exp=rows, store=g)
        elif name == "router":
            # Repeated max+index removal is a candidate design; not a recovered native sort.
            c = dict(
                predicate=g,
                load=2 * g,
                arithmetic=4 * g + n * g,
                exp=g,
                divide=g + rows,
                reduce=n * (g + tree) + rows,
                permute=n * rows,
                store=n * rows,
            )
        elif name == "att_weights":
            c = dict(load=2 * g, convert=2 * g, arithmetic=g, store=g)
        elif name == "att_prefix":
            c = dict(load=2 * g, arithmetic=g, store=g)
        elif name == "att_score":
            c = dict(
                load=2 * n * g,
                convert=n * g,
                arithmetic=5 * n * g,
                reduce=n * (g + tree),
                exp=n * rows,
                rsqrt=n * rows,
                divide=rows,
                store=g,
            )
        else:
            raise ValueError("Unknown SIMD program: " + name)
        ii = {
            k: d.get("vf_" + k + "_ii_cycles") or {"rsqrt": 4.0, "exp": 4.0, "divide": 4.0}.get(k, 1.0)
            for k in c
        }
        service = {k: num * ii[k] for k, num in c.items()}
        bounds = {
            **service,
            "issue": sum(c.values()) / (d.get("vf_issue_width") or 2.0),
            "ub": (service.get("load", 0) + service.get("store", 0)) / (d.get("vf_ub_ports") or 2.0),
            "alu": sum(v for k, v in service.items() if k not in ("load", "store", "predicate"))
            / (d.get("vf_alu_lanes") or 2.0),
        }
        # One representative producer chain, with no multiplication by the number of
        # independent register groups. Row and column phases of Sinkhorn are serial.
        chain = {
            "mx4": ("load", "reduce", "convert", "permute", "store"),
            "mx8": ("load", "reduce", "convert", "store"),
            "antimx": ("load", "permute", "convert", "arithmetic", "store"),
            "rms_reduce": ("load", "convert", "arithmetic", "reduce", "store"),
            "rms_stat": ("load", "arithmetic", "rsqrt", "store"),
            "rms_norm": ("load", "convert", "arithmetic", "arithmetic", "convert", "store"),
            "post_mix": ("load", "convert") + ("arithmetic",) * n + ("store",),
            "cast_out": ("load", "convert", "store"),
            "pre_cast": ("load", "convert", "arithmetic", "reduce", "store"),
            "pre_mix": ("load", "convert", "arithmetic", "rsqrt", "arithmetic", "store"),
            "sink_normalize": ("load", "permute", "arithmetic", "arithmetic", "divide", "store"),
            "router": ("load", "exp", "arithmetic", "divide")
            + ("reduce", "permute") * (n if name == "router" else 1)
            + ("store",),
            "att_weights": ("load", "convert", "arithmetic", "store"),
            "att_prefix": ("load", "arithmetic", "store"),
            "att_score": (
                "load",
                "convert",
                "arithmetic",
                "reduce",
                "rsqrt",
                "arithmetic",
                "exp",
                "divide",
                "store",
            ),
        }[name]
        latency = {
            "load": 9.0,
            "store": 9.0,
            "convert": 7.0,
            "arithmetic": 7.0,
            "reduce": 16.0,
            "exp": 18.0,
            "divide": 17.0,
            "rsqrt": 17.0,
            "permute": 7.0,
        }
        bounds["dependency"] = sum(d.get("vf_" + k + "_latency_cycles") or latency[k] for k in chain)
        cycles = max(bounds.values())
        self.samples.setdefault(
            name,
            dict(
                tile_rows=rows,
                tile_width=width,
                parameter=n,
                counts=c,
                representative_chain=list(chain),
                bounds_cycles=bounds,
                cycles=cycles,
            ),
        )
        self.cache[key] = cycles
        return cycles


def phase_alternative(stages, price, chip_floor, old="SIMD"):
    out = {k: v for k, v in stages.items() if k not in (old, "shared_io", "shared_link_floor")}
    out["instruction_service"] = price
    out["exposed_shared"] = max(0.0, chip_floor - sum(out.values()))
    return out


def _pre(shape, chip, vf):
    from operator_model.operators.mhc import pre_dispatch

    branch, work = pre_dispatch.partition("pre", shape)
    phases = pre_dispatch.profile()["stages"]["pre"][branch]
    t, n, d = shape
    rows = ceil(t / chip.vector_cores)
    local_k = min(n * d, 256 * ceil(n * d / (chip.cube_cores * 256)))
    stages = {
        k: (p["startup_us"] + p["unit_us"] * work[p["unit"]]) * 1.65 / chip.frequency_ghz
        for k, p in phases.items()
        if k != "vector"
    }
    stages["vector_local_k_and_mixing"] = (
        vf("pre_cast", rows, local_k) + vf("pre_mix", rows, d, n)
    ) / vf.clock
    return stages


def _post(shape, attrs, chip, vf):
    from operator_model.operators.mhc.resources import resource_post
    from operator_model.operators.mhc.history import _scenarios

    _, aic, aiv, base = resource_post(shape, chip, attrs)
    p = base["target_plan"]
    rows = p["rows_per_core"]
    width = p["fragment_width"]
    n = shape[1]
    # SIMD intercept includes VF entry/synchronization on these disjoint occupancy phases.
    from operator_model.operators.mhc.resources import parameters

    coeff = parameters()["phases"][p["route"]]["SIMD"]
    phases = {
        s: phase_alternative(
            st,
            vf("post_mix", rows, width, n) / vf.clock + coeff[s] * 1.65 / chip.frequency_ghz,
            base["shared_floor_us"],
        )
        for s, st in base["state_phases_us"].items()
    }
    base = {
        **base,
        "state_phases_us": phases,
        "context_predictions_us": {s: sum(st.values()) for s, st in phases.items()},
    }
    if attrs.get("resident_aiv") is None:
        scenarios = _scenarios(shape, chip, base, vf)
        values = {k: v.latency_us for k, v in scenarios.items()}
        lo, hi = min(values.values()), max(values.values())
        stages = {"unknown_history_center": 2 * lo * hi / (lo + hi)}
        base.update(state_scenarios_us=values, state_range_us=[lo, hi])
    else:
        stages = phases[base["context"]]
    return sum(stages.values()), aic, aiv, {**base, "stages_us": stages}


def reprice(op, shape, attrs, target, dtype, result):
    """Replace only an eligible schedule; work-only/unsupported contracts remain unpriced."""
    from operator_model.backend import _rates

    if result.latency_us is None:
        return result
    attrs = dict(attrs or {})
    attrs.pop("dtype", None)
    attrs.pop("vector_model", None)
    vf = Pricer(target)
    clock = vf.clock
    p = result.projection or {}
    diag = p.get("timing_diagnostics", p)
    plan = p.get("target_plan", {})
    aic, aiv = result.active_aic, result.active_aiv
    if op == "quant":
        from operator_model.operators.quant.resources import resource_quant

        us, aic, aiv, diag = resource_quant(shape, target, vf=vf)
    elif op == "quant_mxfp4":
        from operator_model.operators.quant.mxfp4 import price

        us, _, diag = price(shape, target, attrs, vf=vf)
    elif op == "cache_load":
        if attrs.get("implementation") != "aclnnAntiMxQuant":
            return replace(
                result,
                status="unavailable",
                latency_us=None,
                rates=None,
                reason="No timed SIMD program for this cache implementation",
            )
        from operator_model.operators.vector.cache_load_antimx import schedule

        us, plan, _ = schedule(p["valid_rows"], shape[-1], result.work, target, vf=vf)
        diag = {"target_plan": plan, "stages_us": {"critical_pipeline": us}}
    elif op == "rmsnorm":
        from operator_model.operators.rmsnorm.pipeline import (
            _reference_prices,
            _per_tile_dag,
            RmsGeometry,
            _unknown_point,
        )

        if not plan:
            from operator_model.operators.rmsnorm.dispatch import price_rms

            _, aic, aiv, diag = price_rms(shape, target, attrs)
            plan = diag["target_plan"]
        prices = _reference_prices()
        geo = RmsGeometry(vector_register_bytes=int(target.details.get("vector_register_bytes") or 256))

        def state(h, rep):
            return _per_tile_dag(shape, target, h, rep, diag, prices, geo, False, vf)

        h = attrs.get("resident_aiv")
        pred = _unknown_point(aiv, state) if h is None else state(h, False)
        us = pred["us"]
        diag = dict(target_plan=plan, issue_graph=pred, stages_us={"issue_graph": us})
    elif op == "mhc_post":
        if attrs.get("implementation") == "vllm_hc_post":
            from operator_model.operators.mhc.vllm_post import resource_hc_post

            us, _, diag = resource_hc_post(shape, target, attrs, vf_price=vf)
        else:
            us, aic, aiv, diag = _post(shape, attrs, target, vf)
    elif op == "mhc_sinkhorn":
        if attrs.get("resident_aiv") is None:
            from operator_model.operators.mhc.sink_unknown import price

            attrs.pop("resident_aiv", None)
        else:
            from operator_model.operators.mhc.sink_pipeline import price
        us, aic, aiv, diag = price(shape, target, attrs, vf=vf)
    elif op in ("mhc", "mhc_pre"):
        stages = _pre(shape, target, vf)
        if op == "mhc":
            from operator_model.operators.mhc.sink_unknown import price

            s = price(shape[:2], target, {"num_iters": attrs.get("num_iters", 20)}, vf=vf)[0]
            if s is None:
                return replace(
                    result,
                    latency_us=None,
                    status="unavailable",
                    rates=None,
                    reason="No timed Sinkhorn child",
                )
            stages.update(sinkhorn=s, post=_post(shape, {}, target, vf)[0])
        us = sum(stages.values())
        diag = {"target_plan": plan, "stages_us": stages, "schedule": "split kernel budgets"}
    elif op == "mhc_pre_sinkhorn":
        from operator_model.operators.mhc.pre_sinkhorn import _cost, fused
        from operator_model.chips import load_chip

        t, n, d = shape
        st = dict(diag["stages_us"])
        ref = load_chip()
        # The alternative team layout has no isolated SIMD-only CA window. Use the
        # existing K-split work plan as an explicit implementation hypothesis.
        # The L1/L0 fragment calculation consumes one busy-core M panel.
        # Reserve the same 128-row panel bound as the extended system schedule.
        # Launch and the separate post team are paid once outside this loop.
        local_rows = ceil(t / plan["m_blocks"])
        panels, tail = divmod(local_rows, 128)
        cube_us = cast_us = 0.0
        rd = target.details.get("mte2_bandwidth_gbs")
        for count, height in ((panels, 128), (int(tail > 0), tail)):
            if count:
                cost, units = _cost((height, n, d), target, plan, ref)
                elements = units["critical_aiv_stage1_elements"]
                cube_us += count * cost["projection"]
                cast_us += count * max(vf("pre_cast", 1, elements) / clock,
                                       2 * elements / (rd * 1000) if rd else 0.0)
        if "dispatch_and_handoff" not in st:
            st = {
                "dispatch_and_handoff": fused((1, 4, 4096), ref, {})[3]["stages_us"]["dispatch_and_handoff"]
                * ref.frequency_ghz
                / target.frequency_ghz
            }
        rows = plan["rows_per_aiv"]
        st.update(
            stage1_cube=cube_us,
            stage1_cast=cast_us,
            stage2_rows=(
                vf("pre_mix", rows, d, n) + 2 * attrs.get("num_iters", 20) * vf("sink_normalize", rows, n)
            )
            / clock,
        )
        st.pop("exposed_hbm_floor", None)
        byte = 2 * t * n * d + 4 * n * d * (n * n + 2 * n) + 2 * t * d
        st["exposed_hbm_floor"] = max(0.0, byte / (target.hbm_bandwidth_gbs * 1000) - sum(st.values()))
        us = sum(st.values())
        diag = dict(target_plan=plan, stages_us=st, schedule="K-split and joint normalization hypothesis")
    elif op == "attn_res_fwd":
        from operator_model.operators.attention.residual_timing import resource_attention

        us, _, diag = resource_attention(shape, target, attrs, vf=vf)
    elif op == "moe_gating_top_k":
        t, e, k = shape
        st = dict(diag["stages_us"])
        st["rank_and_normalize"] = vf("router", plan["rows_per_busy_core"], e, k) / clock
        us = sum(st.values())
        diag = {"target_plan": plan, "stages_us": st}
    else:
        raise ValueError("No native SIMD pricing program: " + op)
    if us is None:
        return replace(
            result, latency_us=None, rates=None, status="unavailable", reason="No eligible physical schedule"
        )
    diag = dict(diag, vector_model="instruction_classes", instruction_samples=vf.samples)
    projection = dict(
        method="instruction_class_schedule",
        vector_model="instruction_classes",
        latency_us=us,
        stages_us=diag.get("stages_us", {}),
        target_plan=diag.get("target_plan", {}),
        timing_diagnostics=diag,
        instruction_samples=vf.samples,
        inherited_fields=list(target.inherited_fields),
    )
    if "implementation" in p:
        projection["implementation"] = p["implementation"]
    return replace(
        result,
        latency_us=us,
        status="extrapolated",
        reason="Instruction-class SIMD program; issue ports/lowering not independently calibrated",
        active_aic=aic,
        active_aiv=aiv,
        projection=projection,
        rates=_rates(result.work, us, target),
    )
