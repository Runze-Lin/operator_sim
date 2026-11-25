"""Register-program alternatives for native kernels. Costs are in busy-AIV cycles.

The program is an implementation hypothesis, not a disassembly. The same tiler,
launch/DMA costs, resource floors and kernel schedule are reused by both prices.
"""

from dataclasses import replace
from math import ceil
from statistics import median

OPS = {
    "matmul",
    "addmm",
    "gmm2",
    "routed_grouped_matmul",
    "lightning_indexer",
    "candidate_filtered_indexer",
    "grouped_matmul_swiglu_quant_v2",
    "moe_ffn",
    "kda_recurrent",
    "kda_chunk",
}


class Programs:
    def __init__(self, chip):
        d = chip.details
        self.reg_bytes = int(d.get("vector_register_bytes") or 256)
        self.fp32 = self.reg_bytes // 4
        self.ii = {
            k: d.get("vf_" + k + "_ii_cycles") or v
            for k, v in dict(
                arithmetic=1, reduce=1, exp=4, divide=4, load=1, store=1, convert=1, predicate=1
            ).items()
        }
        self.lat = {
            k: d.get("vf_" + k + "_latency_cycles") or v
            for k, v in dict(
                arithmetic=7, reduce=16, exp=18, divide=17, load=9, store=9, convert=7, predicate=1
            ).items()
        }
        self.issue = d.get("vf_issue_width") or 2
        self.alu = d.get("vf_alu_lanes") or 2
        self.ub = d.get("vf_ub_ports") or 2
        self.records = {}

    def phase(self, name, counts, chain):
        # FP32 RegBase producer-chain priors inherited from the FA probe;
        # predicate latency is a one-cycle assumption, not a measurement.
        demand = {k: n * self.ii[k] for k, n in counts.items()}
        bounds = {
            "issue": sum(counts.values()) / self.issue,
            "ub": (demand.get("load", 0) + demand.get("store", 0)) / self.ub,
            "store": demand.get("store", 0),
            "alu": sum(v for k, v in demand.items() if k not in ("load", "store", "predicate")) / self.alu,
            "port": max(demand.values(), default=0),
            "dependency": sum(self.lat[k] * n for k, n in chain.items()),
        }
        cycles = max(bounds.values())
        self.records[name] = dict(counts=counts, chain=chain, bounds_cycles=bounds, cycles=cycles)
        return cycles

    def unpack(self, rows, depth):
        # One packed 256-B register -> two expanded FP8 registers. Stride is per weight row.
        g = rows * ceil(depth / (2 * self.reg_bytes))
        return self.phase(
            "W4 unpack %dx%d" % (rows, depth),
            dict(load=g, predicate=2 * g, arithmetic=4 * g, convert=2 * g, store=2 * g),
            dict(load=1, arithmetic=2, convert=1, store=1),
        )

    def score(self, keys, heads, rows=1, mask=False):
        g = ceil(keys / self.fp32)
        x = heads * g
        return rows * self.phase(
            "score %dx%d mask=%d" % (keys, heads, mask),
            dict(load=x + g, arithmetic=3 * x - g, predicate=(2 * g if mask else g), store=g),
            dict(load=1, arithmetic=2 + (heads - 1).bit_length(), store=1),
        )

    def hist(self, entries):
        # Eight uint16 radix passes per 256-key group (as in indexer._sort_row).
        # The implementation-specific histogram/permutation primitive throughput is unknown.
        blocks = ceil(entries / 256)
        count = 8 * blocks
        return self.phase(
            "hist256 x%d" % blocks,
            dict(
                load=ceil(512 / self.reg_bytes) * count,
                predicate=count,
                arithmetic=2 * count,
                reduce=count,
                store=count,
            ),
            dict(load=1, arithmetic=2, reduce=1, store=1),
        )

    def sort_row(self, n):
        scans = ceil(n / 8192)
        last = (n - 1) % 8192 + 1
        if scans == 1:
            return self.hist(n)
        return self.hist(8192) + (scans - 2) * self.hist(8704) + self.hist(last + 512)

    def sort_map(self, k, compact):
        levels = (k - 1).bit_length()
        layers = levels * (levels + 1) // 2
        g = ceil((1 << levels) / self.fp32)
        # Each compare/select layer consumes the previous layer's output.
        return layers * self.phase(
            "compare-exchange %d" % k, dict(load=g, arithmetic=3 * g, store=g), dict(arithmetic=2)
        ) + self.phase(
            "remap %d" % k,
            dict(load=g, arithmetic=(4 if compact else 2) * g, store=g),
            dict(load=1, arithmetic=1, store=1),
        )

    def swiglu(self, n):
        g = ceil(n / self.fp32)
        scales = ceil(n / 32)
        return self.phase(
            "swiglu-quant %d" % n,
            dict(
                load=2 * g,
                exp=g,
                divide=2 * g,
                arithmetic=5 * g,
                reduce=g,
                convert=g,
                store=ceil(n / self.reg_bytes) + ceil(scales / self.reg_bytes),
            ),
            dict(load=1, exp=1, divide=2, arithmetic=2, reduce=1, convert=1, store=1),
        )

    def recurrent(self, s, k, v):
        rows = k * ceil(v / self.fp32)
        vv = ceil(v / self.fp32)
        kk = ceil(k / self.fp32)
        # Decay, K/state dot, corrected V, rank-one update, Q/state dot in that order.
        matrix = self.phase(
            "state update",
            dict(load=4 * rows, arithmetic=4 * rows, reduce=2 * rows, store=rows),
            dict(load=1, arithmetic=3, reduce=2, store=1),
        )
        return s * (
            matrix
            + self.phase(
                "K gate and V correction",
                dict(load=3 * kk + 2 * vv, exp=kk, arithmetic=2 * vv, convert=2 * kk + 2 * vv, store=vv),
                dict(load=1, exp=1, arithmetic=2, store=1),
            )
        )

    def fla(self, family, i, n):
        # Each descriptor maps to a separate VF in the FLA source schedule.
        if family == "chunk_fwd_h":
            vals = (16384, n * 128, 16384)[i]
            mix = (
                {"load": 1, "convert": 1, "store": 1},
                {"load": 2, "arithmetic": 1, "convert": 1, "store": 2},
                {"load": 3, "exp": 1, "arithmetic": 2, "convert": 1, "store": 1},
            )[i]
        else:
            vals = (
                128 * n,
                128 * (n + sum(min(n, j * 16) for j in range(1, ceil(n / 16) + 1))),
                2 * (31 * 32 // 2) * 32,
                128 * n,
            )[i]
            mix = (
                {"load": 3, "arithmetic": 3, "exp": 1, "store": 3},
                {"load": 2, "arithmetic": 2, "convert": 1, "store": 2},
                {"load": 2, "arithmetic": 2, "reduce": 1, "store": 1},
                {"load": 3, "arithmetic": 2, "convert": 2, "store": 2},
            )[i]
        counts = {k: v * ceil(vals / self.fp32) for k, v in mix.items()}
        return self.phase("%s/%d rows%d" % (family, i, n), counts, mix), counts


def _overlap(op, shape, call, target, vf_us):
    """Replace the SIMD budget inside each donor core's concurrent-queue masks."""
    from operator_model.chips import load_chip
    from operator_model.scheduling.projection import _queue_factors
    from operator_model.operators.fused import templates
    from operator_model.workload import account_work
    from operator_model.scheduling.reference import _profile, predict

    demand = templates.work(op, shape, call)
    raw = predict(op, shape, call)
    f, _ = _queue_factors(op, shape, call, raw, account_work(op, shape, call), load_chip(), target, [])
    candidates = []
    for ref in _profile("fusion_overlap")["templates"][op]:
        ratios = {q: f.get(q, f["SCALAR"]) * n / ref["work"][q] for q, n in demand.items()}
        # Normalize against the slowest vector-bearing core, not against the
        # total across AICs and AIVs. Other donors retain relative SIMD shares.
        vf_max = max(
            sum(us for mask, us in core.items() if "SIMD" in mask.split("+")) for core in ref["cores"]
        )
        if vf_max <= 0:
            raise ValueError("Overlap template has no identifiable VF interval")
        ratios["SIMD"] = vf_us / vf_max
        candidates.append(
            max(
                sum(us * max(ratios[q] for q in mask.split("+")) for mask, us in core.items())
                for core in ref["cores"]
            )
        )
    return median(candidates)


def reprice(op, shape, attrs, target, dtype, result):
    """Same scheduling contract; replace VF at its producer site, never fit elapsed time."""
    from operator_model.backend import _call_attrs, _rates
    from operator_model.scheduling.reference import predict
    from operator_model.operators.attention import indexer, candidate_indexer, flash_sparse
    from operator_model.operators.fused import templates

    call, chosen = _call_attrs(op, {k: v for k, v in (attrs or {}).items() if k != "vector_model"}, dtype)
    p = Programs(target)
    shape = tuple(shape)
    clock = 1000 * target.frequency_ghz
    diag = dict(result.projection or {})
    work = result.work
    if result.latency_us is None:
        return replace(
            result,
            projection={
                **diag,
                "vector_model": "instruction_classes",
                "vector_model_scope": "unpriced contract; no timing synthesized",
            },
        )
    if op in ("addmm", "gmm2", "routed_grouped_matmul") or (op == "matmul" and (chosen != "fp8_e4m3fn" or work.get("no_vector_engine"))):
        return replace(
            result,
            projection={
                **diag,
                "vector_model": "not_applicable",
                "vector_model_scope": "No VF on this Cube/FIXP implementation; identical schedule",
            },
        )
    if op == "matmul":
        from operator_model.operators.matmul.mx import price

        us, work, diag, _ = price(shape, target, chosen, call, programs=p)
        diag["price_sources"]["vector"] = "packed W4 register expansion class hypothesis"
        work["implementation_work"]["unpack_peak_equivalents_per_weight"] = None
        plan = diag["target_plan"]
        for key in ("vf_row_issue_cycles_950", "vf_packed_repeat_issue_cycles_950", "vf_register_bytes_950"):
            plan.pop(key, None)
        plan["vf_register_bytes"] = p.reg_bytes
        diag["prices"]["weights_per_cycle_per_aiv"] = None  # width/tail dependent; see each register program
    elif op == "lightning_indexer" and indexer.applies(shape, call, target):
        us, plan, info = indexer.schedule(shape, target, programs=p)
        diag.update(target_plan=plan, timing_diagnostics=info, stages_us=info["stages_us"])
    elif op == "candidate_filtered_indexer":
        if call.get("implementation") == flash_sparse.IMPLEMENTATION:
            us, work, diag = flash_sparse.price(shape, target, programs=p, **flash_sparse.options(call))
        else:
            us, work, diag = candidate_indexer.price(shape, target, call, programs=p)
    elif op == "kda_chunk":
        from operator_model.operators.kda.fla import resource

        us, _, _, work, diag = resource(shape, target, call, programs=p)
        diag["legacy_prediction_freeze_sha256"] = diag.pop("prediction_freeze_sha256", None)
    elif op == "kda_recurrent":
        from operator_model.chips import load_chip
        from operator_model.scheduling.projection import project
        from operator_model.workload import account_work

        b, s, hq, hv, k, v = shape
        one = (1, s, 1, 1, k, v)
        # Single-chain template contracts are fixed and validated upstream.
        raw = predict(op, one, {})
        ref = load_chip()
        st = dict(
            raw["stages_us"]
            if target == ref
            else project(op, one, {}, raw, account_work(op, one), ref, target)["stages_us"]
        )
        st["SIMD"] = p.recurrent(s, k, v) / clock
        us = sum(st.values())
        chain = diag.get("timing_diagnostics") or {}
        if "chains_per_busiest_aiv" in chain:
            prefix, drain = chain["once_prefix_us"], chain["once_drain_us"]
            rounds = chain["chains_per_busiest_aiv"]
            body = max(0.0, us - prefix - drain)
            shared = chain["external_service_floor_us"]
            diag["instruction_chain_us"] = st
            local = prefix + rounds * body + drain
            us = max(local, shared)
            # Keep the original prefetch sensitivity bounds explicitly separate.
            diag["legacy_timing_diagnostics"] = diag.pop("timing_diagnostics")
            diag["timing_diagnostics"] = dict(
                single_chain_us=sum(st.values()),
                chains_per_busiest_aiv=rounds,
                per_chain_body_us=body,
                external_service_floor_us=shared,
            )
            st = dict(
                prefix=prefix,
                repeated_chain=rounds * body,
                drain=drain,
                exposed_shared=max(0.0, shared - local),
            )
        diag.update(stages_us=st)
    elif op in (templates.FIRST, templates.CHAIN, "lightning_indexer"):
        if op == "lightning_indexer":
            _, q, l, h, _ = shape
            live = min(q, target.vector_cores)
            vf = ceil(q / live) * (p.score(l, h) + p.sort_row(l)) / clock
            us = _overlap(op, shape, call, target, vf)
            diag["stages_us"] = {op: us}
        else:
            _, _, width, _ = shape
            counts = call["expert_counts"]
            jobs = sum(ceil(n * width / 256) for n in counts if n)
            vf = ceil(jobs / target.vector_cores) * p.swiglu(min(max(counts) * width, 256)) / clock
            first = _overlap(templates.FIRST, shape, call, target, vf)
            stages = {templates.FIRST: first}
            if op == templates.CHAIN:
                old = diag.get("stages_us") or predict(op, shape, call)["stages_us"]
                stages[templates.SECOND] = old[templates.SECOND]
            us = sum(stages.values())
            diag["stages_us"] = stages
        diag["busy_vector_work_us"] = vf
    else:
        raise ValueError("Missing instruction-program contract: " + op)
    diag.update(
        latency_us=us,
        vector_model="instruction_classes",
        timing_qualification="extrapolated",
        instruction_programs=p.records,
        vector_model_scope="Per-template register/issue proxy; no new calibration",
    )
    return replace(
        result,
        latency_us=us,
        status="extrapolated",
        work=work,
        projection=diag,
        rates=_rates(work, us, target),
        reason="Experimental register-class schedule: source-informed program, inherited 950 class rates",
    )
