"""Reprice the 33 frozen 950DT rank-0 request graphs under QSLI locality scenarios.

The graph stores shape/control metadata and no candidate addresses. Default calls
omit locality input. Named scenarios apply operation attributes without changing
any other node's price, resource reservation or dependency.
"""

import argparse
import csv
import gzip
import json
import math
import pickle
from dataclasses import replace
from functools import partial
from hashlib import sha256
from pathlib import Path

from calibration.candidate_indexer.context_sensitivity import (
    CANDIDATE,
    anchor_error,
    bundle,
    caps,
    geometry,
    hardware,
    integrate,
    jobs,
    page_table,
    quote,
)
from calibration.candidate_indexer.v112_context import HERE, LANES, PHASES, check, key, v111
from calibration.context_report import write
from calibration.context_sweep import case_id, inputs, profiles

DIGESTS = {
    "requests": "8c5fff59288b9719386538cdbbfb56131673c41b4b86eede19fda3a2438fcb5d",
    "anchors": "961642c1ebb5f692229ed55b47661d659bc5103204942ee4c1d4df29acf3b2f2",
}
LAST = ("v41-flash", 1048576, 1, 32, "prefill", 1044480, 4096)


def v112(name):
    p = HERE / ("v112_context_" + name + ".csv")
    check(sha256(p.read_bytes()).hexdigest() == DIGESTS[name], f"Modified v1.12 {name}")
    with p.open() as stream:
        return list(csv.DictReader(stream))


def price(shape, attrs, target, hw):
    from operator_model import estimate

    est = estimate(
        "candidate_filtered_indexer", shape, attrs, chip=target, dtype="mxfp4", memory_policy="cold_io"
    )
    ref = est.rates["logical_roofline"]
    cube, vec, memory = (ref[k] for k in ("cube_us", "vector_us", "logical_io_us"))
    peak = max(cube, vec, memory)
    check(
        all(math.isfinite(v) and v >= 0 for v in (cube, vec, memory, est.latency_us))
        and est.latency_us >= peak * (1 - 1e-12),
        f"Invalid aligned quote for {shape}",
    )
    effective = max(
        cube / hw.chip.cube.utilization.ratio,
        vec / hw.chip.vec.utilization.ratio,
        memory / hw.chip.hbm.utilization.ratio,
    )
    return est.latency_us, peak, effective


def replay(spec, scratch, cache, hypotheses):
    from operator_model.sim_backend import OperatorSimBackend
    from sim_platform.backend.l1 import simulate_graph

    record = quote(scratch, spec)
    if not record["operators"].get(CANDIDATE):
        return [
            dict(
                full=stage["full_us"],
                peak=stage["peak_us"],
                effective=stage["effective_us"],
                **{k: stage["full_us"] for k in hypotheses},
            )
            for stage in record["stages"]
        ]
    workload, opts, _, _ = bundle(scratch, spec)
    hw = hardware(opts)
    target = OperatorSimBackend()._resolve_chip(hw.name, hw.chip)
    capacity, pages = page_table(scratch, spec)
    attrs = {"implementation": "a5_qsli_projection", "pa_block_size": 128, "table_pages": pages}
    with gzip.open(cache / (case_id(spec) + ".gz"), "rb") as stream:
        cached = pickle.load(stream)
    stages = []
    prices = {}
    for graph, nodes in zip(workload.graphs, cached, strict=True):
        check(len(graph.nodes) == len(nodes), "Cached and captured graph have different sizes")
        costs = {k: {} for k in (*LANES, *hypotheses)}
        subtotals = {k: 0.0 for k in ("full", *hypotheses)}
        owners = []
        for n in graph.nodes:
            fine, peak, effective, candidate = nodes[n.ref.node_id]
            check(
                (candidate is not None) == (n.raw_name == CANDIDATE),
                "Capture and cached candidate identities differ",
            )
            if candidate is None:
                node_prices = dict(
                    full=fine.duration_us,
                    peak=peak,
                    effective=effective,
                    **{k: fine.duration_us for k in hypotheses},
                )
            else:
                shape, _ = candidate
                check(shape == geometry(n), "Captured geometry changed")
                owners.append(shape)
                if shape not in prices:
                    full, peak, effective = price(shape, attrs, target, hw)
                    out = dict(full=full, peak=peak, effective=effective)
                    for k, extra in hypotheses.items():
                        scenario, scenario_peak, scenario_eff = price(shape, attrs | extra, target, hw)
                        check(
                            (scenario_peak, scenario_eff) == (peak, effective),
                            "Locality changed the logical contract",
                        )
                        out[k] = scenario
                    prices[shape] = out
                node_prices = prices[shape]
                for k in subtotals:
                    subtotals[k] += node_prices[k]
            for lane, value in node_prices.items():
                costs[lane][n.ref] = replace(fine, duration_us=value)
        stage = {k: simulate_graph(graph, c, caps(opts)).duration_us for k, c in costs.items()}
        stage.update(
            candidates=subtotals,
            nodes=len(owners),
            shape=";".join(sorted({"x".join(map(str, s)) for s in owners})),
            kv_capacity=capacity,
            table_pages=pages,
        )
        stages.append(stage)
    check(sum(r["nodes"] for r in stages) == record["operators"][CANDIDATE], "Incomplete candidate census")
    return stages


def sweep(args, hypotheses):
    previous = {key(row): row for row in v112("requests")}
    original = {key(row): row for row in v111("requests")}
    archived = {(key(row), row["phase"], row["stage"], int(row["history"])): row for row in v112("anchors")}
    specs = [s for profile in profiles() for s in inputs(profile)]
    fn = partial(replay, scratch=args.scratch, cache=args.cache, hypotheses=hypotheses)
    samples = dict(zip(specs, jobs(fn, specs, args.workers), strict=True))
    requests, anchors = [], []
    lanes = (*LANES, *hypotheses)
    for profile in profiles():
        model, prompt, batch, ep, chunk = profile
        old, ref = previous[profile[:4]], original[profile[:4]]
        row = {
            k: old[k]
            for k in (
                "model",
                "prompt",
                "output",
                "batch",
                "tp",
                "ep",
                "decode_sampling",
                "decode_positions",
                "fits_144GB",
            )
        }
        for phase in PHASES[:2]:
            curves = {k: {} for k in lanes}
            hits = 0
            for spec in (s for s in inputs(profile) if s[4] == phase):
                pos = spec[5] // chunk if phase == "prefill" else spec[5]
                for i, stage in enumerate(samples[spec]):
                    for k in lanes:
                        curves[k][pos] = curves[k].get(pos, 0.0) + stage[k]
                    if stage.get("nodes"):
                        name = "prefill" if phase == "prefill" else ("target", "draft")[i]
                        old_stage = archived[(profile[:4], phase, name, spec[5])]
                        check(
                            int(old_stage["candidate_nodes"]) == stage["nodes"]
                            and old_stage["shapes"] == stage["shape"],
                            "Captured shape or multiplicity changed",
                        )
                        hits += stage["nodes"]
                        a = dict(
                            model=model,
                            prompt=prompt,
                            batch=batch,
                            ep=ep,
                            phase=phase,
                            stage=name,
                            history=spec[5],
                            shapes=stage["shape"],
                            kv_capacity=stage["kv_capacity"],
                            table_pages=stage["table_pages"],
                            candidate_nodes=stage["nodes"],
                        )
                        a.update(
                            v112_candidate_us=float(old_stage["v112_candidate_us"]),
                            v112_stage_us=float(old_stage["v112_stage_us"]),
                        )
                        a.update({"v113_" + k + "_candidate_us": v for k, v in stage["candidates"].items()})
                        a.update({"v113_" + k + "_stage_us": stage[k] for k in lanes})
                        anchors.append(a)
            row[phase + "_candidate_nodes"] = hits
            for k in lanes:
                row[phase + "_v113_" + k + "_s"] = integrate(curves[k], phase, profile)
            row[phase + "_anchor_error_pct"] = anchor_error(curves["full"], phase, prompt, chunk)
        for k in lanes:
            row["request_v113_" + k + "_s"] = sum(row[p + "_v113_" + k + "_s"] for p in PHASES[:2])
        for phase in PHASES:
            row[phase + "_v112_full_s"] = float(old[phase + "_v112_full_s"])
            row[phase + "_legacy_whole_s"] = float(ref[phase + "_legacy_whole_s"])
            row[phase + "_reference_bf16_s"] = float(ref[phase + "_reference_full_s"])
            row[phase + "_candidate_only_legacy_s"] = float(ref[phase + "_restore_candidate_s"])
            row[phase + "_delta_us"] = 1e6 * (row[phase + "_v113_full_s"] - row[phase + "_v112_full_s"])
            for k in ("peak", "effective"):
                check(
                    row[phase + "_v113_" + k + "_s"] == float(old[phase + "_v112_" + k + "_s"]),
                    "Aligned work changed",
                )
            if model != "v41-flash":
                check(
                    all(
                        row[phase + "_v113_" + k + "_s"] == row[phase + "_v112_full_s"]
                        for k in ("full", *hypotheses)
                    ),
                    "Non-Flash workload changed",
                )
        requests.append(row)
    check(len(requests) == 33 and len(archived) == len(anchors) == 138, "Incomplete v1.13 sweep")
    if args.initial:
        write(args.output / "v113_context_initial_requests.csv", requests)
        write(args.output / "v113_context_initial_anchors.csv", anchors)
        return

    # v1.13.0 snapshots precede all three controls. v1.13.1 is frozen after
    # the shared/rotated timing review; preserve their distinct data roles.
    digests = {
        "requests": "856d79e342eb6d1a94ba1aa4e0c8b394743272f02604ce759c5bdfb9e21879b2",
        "anchors": "64ae005b642d09ccfd417ec2240b312fbc420c3671d966ccc856f6149deb1865",
    }
    for name, totals in (("requests", requests), ("anchors", anchors)):
        path = HERE / f"v113_context_initial_{name}.csv"
        check(sha256(path.read_bytes()).hexdigest() == digests[name], f"Initial {name} snapshot changed")
        with path.open() as stream:
            for cur, first in zip(totals, csv.DictReader(stream), strict=True):
                check(key(cur) == key(first), "Profile ordering changed")
                if name == "requests":
                    for phase in PHASES:
                        old_s = float(first[phase + "_v113_full_s"])
                        cur[phase + "_initial_full_s"] = old_s
                        cur[phase + "_delta_from_initial_us"] = 1e6 * (cur[phase + "_v113_full_s"] - old_s)
                else:
                    check(
                        cur["stage"] == first["stage"] and cur["history"] == int(first["history"]),
                        "Captured anchor ordering changed",
                    )
                    for part in ("stage", "candidate"):
                        cur["initial_full_" + part + "_us"] = float(first["v113_full_" + part + "_us"])
        write(args.output / f"v113_context_{name}.csv", totals)


def last_chunk_prices(scratch, cache, amounts):
    """Apply each single-node quote to all four consumers and rerun resource arbitration."""
    from sim_platform.backend.l1 import simulate_graph

    work, opts, _, _ = bundle(scratch, LAST)
    (graph,) = work.graphs
    with gzip.open(cache / (case_id(LAST) + ".gz"), "rb") as stream:
        (stored,) = pickle.load(stream)
    other = {node.ref: stored[node.ref.node_id][0] for node in graph.nodes}
    owners = [node.ref for node in graph.nodes if node.raw_name == CANDIDATE]
    check(
        len(owners) == 4
        and all(stored[n.node_id][3][0] == (1, 4096, 32, 128, 1048576, 2048, 512) for n in owners),
        "Last prefill graph must include four identical candidate consumers",
    )
    durations = []
    for amount in amounts:
        costs = other.copy()
        for ref in owners:
            costs[ref] = replace(costs[ref], duration_us=amount)
        durations.append(simulate_graph(graph, costs, caps(opts)).duration_us)
    return durations


def waves(args):
    arm_names = {"shared", "rotated"} if args.order_contrast_only else {"shared", "moving", "rotated"}
    check(len(args.receipts) == len(arm_names), "Supply one DT Q64 receipt per requested arm")
    rows = []
    hashes = set()
    seen_arms = set()
    freeze_digest = sha256((HERE / "v113_frozen.json").read_bytes()).hexdigest()
    startup = 8568 / 1650  # DT ordinary-QLI startup proxy; not a measured multiwave intercept.
    for pathname in args.receipts:
        rec = json.loads(pathname.read_text())
        arm = rec["arm"]
        check(arm in arm_names and arm not in seen_arms, f"Repeated or unknown control {arm}")
        seen_arms.add(arm)
        fixture = HERE / f"v113_control_{arm}_fixture.json"
        facts = json.loads(fixture.read_text())
        check(
            rec["shape"] == [1, 64, 32, 128, 1048576, 2048, 512]
            and rec["sku"] == "Ascend950DT_9582"
            and facts["table_pages"] == 8201
            and rec["cpu_control_sha256"] == sha256(fixture.read_bytes()).hexdigest()
            and rec["freeze_sha256"] == freeze_digest
            and rec["qualified_end_events"] == 96
            and rec["all_ordered_lane_streams_match"]
            and rec["mmad_count"] == 2048
            and rec["golden_positions"] == 64 * 512
            and rec["status"] == "qualified full device window",
            "Only complete 1Mi/Q64 DT controls are comparable",
        )
        hashes.add(rec["trace_set_sha256"])
        for mode, credit in (("each_segment", 0.0), ("first_only", startup)):
            amount = 64 * rec["device_us"] - 63 * credit
            rows.append(
                dict(
                    fixture=arm,
                    candidate_order_overlap=facts["all_unordered_row_pairs_same_rank_fraction"],
                    receipt=pathname.name,
                    trace_set_sha256=rec["trace_set_sha256"],
                    measured_rows=64,
                    measured_device_us=rec["device_us"],
                    target_rows=4096,
                    segments=64,
                    startup=mode,
                    credit_per_additional_segment_us=credit,
                    projected_single_node_us=amount,
                )
            )
    check(len(hashes) == len(arm_names), "Each receipt must describe an independent capture")
    results = last_chunk_prices(args.scratch, args.cache, [r["projected_single_node_us"] for r in rows])
    for row, duration in zip(rows, results, strict=True):
        row["des_last_chunk_us"] = duration
    filename = (
        "v113_context_order_pair.csv" if args.order_contrast_only else "v113_context_measured_waves.csv"
    )
    write(args.output / filename, rows)


def warm_rows(args):
    """Replay the independent per-AIC extrapolations on the exact final chunk graph."""
    source = HERE / "v113_warm_rows.json"
    source_hash = sha256(source.read_bytes()).hexdigest()
    arms = json.loads(source.read_text())["arms"]
    check(
        {a["arm"] for a in arms} == {"shared", "rotated", "moving"} and len(arms) == 3,
        "Incomplete warm-row cases",
    )
    output = []
    for arm in arms:
        name = arm["arm"]
        receipt = json.loads((HERE / f"v113_control_{name}.json").read_text())
        check(
            arm["source_trace"] == receipt["trace_set_sha256"]
            and arm["observed_q64_us"] == receipt["device_us"],
            f"The {name} warm-row scenario must use the matching measured arm",
        )
        rows = arm["per_team_cycles"]
        check(
            len({r["core"] for r in rows}) == 32 and arm["observed_tail_cycles"] >= 0,
            "All native teams and the measured nonnegative final drain are required",
        )
        ticks = max(r["final_retire"] + 126 * (r["second_row_span"] + r["row_transition"]) for r in rows)
        projected = (ticks - arm["origin"] + arm["observed_tail_cycles"]) / receipt["frequency_cycles_per_us"]
        check(projected == arm["projected_q4096_us"], f"Cannot reproduce per-team projection for {name}")
        output.append(
            dict(
                fixture=name,
                source_table_sha256=source_hash,
                trace_set_sha256=arm["source_trace"],
                q64_ca_us=arm["observed_q64_us"],
                critical_aic=arm["critical_aic"],
                additional_rows_per_team=126,
                projected_single_node_us=projected,
            )
        )
    durations = last_chunk_prices(args.scratch, args.cache, [r["projected_single_node_us"] for r in output])
    for result, duration in zip(output, durations, strict=True):
        result["des_last_chunk_us"] = duration
    write(args.output / "v113_context_warm_rows.csv", output)


def check_freeze(initial):
    import operator_model

    freeze = json.loads((HERE / ("v113_frozen.json" if initial else "v113b_frozen.json")).read_text())
    package = Path(operator_model.__file__).parent
    check(operator_model.__version__ == freeze["version"], "Run this release with its frozen model")
    for relative, expected in freeze["model_sha256"].items():
        check(
            sha256((package / relative).read_bytes()).hexdigest() == expected,
            f"Modified frozen file {relative}",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["sweep", "waves", "warm"])
    parser.add_argument("scratch", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--receipts", nargs="+", type=Path)
    parser.add_argument(
        "--order-contrast-only",
        action="store_true",
        help="Use only the named shared and rotated receipts; save a separate pair report",
    )
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--initial", action="store_true", help="Reproduce the first freeze using v1.13.0")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    if args.mode == "waves":
        if not args.receipts:
            parser.error("waves requires --receipts")
        waves(args)
    elif args.mode == "warm":
        warm_rows(args)
    else:
        check_freeze(args.initial)
        hypotheses = json.loads((HERE / "v113_context_hypotheses.json").read_text())
        for arm in ("shared", "moving", "rotated"):
            inputs = json.loads((HERE / f"v113_control_{arm}_fixture.json").read_text())
            check(
                inputs["all_unordered_row_pairs_same_rank_fraction"]
                == hypotheses[arm]["candidate_order_overlap"],
                f"Scenario and fixture pairwise overlap disagree for {arm}",
            )
        sweep(args, hypotheses)


if __name__ == "__main__":
    main()
