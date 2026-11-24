"""Replay only Flash QSLI candidate nodes; keep v1.11 archives read-only.

Requires the trusted graph/cache directories from context_sensitivity.prepare.
Phase and request totals use the original study's grid and interpolation.
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
    near,
    page_table,
    quote,
)
from calibration.context_report import write
from calibration.context_sweep import case_id, inputs, profiles

HERE = Path(__file__).parent
ARCHIVES = {
    "requests": "ccdfc35a9fe80fc7d585f706ba2ffa7f0ed40bb30e6b3fd0f7cfa6c8bc0f04d0",
    "flash-anchors": "c2c2caa23aed31c51b24ae436a30c77301ea0eed1acc78c2910e4ef18bb59d70",
}
LANES = ("full", "peak", "effective")
PHASES = ("prefill", "decode", "request")


def check(ok, message):
    if not ok:
        raise ValueError(message)


def v111(name):
    p = HERE / ("context-sensitivity-" + name + ".csv")
    check(sha256(p.read_bytes()).hexdigest() == ARCHIVES[name], "v1.11 baseline changed: " + str(p))
    with p.open() as f:
        return list(csv.DictReader(f))


def replay(spec, scratch, cache):
    from operator_model import estimate
    from operator_model.sim_backend import OperatorSimBackend
    from sim_platform.backend.l1 import simulate_graph

    q = quote(scratch, spec)
    if not q["operators"].get(CANDIDATE):
        return [{k: row[k + "_us"] for k in LANES} for row in q["stages"]]
    workload, o, _, _ = bundle(scratch, spec)
    hw = hardware(o)
    target = OperatorSimBackend()._resolve_chip(hw.name, hw.chip)
    with gzip.open(cache / (case_id(spec) + ".gz"), "rb") as f:
        receipt = pickle.load(f)
    capacity, pages = page_table(scratch, spec)
    attrs = dict(implementation="a5_qsli_projection", pa_block_size=128, table_pages=pages)
    by_shape = {}
    result = []
    for graph, stored in zip(workload.graphs, receipt, strict=True):
        check(len(stored) == len(graph.nodes), "Cached graph size changed")
        costs = {k: {} for k in LANES}
        hits, subtotal = 0, 0.0
        shapes = []
        for n in graph.nodes:
            fine, peak, eff, candidate = stored[n.ref.node_id]
            check((candidate is not None) == (n.raw_name == CANDIDATE), "Candidate identity mismatch")
            durations = (fine.duration_us, peak, eff)
            if candidate is not None:
                shape, _ = candidate
                check(shape == geometry(n), "Capture and receipt geometry differ")
                if shape not in by_shape:
                    est = estimate(
                        "candidate_filtered_indexer",
                        shape,
                        attrs,
                        chip=target,
                        dtype="mxfp4",
                        memory_policy="cold_io",
                    )
                    logical = est.rates["logical_roofline"]
                    tc, tv, tm = (logical[k] for k in ("cube_us", "vector_us", "logical_io_us"))
                    adjusted = max(
                        tc / hw.chip.cube.utilization.ratio,
                        tv / hw.chip.vec.utilization.ratio,
                        tm / hw.chip.hbm.utilization.ratio,
                    )
                    peak_us = max(tc, tv, tm)
                    check(
                        all(math.isfinite(v) and v >= 0 for v in (tc, tv, tm, est.latency_us)),
                        f"Incomplete price for {shape}",
                    )
                    check(
                        est.latency_us >= peak_us or near(est.latency_us, peak_us),
                        f"Candidate below aligned peak for {shape}",
                    )
                    by_shape[shape] = (est.latency_us, peak_us, adjusted)
                durations = by_shape[shape]
                hits += 1
                subtotal += durations[0]
                shapes.append("x".join(map(str, shape)))
            for k, value in zip(LANES, durations):
                costs[k][n.ref] = replace(fine, duration_us=value)
        row = {k: simulate_graph(graph, by_ref, caps(o)).duration_us for k, by_ref in costs.items()}
        row.update(
            nodes=hits,
            candidates_us=subtotal,
            shapes=";".join(sorted(set(shapes))),
            kv_capacity=capacity,
            table_pages=pages,
        )
        result.append(row)
    check(sum(r["nodes"] for r in result) == q["operators"][CANDIDATE], "Census mismatch")
    return result


def key(r):
    return r["model"], int(r["prompt"]), int(r["batch"]), int(r["ep"])


def sweep(args):
    import operator_model

    package = Path(operator_model.__file__).parent
    freeze = json.loads((HERE / "v112_frozen.json").read_text())
    check(freeze["version"] == operator_model.__version__, "Expected the frozen model release")
    for relative, expected in freeze["sha256"].items():
        check(
            sha256((package / relative).read_bytes()).hexdigest() == expected,
            f"Frozen source or parameter changed: {relative}",
        )
    old = {key(r): r for r in v111("requests")}
    prior = {(key(r), r["phase"], r["stage"], int(r["history"])): r for r in v111("flash-anchors")}
    specs = [spec for p in profiles() for spec in inputs(p)]
    samples = dict(
        zip(
            specs,
            jobs(partial(replay, scratch=args.scratch, cache=args.cache), specs, args.workers),
            strict=True,
        )
    )
    requests, anchors = [], []
    for profile in profiles():
        model, prompt, batch, ep, chunk = profile
        o = old[profile[:4]]
        row = {
            k: o[k]
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
            values = {k: {} for k in LANES}
            hits = 0
            for s in (x for x in inputs(profile) if x[4] == phase):
                pos = s[5] // chunk if phase == "prefill" else s[5]
                for i, ev in enumerate(samples[s]):
                    stage = "prefill" if phase == "prefill" else ("target", "draft")[i]
                    for k in LANES:
                        values[k][pos] = values[k].get(pos, 0.0) + ev[k]
                    if ev.get("nodes"):
                        anchor = prior[(profile[:4], phase, stage, s[5])]
                        check(
                            int(anchor["candidate_nodes"]) == ev["nodes"]
                            and anchor["candidate_shapes_bqhd_lck"] == ev["shapes"],
                            "Changes outside node price",
                        )
                        hits += ev["nodes"]
                        anchors.append(
                            dict(
                                model=model,
                                prompt=prompt,
                                batch=batch,
                                ep=ep,
                                phase=phase,
                                stage=stage,
                                history=s[5],
                                kv_capacity=ev["kv_capacity"],
                                table_pages=ev["table_pages"],
                                shapes=ev["shapes"],
                                candidate_nodes=ev["nodes"],
                                v111_candidate_us=float(anchor["candidate_packed_us"]),
                                v112_candidate_us=ev["candidates_us"],
                                v111_stage_us=float(anchor["packed_full_us"]),
                                v112_stage_us=ev["full"],
                                stage_delta_us=ev["full"] - float(anchor["packed_full_us"]),
                                v111_peak_us=float(anchor["packed_peak_us"]),
                                v112_peak_us=ev["peak"],
                                v111_effective_us=float(anchor["packed_effective_us"]),
                                v112_effective_us=ev["effective"],
                            )
                        )
            for k in LANES:
                row[phase + "_v112_" + k + "_s"] = integrate(values[k], phase, profile)
            row[phase + "_v112_anchor_check_pct"] = anchor_error(values["full"], phase, prompt, chunk)
            row[phase + "_sampled_candidate_nodes"] = hits
        for k in LANES:
            row["request_v112_" + k + "_s"] = sum(row[p + "_v112_" + k + "_s"] for p in PHASES[:2])
        for phase in PHASES:
            row[phase + "_legacy_s"] = float(o[phase + "_legacy_whole_s"])
            row[phase + "_v111_full_s"] = float(o[phase + "_packed_full_s"])
            row[phase + "_v111_peak_s"] = float(o[phase + "_packed_peak_s"])
            row[phase + "_v111_effective_s"] = float(o[phase + "_packed_effective_s"])
            row[phase + "_delta_us"] = 1e6 * (row[phase + "_v112_full_s"] - row[phase + "_v111_full_s"])
            check(row[phase + "_v112_full_s"] >= row[phase + "_v112_peak_s"], "below aligned peak")
            if model != "v41-flash":
                check(
                    all(
                        near(row[phase + "_v112_" + k + "_s"], row[phase + "_v111_" + k + "_s"])
                        for k in LANES
                    ),
                    "Unexpected control change",
                )
        requests.append(row)
    check(len(requests) == 33 and len(anchors) == len(prior) == 138, "Incomplete sweep")
    write(args.output / "v112_context_requests.csv", requests)
    write(args.output / "v112_context_anchors.csv", anchors)


def measured_waves(args):
    from sim_platform.backend.l1 import simulate_graph

    spec = ("v41-flash", 1048576, 1, 32, "prefill", 1044480, 4096)
    first = json.loads((HERE / "prefill-wave.json").read_text())
    second = json.loads(args.wave64.read_text())
    for q, w in ((32, first), (64, second)):
        check(
            w["shape"] == [1, q, 32, 128, 1048576, 2048, 512]
            and w["sku"] == "Ascend950DT_9582"
            and w["golden_positions"] == q * 512
            and w["attrs"]["table_pages"] == 8201,
            "Wave receipt mismatch",
        )
    check(second["status"] == "qualified device window", "Q64 requires an exact golden and all core traces")
    t1, t2 = first["device_us"], second["device_us"]
    check(t1 > 0 and t2 > 0, "Expected two positive complete device windows")
    workload, o, _, _ = bundle(args.scratch, spec)
    (g,) = workload.graphs
    with gzip.open(args.cache / (case_id(spec) + ".gz"), "rb") as f:
        (stored,) = pickle.load(f)
    base = {n.ref: stored[n.ref.node_id][0] for n in g.nodes}
    owners = [n.ref for n in g.nodes if n.raw_name == CANDIDATE]
    check(len(owners) == 4, "Expected four consumers")
    startup_us = 8568 / 1650  # Ordinary-QLI proxy at DT frequency, not measured multiwave startup.
    rows = []
    # Two-point slope already absorbs startup once; do not subtract startup again.
    for name, us in [
        ("repeat_32_with_startup", 128 * t1),
        ("repeat_32_credit_startup", 128 * t1 - 127 * startup_us),
        ("repeat_64_with_startup", 64 * t2),
        ("repeat_64_credit_startup", 64 * t2 - 63 * startup_us),
        ("first_32_plus_127_increments", t1 + 127 * (t2 - t1)),
    ]:
        # A locality change can make a two-point slope negative. Keep the
        # diagnostic and do not feed an impossible duration to the DES.
        duration = None
        if us > 0:
            costs = base.copy()
            for ref in owners:
                costs[ref] = replace(costs[ref], duration_us=us)
            duration = simulate_graph(g, costs, caps(o)).duration_us
        rows.append(
            dict(
                scenario=name,
                q32_ca_us=t1,
                q64_ca_us=t2,
                q32_trace_set_sha256=first["trace_set_sha256"],
                q64_trace_set_sha256=second["trace_set_sha256"],
                startup_proxy_us=startup_us,
                q64_minus_q32_us=t2 - t1,
                implied_affine_startup_us=2 * t1 - t2,
                projected_node_us=us,
                des_last_chunk_us=duration,
                status="conditional extrapolation"
                if duration is not None
                else "nonphysical affine extrapolation",
            )
        )
    write(args.output / "v112_context_multiwave.csv", rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["sweep", "waves"])
    p.add_argument("scratch", type=Path)
    p.add_argument("cache", type=Path)
    p.add_argument("--output", type=Path, default=HERE)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--wave64", type=Path)
    a = p.parse_args()
    a.output.mkdir(exist_ok=True, parents=True)
    if a.mode == "sweep":
        sweep(a)
    elif not a.wave64:
        p.error("waves requires --wave64 with a qualified DT CA trace receipt")
    else:
        measured_waves(a)


if __name__ == "__main__":
    main()
