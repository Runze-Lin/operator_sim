"""Replay trusted SIMPlatform captures with the three v2.3 Vector policies and same-graph Rooflines."""

import argparse
import csv
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from hashlib import sha256
import json
import logging
import math
from pathlib import Path
from time import perf_counter

from calibration.candidate_indexer.context_sensitivity import (
    anchor_error,
    bundle,
    caps,
    hardware,
    integrate,
    page_table,
)
from calibration.context_report import write
from calibration.context_sweep import ACCEPT, OUTPUT, EXACT_DECODE, case_id, inputs, profiles
from calibration.roofline_audit import nominal_engine_us

MODES = ("calibrated", "auto", "instruction_classes")
LANES = ("legacy", "peak", "effective", *MODES)
ROOT = Path(__file__).resolve().parents[2]


def replay(spec, *, scratch, cache):
    from operator_model.sim_backend import FullOperatorSimBackend
    from operator_model.system_routes import STRUCTURAL, communication
    from sim_platform.api.inference_evaluation import evaluate_inference_workload
    from sim_platform.backend.factory import build_predictor
    from sim_platform.backend.l1 import simulate_graph
    from sim_platform.config.execution import PredictionOptions

    path = cache / (case_id(spec) + ".json")
    if path.exists():
        return
    logging.disable(logging.WARNING)
    start = perf_counter()
    workload, options, previous, memory = bundle(scratch, spec)
    hw = hardware(options)
    _, pages = page_table(scratch, spec)
    rates = [dict(engine.tflops_by_bits) for engine in (hw.chip.cube, hw.chip.vec)]
    stages = [dict() for _ in previous]
    attribution = [defaultdict(Counter) for _ in previous]
    aligned_costs = []
    owners = []
    refkeys = []

    def check(condition, message):
        if not condition:
            raise ValueError(f"{case_id(spec)}: {message}")

    for mode in (*MODES, "legacy"):
        backend = (
            None
            if mode == "legacy"
            else FullOperatorSimBackend(
                candidate_implementation="a5_qsli_projection",
                candidate_options={"pa_block_size": 128, "table_pages": pages},
                memory_policy="cold_io",
                vector_model=mode,
            )
        )
        name = backend.name if backend else "roofline"
        predictor = build_predictor(
            PredictionOptions((name,), 8192), (backend,) if backend else (), hardware=hw
        )
        evaluations = evaluate_inference_workload(workload, predictor=predictor, options=options)
        for i, (stage, old) in enumerate(zip(evaluations, previous, strict=True)):
            record = stages[i]
            record[mode] = stage.duration_us
            if mode == "legacy":
                check(
                    math.isclose(old["raw_us"], stage.duration_us, rel_tol=1e-8, abs_tol=1e-6),
                    "historical Roofline regression",
                )
                continue
            nodes = stage.graph.nodes
            costs = dict(stage.execution_costs)
            preds = dict(stage.predictions)
            check(len(costs) == len(nodes) == len(preds) == old["nodes"], "coverage")
            if mode == MODES[0]:
                peak, effective = {}, {}
                aligned_costs.append({})
                owners.append(Counter(n.raw_name for n in nodes))
                refkeys.append({})
            for node in nodes:
                ref = node.ref
                pred = preds[ref]
                check(
                    pred.status == "success" and pred.backend_name == name,
                    f"{mode} fallback on {node.raw_name}",
                )
                cost = costs[ref]
                attribution[i][node.raw_name][mode] += cost.duration_us
                if node.raw_name in STRUCTURAL or node.raw_name in communication.NAMES:
                    if mode == MODES[0]:
                        peak[ref] = effective[ref] = cost
                    continue
                details = pred.estimate.details
                work = (
                    details.cube_work_by_dtype,
                    details.vec_work_by_dtype,
                    details.read_bytes,
                    details.write_bytes,
                )
                if mode == MODES[0]:
                    refkeys[i][ref] = work
                    tc, tv = (nominal_engine_us(v, p) for v, p in zip(work[:2], rates))
                    tm = sum(work[2:]) / (hw.chip.hbm.peak_bytes_per_second / 1e6)
                    ideal = max(tc, tv, tm)
                    aligned_costs[i][ref] = ideal
                    eff = max(
                        tc / hw.chip.cube.utilization.ratio,
                        tv / hw.chip.vec.utilization.ratio,
                        tm / hw.chip.hbm.utilization.ratio,
                    )
                    peak[ref] = replace(cost, duration_us=ideal)
                    effective[ref] = replace(cost, duration_us=eff)
                else:
                    check(work == refkeys[i][ref], f"{mode} changed logical work on {node.raw_name}")
                check(
                    cost.duration_us + 1e-7 >= aligned_costs[i][ref] * (1 - 1e-10),
                    "below same-work peak " + node.raw_name,
                )
            if mode == MODES[0]:
                record.update(
                    peak=simulate_graph(stage.graph, peak, caps(options)).duration_us,
                    effective=simulate_graph(stage.graph, effective, caps(options)).duration_us,
                    nodes=len(nodes),
                )
        del evaluations
    receipt = dict(
        spec=spec,
        stages=stages,
        attribution=attribution,
        operators=owners,
        peak_bytes=memory,
        pages=pages,
        wall_s=perf_counter() - start,
    )
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, allow_nan=False))
    tmp.replace(path)


def summarize(cache, output):
    def archived(path):
        with path.open() as stream:
            return {
                (r["model"], int(r["prompt"]), int(r["batch"]), int(r["ep"])): r
                for r in csv.DictReader(stream)
            }

    old = archived(ROOT / "calibration/candidate_indexer/v113_context_requests.csv")
    old_baseline = archived(ROOT / "calibration/reference/context-950dt/requests.csv")
    requests, graphs, accounts, interpolation = [], [], [], []
    total_nodes = Counter()
    observed_graphs = observed_captures = 0
    graph_wall = 0.0
    fingerprints = sha256()
    for profile in profiles():
        model, length, batch, ep, chunk = profile
        identity = dict(model=model, prompt=length, output=OUTPUT, batch=batch, tp=8, ep=ep)
        row = dict(identity, prefill_chunks=length // chunk, decode_rounds=math.ceil((OUTPUT - 1) / ACCEPT))
        for phase in ("prefill", "decode"):
            curves = {lane: {} for lane in LANES}
            node_series = {}
            memory = 0
            for spec in inputs(profile):
                if spec[4] != phase:
                    continue
                filename = cache / (case_id(spec) + ".json")
                data = json.loads(filename.read_text())
                if data["spec"] != list(spec):
                    raise ValueError(f"Wrong receipt: {filename}")
                observed_captures += 1
                graph_wall += data["wall_s"]
                memory = max(memory, data["peak_bytes"])
                pos = spec[5] // chunk if phase == "prefill" else spec[5]
                for lane in LANES:
                    curves[lane][pos] = sum(r[lane] for r in data["stages"])
                node_series[pos] = defaultdict(Counter)
                stages = ("prefill",) if phase == "prefill" else ("target", "draft")
                for name, r, costs, nodes in zip(
                    stages, data["stages"], data["attribution"], data["operators"], strict=True
                ):
                    observed_graphs += 1
                    if sum(nodes.values()) != r["nodes"] or set(costs) != set(nodes):
                        raise ValueError("Incomplete node census")
                    total_nodes.update(nodes)
                    graphs.append(
                        dict(
                            identity,
                            phase=phase,
                            stage=name,
                            history=spec[5],
                            query=spec[6],
                            table_pages=data["pages"],
                            nodes=r["nodes"],
                            **{k + "_us": r[k] for k in LANES},
                        )
                    )
                    for op, amounts in costs.items():
                        node_series[pos][op].update(amounts)
                fingerprints.update(
                    json.dumps([spec, data["stages"], data["peak_bytes"]], sort_keys=True).encode()
                )
            for k in LANES:
                curve = curves[k]
                row[phase + "_" + k + "_s"] = integrate(curve, phase, profile)
                ape = anchor_error(curve, phase, length, chunk)
                interpolation.append(
                    dict(
                        identity,
                        phase=phase,
                        backend=k,
                        observed_positions=len(curve),
                        max_internal_anchor_error_pct=ape,
                    )
                )
            for op in sorted(set().union(*(x.keys() for x in node_series.values()))):
                demands = {
                    k + "_s": integrate(
                        {pos: w.get(op, {}).get(k, 0.0) for pos, w in node_series.items()}, phase, profile
                    )
                    for k in MODES
                }
                accounts.append(dict(identity, phase=phase, raw_name=op, **demands))
            row[phase + "_peak_GB"] = memory / 1e9
        for k in LANES:
            row["request_" + k + "_s"] = row["prefill_" + k + "_s"] + row["decode_" + k + "_s"]
        row["fits_144GB"] = max(row["prefill_peak_GB"], row["decode_peak_GB"]) <= 144
        row["decode_sampling"] = "all_rounds" if profile[:4] in EXACT_DECODE else "three_rounds"
        row["decode_positions"] = "beyond_preset_extrapolation" if length == 1048576 else "in_preset"
        previous = old[profile[:4]]
        for phase in ("prefill", "decode", "request"):
            row[phase + "_previous_full_s"] = float(previous[phase + "_v113_full_s"])
            if not math.isclose(
                row[phase + "_legacy_s"], float(old_baseline[profile[:4]][phase + "_legacy_s"]), rel_tol=1e-8
            ):
                raise ValueError("Legacy request aggregation differs from capture")
            if any(row[phase + "_" + m + "_s"] + 1e-8 < row[phase + "_peak_s"] for m in MODES):
                raise ValueError("Aggregate below aligned peak")
        requests.append(row)
    output.mkdir(parents=True, exist_ok=True)
    for name, records in [
        ("requests", requests),
        ("graphs", graphs),
        ("operator-deltas", accounts),
        ("interpolation-check", interpolation),
    ]:
        write(output / (name + ".csv"), records)
    summary = dict(
        profiles=len(requests),
        captures=observed_captures,
        stage_graphs=observed_graphs,
        priced_node_instances_per_fine_mode=sum(total_nodes.values()),
        distinct_raw_names=len(total_nodes),
        operators=dict(sorted(total_nodes.items())),
        graph_receipt_sha256=fingerprints.hexdigest(),
        summed_worker_wall_s=graph_wall,
        status="same captured graphs; no new hardware or CA timing",
        modes=list(MODES),
        memory_policy="cold_io",
        candidate_implementation="a5_qsli_projection",
        candidate_order_overlap="implementation default; inferred page order",
        pa_block_size=128,
    )
    (output / "sweep.json").write_text(json.dumps(summary, indent=2) + "\n")


def input_identity(scratch):
    """Fingerprint portable model resources and trusted local graphs; exclude output files."""
    from operator_model import __version__
    from sim_platform import __file__ as platform_path
    from operator_performance_models import __file__ as roofline_path
    from importlib.metadata import version

    def tree(root, suffixes):
        h = sha256()
        for p in sorted(x for x in root.rglob("*") if x.suffix in suffixes):
            h.update(str(p.relative_to(root)).encode() + b"\0" + sha256(p.read_bytes()).digest())
        return h.hexdigest()

    captured = sha256()
    count = 0
    for profile in profiles():
        for spec in inputs(profile):
            for name in ("graph.pkl.gz", "task.toml"):
                key = Path(case_id(spec)) / name
                captured.update(str(key).encode() + b"\0" + sha256((scratch / key).read_bytes()).digest())
            count += 1
    return dict(
        schema=1,
        model_version=__version__,
        operator_model_sha256=tree(ROOT / "src/operator_model", {".py", ".json"}),
        sim_platform_version=version("sim-platform"),
        sim_platform_sha256=tree(Path(platform_path).parent, {".py"}),
        legacy_backend_version=version("operator-performance-models"),
        legacy_backend_sha256=tree(Path(roofline_path).parent, {".py", ".json"}),
        captures=count,
        graph_and_task_sha256=captured.hexdigest(),
        sweep_script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
    )


def check_identity(scratch, cache):
    path = cache / "input-identity.json"
    identity = input_identity(scratch)
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Inputs or replay code changed; select a separate receipt directory")
    path.write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("run", "report"))
    cli.add_argument("scratch", type=Path)
    cli.add_argument("cache", type=Path)
    cli.add_argument("--workers", type=int, default=8)
    cli.add_argument("--limit", type=int)
    cli.add_argument("--output", type=Path, default=ROOT / "results/v2.3/system-e2e")
    args = cli.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    identity = check_identity(args.scratch, args.cache)
    if args.action == "run":
        specs = [s for p in profiles() for s in inputs(p)]
        if args.limit:
            specs = specs[: args.limit]
        with ProcessPoolExecutor(args.workers) as pool:
            for i, _ in enumerate(
                pool.map(partial(replay, scratch=args.scratch, cache=args.cache), specs), 1
            ):
                if i % 30 == 0:
                    print(i, "/", len(specs), flush=True)
    else:
        summarize(args.cache, args.output)
        (args.output / "input-identity.json").write_text(json.dumps(identity, indent=2) + "\n")


if __name__ == "__main__":
    main()
