"""Audit the published 27-task matrix with same-graph node substitutions."""
import argparse
import csv
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tomllib

from calibration.system_compare import cases, near
from calibration.system_full_compare import FULL, write

CANDIDATE = "inference::candidate_filtered_indexer"
GROUPED = "npu::npu_grouped_matmul"
BITS = {"float4_e2m1fn_x2": 4, "float8_e4m3fn": 8, "float16": 16,
        "bfloat16": 16, "float32": 32}
REFERENCE = Path(__file__).with_name("reference")


def nominal_engine_us(typed_work, peaks):
    """Useful work priced at the task's nominal peak. Missing types stay unknown."""
    if typed_work is None:
        return None
    total = 0.0
    for dtype, count in typed_work:
        if count:
            peak = peaks.get(BITS.get(dtype))
            if not peak:
                return None
            total += count / (peak * 1e6)
    return total


def types_used(work):
    return "+".join(sorted(dtype or "unspecified" for dtype, count in (work or ()) if count))


def audit_case(job):
    from sim_platform.api.inference import run_inference
    from sim_platform.backend.l1 import simulate_graph
    from sim_platform.config.inference import load_inference_task
    from sim_platform.hardware.loading import load_hardware
    from operator_model.system_routes import STRUCTURAL, communication

    case, model, layout, batch, scratch, reference = job
    task = case / "task.toml"
    overrides = {"prediction.backends": [FULL], "output.directory": str(scratch/case.name),
                 "exports.enabled": False, "logging.console_level": "ERROR"}
    options = load_inference_task(task, overrides=overrides).options
    capacities = dict(options.capacities)
    if not options.cube_vec_overlap:
        capacities["compute"] = 1
    hw_path = tomllib.loads(task.read_text())["platform"]["hardware"]
    hw = load_hardware(case / hw_path).to_model_hardware().chip
    cube_peaks, vec_peaks = dict(hw.cube.tflops_by_bits), dict(hw.vec.tflops_by_bits)
    run = run_inference(task, overrides=overrides)
    [stats] = run.statistics
    totals, operators, examples, signatures = Counter(), defaultdict(Counter), {}, defaultdict(Counter)
    graph_hash = sha256()
    for idx, ((stage, full, repeat), ev) in enumerate(
            zip(stats.stage_durations_us, run.evaluations, strict=True)):
        assert repeat == 1 and stage == ("target", "draft")[idx]
        for entity in (*ev.graph.nodes, *ev.graph.dependencies):
            graph_hash.update(repr(entity).encode() + b"\n")
        with (case/"results/l1"/f"{idx:03}_verify_{stage}.csv").open(newline="") as inp:
            old = {r["id"]: r for r in csv.DictReader(inp) if r["record_type"] == "operator"}
        costs, preds = dict(ev.execution_costs), dict(ev.predictions)
        assert len(old) == len(costs) == len(preds)
        names, prior, original_costs = {}, {}, {}
        aligned_peak, aligned_effective = costs.copy(), costs.copy()
        for n in ev.graph.nodes:
            r = old[n.ref.node_id]
            assert (n.raw_name, ev.graph.graph_id) == (r["raw_name"], r["graph_id"])
            name = names[n.ref] = n.raw_name
            prev = prior[n.ref] = float(r["duration_us"])
            cost = costs[n.ref]
            original_costs[n.ref] = replace(cost, duration_us=prev,
                                            resources=tuple(json.loads(r["resources"]).items()))
            value = cost.duration_us
            det = preds[n.ref].estimate.details
            assert preds[n.ref].backend_name == FULL and near(value, preds[n.ref].estimate.duration_us)
            for terms, work in ((det.cube_work_by_dtype, det.cube_flops),
                                (det.vec_work_by_dtype, det.vec_flops)):
                assert terms is None or near(sum(count for _, count in terms), work)
            for tensors, field in ((n.inputs, "inputs"), (n.outputs, "outputs")):
                assert [(t.dtype.removeprefix("torch."), tuple(t.shape)) for t in tensors] == [
                    (t["dtype"].removeprefix("torch."), tuple(t["shape"])) for t in json.loads(r[field])]
            bucket = operators[name]
            bucket.update(nodes=1, original_us=prev, full_us=value,
                          faster=int(value < prev and not near(value, prev)),
                          slower=int(value > prev and not near(value, prev)))
            compare = {"cube": nominal_engine_us(det.cube_work_by_dtype, cube_peaks),
                       "vec_proxy": nominal_engine_us(det.vec_work_by_dtype, vec_peaks),
                       "logical_io": None if None in (det.read_bytes, det.write_bytes) else
                       (det.read_bytes + det.write_bytes) / (hw.hbm.peak_bytes_per_second / 1e6)}
            role = "local"
            if name in communication.NAMES:
                role = "same_fabric"
            elif name in STRUCTURAL:
                role = "same_metadata"
            elif None in compare.values():
                role = "unpriced_terms_keep_full"
            if role == "local":
                # The published system task has constant efficiencies and zero head/tail.
                # Keep vector equivalent work and logical bytes visible as proxy terms.
                assert hw.head_tail_time == hw.head_tail_time_for_fused_op == 0
                peak_us = max(compare.values())
                effective_us = max(compare["cube"] / hw.cube.utilization.ratio,
                                   compare["vec_proxy"] / hw.vec.utilization.ratio,
                                   compare["logical_io"] / hw.hbm.utilization.ratio)
                aligned_peak[n.ref] = replace(cost, duration_us=peak_us)
                aligned_effective[n.ref] = replace(cost, duration_us=effective_us)
                bucket["local_priced"] += 1
                bucket["below_aligned_peak"] += int(value < peak_us and not near(value, peak_us))
                bucket["below_aligned_effective"] += int(value < effective_us and not near(value, effective_us))
            else:
                peak_us = effective_us = value
            semantic = dict(dict(n.attrs).get("semantic", ()))
            sig = (name, det.model_name, role,
                   ",".join(t.dtype.removeprefix("torch.") for t in n.inputs),
                   ",".join(t.dtype.removeprefix("torch.") for t in n.outputs),
                   str(semantic.get("compute_precision", "")), str(semantic.get("accumulator_dtype", "")),
                   types_used(json.loads(r["roofline_cube_work_by_dtype"] or "[]")),
                   types_used(json.loads(r["roofline_vec_work_by_dtype"] or "[]")),
                   types_used(det.cube_work_by_dtype), types_used(det.vec_work_by_dtype))
            signatures[sig].update(nodes=1,original_us=prev,aligned_peak_us=peak_us,
                                   aligned_effective_us=effective_us,full_us=value)
            for term, ideal in compare.items():
                if ideal is None or ideal <= 0:
                    continue
                bucket[term+"_positive"] += 1
                if ideal <= value or near(ideal, value):
                    continue
                bucket[term+"_over"] += 1
                key = (name, term)
                if key not in examples or ideal/value > examples[key]["ratio"]:
                    examples[key] = dict(case=case.name, stage=stage, node=n.ref.node_id,
                                         raw_name=name, method=det.model_name, term=term,
                                         ideal_us=ideal, full_us=value, ratio=ideal/value,
                                         inputs=json.loads(r["inputs"]))
        def replay(prices):
            return simulate_graph(ev.graph, prices, capacities).duration_us
        assert near(replay(costs), full)
        totals["original_us"] += replay(original_costs)
        totals["full_us"] += full
        totals["original_serial_us"] += sum(prior.values())
        totals["full_serial_us"] += sum(c.duration_us for c in costs.values())
        totals["aligned_peak_us"] += replay(aligned_peak)
        totals["aligned_effective_us"] += replay(aligned_effective)
        for label, restored in (("candidate", {CANDIDATE}), ("gmm", {GROUPED}),
                                ("both", {CANDIDATE, GROUPED})):
            prices = {ref: replace(cost, duration_us=prior[ref]) if names[ref] in restored else cost
                      for ref, cost in costs.items()}
            totals["restore_"+label+"_us"] += replay(prices)
    assert graph_hash.hexdigest() == reference["graph_sha256"]
    assert sha256(task.read_bytes()).hexdigest() == reference["task_sha256"]
    assert near(totals["original_us"], float(reference["original_us"]))
    row = dict(case=case.name, model=model, layout=layout, batch=batch,
               fits_144gb=reference["fits_144gb"], v19_us=float(reference["full_us"]), **totals)
    return row, dict(operators), examples, dict(signatures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("scratch", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    with (REFERENCE / "system-full-comparison.csv").open() as f:
        refs = {r["case"]: r for r in csv.DictReader(f)}
    jobs = [(*item, args.scratch.resolve(), refs[item[0].name]) for item in cases(args.matrix)]
    rows, ops, examples, signatures = [], defaultdict(Counter), {}, defaultdict(Counter)
    with ProcessPoolExecutor(args.workers) as pool:
        for row, buckets, worst, sigs in pool.map(audit_case, jobs):
            rows.append(row)
            for name, bucket in buckets.items():
                ops[name].update(bucket)
            for key, ex in worst.items():
                if key not in examples or ex["ratio"] > examples[key]["ratio"]:
                    examples[key] = ex
            for sig, values in sigs.items():
                signatures[sig].update(values)
    assert sum(b["nodes"] for b in ops.values()) == sum(int(r["nodes"]) for r in refs.values())
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output/"roofline-audit-cases.csv", rows)
    keys = ("nodes", "faster", "slower", "original_us", "full_us",
            "cube_positive", "cube_over", "vec_proxy_positive", "vec_proxy_over",
            "logical_io_positive", "logical_io_over", "local_priced", "below_aligned_peak",
            "below_aligned_effective")
    write(args.output/"roofline-audit-operators.csv",
          [dict(raw_name=name, **{k: bucket[k] for k in keys}) for name, bucket in sorted(ops.items())])
    samples = [examples[key] for key in sorted(examples)]
    (args.output/"roofline-audit-exceptions.json").write_text(json.dumps(samples, indent=1)+"\n")
    columns = ("raw_name", "method", "role", "input_storage_dtypes", "output_storage_dtypes",
               "capture_compute_precision", "capture_accumulator_dtype",
               "old_cube_types", "old_vec_types", "aligned_cube_types", "aligned_vec_proxy_types")
    write(args.output/"roofline-audit-precision.csv",
          [dict(zip(columns, key, strict=True), **counts) for key, counts in sorted(signatures.items())])


if __name__ == "__main__":
    main()
