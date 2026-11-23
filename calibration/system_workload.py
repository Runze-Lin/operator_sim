"""Read a system simulator decode export, join rank-zero nodes to its L1 baseline."""
import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


STATUS_NAMES = ("success", "miss", "fail", "placeholder", "skipped_zero", "skipped_control")
VARIANT_ATTRIBUTES = ("storage_precision", "logical_dtype", "gemm_grouping",
                      "weight_precision", "activation_precision")


def read_json(path):
    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz"
          else path.open(encoding="utf-8")) as source:
        return json.load(source)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        yield from csv.DictReader(source)


def require(value, message):
    if not value:
        raise ValueError(message)


def tensor(graph, index):
    t = graph["tensors"][index]
    storage = graph["storages"][t["storage_id"]]
    return dict(t, storage_category=storage["category"], backing_storage_bytes=storage["nbytes"])


def bucket():
    return {"nodes": 0, "serial_us": 0., "exposed_us": 0., "status": Counter()}


def add(item, row):
    item["nodes"] += 1
    item["serial_us"] += float(row["duration_us"] or 0)
    item["exposed_us"] += float(row["exposed_duration_us"] or 0)
    item["status"][row["prediction_status"]] += 1


def audit(root, case, op, boundary, check_trace):
    cases = list(read_rows(root / "summary.csv"))
    cases = [r for r in cases if (boundary or int(r["history_tokens"]) == 131072)
             and (case is None or r["case"] == case)]
    require(cases, "No case matches these filters")
    unique = set()
    operators = defaultdict(bucket)
    splits = defaultdict(lambda: defaultdict(bucket))
    case_stats = {}
    for entry in cases:
        name = entry["case"]
        require(name not in unique, f"Repeated case {name}")
        unique.add(name)
        directory = root / name
        result_dir = directory / "results" / "l1"
        result = read_json(result_dir / "inference.json")
        runtime = read_json(directory / "validation.json")
        meta = read_json(directory / "native" / "000_round_metadata.json")
        rt = runtime["config"]["runtime"]
        expected = {"tp": int(entry["tp"]), "dp": int(entry["dp"]), "ep": int(entry["ep"]),
                    "batch_size": int(entry["batch"]), "kv_seq_len": int(entry["history_tokens"])}
        require(all(meta["target_runtime"][key] == value for key, value in expected.items()),
                f"{name}: native parallelism, batch or history differs")
        stat, = result["statistics"]
        peak = stat["memory"]["peak_gb"]
        require(math.isclose(peak, float(entry["static_peak_gb"]), abs_tol=1e-9)
                and stat["local_batch_size"] == expected["batch_size"]
                and stat["global_batch_size"] == expected["batch_size"] * expected["dp"],
                f"{name}: memory or batch accounting differs")
        require(runtime["valid"] and all(rt[k] == int(entry[col]) for k, col in
                (("kv_seq_len", "history_tokens"), ("batch_size", "batch"), ("tp", "tp"), ("dp", "dp"))),
                f"{name}: settings disagree with exported summary")
        stats = bucket()
        stages = {}
        for item in result["results"]:
            stage = item["stage"]
            require(item["rank"] == 0 and item["declared_execution_count"] == 1 and
                    item["phase"] == "verify" and stage not in stages, f"{name}: unexpected stage")
            graph = read_json(directory / "native" / f"000_rank0_{stage}.json.gz")
            csv_rows = list(read_rows(result_dir / item["files"]["csv"]))
            rows = {r["id"]: r for r in csv_rows if r["record_type"] == "operator"}
            ids = {str(node["id"]) for node in graph["nodes"]}
            require(graph["rank"] == 0 and len(rows) == len(csv_rows) == len(ids) == len(graph["nodes"])
                    and set(rows) == ids, f"{name}/{stage}: rank, duplicate or missing node")
            trace_count = None
            if check_trace:
                trace_data = read_json((result_dir / item["files"]["csv"]).with_suffix(".trace.json"))
                events = [e["args"] for e in trace_data["traceEvents"] if e["ph"] in ("X", "i")
                          and e.get("args", {}).get("record_type") == "operator"]
                trace_ids = {str(e["id"]) for e in events}
                require(len(events) == len(trace_ids) and trace_ids <= ids and
                        all(rows[str(e["id"])]["raw_name"] == e["raw_name"] for e in events),
                        f"{name}/{stage}: trace IDs disagree with L1")
                trace_count = len(events)
            stages[stage] = {"nodes": len(rows), "des_us": item["duration_us"],
                             "trace_operators": trace_count}
            stage_status = Counter(r["prediction_status"] for r in rows.values())
            require(all(stage_status[k] == count for k, count in item["prediction_coverage"].items()
                        if k in STATUS_NAMES),
                    f"{name}/{stage}: status totals disagree")
            for node in graph["nodes"]:
                raw = node["name"]
                r = rows[str(node["id"])]
                require(raw == r["raw_name"], f"{name}/{stage}/{node['id']}: name mismatch")
                add(stats, r)
                add(operators[raw], r)
                sem = node["attributes"].get("semantic", {})
                dtypes = ",".join(graph["tensors"][i]["dtype"].removeprefix("torch.") for i in node["outputs"])
                # Keep compact dimensions that change the operator contract. Full tensor shape,
                # stride, physical storage and attributes are available for each individual event.
                parts = [dtypes] + [f"{k}={sem[k]}" for k in VARIANT_ATTRIBUTES if k in sem]
                variant = " ".join(parts)
                add(splits[raw][variant], r)
                if raw == op:
                    event = {
                        "case": name, "rank": 0, "stage": stage, "id": node["id"], "name": raw,
                        "status": r["prediction_status"], "baseline_us": float(r["duration_us"] or 0),
                        "inputs": [tensor(graph, i) for i in node["inputs"]],
                        "outputs": [tensor(graph, i) for i in node["outputs"]],
                        "attributes": node["attributes"], "scope": node["scope"],
                        "dependencies": node["dependencies"],
                    }
                    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
        require(set(stages) == {"target", "draft"}, f"{name}: missing stage")
        require(all(stats["status"][k] == int(entry[k]) for k in
                    STATUS_NAMES),
                f"{name}: coverage counts differ")
        require(stats["nodes"] == int(entry["target_nodes"]) + int(entry["draft_nodes"]),
                f"{name}: node count differs")
        stats.update(stages=stages, family=meta["family"], tp=rt["tp"], dp=rt["dp"],
                     ep=int(entry["ep"]), local_batch=rt["batch_size"], history=rt["kv_seq_len"],
                     static_peak_gb=peak,
                     partial_des_us=float(entry["partial_round_des_us"]))
        case_stats[name] = stats
    # Cross-check the compact upstream table, which aggregates every operator including zero nodes.
    declared = defaultdict(bucket)
    for r in read_rows(root / "operators.csv"):
        if r["case"] not in unique:
            continue
        target = declared[r["raw_name"]]
        target["nodes"] += int(r["nodes"])
        for key in ("serial_us", "exposed_us"):
            target[key] += float(r[key])
        target["status"].update({k: int(r[k]) for k in
                                STATUS_NAMES
                                if int(r[k])})
    require(set(declared) == set(operators), "Operator names differ from summary")
    for n, b in operators.items():
        other = declared[n]
        require(b["nodes"] == other["nodes"] and b["status"] == other["status"] and
                all(math.isclose(b[k], other[k], rel_tol=1e-9, abs_tol=1e-6)
                    for k in ("serial_us", "exposed_us")), f"{n}: aggregate differs from source ledger")
    if op:
        require(op in operators, f"Operator {op!r} absent in selected cases")
    else:
        print(json.dumps({"cases": case_stats, "operators": operators, "variants": splits},
                         ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path, help="decode-950dt-128k directory")
    parser.add_argument("--case", help="Filter to an exact case directory name")
    parser.add_argument("--op", help="Print JSONL with all occurrences, full tensor and operator metadata")
    parser.add_argument("--include-boundary", action="store_true", help="Include the 131198 diagnostic")
    parser.add_argument("--verify-trace", action="store_true", help="Also check visualization event IDs")
    args = parser.parse_args()
    audit(args.export, args.case, args.op, args.include_boundary, args.verify_trace)


if __name__ == "__main__":
    main()
