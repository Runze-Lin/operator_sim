"""Price every captured catalogue signature using the public SIMPlatform backend."""

import argparse
from collections import Counter, defaultdict
from statistics import median
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache, partial
import gzip
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter_ns

from calibration.examples.catalog_capture import CHIP, ROOT, case_id, write_packet
from calibration.roofline_audit import nominal_engine_us

CANDIDATE = "inference::candidate_filtered_indexer"


def freeze(x):
    if isinstance(x, list):
        return tuple(freeze(v) for v in x)
    if isinstance(x, dict):
        return {k: freeze(v) for k, v in x.items()}
    return x


def unpack(data):
    from operator_performance_models.core import OpFeatures, TensorFeatures
    from operator_performance_models.collective import BalancedEpProof, CollectiveFacts

    f = freeze(data)
    for field in ("inputs", "outputs"):
        f[field] = tuple(TensorFeatures(**t) for t in f[field])
    if f["collective"] is not None:
        c = f["collective"]
        if c["balanced_ep"] is not None:
            c["balanced_ep"] = BalancedEpProof(**c["balanced_ep"])
        f["collective"] = CollectiveFacts(**c)
    return OpFeatures(**f)


@lru_cache(maxsize=1)
def hardware():
    from sim_platform.hardware.loading import load_hardware

    return load_hardware(CHIP).to_model_hardware()


@lru_cache(maxsize=None)
def backend(pages, mode):
    from operator_model.sim_backend import FullOperatorSimBackend

    return (
        FullOperatorSimBackend(
            vector_model=mode,
            candidate_implementation="a5_qsli_projection",
            candidate_options={"pa_block_size": 128, "table_pages": pages},
        )
        if pages
        else FullOperatorSimBackend(vector_model=mode)
    )


def price(item, mode):
    from operator_model.system_routes import communication

    key, pages, raw = item
    op = raw["raw_name"]
    try:
        f = unpack(raw)
        hw = hardware()
        model = backend(pages, mode)
        start = perf_counter_ns()
        q = model.predict(f, hw)
        query_us = (perf_counter_ns() - start) / 1000
        d = q.details
        vals = (q.duration_us, d.cube_flops, d.vec_flops, d.read_bytes, d.write_bytes)
        if d.adaptor_name != model.name or any(
            x is None or not math.isfinite(x) or x < 0 for x in vals
        ):
            raise ValueError("invalid backend or non-finite/negative resource quote")
        if q.duration_us == 0 and d.assumptions[0] != "structural":
            if any(vals[1:]) or not any(0 in t.shape for t in (*f.inputs, *f.outputs)):
                raise ValueError("unexplained zero-work device window")
        for terms, total in (
            (d.cube_work_by_dtype, d.cube_flops),
            (d.vec_work_by_dtype, d.vec_flops),
        ):
            if (
                terms is None
                or sum(c for _, c in terms) != total
                or any(c < 0 or not t for t, c in terms)
            ):
                raise ValueError("invalid typed work")
        if d.assumptions[0] != "structural" and op not in communication.NAMES:
            tc = nominal_engine_us(
                d.cube_work_by_dtype, dict(hw.chip.cube.tflops_by_bits)
            )
            tv = nominal_engine_us(
                d.vec_work_by_dtype, dict(hw.chip.vec.tflops_by_bits)
            )
            tm = (
                (d.read_bytes + d.write_bytes) / hw.chip.hbm.peak_bytes_per_second * 1e6
            )
            if (
                tc is None
                or tv is None
                or max(tc, tv, tm) > q.duration_us * (1 + 1e-8) + 1e-8
            ):
                raise ValueError(f"unpriced dtype or below matched peak {tc},{tv},{tm}")
        return (
            key,
            pages,
            (
                op,
                q.duration_us,
                d.model_name,
                d.assumptions[0],
                d.cube_work_by_dtype,
                d.vec_work_by_dtype,
                d.read_bytes,
                d.write_bytes,
                query_us,
            ),
            None,
        )
    except Exception as e:
        return key, pages, None, (op, type(e).__name__, str(e))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def model_digest():
    base = ROOT / "src/operator_model"
    d = hashlib.sha256()
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.suffix in {".py", ".json"}:
            d.update(
                str(p.relative_to(base)).encode()
                + b"\0"
                + hashlib.sha256(p.read_bytes()).digest()
            )
    return d.hexdigest()


def replay(data, out, workers, quick=False, mode="auto", summary_only=False):
    digest = model_digest()
    start = perf_counter_ns()
    features = json.loads(gzip.decompress((data / "features.json.gz").read_bytes()))
    cases = json.loads(gzip.decompress((data / "cases.json.gz").read_bytes()))
    requests = {}
    total = Counter()
    task_keys = []
    for case in cases:
        s = case["task"]
        pages = math.ceil((s["history"] + s["query"]) / 128)
        counts = Counter()
        for stage in case["stages"]:
            for key, n in stage["signatures"]:
                f = features[key]
                k = (key, pages if f["raw_name"] == CANDIDATE else 0)
                requests[k] = f
                counts[k] += n
        total.update(counts)
        task_keys.append((case_id(s), counts))
    tasks = [(k, p, f) for (k, p), f in requests.items()]
    if quick:
        seen = set()
        sample = []
        for k, p, f in tasks:
            identity = (
                f["raw_name"],
                tuple((t["dtype"], len(t["shape"])) for t in f["inputs"]),
            )
            if identity not in seen:
                seen.add(identity)
                sample.append((k, p, f))
        tasks = sample
    out.mkdir(parents=True, exist_ok=True)
    quoted = {}
    errors = {}
    issues = Counter()
    ops = defaultdict(Counter)
    with ProcessPoolExecutor(workers) as pool:
        for key, p, row, error in pool.map(
            partial(price, mode=mode), tasks, chunksize=48
        ):
            k = (key, p)
            count = total[k]
            name = requests[k]["raw_name"]
            tally = ops[name]
            tally["nodes"] += count
            tally["signatures"] += 1
            if error:
                issues[error] += count
                errors[k] = error
            else:
                quoted[k] = row
                tally["priced_nodes"] += count
                tally["priced_signatures"] += 1
                tally["zero_nodes"] += count * (row[1] == 0)
                tally["serial_us"] += count * row[1]

    def file_key(k):
        return f"{k[0]}:{k[1]}" if k[1] else k[0]

    if not summary_only:
        write_packet(
            out / "quotes.json.gz",
            dict(
                fields=[
                    "raw_name",
                    "duration_us",
                    "contract",
                    "status",
                    "cube_by_dtype",
                    "vec_by_dtype",
                    "read_bytes",
                    "write_bytes",
                    "query_us",
                ],
                quotes={file_key(k): r for k, r in quoted.items()},
                errors={file_key(k): r for k, r in errors.items()},
            ),
        )
    geometries = defaultdict(set)
    dtypes = defaultdict(set)
    costs = defaultdict(list)
    for key, p, row in tasks:
        op = row["raw_name"]
        tensor_list = row["inputs"] + row["outputs"]
        geometries[op].add(tuple(tuple(t["shape"]) for t in tensor_list))
        dtypes[op].update(t["dtype"] for t in tensor_list)
        if (key, p) in quoted:
            costs[op].append(quoted[key, p][-1])
    by_op = [
        dict(
            raw_name=n,
            physical_shape_tuples=len(geometries[n]),
            captured_tensor_dtypes="|".join(sorted(dtypes[n])),
            median_query_us=median(costs[n]) if costs[n] else "",
            **{
                field: c[field]
                for field in (
                    "nodes",
                    "signatures",
                    "priced_nodes",
                    "priced_signatures",
                    "zero_nodes",
                    "serial_us",
                )
            },
        )
        for n, c in sorted(ops.items())
    ]
    write_csv(out / "operators.csv", by_op)
    if not quick:
        by_case = [
            dict(
                case=k,
                nodes=sum(c.values()),
                priced_nodes=sum(n for p, n in c.items() if p in quoted),
                signatures=len(c),
                serial_us=sum(n * quoted[p][1] for p, n in c.items() if p in quoted),
            )
            for k, c in task_keys
        ]
        write_csv(out / "tasks.csv", by_case)
    result = dict(
        sampled=quick,
        attempts=len(tasks),
        priced=len(quoted),
        failures=len(errors),
        elapsed_s=(perf_counter_ns() - start) / 1e9,
        mode=mode,
        workers=workers,
        model_sha256=digest,
        input_hashes={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (data / "cases.json.gz", data / "features.json.gz")
        },
        chip_sha256=hashlib.sha256(CHIP.read_bytes()).hexdigest(),
        errors=[[*e, n] for e, n in issues.most_common()],
    )
    (out / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    if model_digest() != digest:
        raise ValueError("Model changed during replay; use a fresh frozen run")
    return not errors


def recheck_slow(data, out):
    """Repeat the defaults above the 100 ms query budget without competing replay workers."""
    features = json.loads(gzip.decompress((data / "features.json.gz").read_bytes()))
    quotes = json.loads(gzip.decompress((out / "quotes.json.gz").read_bytes()))[
        "quotes"
    ]
    subset = sorted(
        (k for k, row in quotes.items() if row[-1] >= 100_000),
        key=lambda k: -quotes[k][-1],
    )
    hw = hardware()
    fields = ("key", "raw_name", "concurrent_us", "median_us", "max_us")
    with (out / "slow-query-recheck.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key in subset:
            sig, _, pages = key.partition(":")
            feat = unpack(features[sig])
            b = backend(int(pages or 0), "auto")
            times = []
            for _ in range(5):
                start = perf_counter_ns()
                result = b.predict(feat, hw)
                times.append((perf_counter_ns() - start) / 1000)
            if not math.isclose(result.duration_us, quotes[key][1], abs_tol=1e-8):
                raise ValueError(f"Unstable device estimate for {key}")
            writer.writerow(
                dict(
                    key=key,
                    raw_name=feat.raw_name,
                    concurrent_us=quotes[key][-1],
                    median_us=median(times),
                    max_us=max(times),
                )
            )



def compare_methods(data, out):
    """Join identically weighted per-operator ledgers from the three frozen runs."""
    modes = ("prices", "calibrated", "instruction_classes")
    audits = [json.loads((data / mode / "audit.json").read_text()) for mode in modes]
    for audit in audits:
        for key in ("input_hashes", "model_sha256", "chip_sha256", "attempts"):
            if audit[key] != audits[0][key] or audit["failures"] or audit["sampled"]:
                raise ValueError("Comparison requires three complete, matched runs")
    tables = []
    for directory in modes:
        with (data / directory / "operators.csv").open() as stream:
            tables.append({row["raw_name"]: row for row in csv.DictReader(stream)})
    auto, calibrated, instructions = tables
    if auto.keys() != calibrated.keys() or auto.keys() != instructions.keys():
        raise ValueError("Operator sets differ")
    joined = []
    keys = ("raw_name", "physical_shape_tuples", "captured_tensor_dtypes", "nodes", "signatures")
    for name, row in sorted(auto.items()):
        record = {key: row[key] for key in keys}
        for label, table in zip(("auto", "calibrated", "instruction_classes"), tables):
            if any(record[k] != table[name][k] for k in keys):
                raise ValueError(f"Unmatched coverage for {name}")
            record[label + "_serial_us"] = table[name]["serial_us"]
        base = float(calibrated[name]["serial_us"])
        fine = float(instructions[name]["serial_us"])
        record["instruction_vs_calibrated_pct"] = f"{100 * (fine / base - 1):.4f}" if base else ""
        joined.append(record)
    write_csv(out, joined)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "results/v2.4/catalog")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--mode", choices=("auto", "calibrated", "instruction_classes"), default="auto"
    )
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--recheck-slow", action="store_true")
    parser.add_argument("--compare-modes", action="store_true", help="write a joined CSV from the three published mode ledgers")
    args = parser.parse_args()
    if args.compare_modes:
        compare_methods(args.data, args.out)
        raise SystemExit(0)
    if args.recheck_slow:
        recheck_slow(args.data, args.out)
        raise SystemExit(0)
    raise SystemExit(
        0
        if replay(
            args.data, args.out, args.workers, args.quick, args.mode, args.summary_only
        )
        else 1
    )
