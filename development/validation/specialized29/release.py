"""Check the installed public backend, timing candidates and archived CA in isolation."""
from collections import defaultdict
from hashlib import sha256
import json
from math import ceil, isclose
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/v0.29"
OLD = ROOT / "results/v0.28"
DT, PR, FUT = "Ascend950DT_9582", "Ascend950PR_957b", "Ascend970DT"


def read(path):
    return json.loads(path.read_bytes())


def store(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def run():
    import operator_model as api
    from operator_model.matmul_placement import matmul_placement
    from operator_model.matmul_full_load import estimate_matmul_full_load
    from operator_model.quant_stream_memory import scenario
    from operator_model.attn_res_prefill_plan import prefill_service_plan

    if not sys.flags.isolated or api.__version__ != "0.29.0" or Path(api.__file__).is_relative_to(ROOT):
        raise RuntimeError("Run with python -I from the installed v0.29 wheel")
    cache, calls = {}, defaultdict(list)

    def timed(name, func, *args, **kwargs):
        start = perf_counter_ns()
        result = func(*args, **kwargs)
        calls[name].append((perf_counter_ns() - start) / 1e6)
        return result

    def query(op, shape, attrs, target):
        key = (op, tuple(shape), json.dumps(attrs, sort_keys=True), target)
        if key not in cache:
            cache[key] = timed("estimate", api.estimate, op, shape, attrs, chip=target)
        return cache[key]

    def equal(actual, expected):
        if actual is None or expected is None:
            assert actual is expected
        else:
            assert isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-10), (actual, expected)

    groups = defaultdict(list)
    rows = []
    for old in read(OLD / "replay.json")["rows"]:
        name = {"fla_kda_v2": "kda_chunk", "vllm_hc_post": "mhc_post"}.get(old["op"], old["op"])
        shape, attrs, chip = old["shape"], old["attrs"], old["chip"]
        current = query(name, shape, attrs, chip)
        future = query(name, shape, attrs, FUT)
        equal(current.latency_us, old["current_us"])
        equal(future.latency_us, old["future_970dt_extrapolated_us"])
        row = dict(old, previous_us=old["current_us"], current_us=current.latency_us,
                   future_970dt_extrapolated_us=future.latency_us, status=current.status)
        rows.append(row)
        if current.latency_us is not None:
            groups[old["op"], chip].append(row)
    summaries = []
    for (op, chip), group in sorted(groups.items()):
        errors = [abs(x["ca_us"]-x["current_us"]) for x in group]
        summaries.append(dict(op=op, chip=chip, shapes=len({tuple(x["shape"]) for x in group}),
                              windows=len(group), wape=100*sum(errors)/sum(x["ca_us"] for x in group),
                              maximum=max(100*e/x["ca_us"] for e, x in zip(errors, group))))
    store(OUT / "replay.json", dict(role="Qualified historical full CA windows repriced with the installed "
          "wheel. Development replay; does not establish new independent coverage.", summary=summaries, rows=rows))
    # The two prices below have different roles; v0.28's public price came from a frozen release wheel.
    for job in read(OLD / "matmul/overview-prices.json")["rows"]:
        for chip, frozen in job["predictions"].items():
            equal(query("matmul", job["shape"], {}, chip).latency_us, frozen["us"])
    for job in read(OUT / "matmul/timing-freeze.json")["rows"]:
        for chip, frozen in job["predictions"].items():
            p = timed("matmul_placement", matmul_placement, job["shape"], chip=chip)
            equal(p["latency_us"], frozen["latency_us"])
            assert p["route"] == frozen["route"]
    for job in read(OUT / "matmul/full-load/freeze.json")["rows"]:
        observed = timed("matmul_full_load", estimate_matmul_full_load, job["shape"], chip=job["chip"])
        equal(observed["latency_us"], job["candidate"]["latency_us"])
        assert observed["status"] == job["candidate"]["status"]
        assert observed.get("route") == job["candidate"].get("route")
    quant = read(OUT / "quant/freeze.json")
    for group, shape in quant["shapes"].items():
        for chip, frozen in quant["predictions"][group].items():
            cfg = api.load_chip(chip)
            base = query("quant", shape, {}, chip).latency_us
            p = timed("quant_stream_memory", scenario, shape, cfg, base,
                      mixed_bus_efficiency=frozen["mixed_bus_efficiency"],
                      wire_to_configured_bandwidth=frozen["wire_to_configured_bandwidth"])
            equal(p["latency_us"], frozen["latency_us"])
    for chip in (DT, PR, FUT):
        job=next(j for j in read(OUT/"large/vector/freeze.json")["jobs"] if j["op"]=="attn_res_fwd")
        value=timed("attn_res_prefill_service",prefill_service_plan,job["shape"],chip)
        equal(value["conditional_schedule_us"],job["prices"][chip]["ideal_service"]["conditional_schedule_us"])
        assert value["qualified_latency_us"] is None
        stream = api.OperatorStream(chip, track_post_history=True)
        for shape in ((64,4,256),(63,4,320),(63,4,320),(65,4,5120)):
            assert timed("ordered_post", stream.advance, "mhc_post", shape).latency_us > 0
        plan = timed("kimi_kda_v1", api.plan_kimi_kda_prefill, [2048]*8, attention_tp=8, chip=chip)
        assert plan["main_kernels_us"] is None and plan["full_dispatch_us"] is None
        assert plan["v1_work"]["finalize"]["logical_cube_ops"] == 9663676416

    costs = {name: dict(n=len(v), median_ms=median(v), p95_ms=sorted(v)[ceil(.95*len(v))-1],
                        maximum_ms=max(v)) for name, v in calls.items()}
    wheel, = (ROOT / "build/wheels/v029").glob("*.whl")
    store(OUT / "wheel-check.json", dict(version=api.__version__, wheel_sha256=sha256(wheel.read_bytes()).hexdigest(),
          history_rows=len(rows), cost=costs, protocol="Python call only after import, one public query per "
          "distinct input, no CANN or CA. Explicit quant stage excludes its separately checked base query."))
    assert all(v["maximum_ms"] < 100 for v in costs.values()), costs
    def fmt(v):
        return "N/A" if v is None else f"{v:.3f}"
    lines = ["# 各算子对历史合格 CA 窗的同集回放", "",
             "本表的单位是 µs。DT/PR 均按目标配置重新预测。每行保留当次调用的属性与实际 kernel。同形不同前序可有多行。采前冻结的新设备窗及原生 wrapper 另计。[专项与全套指标](../../docs/error-analysis.md) · [逐窗指纹和数值](replay.json)", "",
             "| 算子 · 库 kernel | SKU | shape · 属性 | v0.28 | v0.29 | CA | 偏差 | 970DT 外推 |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        us = r["current_us"]
        err = "N/A" if us is None else f"{100*(us/r['ca_us']-1):+.1f}%"
        lines.append(f"| {r['op']} · {r['kernel']} | {r['chip']} | {r['shape']} · {r['attrs']} | "
                     f"{fmt(r['previous_us'])} | {fmt(us)} | {fmt(r['ca_us'])} | {err} | "
                     f"{fmt(r['future_970dt_extrapolated_us'])} |")
    (OUT / "shape-errors.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(costs))


if __name__ == "__main__":
    run()
