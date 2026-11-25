"""Replay the published device windows, retaining original calls, prefixes and evidence roles."""

import csv
import gzip
import io
import json
from collections import defaultdict
from itertools import chain
from math import ceil
from pathlib import Path
from time import perf_counter_ns

import operator_model as api


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/v2.3"
REF = ROOT / "calibration/reference"
DT, PR, FUTURE = "Ascend950DT_9582", "Ascend950PR_957b", "Ascend970DT"
MODES = ("calibrated", "instruction_classes")


def cases():
    with (REF / "windows.jsonl").open() as f:
        for i, line in enumerate(f, 1):
            yield {**json.loads(line), "source": f"windows:{i}"}
    for filename in (
        "v2.1/new-ca/receipts.json",
        "v2.2/fa-validation/comparison.json",
        "v2.2/fa-validation/long/comparison.json",
    ):
        document = json.loads((ROOT / "results" / filename).read_text())
        for i, r in enumerate(document["rows"]):
            if r["status"] != "qualified":
                continue
            dtype = r.get("dtype", "bf16")
            yield dict(
                op="fa",
                contract="fa · " + dtype + " · supplemental",
                shape=r["shape"],
                attrs={},
                dtype=dtype,
                chip=r["chip"],
                api="aclnnFlashAttentionScoreV2",
                ca_us=r["ca_us"],
                data_role="published supplemental",
                source=f"{filename}#{i}",
            )


def timed_window(case, chip, mode):
    options = {**(case.get("attrs") or {}), "vector_model": mode}
    prefix = case.get("prefix")
    if prefix is None:

        def query():
            return api.estimate(case["op"], case["shape"], options, chip=chip, dtype=case.get("dtype"))
    else:
        stream = api.OperatorStream(chip)
        for event in prefix:
            if event == {"forget": True}:
                stream.forget_history()
            else:
                stream.advance(event["op"], event["shape"], event.get("attrs"), dtype=event.get("dtype"))

        def query():
            return stream.advance(case["op"], case["shape"], options, dtype=case.get("dtype"))

    before = perf_counter_ns()
    result = query()
    return result, (perf_counter_ns() - before) / 1e3


def score(group, key):
    if not group:
        return None
    errors = [abs(r[key] - r["ca_us"]) / r["ca_us"] for r in group]
    return [
        100 * sum(abs(r[key] - r["ca_us"]) for r in group) / sum(r["ca_us"] for r in group),
        100 * max(errors),
    ]


def side(group):
    common = [r for r in group if all(r.get(m + "_us") is not None for m in MODES)]
    cm = {m: score(common, m + "_us") for m in MODES}
    costs = {}
    for m in MODES:
        times = sorted(r[m + "_cpu_us"] for r in common)
        costs[m] = times[ceil(0.95 * len(times)) - 1] if times else None
    return dict(
        n=len(group),
        paired=len(common),
        paired_scores=cm,
        query_p95_us=costs,
        all_available={m: score([r for r in group if r[m + "_us"] is not None], m + "_us") for m in MODES},
    )


def display(v):
    return "N/A" if v is None else f"{v[0]:.2f} / {v[1]:.2f}"


def run():
    rows = []
    original = json.loads((REF / "summary.json").read_text())
    for i, r in enumerate(cases()):
        one = {
            k: r.get(k)
            for k in (
                "source",
                "op",
                "contract",
                "shape",
                "dtype",
                "chip",
                "data_role",
                "api",
                "attrs",
                "prefix",
                "ca_us",
            )
        }
        if one["dtype"] is None:
            from operator_model.workload import default_dtype

            one["dtype"] = default_dtype(one["op"])
        for target in (r["chip"], FUTURE):
            chip = api.load_chip(target)
            for mode in MODES if i % 2 else MODES[::-1]:
                result, wall = timed_window(r, chip, mode)
                if target == FUTURE:
                    one[mode + "_970dt_us"] = result.latency_us
                    continue
                one[mode + "_us"] = result.latency_us
                one[mode + "_cpu_us"] = round(wall, 3)
                one[mode + "_error_pct"] = (
                    None if result.latency_us is None else 100 * (result.latency_us / r["ca_us"] - 1)
                )
                one[mode + "_scope"] = (result.projection or {}).get("vector_model_scope", result.reason)
                if mode == "calibrated" and r.get("predicted_us") is not None:
                    if abs(result.latency_us - r["predicted_us"]) > 1e-6 * max(1.0, result.latency_us):
                        raise ValueError(f"Original estimate changed: {r['source']}")
        if one["instruction_classes_us"] is None:
            raise ValueError(
                f"Missing candidate on published measurement {r['source']}: {one['instruction_classes_scope']}"
            )
        rows.append(one)
    OUT.mkdir(exist_ok=True, parents=True)
    with (
        (OUT / "native-windows.csv.gz").open("wb") as f,
        gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as g,
        io.TextIOWrapper(g, encoding="utf8", newline="") as out,
    ):
        w = csv.DictWriter(out, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    k: (
                        json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                        if isinstance(v, (list, dict))
                        else v
                    )
                    for k, v in r.items()
                }
            )
    groups = defaultdict(list)
    for r in rows:
        groups[r["contract"]].append(r)
    summary = {}
    table = []
    for contract in dict.fromkeys(chain(original, groups)):
        group = groups[contract]
        roles = list(dict.fromkeys(r["data_role"] for r in group)) or ["no archived window"]
        # Keep old frozen and development cohorts separate, even when both are now replayed.
        for role in roles:
            sub = [r for r in group if r["data_role"] == role]
            ends = {sku: side([r for r in sub if r["chip"] == sku]) for sku in (DT, PR)}
            summary.setdefault(contract, {})[role] = ends

            def fields(e):
                orig, cl = (e["paired_scores"][m] for m in MODES)
                if not e["paired"] and e["all_available"][MODES[1]]:
                    cl = e["all_available"][MODES[1]]
                return [f"{e['paired']}/{e['n']}", display(orig), display(cl)]

            no_vf = contract in (
                "addmm",
                "gmm2",
                "matmul",
                "matmul_mx_native",
                "routed_grouped_matmul_mxfp4_v5",
            )
            policy = (
                "无 VF，同调度"
                if no_vf
                else "新精度默认指令法"
                if "fp16 · supplemental" in contract
                else "缺配对证据"
                if not sub
                else "原法"
            )
            label = {
                "replay": "开发回放",
                "independent": "历史冻结留出回放",
                "published supplemental": "既有专项 CA",
                "no archived window": "无配对窗",
            }[role]
            cost = []
            for m in MODES:
                values = sorted(r[m + "_cpu_us"] for r in sub if r[m + "_us"] is not None)
                cost.append(f"{values[ceil(0.95 * len(values)) - 1]:.0f}" if values else "—")
            table.append(
                "| "
                + " | ".join(
                    [contract, label, *fields(ends[DT]), *fields(ends[PR]), " / ".join(cost), policy]
                )
                + " |"
            )
    (OUT / "native-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    headings = "| 算子合同 | 数据角色 | DT 配对/CA | DT 原法 | DT 指令法 | PR 配对/CA | PR 原法 | PR 指令法 | 查询 P95 µs 原/指 | 默认或适用性 |"
    md = "\n".join(
        [
            "# 本地算子两种 Vector 方法的同窗误差",
            "",
            "每个有效预测均从发布 API 或有序流运行。两法共用芯片、精度、分核输入与历史，归档行号可以定位原始 API 和 kernel。无配对窗的合同单列。专项 FA 与主档案分开计分。",
            "",
            headings,
            "| " + " | ".join(["---"] * 10) + " |",
            *table,
            "",
            "**列说明**　合同是具体实现或调用历史；数据角色沿用采集时的身份，本轮全部为回放。配对/CA 是该芯片两法都能预测的窗数/合格 CA 窗数。四个误差列均为 WAPE % / 最大单窗绝对误差 %。WAPE 以 CA 设备时间加权。FP16 无原法，指令列单独与 CA 比。P95 是两个 SKU 所有成功查询的第 95 百分位 CPU 微秒，按原法/指令法排列；含该次调用的调度与结果构造，前序重建、CA 和导入时间另计。末列给出默认选择或缺少证据的原因。",
            "",
            "[逐形 CSV，含双端设备时间、CPU 查询、误差与 970DT 外推](../results/v2.3/native-windows.csv.gz) · [按芯片、数据角色的完整统计](../results/v2.3/native-summary.json)\n",
        ]
    )
    (ROOT / "docs/vector-method-comparison.md").write_text(md)
    report = ROOT / "docs/error-analysis.md"
    start, end = "<!-- vector-methods:start -->", "<!-- vector-methods:end -->"
    before, rest = report.read_text().split(start)
    _, after = rest.split(end)
    body = md.split("\n", 1)[1].strip()
    report.write_text(before + start + "\n\n" + body + "\n\n" + end + after)


if __name__ == "__main__":
    run()
