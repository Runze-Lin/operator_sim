"""Report only numerically checked, all-core CA windows against the frozen predictions."""

from pathlib import Path
from time import perf_counter_ns, sleep
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from development.validation.sigmoid20.campaign import (
    DEST,
    DT,
    PR,
    FUTURE,
    read,
    write,
)
from operator_model import estimate, load_chip


def metrics(pairs):
    if not pairs:
        return dict(n=0, wape_pct=None, max_ape_pct=None)
    errs = [abs(p - a) for p, a in pairs]
    return dict(
        n=len(errs),
        wape_pct=100 * sum(errs) / sum(a for _, a in pairs),
        max_ape_pct=100 * max(err / a for err, (_, a) in zip(errs, pairs)),
    )


def report():
    f = read(DEST / "holdout-freeze.json")
    records = read(DEST / "holdout.json") if (DEST / "holdout.json").exists() else []
    by = {(tuple(r["shape"]), r["sku"]): r for r in records}
    paired = []
    samples = {chip: [] for chip in (DT, PR)}
    speed = []
    reuse = []
    query = []
    for j in f["jobs"]:
        sh = tuple(j["shape"])
        row = {"shape": sh, "devices": {}}
        for c in (DT, PR, FUTURE):
            e = j["predictions"][c]
            pred = e["us"]
            chip = load_chip(c)
            for _ in range(24):
                begin = perf_counter_ns()
                actual = estimate("sigmoid_fp32", sh, dtype="fp32", chip=chip)
                query.append((perf_counter_ns() - begin) / 1e6)
            if abs(actual.latency_us - pred) > 1e-9:
                raise ValueError("frozen predictor mismatch")
            result = dict(
                model_us=pred,
                model_aiv=e["aiv"],
                critical_core_tiles=e["plan"]["critical_core_tiles"],
            )
            if c == FUTURE:
                result["role"] = "future extrapolation"
            else:
                r = by.get((sh, c))
                result["role"] = "frozen holdout"
                result["status"] = (
                    "qualified"
                    if r and r["qualified"]
                    else ("unqualified" if r else "pending")
                )
                if r and r["qualified"]:
                    k = r["kernel"]
                    result.update(
                        ca_us=k["ca_us"],
                        ca_aiv=k["active_aiv"],
                        signed_error_pct=100 * (pred / k["ca_us"] - 1),
                        kernel=k["kernel"],
                        ca_wall_seconds=r["cost"]["seconds"],
                    )
                    samples[c].append((pred, k["ca_us"]))
                    if c == PR:
                        reuse.append((j["predictions"][DT]["us"], k["ca_us"]))
                elif r:
                    result["reason"] = r["reason"]
            row["devices"][c] = result
        a, b = [row["devices"][c] for c in (DT, PR)]
        if "ca_us" in a and "ca_us" in b:
            p, a0 = a["model_us"] / b["model_us"], a["ca_us"] / b["ca_us"]
            speed.append((p, a0))
            row["speedup_dt_over_pr"] = dict(predicted=p, measured=a0)
        paired.append(row)
    summary = {c: metrics(v) for c, v in samples.items()}
    summary["paired_speedup_dt_over_pr"] = metrics(speed)
    summary["reuse_dt_prediction_on_pr"] = metrics(reuse)
    qsorted = sorted(query)
    out = dict(
        frozen_utc=f["utc"],
        distinct_shapes=len(f["jobs"]),
        distinct_flattened_element_counts=len(
            {j["shape"][0] * j["shape"][1] for j in f["jobs"]}
        ),
        completed_launches=len(records),
        qualified_launches=sum(r["qualified"] for r in records),
        summary=summary,
        rows=paired,
        query_ms=dict(
            n=len(query),
            p50=median(query),
            p95=qsorted[int(0.95 * (len(query) - 1))],
            max=max(query),
            scope="Fresh analytical estimate per call, preloaded chip, repeated in-process; includes Python arithmetic and output accounting.",
        ),
        ca_cost=dict(
            wall_seconds_sum=sum(r["cost"]["seconds"] for r in records),
            retained_bytes=sum(r["cost"]["raw_bytes"] for r in records),
            peak_bytes_single=max(
                (r["cost"]["peak_bytes"] for r in records), default=0
            ),
        ),
        dev=read(DEST / "development.json"),
    )
    write(DEST / "evaluation.json", out)

    def stage(r, c):
        v = r["devices"][c]
        ca = v.get("ca_us")
        gap = v.get("signed_error_pct")
        cores = v.get("ca_aiv")
        return (
            f"{v['model_us']:.3f} / {ca:.3f} / {gap:+.1f}% / {v['model_aiv']}→{cores}"
            if ca
            else f"{v['model_us']:.3f} / **{v['status']}** / — / {v['model_aiv']}→—"
        )

    lines = [
        "# Kimi K3 beta Sigmoid 冻结留出集",
        "",
        f"已完成 {len(records)} / 20 次离线 CA，其中 {out['qualified_launches']} 个窗口通过全核窗口及全元素数值核验。输入全为连续 FP32 `[T,H_local]`，设备耗时单位 µs。开头三个超长场景分别按 B8×32k、128k、256k 推导；其余记录覆盖 TP、decode 和 tile 边界。10 个 shape 有 9 个不同的展平元素数；`[9,96]` 和 `[72,12]` 均为 864 元素，是可以对照的等工作量调用。",
        "",
        "| Shape | DT 预测 / CA / 误差 / AIV 预测→实测 | PR 预测 / CA / 误差 / AIV 预测→实测 | 970DT 外推 µs |",
        "|---|---|---|---:|",
    ]
    priority = {(262144, 12): 0, (1048576, 12): 1, (2097152, 12): 2}
    for r in sorted(paired, key=lambda r: (priority.get(r["shape"], 3), r["shape"])):
        lines.append(
            f"| {list(r['shape'])} | {stage(r, DT)} | {stage(r, PR)} | {r['devices'][FUTURE]['model_us']:.3f} |"
        )
    lines += [
        "",
        "| 留出指标 | 已完成数 | WAPE | 最大相对误差 |",
        "|---|---:|---:|---:|",
    ]
    for name, s in summary.items():
        w = s["wape_pct"]
        m = s["max_ape_pct"]
        vals = f"{w:.2f}% | {m:.2f}%" if w is not None else "— | —"
        lines.append(f"| {name} | {s['n']} | {vals} |")
    lines += [
        "",
        f"在线查询 {len(query)} 次，P50 {median(query):.3f} ms，P95 {out['query_ms']['p95']:.3f} ms，最大 {max(query):.3f} ms。离线 CA 各进程墙钟的和为 {out['ca_cost']['wall_seconds_sum'] / 60:.1f} 分钟，两项并行时有重叠。当前已完成项合计保留 {out['ca_cost']['retained_bytes'] / 2**30:.2f} GiB。",
        "",
        "速度比的方向取 T_DT/T_PR。每行误差是 `(预测−CA)/CA`。独立窗口采用最早 AIV 开始至最晚 AIV 完成，单位通过逐核 vendor 表二次核对。`pending` 样本不进入误差分母；`unqualified` 表示此次运行未获得满足所有条件的窗口。970DT 只有外推。完整计时、展开通道数、实际 kernel/binary、误差、仿真开销分别见 [冻结清单](holdout-freeze.json)、[逐形状数据](evaluation.json)。",
    ]
    (DEST / "measurements.md").write_text("\n".join(lines) + "\n")
    return len(records)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        sig = None
        while True:
            hold = DEST / "holdout.json"
            stamp = hold.stat().st_mtime_ns if hold.exists() else 0
            if stamp != sig:
                done = report()
                sig = stamp
                print(f"updated: {done}/20 completed", flush=True)
                if done == 20:
                    break
            sleep(30)
    else:
        report()
