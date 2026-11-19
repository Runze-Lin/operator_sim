"""Render a read-only cross-check of the two short native CA controls."""

from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/v0.23/full-hidden"
ACTIVE = OUT / "v2"


def load(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def percent(value):
    return f"{value * 100:.3f}%"


def object_reads(path):
    """Only successful opens prove that the selected vendor object was accessible."""
    expr = re.compile(r'openat\([^,]*, "([^"]*/[Cc]ast_[^"]+\.o)", O_RDONLY\)\s+=\s+\d+')
    objects = sorted(set(expr.findall(path.read_text())))
    return [{"path": str(Path(p).resolve().relative_to(ROOT)),
             "sha256": sha256(Path(p).read_bytes()).hexdigest()} for p in objects]


def controls():
    previous = load(OUT / "previous-partial.json")
    receipts = [load(ACTIVE / (name + ".json")) for name in ("control-original", "control-redirected")]
    archive = load(ROOT / "results/v0.20/large/cast_32_4096_native.json")
    comparison = load(ACTIVE / "control-comparison.json")
    rows = []
    for receipt in receipts:
        raw = ROOT / receipt["cost"]["raw"]
        rows.append({"raw": str(raw.relative_to(ROOT)),
                     "qualified": receipt["qualified_core_envelope"],
                     "core_us": receipt.get("core_envelope_us"), "active_aiv": receipt.get("active_aiv"),
                     "golden_elements": receipt.get("correct_elements"),
                     "only_write_opens": receipt["all_writes_only"],
                     "names_seen": len(receipt["controller_open_flags"]),
                     "periodic_file_bytes": receipt["controller_output_bytes"],
                     "controller_symlinks": receipt["state_symlinks"],
                     "wall_seconds": receipt["cost"]["seconds"],
                     "native_object_opens": object_reads(raw / "open.trace"),
                     "open_trace_sha256": receipt.get("open_trace_sha256")})
    evidence = {"controls": rows, "qualified_control_interchange": comparison["eligible_for_redirect"],
                "archive_reference_us": archive["core_envelope_us"]}
    if all(r["qualified"] for r in rows):
        compare_lanes = []
        a, b = [r["evidence"]["wrapper"] for r in receipts]
        for lane in sorted(a.keys() & b.keys(), key=int):
            x, y = a[lane], b[lane]
            compare_lanes.append({"lane": int(lane), "same_block": x["block"] == y["block"],
                                  "baseline_ticks": [x["start"], x["done"]], "redirect_ticks": [y["start"], y["done"]],
                                  "delta_duration_ticks": (y["done"] - y["start"]) - (x["done"] - x["start"])})
        evidence["same_roster_and_blocks"] = a.keys() == b.keys() and all(r["same_block"] for r in compare_lanes)
        evidence["exact_equal_lanes"] = sum(r["baseline_ticks"] == r["redirect_ticks"] for r in compare_lanes)
        evidence["lane_comparison"] = compare_lanes
    write(ACTIVE / "control-evidence.json", evidence)
    return previous, rows, evidence, comparison


def main():
    prev, rows, ev, cmp = controls()
    missing = sorted(set(map(int, prev["logs"]["block_start"])) - set(map(int, prev["logs"]["block_end"])))
    reg = load(ACTIVE / "registration.json")
    status = load(ACTIVE / "queue-status.json")
    costs = prev["original_cost"]
    ratio = prev["periodic_stat_bytes"] / costs["raw_bytes"]
    lines = [
        "# 950DT 上的完整 prefill Cast",
        "",
        "DeepSeek V4 Flash 场景按每卡 batch 8、每条请求 32,768 个 token、hidden size 5,120 推导。未切 chunk 时，单次 BF16→FP32 Cast 输入为 [262144,5120]，宿主逐项校验 1,342,177,280 个输出。输入约 2.5 GiB，输出约 5 GiB。该实验对应一次调度调用。系统按 chunk 或分段运行时应分别预测各次实际调用。",
        "",
        "## 第一次完整输入采集",
        "",
        f"原始运行的 64 个 AIV 全部开始，只有 {prev['completed_aiv']} 个有配对的 block_end 及同一 block 的 wrapper done。缺少结束的 AIV 为 {', '.join(map(str, missing))}。已经结束的核中最早开始到最晚退出相差 3,465,175 个 wrapper tick。在 1.65 GHz 下完整设备窗至少为 **{prev['partial_wrapper_lower_bound_us']:.3f} µs**。STARS 调度 tick 和全机仿真进度计数各用不同时间基准。进程触及磁盘预算后停止，全量数值校验、整核窗口和指令窗口均未取得，主误差表的 CA 与误差均应填 N/A。",
        "",
        f"上次花费 {costs['seconds']/3600:.3f} 小时，落盘 {costs['raw_bytes']/2**30:.3f} GiB。128 份 lpddr_state 周期状态文本共 {prev['periodic_stat_bytes']/2**30:.3f} GiB，占 {percent(ratio)}。保留了原目录及按核摘要 SHA256。[31 个核的原始 tick 与摘要哈希](previous-partial.json)",
        "",
        "![上部为最长核服务和共享存储两条时间预算，下部为本次 31 个完整核窗口，两个面板共用微秒轴](cast-full-hidden-partial.png)",
        "",
        "图上半的两行都从零起算。它们分别表示最长 AIV 的双缓冲流水耗时、共享存储的全芯片吞吐下界，取较大者。图下半每一条横线都对应一个真实 wrapper 开始和完成事件，以这 31 个核中最早的开始作为零点。",
        "",
        "## 相同 shape 的短控",
        "",
        "两个新进程分别按完整输出和选择性丢弃周期文本运行相同 [32,4096] 输入、相同原生宿主、相同 CPU 亲和性。仅在新的空目录把预登记的 128 个输出名指向 /dev/null。系统调用记录保留文件打开标志和实际打开的 kernel 对象；这项额外记录只在短控启用。",
        "",
        "| 采集 | 设备 wrapper 窗 µs | 活动 AIV | 逐项通过 | 周期日志 MiB | 宿主秒 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 既有全输出对照 | {ev['archive_reference_us']:.6f} | 64 | 131072 | — | 54.1 |"]
    for name, row in zip(("新全输出", "新重定向"), rows):
        tm = "N/A" if row["core_us"] is None else f"{row['core_us']:.6f}"
        lines.append(f"| {name} | {tm} | {row['active_aiv']} | {row['golden_elements']} | {row['periodic_file_bytes']/2**20:.2f} | {row['wall_seconds']:.1f} |")
    lines += ["", f"预登记的重定向门槛{'通过' if cmp['eligible_for_redirect'] else '未通过'}。两次短控分别观察到 {rows[0]['names_seen']} 和 {rows[1]['names_seen']} 个周期文件，以写入方式打开的检查结果分别是 {rows[0]['only_write_opens']} 和 {rows[1]['only_write_opens']}。"]
    if ev.get("same_roster_and_blocks"):
        lines += [f"两个完整设备窗均覆盖相同的 64 个核及 block，{ev['exact_equal_lanes']} 个核的开始和结束 tick 均逐项相等。"]
    if rows[0]['native_object_opens'] and rows[0]['native_object_opens'] == rows[1]['native_object_opens']:
        objects = rows[0]['native_object_opens']
        lines += [f"短控实际打开安装包 ops_math/cast_apt 对象 {', '.join(Path(row['path']).name for row in objects)}。对象 SHA256、逐核 tick 和系统调用 trace 的 SHA256 可复核。这个对象身份由短控直接观测，大 shape 的原生运行按实际取得的证据登记。"]
    lines += ["", "[两次短控证据](v2/control-evidence.json) · [预登记门槛及结果](v2/control-comparison.json)",
              "", "## 完整 shape 重采", "",
              "保留 v0.20 事前冻结的 DT 预测 2014.328 µs 以及首次重试前冻结的资源模型预测 DT 2514.820 µs、PR 5697.327 µs、970DT 1223.154 µs。这里直接观测的平台仅为 Ascend950DT_9582，970DT 数值来自外推。",
              "",
              f"新实验使用原生 aclnnCast，以整个输入逐项校验并要求 64 个 AIV 的 wrapper 起止齐全。预算 24 小时，按短控门槛选择{'仅丢弃周期文本，16 GiB' if cmp['eligible_for_redirect'] else '保留所有日志，48 GiB'}。全量路径关闭逐指令采集和短控的系统调用采集。独立的 Kimi [262144,7168] DynamicMxQuant 占用唯一的长任务槽，其预算为 12 小时、24 GiB。Flash 管理进程会在 Kimi 终止并释放进程后启动。登记时的队列状态为 `{status['state']}`，更新时间 {status['utc']}。最终可取得的计时口径为全核 wrapper 窗。", "",
              "[原输入、宿主和预算冻结](v2/registration.json) · [实时队列](v2/queue-status.json)"]
    terminal = ACTIVE / "flash-retry.json"
    if terminal.exists():
        full = load(terminal)
        cost = full["cost"]
        lines += ["", "### 完整输入终态", "",
                  f"宿主耗费 {cost['seconds']/3600:.3f} 小时，原始输出 {cost['raw_bytes']/2**30:.3f} GiB。"]
        if full["qualified_core_envelope"]:
            us = full["core_envelope_us"]
            pred = reg["previous_frozen_us"]["v021_before_retry"]["dt"]
            lines.append(f"{full['active_aiv']} 个活动 AIV 提供完整 wrapper 窗，逐项正确 {full['correct_elements']:,}。实测设备窗口 {us:.3f} µs，相对该冻结 DT 预测的有符号差为 {(pred/us-1)*100:+.2f}%。原生采集仍仅覆盖 wrapper 口径。")
        else:
            lines.append(f"完整数值和计时资格未取得。退出码 {cost['exit_code']}，预算终止原因 {cost['stop_reason']}。完整窗及误差 N/A。")
        lines.append("[完整输入回执](v2/flash-retry.json)")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


def follow():
    """Finalise evidence once the supervisor publishes its controls and long-run receipt."""
    for receipt in ("control-comparison.json", "flash-retry.json"):
        deadline = time.monotonic() + 40 * 3600
        while not (ACTIVE / receipt).exists():
            if time.monotonic() > deadline:
                raise TimeoutError(receipt)
            time.sleep(30)
        main()


if __name__ == "__main__":
    if sys.argv[1:] == ["follow"]:
        follow()
    else:
        main()
