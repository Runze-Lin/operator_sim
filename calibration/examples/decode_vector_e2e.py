"""Reprice the published 128Ki, rank-0 target+draft rounds, keeping the native DES.

Run with the matching preview SIMPlatform Python, PYTHONPATH=src:.; scratch pickles
must be local and trusted. See docs/decode-vector-comparison.md for exact commands.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import pickle
from time import perf_counter
import tomllib

from calibration.candidate_indexer.context_sensitivity import PA_BLOCK, caps, check
from calibration.roofline_audit import nominal_engine_us
from calibration.system_compare import cases, near
from calibration.system_full_compare import FULL, write

ROOT = Path(__file__).resolve().parents[2]
MODES = ("auto", "calibrated", "instruction_classes")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def graph_digest(workload):
    h = hashlib.sha256()
    for graph in workload.graphs:
        for entity in (*graph.nodes, *graph.dependencies):
            h.update(repr(entity).encode() + b"\n")
    return h.hexdigest()


def capture(job):
    from sim_platform.api.inference import run_inference
    from sim_platform.config.inference import load_inference_task

    case, model, layout, batch, scratch, ref = job
    task = case / "task.toml"
    dest = scratch / case.name
    dest.mkdir(parents=True, exist_ok=True)
    local = dest / "graphs.pkl.gz"
    check(
        digest(task.read_bytes()) == ref["task_sha256"],
        f"{case.name}: original task changed",
    )
    if local.exists():
        return
    config = load_inference_task(
        task,
        overrides={
            "output.directory": str(dest / "run"),
            "exports.enabled": False,
            "logging.console_level": "ERROR",
        },
    )
    run = run_inference(replace(config, options=None))
    check(
        len(run.summaries) == 2 and all(s.execution_count == 1 for s in run.summaries),
        "stage count",
    )
    options = replace(
        config.options,
        hardware=config.hardware_variants[0].hardware.to_model_hardware(),
    )
    check(
        near(run.static_memory[0].peak_bytes / 1e9, float(ref["peak_gb"])),
        f"{case.name}: memory differs",
    )
    check(
        graph_digest(run.workload) == ref["graph_sha256"],
        f"{case.name}: graph fingerprint differs",
    )
    temp = local.with_suffix(".tmp")
    with gzip.open(temp, "wb", compresslevel=1) as handle:
        pickle.dump((run.workload, options), handle)
    temp.replace(local)


def replay(job):
    from operator_model.sim_backend import FullOperatorSimBackend
    from operator_model.system_routes import STRUCTURAL, communication
    from sim_platform.api.inference_evaluation import evaluate_inference_workload
    from sim_platform.backend.factory import build_predictor
    from sim_platform.backend.l1 import simulate_graph
    from sim_platform.config.execution import PredictionOptions

    case, model, layout, batch, scratch, ref = job
    start = perf_counter()
    with gzip.open(scratch / case.name / "graphs.pkl.gz", "rb") as f:
        workload, options = pickle.load(f)
    check(
        graph_digest(workload) == ref["graph_sha256"],
        f"{case.name}: capture fingerprint changed",
    )
    task = case / "task.toml"
    check(digest(task.read_bytes()) == ref["task_sha256"], f"{case.name}: task changed")
    config = tomllib.loads(task.read_text())["inference"]["runtime"]
    pages = (config["kv_capacity"] + PA_BLOCK - 1) // PA_BLOCK
    hw = options.hardware
    capacity = caps(options)
    cube, vec = (dict(engine.tflops_by_bits) for engine in (hw.chip.cube, hw.chip.vec))
    archive = case / "results/l1"
    hist = json.loads((archive / "inference.json").read_text())["statistics"][0]
    check(
        near(hist["duration_us"], float(ref["original_us"]))
        and hist["accepted_length"] == 7,
        "baseline or token normalization mismatch",
    )
    row = dict(
        case=case.name,
        model=model,
        layout=layout,
        batch=batch,
        peak_gb=float(ref["peak_gb"]),
        fits_144gb=ref["fits_144gb"],
        nodes=int(ref["nodes"]),
        kv_capacity=config["kv_capacity"],
        table_pages=pages,
        graph_sha256=ref["graph_sha256"],
        task_sha256=ref["task_sha256"],
    )
    substitutions = {}
    costs_per_mode = {}
    node_deltas = Counter()
    selected = {m: Counter() for m in MODES}
    typed_peak = []
    for mode in MODES:
        backend = FullOperatorSimBackend(
            vector_model=mode,
            memory_policy="cold_io",
            candidate_implementation="a5_qsli_projection",
            candidate_options={"pa_block_size": PA_BLOCK, "table_pages": pages},
        )
        predictor = build_predictor(
            PredictionOptions((FULL,), 8192), (backend,), hardware=hw
        )
        evaluations = evaluate_inference_workload(
            workload, predictor=predictor, options=options
        )
        costs_per_mode[mode] = []
        for i, (ev, (stage, baseline, repeat)) in enumerate(
            zip(evaluations, hist["stage_durations_us"], strict=True)
        ):
            check(
                stage == ("target", "draft")[i] and repeat == 1,
                "target/draft repeat mismatch",
            )
            costs = dict(ev.execution_costs)
            preds = dict(ev.predictions)
            costs_per_mode[mode].append(costs)
            check(len(costs) == len(preds) == len(ev.graph.nodes), "incomplete pricing")
            peak = costs.copy()
            for node in ev.graph.nodes:
                p, cost = preds[node.ref], costs[node.ref]
                check(
                    p.status == "success"
                    and p.backend_name == FULL
                    and near(cost.duration_us, p.estimate.duration_us),
                    f"{node.ref}: unpriced or altered ({p.status}, {p.backend_name})",
                )
                d = p.estimate.details
                selected[mode][
                    str(dict(d.model_parameters).get("vector_model", "not_applicable"))
                ] += 1
                if (
                    node.raw_name not in STRUCTURAL
                    and node.raw_name not in communication.NAMES
                ):
                    c = nominal_engine_us(d.cube_work_by_dtype, cube)
                    v = nominal_engine_us(d.vec_work_by_dtype, vec)
                    check(
                        c is not None and v is not None,
                        f"{node.raw_name}: missing precision",
                    )
                    ideal = max(
                        c,
                        v,
                        (d.read_bytes + d.write_bytes)
                        / (hw.chip.hbm.peak_bytes_per_second / 1e6),
                    )
                    check(
                        ideal <= cost.duration_us or near(ideal, cost.duration_us),
                        f"{node.raw_name}: below matched peak",
                    )
                    peak[node.ref] = replace(cost, duration_us=ideal)
                if mode == "instruction_classes":
                    old = costs_per_mode["calibrated"][i][node.ref]
                    check(
                        old.resources == cost.resources,
                        "resource claims differ between VF prices",
                    )
                    node_deltas[node.raw_name] += cost.duration_us - old.duration_us
            matched = simulate_graph(ev.graph, peak, capacity).duration_us
            if mode == "auto":
                typed_peak.append(matched)
                with (archive / f"{i:03}_verify_{stage}.csv").open() as f:
                    ledger = {
                        r["id"]: r
                        for r in csv.DictReader(f)
                        if r["record_type"] == "operator"
                    }
                check(len(ledger) == len(costs), "original ledger count")
                historical = {}
                for n in ev.graph.nodes:
                    r = ledger[n.ref.node_id]
                    check(
                        (r["raw_name"], r["graph_id"])
                        == (n.raw_name, ev.graph.graph_id),
                        "original node identity",
                    )
                    historical[n.ref] = replace(
                        costs[n.ref],
                        duration_us=float(r["duration_us"]),
                        resources=tuple(json.loads(r["resources"]).items()),
                    )
                original = simulate_graph(ev.graph, historical, capacity).duration_us
                check(near(baseline, original), f"{case.name}: historical DES differs")
                row[f"historical_{stage}_us"] = original
                row[f"peak_{stage}_us"] = matched
            else:
                check(
                    near(matched, typed_peak[i]),
                    f"{case.name}: VF method changed logical-work baseline",
                )
            row[f"{mode}_{stage}_us"] = ev.duration_us
    for mode in ("historical", "peak", *MODES):
        row[f"{mode}_us"] = row[f"{mode}_target_us"] + row[f"{mode}_draft_us"]
    for m in MODES:
        check(sum(selected[m].values()) == row["nodes"], "method coverage")
    leaders = sorted(
        (op for op, v in node_deltas.items() if abs(v) > 1e-8),
        key=lambda op: -abs(node_deltas[op]),
    )[:3]
    for op in leaders:
        times = []
        for i, graph in enumerate(workload.graphs):
            swap = costs_per_mode["calibrated"][i].copy()
            for n in graph.nodes:
                if n.raw_name == op:
                    swap[n.ref] = costs_per_mode["instruction_classes"][i][n.ref]
            times.append(simulate_graph(graph, swap, capacity).duration_us)
        substitutions[op] = dict(
            serial_delta_us=node_deltas[op],
            counterfactual_delta_us=sum(times) - row["calibrated_us"],
        )
    row["replay_wall_s"] = perf_counter() - start
    receipt = dict(
        case=row, single_operator_substitutions=substitutions, branch_counts=selected
    )
    (scratch / case.name / "vector-replay.json").write_text(
        json.dumps(receipt, allow_nan=False)
    )
    return receipt


def report(jobs, output, path):
    receipts = [
        json.loads((scratch / case.name / "vector-replay.json").read_text())
        for case, _, _, _, scratch, _ in jobs
    ]
    rows = [r["case"] for r in receipts]
    check(
        len(rows) == len({r["case"] for r in rows}) == 27,
        "incomplete original-27 matrix",
    )
    substitutions = [
        dict(case=r["case"]["case"], operator=op, **measure)
        for r in receipts
        for op, measure in r["single_operator_substitutions"].items()
    ]
    output.mkdir(parents=True, exist_ok=True)
    write(output / "original27.csv", rows)
    write(output / "original27-operator-swaps.csv", substitutions)
    (output / "original27-routing.json").write_text(
        json.dumps(
            {r["case"]["case"]: r["branch_counts"] for r in receipts},
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    names = dict(
        zip(
            ("v4-pro", "v41-flash", "kimi-k31"),
            ("DeepSeek-V4 Pro", "DeepSeek-V4.1 Flash", "Kimi K3.1 Pro"),
        )
    )
    lines = [
        "# 128Ki 历史的 27 组 decode：Vector 阶段的选择对整轮有多大影响？",
        "",
        "这是 950DT 系统模型的 rank 0 预测。每组有一张 target 图和一张 draft 图，原始节点、依赖、资源容量和精度在各方法之间保持一致。每图按完整资源约束重新调度；比较的是一次推测轮次，没有上板或整网 CA 真值。",
        "",
        "## 三个模型各自的变化",
        "",
        "| 模型（各 9 组） | 全指令相对原法，单组范围 | 自动相对原法，单组范围 | 自动比同口径峰值慢多少，单组范围 |",
        "|---|---:|---:|---:|",
    ]

    def delta(r, a, b):
        return (r[a + "_us"] / r[b + "_us"] - 1) * 100

    def span(values):
        return f"{min(values):+.2f}% 至 {max(values):+.2f}%"

    for model, name in names.items():
        group = [r for r in rows if r["model"] == model]
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(
                span([delta(r, a, b) for r in group])
                for a, b in [
                    ("instruction_classes", "calibrated"),
                    ("auto", "calibrated"),
                    ("auto", "peak"),
                ]
            )
            + " |"
        )
    routed = Counter()
    for receipt in receipts:
        routed.update(receipt["branch_counts"]["auto"])
    over = sum(r["fits_144gb"] == "False" for r in rows)
    lines += [
        "",
        "变化 = (该列方法的整轮耗时 / 参照耗时 − 1) × 100%。负号表示该方法预测得更快，不代表误差更小。各组单独比较；表中区间为九个比值的最小至最大值。",
        "",
        "![27 组单轮模型的自动与全指令报价相对原法的百分比，叉号表示超出方案显存](figures/decode-vector-comparison.png)",
        "",
        "图中每个点对应一个布局和一个本卡 batch。圆点是自动，菱形是全指令；加叉的行超出任务显存。零线表示同一预测，所有 panel 共用一个比例尺。",
        "",
        "## 27 组的单轮耗时",
        "",
        "| 模型 | 布局 | 单卡 B | 原峰值显存 GB | 历史 Roofline ms | 同精度峰值 ms | 原法 ms | 全指令 ms | 自动 ms | 自动 target / draft ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        times = " | ".join(
            f"{r[k + '_us'] / 1000:.3f}"
            for k in ("historical", "peak", "calibrated", "instruction_classes", "auto")
        )
        flag = " †" if r["fits_144gb"] == "False" else ""
        lines.append(
            f"| {names[r['model']]} | {r['layout']} | {r['batch']} | {r['peak_gb']:.2f}{flag} | {times} | {r['auto_target_us'] / 1000:.3f} / {r['auto_draft_us'] / 1000:.3f} |"
        )
    lines += [
        "",
        f"**读表。** B 是本卡 batch；† 表示超过当前系统场景的 144 GB（{over}/27 组），保留图模型时间作容量敏感性参考。原表里的 114.35、33.49 等是显存 GB。三种布局 (TP,DP,EP) 分别是 (8,1,8)、(8,4,32)、(1,64,64)。历史为 131072 token，8 个 draft 加一个 anchor，按接受 6 个 draft 加目标 token 计 7 个产出。单轮含 target 一次、draft 一次；不是完整 1024 token 的 decode 总时间。",
        "",
        "**两条基线。** “历史”从原逐节点账本恢复时长与资源再跑 DES，逐阶段对回归档。它保留旧后端的算法、精度代理与未定价节点，不是精细法的物理下界。“同精度峰值”使用当前各节点的 Cube/Vector 工作量及各 dtype 吞吐峰值、冷读写逻辑字节及 HBM 峰值，取最多的资源时间；通信和结构节点沿用同一合同，再跑 DES。三种 VF 方法分别重算峰值，并验证结果一致。",
        "",
        "**三种方法。** 原法 `calibrated` 选该实现原有的阶段报价，并不表示每条路径都有 CA；全指令 `instruction_classes` 换算具备 Vector 阶段的工作量；`auto` 按算子合同选择。元数据、纯通信或无 Vector 工作的路径没有可切换的 VF 时间。固定 `cold_io`；Flash 使用 A5 候选投影 `a5_qsli_projection`，PA 块 128、页数从该任务的 KV 容量向上取整。与 [33 组整批请求的三种方法对照](system-vector-e2e.md) 的使用假设一致。",
        "",
        f"默认路由在节点账本中记录：{routed['calibrated']:,} 次原法，{routed['instruction_classes']:,} 次指令法，{routed['calibrated+instruction_classes']:,} 次融合节点内混用两种，另有 {routed['not_applicable']:,} 次无适用 VF 报价。这里按捕获图中节点次数计数，不是独立算子名。",
        "",
        "## 哪些 VF 调价能传到整轮？",
        "",
        "对每组先按该算子所有节点的有符号耗时差绝对值选前三，再从“全原法”单独把该算子换成指令法并重跑 DES。下表列每个模型 9 组实验中绝对反事实变化最大的一例。此处反事实列真实包含图中的依赖等待和通信重叠；不同反事实通常不能相加。",
        "",
        "| 模型 | 布局 / B | 单独切换的捕获算子 | 该算子节点时长之和变化 ms | 只切换它的整轮变化 ms | 同组所有 VF 都切换的整轮变化 ms |",
        "|---|---|---|---:|---:|---:|",
    ]
    lookup = {r["case"]: r for r in rows}
    for model, name in names.items():
        best = max(
            (v for v in substitutions if lookup[v["case"]]["model"] == model),
            key=lambda v: abs(v["counterfactual_delta_us"]),
        )
        r = lookup[best["case"]]
        lines.append(
            f"| {name} | {r['layout']} / {r['batch']} | `{best['operator']}` | {best['serial_delta_us'] / 1000:+.3f} | {best['counterfactual_delta_us'] / 1000:+.3f} | {(r['instruction_classes_us'] - r['calibrated_us']) / 1000:+.3f} |"
        )
    lines += [
        "",
        "“节点时长之和变化”不扣除与其他节点重叠的时间；“整轮变化”均为重排后减去原法整轮，反映的是模型敏感性。它不能用来判断哪种方法更准确。应结合逐算子完整 CA 窗口，[查询 VF 误差和证据边界](system-vector-methods.md)。",
        "",
        "## 复核与复现",
        "",
        f"从原任务生成的 {len(rows) * 2} 张图共 {sum(r['nodes'] for r in rows):,} 次节点调用。每组三个精细后端均要求所有节点成功解析、执行时长与后端报价一致、逻辑峰值不越界；归档任务和图指纹逐一核对。原始图和 CA 均不进入结果 CSV。",
        "",
        "[逐配置数据（微秒，含两阶段与任务、图 SHA-256）](../results/v2.3/system-e2e/original27.csv) · [逐配置单算子切换的反事实](../results/v2.3/system-e2e/original27-operator-swaps.csv) · [分支命中详情](../results/v2.3/system-e2e/original27-routing.json)。CSV 的 `replay_wall_s` 是离线三种方法、同精度基线、反事实和一致性检查的墙钟，不是单次在线算子开销。",
        "",
        "使用匹配原任务的 SIMPlatform preview 和本仓源码。当前工作区的命令如下；`/tmp` 承载本机生成的可信捕获，首次形状捕获需与预览环境匹配的 torch_npu 依赖。",
        "",
        "```bash",
        "source toolchains/cann/9.2.1/ascend-toolkit/set_env.sh",
        "SIM=system-sim-workspace/preview-0.1.0-preview.1",
        'PYTHONPATH="src:.:$PYTHONPATH" "$SIM/.venv-runtime/bin/python" -m calibration.examples.decode_vector_e2e all "$SIM/experiments/decode-950dt-128k" /tmp/operator-vector-decode27 --workers 6',
        "```",
        "",
        "若需重绘图，使用有 matplotlib 的 Python，在相同参数下将 `all` 改成 `figure`；它只读取结果 CSV。",
        "",
    ]
    path.write_text("\n".join(lines))


def figure(data, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with data.open() as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.8), sharex=True, sharey=True)
    for ax, (model, title) in zip(
        axes,
        (
            ("v4-pro", "DeepSeek-V4 Pro"),
            ("v41-flash", "DeepSeek-V4.1 Flash"),
            ("kimi-k31", "Kimi K3.1 Pro"),
        ),
        strict=True,
    ):
        block = [r for r in rows if r["model"] == model]
        if len(block) != 9:
            raise ValueError("Expected nine layout/batch combinations")
        for i, r in enumerate(block):
            a, b = [
                100 * (float(r[m + "_us"]) / float(r["calibrated_us"]) - 1)
                for m in ("auto", "instruction_classes")
            ]
            ax.plot([a, b], [i, i], color="#bac4ca", linewidth=1.2)
            for x, marker, color in ((a, "o", "#177665"), (b, "D", "#b45221")):
                ax.scatter(x, i, marker=marker, s=42, color=color, zorder=4)
                if r["fits_144gb"] == "False":
                    ax.scatter(
                        x, i, marker="x", s=17, color="white", linewidth=0.9, zorder=5
                    )
        ax.set_title(title, size=11, weight="bold")
        ax.axvline(0, color="#718089", ls="--", lw=1)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.18)
        ax.tick_params(length=0, labelsize=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_yticks(range(9), [f"{r['layout']}  B{r['batch']}" for r in rows[:9]])
    axes[0].invert_yaxis()
    fig.suptitle(
        "128Ki decode: change in one speculative round",
        x=0.13,
        y=0.99,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    fig.legend(
        handles=[
            Line2D([], [], ls="none", color=c, marker=m, label=t)
            for c, m, t in [
                ("#177665", "o", "Default (auto)"),
                ("#b45221", "D", "Force instruction classes"),
            ]
        ],
        loc="upper left",
        bbox_to_anchor=(0.12, 0.935),
        frameon=False,
        ncol=2,
        fontsize=9,
    )
    fig.supxlabel("DES time relative to calibrated (%)", y=0.065, fontsize=10)
    fig.text(
        0.13,
        0.018,
        "White cross: static memory exceeds 144 GB. Same scales and same historical graph for each pair.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.86))
    fig.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("capture", "replay", "report", "all", "figure"))
    p.add_argument("matrix", type=Path)
    p.add_argument("scratch", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", type=Path, default=ROOT / "results/v2.3/system-e2e")
    p.add_argument(
        "--document", type=Path, default=ROOT / "docs/decode-vector-comparison.md"
    )
    args = p.parse_args()
    if args.mode == "figure":
        figure(
            args.output / "original27.csv",
            args.document.parent / "figures/decode-vector-comparison.png",
        )
        return
    with (ROOT / "calibration/reference/system-full-comparison.csv").open() as f:
        references = {r["case"]: r for r in csv.DictReader(f)}
    matrix, scratch = args.matrix.resolve(), args.scratch.resolve()
    check(
        matrix != scratch
        and matrix not in scratch.parents
        and scratch not in matrix.parents,
        "scratch and historical artifacts must be separate",
    )
    jobs = [(*c, scratch, references[c[0].name]) for c in cases(matrix)]
    for name, fn in [("capture", capture), ("replay", replay)]:
        if args.mode in (name, "all"):
            with ProcessPoolExecutor(args.workers) as pool:
                list(pool.map(fn, jobs))
    if args.mode in ("report", "all"):
        report(jobs, args.output, args.document)


if __name__ == "__main__":
    main()
