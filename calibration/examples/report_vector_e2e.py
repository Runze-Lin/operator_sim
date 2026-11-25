"""Render the 33-request Vector-selector report from the retained DES replay CSVs."""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "v4-pro": "DeepSeek-V4 Pro",
    "v41-flash": "DeepSeek-V4.1 Flash",
    "kimi-k31": "Kimi K3.1 Pro",
}
SHORT = {"v4-pro": "V4 Pro", "v41-flash": "V4.1 Flash", "kimi-k31": "K3.1 Pro"}
METHODS = ("calibrated", "auto", "instruction_classes")
PHASES = ("prefill", "decode", "request")
KEY = ("model", "prompt", "batch", "tp", "ep")


def read(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def length(r):
    n = int(r["prompt"])
    return "1Mi" if n == 1048576 else f"{n // 1024}Ki"


def price(r, phase, method):
    return float(r[f"{phase}_{method}_s"])


def gap(r, phase, method, base="calibrated"):
    return 100 * (price(r, phase, method) / price(r, phase, base) - 1)


def pc(x):
    return f"{x:+.2f}%" if abs(x) >= 0.005 else ("≈0.00%" if x != 0 else "0.00%")


def span(numbers):
    nums = list(numbers)
    return f"{pc(min(nums))} 至 {pc(max(nums))}"


def key(r):
    return tuple(r[k] for k in KEY)


def plot(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axs = plt.subplots(3, 3, figsize=(11.2, 9), sharex="col", sharey=True)
    labels = ("Prefill", "Decode", "Full request")
    for i, m in enumerate(MODELS):
        subset = [r for r in rows if r["model"] == m]
        ticks = [
            f"{length(r)}{'*' if length(r) == '1Mi' else ''}  {r['batch']}/{r['ep']}"
            for r in subset
        ]
        for j, p in enumerate(PHASES):
            ax = axs[i, j]
            y = list(range(len(subset)))
            auto = [gap(r, p, "auto") for r in subset]
            ins = [gap(r, p, "instruction_classes") for r in subset]
            ax.axvline(0, color="#6c7681", lw=0.8)
            ax.hlines(y, auto, ins, color="#c3ccd2", lw=1.3)
            ax.scatter(auto, y, color="#137266", marker="o", s=25, zorder=3)
            ax.scatter(
                ins,
                [v + 0.12 for v in y],
                facecolor="none",
                edgecolor="#b14d1e",
                marker="D",
                s=28,
                zorder=3,
            )
            ax.grid(axis="x", color="#d7dce1", lw=0.6)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="both", length=0, labelsize=8)
            if i == 0:
                ax.set_title(labels[j], fontsize=12, weight="bold")
            if j == 0:
                ax.set_yticks(y, ticks)
                ax.set_ylabel(SHORT[m], fontsize=10, weight="bold")
            if i == 2:
                ax.set_xlabel("Difference from calibrated (%)", fontsize=9)
    axs[0, 0].invert_yaxis()
    fig.suptitle(
        "How Vector pricing changes simulated device time",
        x=0.1,
        ha="left",
        fontsize=17,
        weight="bold",
        y=0.995,
    )
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color="#137266",
                label="Default (auto)",
            ),
            Line2D(
                [],
                [],
                marker="D",
                linestyle="none",
                markerfacecolor="none",
                color="#b14d1e",
                label="Force instruction classes",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.09, 0.959),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.10,
        0.017,
        "Rows: prompt length, local batch / EP. TP = 8.  * Decode and request extend beyond the position preset.  Axes differ by phase.",
        fontsize=9,
        color="#394753",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92), w_pad=1.3, h_pad=0.6)
    fig.savefig(path, dpi=175, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render(rows, ops, graphs, checks, sweep, swaps):
    if (
        len(rows) != 33
        or len({key(r) for r in rows}) != 33
        or any(sum(r["model"] == m for r in rows) != 11 for m in MODELS)
    ):
        raise ValueError("expected 33 unique request configurations, 11 per model")
    if {key(r) for r in rows} != {key(r) for r in ops}:
        raise ValueError("operator and request configuration keys differ")
    for r in rows:
        for m in (*METHODS, "legacy", "effective", "peak"):
            if (
                abs(
                    price(r, "prefill", m)
                    + price(r, "decode", m)
                    - price(r, "request", m)
                )
                > 1e-6
            ):
                raise ValueError(f"phase totals do not match: {key(r)} {m}")
    faster = sum(
        price(r, "request", "auto") < price(r, "request", "legacy") for r in rows
    )
    floor = sum(
        all(
            price(r, "request", m) >= price(r, "request", "peak") * (1 - 1e-10)
            for m in METHODS
        )
        for r in rows
    )
    previous_diff = max(
        abs(price(r, p, "calibrated") - price(r, p, "previous_full"))
        for r in rows
        for p in PHASES
    )
    largest = max(rows, key=lambda r: abs(gap(r, "request", "auto")))
    flash = next(
        g
        for g in graphs
        if g["model"] == "v41-flash"
        and int(g["prompt"]) == 2**20
        and int(g["batch"]) == 1
        and int(g["ep"]) == 32
        and g["phase"] == "prefill"
        and int(g["history"]) == 2**20 - 4096
    )
    text = [
        "# 三个模型的端到端复算：Vector 方法会改变多少请求时间？",
        "",
        "这里固定 950DT 的同一份 rank 0 捕获图、执行精度、核数、通信价格和资源依赖，仅切换 Vector（VF）阶段的定价方法。对三个模型各 11 种配置，分别计算读完整 prompt 的 prefill、随后生成余下 token 的 decode，两者相加就是该批请求在这张图上的完成时间；不除以 B，也不含线上排队。所有秒数都是系统离散事件仿真（DES）预测；与单算子的 CA 校准分开报告。",
        "",
        "## 先看结果",
        "",
        f"按配置逐一对照，默认策略相对原法的**完整请求**变化为 **{span(gap(r, 'request', 'auto') for r in rows)}**；强制使用指令类别法为 **{span(gap(r, 'request', 'instruction_classes') for r in rows)}**。范围取各配置的最小、最大相对差，不以一次长请求的耗时给其他配置加权。",
        "",
        "| 模型 | 默认 vs 原法：prefill | 默认 vs 原法：decode | 全指令 vs 原法：prefill | 全指令 vs 原法：decode | 默认 vs 历史 Roofline：请求 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m, name in MODELS.items():
        rs = [r for r in rows if r["model"] == m]
        text.append(
            "| "
            + name
            + " | "
            + " | ".join(
                span(gap(r, p, h, b) for r in rs)
                for p, h, b in [
                    ("prefill", "auto", "calibrated"),
                    ("decode", "auto", "calibrated"),
                    ("prefill", "instruction_classes", "calibrated"),
                    ("decode", "instruction_classes", "calibrated"),
                    ("request", "auto", "legacy"),
                ]
            )
            + " |"
        )
    text += [
        "",
        "表头说明：每格为该模型 11 组部署的相对差范围。差值 = 100 ×（当前方法时间 / 对照方法时间 − 1），负号表示预测时间缩短。只有前四列隔离了 Vector 方法；历史 Roofline 还使用原有引擎分配、效率系数和占位路径。方法间差值不代表预测对真机的误差。",
        "",
        f"默认与原法的请求差值比例最大的是 **{MODELS[largest['model']]}，{length(largest)}、B={largest['batch']}、EP={largest['ep']}**：原法 {price(largest, 'request', 'calibrated'):.3f} s，默认 {price(largest, 'request', 'auto'):.3f} s；其中 prefill 改变 {price(largest, 'prefill', 'auto') - price(largest, 'prefill', 'calibrated'):+.3f} s，decode 改变 {price(largest, 'decode', 'auto') - price(largest, 'decode', 'calibrated'):+.3f} s。{'该组的 decode 超出模型原位置预置，含这段时间的请求总量也属于外推。' if largest['decode_positions'] != 'in_preset' else ''}整个矩阵有 {faster}/33 组默认请求比历史 Roofline 短；{floor}/33 组的三种请求时间都不短于同口径峰值下界。",
        "",
        "![三个模型各 11 种配置的 prefill、decode、请求时间相对原法的变化](figures/system-vector-e2e.png)",
        "",
        "图中实心圆为默认方法、空心菱形为强制指令法。每行标出输入长度和 B/EP；三列分别画 prefill、decode 与请求。横轴 0 为原法，各阶段独立设定横轴刻度，便于看清较小的差异。",
        "",
        "## 为什么单算子变化不会等比例传到整网",
        "",
        "两种定价只替换有 Vector 工作的节点。没有 VF 的纯 Cube、视图、纯搬运和纯通信保持共同路径；Cube 或访存已经主导的节点，即使 VF 估计改变，整窗也可能不变。系统重放保留原图资源互斥与依赖，一个分支提早完成并不保证请求也同样提前。以下表格对每个模型、每个阶段取**默认 vs 原法绝对百分比差最大的配置**，再在该配置中按各调用名累计的绝对设备时间差排序，给出前两项。",
        "",
        "| 模型 · 阶段 | 输入 / B / EP | 阶段默认 − 原法（s） | 占原阶段 | 节点累计差前两项：默认 − 原法（s） |",
        "|---|---|---:|---:|---|",
    ]
    for m, name in MODELS.items():
        for p in PHASES[:2]:
            r = max(
                (r for r in rows if r["model"] == m),
                key=lambda r: abs(gap(r, p, "auto")),
            )
            local = [
                (float(o["auto_s"]) - float(o["calibrated_s"]), o["raw_name"])
                for o in ops
                if key(o) == key(r) and o["phase"] == p
            ]
            local = sorted(local, key=lambda a: -abs(a[0]))
            parts = (
                "；".join(f"`{n}` {d:+.3f}" for d, n in local[:2] if d != 0)
                or "两法同价"
            )
            text.append(
                f"| {name} · {p} | {length(r)} / {r['batch']} / {r['ep']} | {price(r, p, 'auto') - price(r, p, 'calibrated'):+.3f} | {pc(gap(r, p, 'auto'))} | {parts} |"
            )
    text += [
        "",
        "表头说明：阶段差是两次完整 DES 调度后的结果；节点累计差按同名节点的预计设备时间、出现次数及同一阶段积分口径计算，可用来定位应检查的算子。它没有分摊重叠，不能作为可相加的关键路径贡献。",
        "",
    ]
    text += [
        "### 固定其他节点，只替换一种调用名的反事实检验",
        "",
        "每个模型取两个点：1Mi、B=1、EP32 的末 prefill 块，以及 128Ki、B=1、EP32 的首个 decode 轮（target + draft）。从同一个基准方法出发，只切换所列调用名的所有实例，其他节点保持原价，再运行一次 DES。两个切换方向（原法 → 默认、默认 → 全指令）各取节点时间总差绝对值最大的三个调用名。每个点列出这六种干预中，对阶段时间影响绝对值最大的一个；这比单纯排列节点的耗时和更接近“如果只改这个节点，系统会怎样”。",
        "",
        "| 模型 · 锚点 | 基准 → 对照方法 | 单独切换的调用名（次数） | 节点时间差之和（ms） | 只切此调用的阶段差（ms） | 整图全切的阶段差（ms） |",
        "|---|---|---|---:|---:|---:|",
    ]
    for m, name in MODELS.items():
        for p in PHASES[:2]:
            cases = [r for r in swaps if r["model"] == m and r["phase"] == p]
            if not cases:
                raise ValueError(f"missing counterfactual for {m} {p}")
            ex = max(cases, key=lambda r: abs(float(r["swap_delta_us"])))
            text.append(
                f"| {name} · {p} | `{ex['base_mode']}` → `{ex['changed_mode']}` | `{ex['raw_name']}` ({ex['nodes']}) | {float(ex['serial_delta_us']) / 1e3:+.4f} | {float(ex['swap_delta_us']) / 1e3:+.4f} | {(float(ex['changed_all_us']) - float(ex['base_us'])) / 1e3:+.4f} |"
            )
    # Show a real scheduling mask and a real opposing change at the same stage.
    mask = max(
        swaps,
        key=lambda r: abs(float(r["serial_delta_us"]) - float(r["swap_delta_us"])),
    )
    contrary = next(
        r
        for r in swaps
        if r["model"] == "v4-pro"
        and r["phase"] == "prefill"
        and r["base_mode"] == "calibrated"
        and r["raw_name"] == "_C_ascend::moe_gating_top_k_hash"
    )
    comp = next(
        r
        for r in swaps
        if r["model"] == "v4-pro"
        and r["phase"] == "prefill"
        and r["base_mode"] == "calibrated"
        and r["raw_name"] == "_C_ascend::compressor"
    )
    text += [
        "",
        "表头说明：节点时间差之和可含重叠；只切此调用的阶段差来自重新调度，带符号，负值表示阶段缩短。全切列则让所有节点采用对照方法。随报价改变，忙资源或最后完成的分支可能改变，各单项反事实不可直接相加。这里只覆盖六个锚点；详细干预见[同图反事实账](../results/v2.3/system-e2e/counterfactuals.csv)。",
        "",
        f"例如 {MODELS[mask['model']]} 的 {mask['phase']}，单独切换 `{mask['raw_name']}`，节点时间差之和为 {float(mask['serial_delta_us']) / 1000:+.3f} ms；重跑同图只有 {float(mask['swap_delta_us']) / 1000:+.3f} ms 传到阶段末尾，说明原图有重叠。另一个现象是正负抵消：V4 Pro 的 1Mi 末块，从原法切到默认时，单独切 compressor 增加 {float(comp['swap_delta_us']) / 1000:.3f} ms，单独切 hash gating 缩短 {abs(float(contrary['swap_delta_us'])) / 1000:.3f} ms，所有改变一起排程最终增加 {(float(comp['changed_all_us']) - float(comp['base_us'])) / 1000:.3f} ms。",
        "",
        "### 长上下文实例：Flash 1Mi、B=1、TP8/EP32 的最后一个块",
        "",
        f"该块已有 {int(flash['history']):,} 个历史位置，新输入 4,096 个位置。历史 Roofline 的块时间为 {float(flash['legacy_us']) / 1e6:.3f} s；在同一捕获图上，原 VF 为 {float(flash['calibrated_us']) / 1e6:.6f} s，默认 {float(flash['auto_us']) / 1e6:.6f} s，全指令 {float(flash['instruction_classes_us']) / 1e6:.6f} s。这里区分**末块**与下表的**整段 prefill**。候选消费按 A5 的稀疏 MXFP4 方案报价，地址重合度采用实现默认；设备 CA 只验证过它的若干短窗，没有采集这个 4,096 行末块的完整设备窗。",
        "",
        "## 33 组结果：逐一核对 prefill、decode 和 Roofline",
        "",
        "表头说明：B 为每个 attention DP 副本的本地请求数，TP 固定为 8，DP = EP/8。Prefill 和 Decode 单元格依次给 **原 / 默认 / 全指令** 三种价格（秒）；请求列为默认 prefill + 默认 decode。历史列复现原后端；峰值列是同执行精度、工作量和资源合同下的理想 Roofline。`*` 表示 1Mi 输入以后的 decode 超出位置预置；`†` 表示 Kimi 该配置直接累加 147 轮。",
        "",
    ]
    for m, name in MODELS.items():
        text += [
            f"### {name}",
            "",
            "| 输入 | B / EP | Prefill 原 / 默认 / 全指令（s） | Decode 原 / 默认 / 全指令（s） | 默认请求（s） | 历史 Roofline 请求（s） | 同精度峰值请求（s） |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            if r["model"] != m:
                continue
            postfix = ("\\*" if r["decode_positions"] != "in_preset" else "") + (
                "†" if r["decode_sampling"] == "all_rounds" else ""
            )
            phases = [
                " / ".join(f"{price(r, p, h):.3f}" for h in METHODS) for p in PHASES[:2]
            ]
            text.append(
                f"| {length(r)}{postfix} | {r['batch']} / {r['ep']} | {' | '.join(phases)} | {price(r, 'request', 'auto'):.3f} | {price(r, 'request', 'legacy'):.3f} | {price(r, 'request', 'peak'):.3f} |"
            )
        text.append("")
    text += [
        "## 采样、外推与复现",
        "",
        f"- 重放了 {sweep['captures']:,} 次捕获，共 {sweep['stage_graphs']:,} 张阶段图；每种精细策略均为 {sweep['priced_node_instances_per_fine_mode']:,} 个捕获节点实例定价，涉及 {sweep['distinct_raw_names']} 个原始调用名。同一节点在多个层、锚点或阶段重复出现会重复统计；这不是不同 kernel 的数目。",
        "- 相同配置的三次重放仅改变方法选择。`calibrated` 是原 VF 排程（某些没有实现的路径本来就是设计估计）；`instruction_classes` 按工作类别为可识别的 VF 工作定价；`auto` 在已有实现上保留原法，在缺少对应实现的新合同上选用类别法。这并不保证任意精度都有两条独立的已校准程序。例如原生 FP16 稠密 FA 没有 `calibrated` 路径，强制它会报不支持；纯 Cube 或纯 DMA 根本没有 VF 可替换。[逐执行合同的方法资格](vector-method-comparison.md)。",
        "- 每请求的 prefill 块上限是 4,096 token；32Ki 输入的 8 个块全部实算。长输入实算首块以及含中点、末点的 5 个续块，未采到的块在相邻 DES 锚点间线性积分。每请求输出总长为 1,024，prefill 产生首 token，之后每轮有 8 个草稿、主模型一般校验 9 个位置，假设每轮推进 7 个 token，故计 147 轮 target + draft。通常实算首、中、末三轮，三组 Kimi 因分块随轮号改变而逐轮计算。",
        "- 把少数锚点排除在拟合外，再用相邻锚点预测它们，默认方法在 prefill、decode 的最大相对偏差分别为 "
        + "、".join(
            f"{max(float(v['max_internal_anchor_error_pct']) for v in checks if v['backend'] == 'auto' and v['phase'] == p and v['max_internal_anchor_error_pct']):.2f}%"
            for p in PHASES[:2]
        )
        + "。这个检查只衡量采样积分的误差；1Mi 位置预置以外没有可供核验的真实锚点。",
        "- 三个模型的原位置预置均到 1,048,576。1Mi 最后一个 prefill 块仍在域内；其后的 1Ki decode 由域内相邻位置延伸末段斜率，未验证更远位置的路由、分块与可执行性。没有完整服务实测，不能据此给出真机端到端 WAPE。",
        f"- 以原图的 rank 0 静态分配和 KV 预留估算，满足 144 GB 条件的配置为 {sum(r['fits_144GB'].lower() == 'true' for r in rows)}/{len(rows)}；它不覆盖其他 rank 的路由倾斜、碎片与服务排队。系统任务的芯片参数和单算子 CA 的 950DT 模板也不完全相同，不能拿系统秒数代替 CA 实测。",
        "",
        "相对上一份发布表的默认列，本次默认请求时间变化为 "
        + span(gap(r, "request", "auto", "previous_full") for r in rows)
        + f"。单独检查上一份发布表与本次重跑的 `calibrated`，33 组的三项时间（两阶段及请求）最大绝对差为 {previous_diff:.8g} s。",
        "",
        "[完整请求与各 Roofline 口径（CSV）](../results/v2.3/system-e2e/requests.csv) · [按算子和阶段的累计价格账（CSV）](../results/v2.3/system-e2e/operator-deltas.csv) · [采样位置检查（CSV）](../results/v2.3/system-e2e/interpolation-check.csv) · [逐算子两法误差及计时资格](system-vector-methods.md) · [历史捕获和配置口径](context-sweep.md)",
        "",
        "从仓库根目录运行 `python -m calibration.examples.report_vector_e2e`，可直接用已发布的 CSV/JSON 重绘此图与本页。有可信的本地捕获、SIMPlatform 和原预测存档时，可按以下顺序重新报价；`CAPTURES` 指原始捕获目录，`CACHE` 指本次专用的空缓存目录（程序按已有记录跳过，同一路径不要跨版本复用）。",
        "",
        "```bash",
        'python -m calibration.examples.context_vector_e2e run "$CAPTURES" "$CACHE" --workers 8',
        'python -m calibration.examples.context_vector_e2e report "$CAPTURES" "$CACHE"',
        'python -m calibration.examples.vector_e2e_swaps "$CAPTURES" --workers 3',
        "python -m calibration.examples.report_vector_e2e",
        "```",
        "",
    ]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "results/v2.3/system-e2e")
    parser.add_argument("--docs", type=Path, default=ROOT / "docs")
    args = parser.parse_args()
    rows = sorted(
        read(args.data / "requests.csv"),
        key=lambda r: (
            list(MODELS).index(r["model"]),
            int(r["prompt"]),
            int(r["batch"]),
            int(r["ep"]),
        ),
    )
    md = render(
        rows,
        read(args.data / "operator-deltas.csv"),
        read(args.data / "graphs.csv"),
        read(args.data / "interpolation-check.csv"),
        json.loads((args.data / "sweep.json").read_text()),
        read(args.data / "counterfactuals.csv"),
    )
    (args.docs / "figures").mkdir(parents=True, exist_ok=True)
    plot(rows, args.docs / "figures/system-vector-e2e.png")
    (args.docs / "system-vector-e2e.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
