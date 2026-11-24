"""Regenerate the paper's system figure and evidence tables from the release ledgers."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

DOC = Path(__file__).resolve().parents[2] / "docs"
MODELS = {"v4-pro": "DeepSeek-V4 Pro", "v41-flash": "DeepSeek-V4.1 Flash",
          "kimi-k31": "Kimi K3.1 Pro"}


def system_results():
    with (DOC.parent / "calibration/candidate_indexer/v113_context_requests.csv").open() as f:
        rows = sorted(csv.DictReader(f), key=lambda r: tuple(int(r[k]) for k in ("prompt", "batch", "ep")))
    fig, axes = plt.subplots(3, 1, figsize=(7.75, 6.1), sharex=True, sharey=True, layout="constrained")
    appendix = ["# 附录 C　33 组系统预测（整批秒数）", "",
                "下表按本卡 B、EP 和输入长度排序。所有配置 TP=8，输出 1024 个 token。峰值与精细共用精度和冷读写假设，历史为原后端。1Mi 组的 decode 及总量含超位置范围外推。", ""]
    keys = list(dict.fromkeys(r["model"] for r in rows))
    if len(keys) != 3 or len(rows) != 33:
        raise ValueError("Unexpected system experiment coverage")
    for axis, model in zip(axes, keys):
        selected = [r for r in rows if r["model"] == model]
        name = MODELS[model]
        labels = [f"{int(r['prompt'])//1024}k\nB{r['batch']}\nE{r['ep']}" for r in selected]
        x = range(len(selected))
        for field, label, color, offset, marker in (
            ("request_legacy_whole_s", "Historical backend", "#a04c24", -.09, "o"),
            ("request_v113_peak_s", "Matched peak", "#14797d", .09, "s")
        ):
            ratio = [float(r["request_v113_full_s"]) / float(r[field]) for r in selected]
            axis.scatter([i+offset for i in x], ratio, s=25, label=label, color=color, marker=marker)
        axis.axhline(1, color="#5b6877", ls="--", lw=.9)
        axis.grid(axis="y", alpha=.2)
        axis.set_title(name, loc="left", size=11)
        axis.set_yscale("log")
        axis.set_yticks([.125, .25, .5, 1, 2, 4])
        axis.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        axis.minorticks_off()
        axis.spines[["right", "top"]].set_visible(False)
        appendix.extend([f"## {name}", "", "| 输入 | B / EP | 历史总 | 同精度峰值总 | 精细 prefill | 精细 decode | 精细总 |",
                         "|---|---:|---:|---:|---:|---:|---:|"])
        for r in selected:
            length = f"{int(r['prompt'])//1024}Ki" if int(r["prompt"]) < 1048576 else "1Mi*"
            times = [r[key] for key in ("request_legacy_whole_s", "request_v113_peak_s",
                     "prefill_v113_full_s", "decode_v113_full_s", "request_v113_full_s")]
            appendix.append(f"| {length} | {r['batch']} / {r['ep']} | " +
                            " | ".join(f"{float(value):.3f}" for value in times) + " |")
        appendix.append("")
    axes[0].legend(frameon=False, ncol=2, fontsize=8, loc="upper right")
    axes[-1].set_xticks(list(x), labels, fontsize=7.5)
    fig.supylabel("Detailed / reference time")
    fig.savefig(DOC / "figures/paper-system-ratios.png", dpi=180)
    plt.close(fig)
    appendix.append("[原始逐配置账单及各次冻结结果](../../calibration/candidate_indexer/v113_context_requests.csv)")
    (DOC / "paper/11-requests.md").write_text("\n".join(appendix) + "\n")


def evidence_table():
    table = ["# 附录 B　逐合同的双端误差与精度", "",
             "DT、PR 栏分别给出不同形状数/窗口数，以及 WAPE/最大绝对误差。开发和独立窗数跨两端合计。冻结独立优先，未有独立数据的合同按开发回放展示。N/A 表示该问题缺可比证据。点击合同名称可查每个 shape 的 CA、预测、970DT 外推和冷读写 U 范围。", ""]
    rows = []
    for line in (DOC / "error-analysis.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not line.startswith("| [") or len(cells) != 9 or "](errors/" not in cells[0]:
            continue
        rows.append([cells[0].replace("](errors/", "](../errors/"), *cells[1:5]])
    table.extend(["| 执行合同 | 主精度 → 输出 | 数据角色 | 950DT | 950PR |", "|---|---|---|---|---|"])
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    table.append("")
    table.append("[完整证据、各合同 U 范围与计分定义](../error-analysis.md)")
    (DOC / "paper/10-evidence.md").write_text("\n".join(table) + "\n")


if __name__ == "__main__":
    system_results()
    evidence_table()
