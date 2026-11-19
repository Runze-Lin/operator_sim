"""Summarize exact B8 prefill launches, including observed scaling and offline cost."""

import os
from pathlib import Path
import sys
from time import sleep

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from development.validation.sigmoid20.campaign import DEST, DT, PR, FUTURE, ROOT, read


def build():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measured = read(DEST / "holdout.json") if (DEST / "holdout.json").exists() else []
    f = read(DEST / "holdout-freeze.json")
    jobs = {tuple(j["shape"]): j for j in f["jobs"]}
    by = {(r["sku"], tuple(r["shape"])): r for r in measured if r["qualified"]}
    lengths = [32, 128, 256]
    shapes = [(l * 1024 * 8, 12) for l in lengths]
    rows = []
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    colors = {DT: "#0c7278", PR: "#994225", FUTURE: "#7054a3"}
    for c in (DT, PR, FUTURE):
        label = c.replace("Ascend", "").split("_")[0]
        preds = [jobs[s]["predictions"][c]["us"] for s in shapes]
        xy = [
            (l, by[c, s]["kernel"]["ca_us"])
            for l, s in zip(lengths, shapes)
            if (c, s) in by
        ]
        ax.plot(
            lengths,
            preds,
            "--",
            color=colors[c],
            alpha=0.75,
            marker="x",
            label=label + (" scenario" if c == FUTURE else " model"),
        )
        if xy:
            ax.plot(*zip(*xy), "-o", color=colors[c], label=label + " CA")
    ax.set_title(
        "Kimi K3 beta Sigmoid · unchunked prefill, B=8, TP=8",
        loc="left",
        fontsize=13,
        weight="bold",
    )
    ax.set_xticks(lengths, ["32k", "128k", "256k"])
    ax.set_xlabel("Prompt length per request (k = 1,024)")
    ax.set_ylabel("Device window (µs)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(ROOT / "docs/examples/figures/sigmoid-context.png", dpi=150)
    plt.close(fig)
    for l, s in zip(lengths, shapes):
        row = []
        for c in (DT, PR):
            actual = by.get((c, s))
            pred = jobs[s]["predictions"][c]
            if actual:
                t = actual["kernel"]["ca_us"]
                m = pred["us"]
                w = actual["cost"]["seconds"] / 60
                row.append(
                    f"{m:.3f} / {t:.3f} ({100 * (m / t - 1):+.1f}%), {w:.1f} min"
                )
            else:
                row.append(f"{pred['us']:.3f} / 等待")
        rows.append(
            f"| B8 × {l}k `{list(s)}` | {jobs[s]['predictions'][DT]['plan']['critical_core_tiles']} / {jobs[s]['predictions'][PR]['plan']['critical_core_tiles']} | {row[0]} | {row[1]} | {jobs[s]['predictions'][FUTURE]['us']:.3f} |"
        )
    lines = [
        "# Kimi K3 beta Sigmoid 在未切 chunk 的长 prefill 上的增长",
        "",
        "固定单卡 B=8、TP=8、局部 head=12，全部 shape 都提交到原生 FP32 `aclnnSigmoid`。计时覆盖所有核；每个元素与 FP64 参考比较。表中设备时间单位 µs，最后的 min 是单次离线 CA 的墙钟。",
        "",
        "| 场景与 shape | 预测关键核 UB 轮数 DT / PR | DT 模型 / 实测（误差），CA 成本 | PR 模型 / 实测（误差），CA 成本 | 970DT 外推 µs |",
        "|---|---:|---|---|---:|",
        *rows,
        "",
        "![模型与全核 CA 随真实 prefill 长度的变化](../../../docs/examples/figures/sigmoid-context.png)",
        "",
        "三个点的逻辑读写总量为 24、96、192 MiB。图中实线点为已经通过校验的 CA，虚线为解析模型。模型中稳定满 tile 的循环量随 token 数成比例，启动、尾块、首个读依赖独立计入，有限长度的增长倍数因此可能偏离工作量倍数。",
    ]
    for chip in (DT, PR):
        if all((chip, s) in by for s in shapes):
            a, b, c = [by[chip, s]["kernel"]["ca_us"] for s in shapes]
            rates = [
                (8 * s[0] * s[1]) / (by[chip, s]["kernel"]["ca_us"] * 1000)
                for s in shapes
            ]
            lines += [
                "",
                f"{chip} 的工作量从 32k 到 128k 增为 4 倍，时间增为 **{b / a:.2f} 倍**；128k 到 256k 的工作量增为 2 倍，时间增为 **{c / b:.2f} 倍**。三点的逻辑读写吞吐分别为 {rates[0]:.0f}、{rates[1]:.0f}、{rates[2]:.0f} GB/s，计入了核函数结束时可以仍驻留在 L2 的输出。",
            ]
    lines += [
        "",
        "[10 个冻结 shape 的逐项记录](measurements.md) · [为何上线模型与 L2 驻留敏感性分列](../../../docs/examples/sigmoid_fp32.md)",
    ]
    (DEST / "long-context.md").write_text("\n".join(lines) + "\n")
    return sum((c, s) in by for c in (DT, PR) for s in shapes)


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/operator-sim-mpl")
    if sys.argv[-1] == "watch":
        prev = None
        while True:
            p = DEST / "holdout.json"
            stamp = p.stat().st_mtime_ns if p.exists() else 0
            if stamp != prev:
                prev = stamp
                if build() == 6:
                    break
            sleep(30)
    else:
        build()
