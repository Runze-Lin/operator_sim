"""Export qualified long-shape terminal-core profiles, then link the generated plots."""

import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from development.validation.sigmoid20.campaign import DEST, DT, PR, ROOT, read
from development.validation.sigmoid20.figure import main


def figures():
    records = read(DEST / "holdout.json") if (DEST / "holdout.json").exists() else []
    by = {(r["sku"], tuple(r["shape"])): r for r in records if r["qualified"]}
    out = [
        "# Kimi K3 长 prefill 的终止核流水",
        "",
        "横轴保留整个 kernel。每图上半是事前冻结的模型，下半来自 CA 中结束最晚的 AIV。每个选中的 tile 展示 MTE2、首个 RV_VLD 至最后 RV_VST 的 SIMD 时间窗和 MTE3。四轮时全部展示；超过六轮时选择首两轮和末两轮，中间工作仍计入总时间。输入均为 B8、TP8 的一次未分 chunk 提交。",
    ]
    count = 0
    for n in (32, 128, 256):
        for chip in (DT, PR):
            shape = (n * 8192, 12)
            row = by.get((chip, shape))
            tag = chip.split("_")[0].replace("Ascend", "")
            slug = f"prefill{n}k" + ("" if chip == DT else "-pr")
            out += ["", f"## {tag} · B8 × {n}k", ""]
            if row is None:
                out.append("等待独立 CA 完成并通过全核时间窗与全部数值检查。")
                continue
            count += 1
            png = ROOT / "docs/examples/figures" / f"sigmoid-{slug}.png"
            dpath = DEST / f"timeline-{slug}.json"
            if not dpath.exists() or not png.exists():
                main("holdout", *shape, slug, chip)
            d = read(dpath)
            k = row["kernel"]
            h = row["cost"]["seconds"] / 60
            out += [
                f"实际 {k['active_aiv']} 个 AIV，每核典型满 tile 搬入 {d['ca_tiles'][0]['bytes']:,} B。终止核 `{d['terminal_core']}` 有 {len(d['ca_tiles'])} 个 tile；模型 {d['model_us']:.3f} µs，CA {d['ca_us']:.3f} µs（{100 * (d['model_us'] / d['ca_us'] - 1):+.1f}%）。离线 CA {h:.1f} 分钟。",
                "",
                f"![{tag} {n}k 预测与实测流水](../../../docs/examples/figures/{png.name})",
                "",
                f"[完整分块时间戳与校验哈希]({dpath.name})",
            ]
    out += [
        "",
        "[场景增长曲线与实测比值](long-context.md) · [计价方法、工作量和参数](../../../docs/examples/sigmoid_fp32.md)",
    ]
    (DEST / "long-timelines.md").write_text("\n".join(out) + "\n")
    return count


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/operator-sim-mpl")
    if sys.argv[-1] == "watch":
        last = None
        while True:
            hold = DEST / "holdout.json"
            stamp = hold.stat().st_mtime_ns if hold.exists() else 0
            if last != stamp:
                if figures() == 6:
                    break
                last = stamp
            time.sleep(30)
    else:
        figures()
