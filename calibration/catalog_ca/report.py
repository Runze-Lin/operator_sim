"""Regenerate paired AddRmsNorm device-window errors from the signed receipts."""

from collections import defaultdict
import json

from calibration.catalog_ca.campaign import RESULTS, ROOT, forecast
from calibration.roofline_audit import nominal_engine_us
from operator_model import load_chip


def main():
    groups = defaultdict(dict)
    out = [
        "# 目录新增算子的 CA 对照",
        "",
        "## BF16 AddRmsNorm",
        "",
        "| 数据角色 | SKU | shape / 完整窗 | WAPE | 最大绝对误差 | 等效冷读写 Roofline U 范围 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for role, file in (("开发", "development"), ("冻结独立", "independent")):
        rows = json.loads((RESULTS / f"add-rms-{file}.json").read_text())
        for row in rows:
            chip = row["chip"]
            r, d = row["shape"]
            p = row.get("auto_us")
            if p is None:
                p = forecast(chip, (r, d))["auto_us"]
            sku = load_chip(chip)
            e = r * d
            peak = max(
                (8 * e + 2 * d + 4 * r) / (1000 * sku.hbm_bandwidth_gbs),
                nominal_engine_us([("float32", 5 * e + 3 * r)], {32: sku.vector_tops["32"]}),
            )
            groups[role, chip][r, d] = (
                p,
                row["measured_us"],
                row["active_aiv"],
                100 * peak / max(p, peak),
            )
    for (role, chip), group in groups.items():
        rows = group.values()
        wape = (
            100 * sum(abs(p - c) for p, c, a, u in rows) / sum(c for p, c, a, u in rows)
        )
        mx = 100 * max(abs(p / c - 1) for p, c, a, u in rows)
        u = [row[3] for row in rows]
        out.append(
            f"| {role} | {chip} | {len(group)}/{len(group)} | {wape:.2f}% | {mx:.2f}% | {min(u):.2f}%–{max(u):.2f}% |"
        )
    out += [
        "",
        "WAPE 是绝对微秒误差之和除以 CA 微秒数之和；最大值取逐窗百分比绝对值。计分预测使用指定 SKU 的设备窗 auto。冷读写 U 按同工作量理想时间除以包含该下界的预测时间计算。理想时间取 FP32 Vector 工作除以峰值、逻辑字节除以 HBM 带宽两项的最大值。",
        "",
        "| 角色 | shape [行,通道] | DT 预测 / CA µs | DT 误差 | PR 预测 / CA µs | PR 误差 | DT / PR 活跃 AIV (CA) | 970DT µs（外推） |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ("开发", "冻结独立"):
        for shape, (p, c, a, _) in sorted(groups[role, "Ascend950DT_9582"].items()):
            pp, cp, ap, _ = groups[role, "Ascend950PR_957b"][shape]
            future = forecast("Ascend970DT", shape)["auto_us"]
            out.append(
                f"| {role} | {list(shape)} | {p:.4f} / {c:.4f} | {100 * (p / c - 1):+.1f}% | {pp:.4f} / {cp:.4f} | {100 * (pp / cp - 1):+.1f}% | {a} / {ap} | {future:.4f} |"
            )
    out += [
        "",
        "误差为预测除以 CA 减一，970DT 使用同一实现假设与目标芯片配置。独立预测在采集前冻结，按原值复核。每窗核查两个完整 BF16 输出和逐行 FP32 逆标准差、实际 SKU/外存配置、安装 OPP ELF 哈希和所有活动核 END。设备时间为最早发起到最晚退休。实际入口为 aclnnAddRmsNorm，安装包对象为 ops_nn/add_rms_norm/AddRmsNorm_bf16_high_performance.o。",
        "",
        "9 行、512 通道的开发样例用于扣除有效载荷、标定控制残差。1 行、4096 通道和 32 行、2048 通道用于观察宽度与并发变化。其余宽度复用同一残差；模型难以完全解释多核同步、端口争用与归约尾部。当前计时无法分别辨识三者。Gemma 权重预加一或不同量化输出的融合路径仍需单独验证。",
        "",
        "[方法与输入输出字节](catalog-vector.md) · [开发收据](../../results/v2.4/add-rms-development.json) · [冻结预测](../../results/v2.4/add-rms-freeze.json) · [独立收据与指令时标](../../results/v2.4/add-rms-independent.json)",
        "",
        "## 同目录其他新增设备对照",
        "",
        "[Indexer LayerNorm 的物化与归一化](catalog-layer-norm-ca.md) · [BatchMatMul 的布局与 CA 对照](catalog-cube.md) · [实际捕获 shape 与精度](../../results/v2.4/catalog/shape-index.md)。新稀疏融合、QSA、GDN 等设计路径的完整 CA 误差仍为 N/A。",
    ]
    (ROOT / "docs/examples/catalog-ca.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
