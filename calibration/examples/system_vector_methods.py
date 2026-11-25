"""Recompute paired predictions on retained system feature contracts and complete CA windows.

Run from the repository root: PYTHONPATH=src:. python -m calibration.examples.system_vector_methods
No SIMPlatform/CANN installation is required: system-inputs stores immutable capture fields.
"""

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from types import SimpleNamespace

from operator_model import estimate, load_chip
from operator_model.system_routes import estimate_full_features, attention, vector, vf
from calibration.system_full_errors import KEYS, VECTOR, ATT
from calibration.system_vector_ca import primitives, report
from calibration.system_cube_ca.predictions import predict

ROOT = Path(__file__).resolve().parents[2]
CA = ROOT / "calibration"
DEST = ROOT / "results/v2.3"
MODES = ("calibrated", "instruction_classes")
SKUS = ("Ascend950DT_9582", "Ascend950PR_957b")


def read(p):
    return json.loads(p.read_text())


def feature(data):
    def freeze(x):
        if isinstance(x, list):
            return tuple(map(freeze, x))
        if isinstance(x, dict):
            return SimpleNamespace(**{k: freeze(v) for k, v in x.items()})
        return x

    return freeze(data)


def _native(row, mode):
    return estimate(
        row["op"],
        row["shape"],
        dict(row["attrs"], vector_model=mode),
        dtype=row.get("dtype"),
        chip=row["chip"],
    ).latency_us


def ca_rows():
    def new(name, chip, shape, api, kernel, actual, role, src, fn):
        a, b = (fn(mode) for mode in MODES)
        return dict(
            name=name,
            chip=chip,
            shape=list(shape),
            contract=api,
            kernel=kernel,
            ca_us=actual,
            calibrated_us=a,
            instruction_classes_us=b,
            archive_role=role,
            source=src,
        )

    for line in (CA / "reference/windows.jsonl").read_text().splitlines():
        row = json.loads(line)
        c = row["contract"]
        if c in KEYS and row["predicted_us"] is not None:
            yield new(
                KEYS[c],
                row["chip"],
                row["shape"],
                c,
                row["kernel"],
                row["ca_us"],
                row["data_role"],
                "reference/windows.jsonl",
                lambda m: _native(row, m),
            )
    for row in csv.DictReader((CA / "system_vector_ca/primitives.csv").open()):
        f = primitives.fixture(row["api"], int(row["rows"]))
        chip = load_chip(row["chip"])

        def run(mode):
            with vf.selection(mode):
                return vector._schedule(vector._primitive(f), chip).latency_us

        yield new(
            row["raw_name"],
            row["chip"],
            [list(t.shape) for t in f.inputs],
            row["api"],
            row["kernel"],
            float(row["ca_us"]),
            row["role"],
            "system_vector_ca/primitives.csv",
            run,
        )
    for row in read(CA / "system_vector_ca/windows.json"):
        f = report.features(row["op"], row["shape"])

        def run(mode):
            with vf.selection(mode):
                return vector.price(f, row["chip"]).latency_us

        yield new(
            VECTOR[row["op"]],
            row["chip"],
            row["shape"],
            row["op"],
            row["kernel"],
            row["measured_us"],
            row["sample_role"],
            "system_vector_ca/windows.json",
            run,
        )
    for file in ("measured", "independent", "router", "router-independent"):
        for chip, group in read(CA / "system_cube_ca" / (file + ".json")).items():
            for row in group["cases"]:
                router = file.startswith("router")
                bias = row["api"] == "aclnnAddmm"
                raw = (
                    "inference::matmul_rhs_transposed"
                    if router
                    else ("aten.addmm.default" if bias else "aten.mm.default")
                )
                spec = (*row["shape_MNK"], "BF16_FP32" if router else row["dtype"], bias)

                def run(mode):
                    with vf.selection(mode):
                        return predict((spec,))[chip][0]["predicted_us"]

                yield new(
                    raw,
                    chip,
                    row["shape_MNK"],
                    row["api"],
                    row["kernel"],
                    row["ca_us"],
                    row.get("reference_role", "archive"),
                    "system_cube_ca/" + file + ".json",
                    run,
                )
    for file in ("development", "independent", "indexer-h64"):
        for row in read(CA / "system_attention_ca" / (file + ".json")):
            op = row["operator"]
            s = row["shape"]
            chip = load_chip(row["sku"])
            t = report.tensor
            if op == "conv":
                b, q, d, w, h = s
                ins = [t((q, d), "bfloat16"), t((w, d), "bfloat16"), t((b, h, d), "bfloat16")]
                outs = ins[:1]
                sem = dict(activation="silu")
                raw = ATT[op]
            elif op == "fia":
                b, q, h, kv, d, k = s
                ins = [
                    t((q, h, d), "bfloat16"),
                    t((q, kv, d), "bfloat16"),
                    t((q, kv, d), "bfloat16"),
                    t((b, k - q, 2, kv, d), "bfloat16"),
                ]
                outs = [ins[0], t((h, q), "float32")]
                sem = dict(
                    kv_lens=(k,),
                    effective_qk_pairs=q * k,
                    query_lens=(q,),
                    num_kv_heads=kv,
                    compute_precision="bf16",
                )
                raw = ATT[op]
            elif op == "indexer":
                b, q, k, h, d, top = s
                ins = [
                    t((q, h, d), "bfloat16"),
                    t((b, k, d), "bfloat16"),
                    t((q, h), "bfloat16"),
                    t((q,), "int64"),
                ]
                outs = [t((q, top), "int32")]
                sem = dict(index_topk=top)
                raw = "vllm_ascend::quant_lightning_indexer_reference"
            else:
                continue
            f = SimpleNamespace(
                raw_name=raw, inputs=ins, outputs=outs, attributes=(("semantic", tuple(sem.items())),)
            )

            def run(mode):
                with vf.selection(mode):
                    return attention.price(f, chip).latency_us

            receipt = new(
                raw,
                chip.name,
                s,
                op,
                " + ".join(i["kernel"] for i in row["stages"]),
                row["device_us"],
                file,
                "system_attention_ca/" + file + ".json",
                run,
            )
            # Quant-indexer reference has a distinct end-to-end boundary. Retain
            # the ordinary BF16 LI timing as evidence, never as its parent CA.
            if op == "indexer":
                receipt["component_only"] = True
            yield receipt


def graph_rows():
    modes = (*MODES, "auto")
    for row in json.loads(gzip.decompress((DEST / "system-inputs.json.gz").read_bytes())):
        f = feature(row["features"])
        for sku in SKUS:
            chip = load_chip(sku)
            quotes = {}
            wall = {}
            plans = {}
            shape = [[t.dtype.removeprefix("torch."), list(t.shape)] for t in f.inputs]
            for mode in modes:
                times = []
                for repeat in range(4):
                    start = perf_counter_ns()
                    out = estimate_full_features(f, chip=chip, vector_model=mode)
                    times.append((perf_counter_ns() - start) / 1e3)
                quotes[mode] = out.latency_us
                wall[mode] = median(times[1:])
                s = (out.work or {}).get("schedule") or {}
                plans[mode] = dict(
                    contract=out.contract,
                    status=out.status,
                    selector=s.get("vector_model_selected")
                    or s.get("vector_model")
                    or (s.get("quant") or {}).get("vector_model"),
                    stage_labels=[(p["stage"], p["method"]) for p in s.get("vector_stages", ())],
                    reason=out.reason if out.latency_us is None else None,
                )
            yield dict(
                name=f.raw_name,
                chip=sku,
                capture=row["capture"],
                node=row["node"],
                inputs=shape,
                device_us=quotes,
                wall_us=wall,
                plans=plans,
            )


def main():
    rows = list(graph_rows())
    ca = list(ca_rows())
    (DEST / "system-windows.json").write_text(json.dumps(ca, ensure_ascii=False, indent=1) + "\n")
    (DEST / "system-quotes.json.gz").write_bytes(
        gzip.compress(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode(), mtime=0)
    )
    doc = [
        "# 系统调用的 Vector 方法对照",
        "",
        "按 **rank-local 原始调用名**列出三模型 prefill/decode 捕获。每个方法使用相同的 shape、dtype、布局、并行元数据和芯片；更换的是 Vector 阶段，而非 Cube 吞吐、HBM 假设或系统 DES。第一张表是系统调用维度的索引；校准数据只属于具体子合同，不能外推为整行或整图实测。",
        "",
        '调用 `FullOperatorSimBackend(vector_model="auto")`；可显式传 `"calibrated"`（原排程，包括未校准的旧设计）或 `"instruction_classes"`。只接原生 lowerings 的 `OperatorSimBackend` 也支持相同选项。选择器进入系统缓存标识。严格未知签名继续返回 unsupported。',
        "",
        "两法保留相同首发、分核、UB 分页与资源依赖。指令法将有意义的工作（如注意力每个 score 的 exp、规约和量化；A8W4 每个展开权重的解包、转换与写 UB）折成寄存器发起次数。对选中忙核，按 256 B 寄存器、计算位宽、分给该核的元素数向上取整；总发起、单类间隔、UB 端口、ALU 端口取最大周期数。默认 2 发起槽、2 端口、exp/divide 4 周期是待检验的架构先验；可在芯片配置中更换。新实现默认类别法，沿用已验证实现的路径默认原法。通信本身不对应 VF；有本地 SUM/反量化融合的通信只替换本地计算。视图及纯 DMA 无此阶段。",
        "",
        "| 原始调用名 | 系统签名数 | auto（样本路径） | DT 同签名整算子差值范围 | DT CA：原 / 类别 | PR CA：原 / 类别 | CA 窗数 DT / PR |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    groups = defaultdict(list)
    windows = defaultdict(list)
    for r in rows:
        groups[r["name"]].append(r)
    for r in ca:
        windows[r["name"]].append(r)
    for name, gr in sorted(groups.items()):
        dt = [r for r in gr if r["chip"] == SKUS[0]]
        ratios = [
            100 * (b / a - 1)
            for r in dt
            if (a := r["device_us"]["calibrated"])
            and (b := r["device_us"]["instruction_classes"]) is not None
        ]
        delta = "—" if not ratios else f"{min(ratios):+.2f} … {max(ratios):+.2f}%"
        counts = []
        metrics = []
        for sku in SKUS:
            sr = [
                r
                for r in windows[name]
                if not r.get("component_only")
                and r["chip"] == sku
                and r["calibrated_us"] is not None
                and r["instruction_classes_us"] is not None
            ]
            counts.append(str(len(sr)))
            metrics.append(
                "N/A"
                if not sr
                else " / ".join(
                    f"{100 * sum(abs(r[k] - r['ca_us']) for r in sr) / sum(r['ca_us'] for r in sr):.2f}%"
                    for k in ("calibrated_us", "instruction_classes_us")
                )
            )
        selectors = sorted(set(str(r["plans"]["auto"]["selector"] or "native") for r in dt))
        label = {
            "not_applicable": "无 VF",
            "calibrated": "原",
            "instruction_classes": "类别",
            "native": "原生（见子合同）",
            "calibrated+instruction_classes": "分阶段：原及类别",
        }
        doc.append(
            f"| `{name}` | {len(dt)} | {', '.join(label.get(v, v) for v in selectors)} | {delta} | {' | '.join(metrics)} | {' / '.join(counts)} |"
        )
    doc.extend(
        [
            "",
            "表头：**原始调用名**是系统图符号；**系统签名数**为本页保留的代表节点数（同一算子可有不同精度、角色和执行阶段，不是所有 shape 数）；**auto** 为自动模式对这些样本实际选择，纯搬运/元数据不存在 VF。**DT 同签名差值**是 `(类别设备时间 / 原设备时间 − 1) × 100%` 的范围，正数为类别法较慢；原时间全零时不作除法。**DT/PR CA** 为对应芯片可重放合同的原法 WAPE / 类别法 WAPE，`WAPE=Σ|预测−CA|/ΣCA`；**CA 窗数**为同样两组 WAPE 的分母样本数。N/A 表示不能给两法配对的合格整窗；范围是方法差异，不是 CA 误差。",
            "",
            f"共 {len(groups)} 个调用名、{len(rows) // 2} 份系统输入，来自三模型各一份 decode 和 prefill 图中的 target/draft。按阶段、输入 dtype/秩和主要实现标签选取首个实际节点（因此不是对所有序列长度或所有系统配置重新跑 DES）。芯片采用原生 DT/PR 配置、CA window 定义；不覆盖系统硬件配置覆盖值或可选 cold-IO 下界。所有 CA 是已存档窗口的回放，包含旧方法开发样本，不是本次新冻结独立验证。类指令在系统设计路线上是源算法推演，不是实测的指令序列。",
            "",
            "[图输入与来源](../results/v2.3/system-inputs.json.gz) · [逐节点双芯片报价、分支、耗时](../results/v2.3/system-quotes.json.gz) · [逐窗口双方法预测、真值、数据角色和 kernel](../results/v2.3/system-windows.json) · [原始完整误差及采集身份](../calibration/reference/system-full-ca.csv)。报价中 `wall_us` 是四次运行去掉首次后取中位数的本机墙钟微秒，不含图转换或 CA；`device_us` 才是预计的芯片运行时间。再现：`PYTHONPATH=src:. python -m calibration.examples.system_vector_methods`。",
        ]
    )
    measured = [
        r
        for r in ca
        if not r.get("component_only")
        and r["calibrated_us"] is not None
        and r["instruction_classes_us"] is not None
    ]
    diff = len(
        {r["name"] for r in rows if r["device_us"]["instruction_classes"] != r["device_us"]["calibrated"]}
    )
    walls = [median(r["wall_us"][k] for r in rows) for k in MODES]
    doc.insert(
        8,
        f"本次 {len(groups)} 个原始名中，有 {diff} 个至少在一个系统样本上改变整窗；其余可能是无 VF、同价、被 Cube/搬运/通信遮住，或低于排程分辨率。{len(set(r['name'] for r in measured))} 个原始名有可配对的计时合同，共 {len(measured)} 个 DT/PR 窗口；普通 H64 LightningIndexer 的组件回放不计入 quant_lightning_indexer_reference 的整窗误差。相同调用的 Python 热态墙钟中位数，原法 {walls[0]:.1f} µs、类别法 {walls[1]:.1f} µs。无证据支持将全部已知原生程序改默认。\n",
    )
    (ROOT / "docs/system-vector-methods.md").write_text("\n".join(doc) + "\n")
    print(
        "nodes",
        len(rows),
        "CA records",
        len(ca),
        "unpriced",
        [
            (r["name"], m, v["reason"])
            for r in rows
            for m, v in r["plans"].items()
            if r["device_us"][m] is None
        ][:8],
    )


if __name__ == "__main__":
    main()
