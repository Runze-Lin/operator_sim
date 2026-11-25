"""Index physical capture geometry and explicitly declared logical formats by model/operator."""

import argparse
from collections import defaultdict
import csv
import gzip
import io
import json
import math
from pathlib import Path

from calibration.examples.catalog_capture import ROOT


class Coverage:
    def __init__(self):
        self.shapes = set()
        self.nodes = 0
        self.cases = set()
        self.physical_types = set()
        self.formats = defaultdict(set)
        self.extremes = [None, None]

    def add(self, feature, key, n, case):
        tensors = feature["inputs"] + feature["outputs"]
        shape = tuple(tuple(t["shape"]) for t in tensors)
        self.shapes.add((len(feature["inputs"]), shape))
        self.nodes += n
        self.cases.add(case)
        self.physical_types.update(t["dtype"].removeprefix("torch.") for t in tensors)
        semantic = dict(dict(feature["attributes"]).get("semantic", []))
        for field, value in semantic.items():
            if any(
                t in field for t in ("precision", "scheme", "quantization", "dtype")
            ) and isinstance(value, str):
                self.formats[field].add(value)
        rank = (sum(math.prod(s) for s in shape), key)
        for i in range(2):
            old = self.extremes[i]
            if old is None or (rank < old if i == 0 else rank > old):
                self.extremes[i] = rank


def geometry(feature):
    out = {}
    for field in ("inputs", "outputs"):
        out[field] = [
            [t["dtype"].removeprefix("torch."), t["shape"]] for t in feature[field]
        ]
    attrs = dict(feature["attributes"])
    for field in ("input_logical_shapes", "output_logical_shapes"):
        if field in attrs:
            out[field] = attrs[field]
    return json.dumps(out, separators=(",", ":"))


def build(folder):
    records = json.loads(gzip.decompress((folder / "features.json.gz").read_bytes()))
    cases = json.loads(gzip.decompress((folder / "cases.json.gz").read_bytes()))
    by_model = defaultdict(Coverage)
    for i, case in enumerate(cases):
        name = case["task"]["preset"]
        for s in case["stages"]:
            for key, count in s["signatures"]:
                f = records[key]
                for group in [(name, f["raw_name"]), ("ALL", f["raw_name"])]:
                    by_model[group].add(f, key, count, i)
    text = io.StringIO()
    rows = []
    for (model, op), c in sorted(by_model.items()):
        small, large = (pair[1] for pair in c.extremes)
        rows.append(
            dict(
                preset=model,
                raw_name=op,
                tasks=len(c.cases),
                nodes=c.nodes,
                shapes=len(c.shapes),
                fewer_than_10=len(c.shapes) < 10,
                captured_tensor_dtypes="|".join(sorted(c.physical_types)),
                semantic_formats=json.dumps(
                    {k: sorted(v) for k, v in sorted(c.formats.items())},
                    separators=(",", ":"),
                ),
                small_key=small,
                large_key=large,
                small_example=geometry(records[small]),
                large_example=geometry(records[large]),
            )
        )
    writer = csv.DictWriter(text, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (folder / "shape-index.csv.gz").write_bytes(
        gzip.compress(text.getvalue().encode(), mtime=0)
    )
    all_ops = [r for r in rows if r["preset"] == "ALL"]
    report = [
        "# 按模型查找算子、精度与真实 shape",
        "",
        "表内的 shape 是完整前向实际捕获的物理输入和输出尺寸元组。相同尺寸的多种精度只计一个 shape；层号、stride、offset 和视图身份也不增加计数。同一张图重复使用的签名按出现次数计入节点数。这里测量工作负载覆盖，完整 CA 校准样本另见总误差分析。",
        "",
        "[逐模型、逐算子的完整 CSV](shape-index.csv.gz) 为每个算子列出 shape 数、是否少于十种、实际 tensor dtype、生产者明确记录的语义精度、最小与最大用例。两个用例按所有输入输出的元素数之和排序（含权重），附有对应的特征键及可用的逻辑 shape，可在 features.json.gz 定位参数与全部张量。空的 semantic_formats 表示该调用没有显式精度标签。",
        "",
        "| 模型 | 捕获任务 | 调用名 | 节点实例 | 该模型中有至少十种 shape 的调用名 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in sorted({r["preset"] for r in rows}):
        subset = [r for r in rows if r["preset"] == model]
        case_count = (
            len(cases)
            if model == "ALL"
            else sum(c["task"]["preset"] == model for c in cases)
        )
        report.append(
            f"| {model} | {case_count} | {len(subset)} | {sum(r['nodes'] for r in subset):,} | {sum(not r['fewer_than_10'] for r in subset)} |"
        )
    report += [
        "",
        "ALL 在全目录按调用名和尺寸去重。最后一列适用于该行的图样本，单个预设有十份任务也不保证每个调用能得到十种尺寸。",
        "",
        "## 全目录仍少于十种物理 shape 的调用",
        "",
        "| 调用名 | shape 数 | 覆盖任务 |",
        "|---|---:|---:|",
    ]
    report.extend(
        f"| `{r['raw_name']}` | {r['shapes']} | {r['tasks']} |"
        for r in all_ops
        if r["fewer_than_10"]
    )
    (folder / "shape-index.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, default=ROOT / "results/v2.4/catalog")
    build(cli.parse_args().data)
