"""Refresh the B8 grid for registered BF16 Matmuls and the current Cast schedule."""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model import estimate
from development.validation.matmul28.overview import DT, PR, FUT, OUT, read, receipt


def refresh():
    frozen = read(OUT / "overview-prices.json")["rows"]
    indexed = {(r["model"], r["node"], tuple(r["shape"])): r for r in frozen
               if r["phase"] == "prefill" and r["requests"] == 8}
    for path in (ROOT / "docs/workloads").glob("*.md"):
        lines = path.read_text().splitlines()
        model = lines[0].removeprefix("# ").split(" · ")[0]
        if model not in {r["model"] for r in frozen}:
            continue
        for i, line in enumerate(lines):
            if not line.startswith("| `"):
                continue
            fields = [col.strip() for col in line.strip("|").split("|")]
            node = fields[0].split("`")[1]
            tensors = {k: json.loads(v) for k, v in
                       re.findall(r"(\w+) <code>(\[[\d,]+\])</code>", fields[1])}
            x = tensors.get("x", [])
            w = tensors.get("weight_logical", [])
            if len(x) == len(w) == 2:
                item = indexed.get((model, node, (x[0], w[0], x[1])))
                if item and x[1] == w[1] and tensors.get("y") == [x[0], w[0]]:
                    fields[2] = "[独立 BF16 ND Matmul](../../results/v0.28/matmul/README.md)"
                    for col, chip, tag in [(3, DT, "dt"), (4, PR, "pr")]:
                        rec = receipt(item, tag)
                        ca = f"{rec['ca_us']:.3f}" if rec["status"] == "qualified" else (
                            "待回收" if rec["status"] == "pending" else "未合格")
                        fields[col] = f"{item['predictions'][chip]['us']:.3f} / {ca}"
                    fields[5] = f"{item['predictions'][FUT]['us']:.3f}"
            if fields[2] == "`cast_bf16_fp32`":
                shape = next(iter(tensors.values()))
                for col, chip in [(3, DT), (4, PR), (5, FUT)]:
                    value = estimate("cast_bf16_fp32", shape, chip=chip).latency_us
                    fields[col] = re.sub(r"^(\[?)\d+\.\d+", lambda m: f"{m[1]}{value:.3f}", fields[col])
            lines[i] = "| " + " | ".join(fields) + " |"
        updated = "\n".join(lines) + "\n"
        if model == "DeepSeek-V4-Pro":
            updated = updated.replace("950PR 为 520.357 / 529.287", "950PR 为 506.517 / 529.287")
        if updated != path.read_text():
            path.write_text(updated)


if __name__ == "__main__":
    refresh()
