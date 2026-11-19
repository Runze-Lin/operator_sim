"""Plot the current resource schedule against archived, qualified whole-kernel windows."""
import argparse
from hashlib import sha256
import json
from math import isclose
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/operator-sim-mpl")
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results/v0.23/prefill"
FIG = ROOT / "docs/examples/figures"
ARCHIVE = ROOT / "results/v0.20"
CASES = (
    ("flash", "DeepSeek V4.1 Flash", 32768, 32, "cast_bf16_fp32", "flash-32ki"),
    ("flash", "DeepSeek V4.1 Flash", 131072, 32, "cast_bf16_fp32", "flash-128ki"),
    ("flash", "DeepSeek V4.1 Flash", 262144, 32, "cast_bf16_fp32", "flash-256ki"),
    ("pro", "DeepSeek V4 Pro", 32768, 64, "cast_bf16_fp32", "pro-32ki"),
    ("kimi", "Kimi K3", 32768, 12, "sigmoid_fp32", "prefill32k"),
    ("kimi", "Kimi K3", 131072, 12, "sigmoid_fp32", "prefill128k"),
    ("kimi", "Kimi K3", 262144, 12, "sigmoid_fp32", "prefill256k"),
)
PIPES = {"Read": "#14798a", "Vector": "#7957a7", "Write": "#b34a26"}
INK = "#1b3341"
MUTED = "#496372"

def read(path):
    return json.loads(path.read_text())

def compare(case, sku):
    from development.validation.vector21.figures import cast, sigmoid
    from operator_model import estimate, __version__

    tag, name, seq, width, op, archive_name = case
    shape = [8 * seq, width]
    chip = {"dt": "Ascend950DT_9582", "pr": "Ascend950PR_957b"}[sku]
    if op == "sigmoid_fp32":
        obs_path = ARCHIVE / "newop" / ("timeline-" + archive_name + ("-pr" if sku == "pr" else "") + ".json")
        record_path = ARCHIVE / "newop/holdout.json"
        row, = [r for r in read(record_path) if r["shape"] == shape and r["sku"] == chip]
        receipt = row["kernel"]
        assert receipt["trace_sha256"] == read(obs_path)["trace_sha256"]
        schedule = sigmoid(shape, chip, read(obs_path))
        counts = [t["bytes"] // 4 for t in schedule["observed"]]
    else:
        obs_path = ARCHIVE / "large" / f"{archive_name}-tiles-{chip}-plot.json"
        record_path = ARCHIVE / "large" / f"cast_{shape[0]}_{width}_{chip}.json"
        row = read(record_path)
        receipt = row["stage"]
        data = read(obs_path)
        assert (row["shape"], row["chip"]) == (shape, chip)
        schedule = cast(shape, chip, data)
        counts = [e["bytes"] // 2 for e in data["dma"]["MTE2"]]
    assert row["qualified"] and isclose(schedule["ca_us"], row.get("ca_us", receipt["ca_us"]))
    plan = schedule["plan"]
    cap = plan["ub_tile_elements"]
    n = len(counts)
    predicted = schedule["model"]
    observed = schedule["observed"]
    assert plan["active_aiv"] == receipt["active_aiv"]
    assert n == len(predicted) == len(observed)
    critical = plan.get("critical_core_elements", plan.get("busy_core_elements"))
    expected = [min(cap, critical - i * cap) for i in range(n)]
    assert 0 < expected[-1] <= cap and counts[:-1] == expected[:-1]
    assert counts[:-1] == [cap] * (n - 1) and 0 < counts[-1] <= cap
    for blocks, end in [(predicted, schedule["model_us"]), (observed, schedule["ca_us"])]:
        for block in blocks:
            assert all(-1e-6 <= block[p][0] <= block[p][1] <= end + .001 for p in PIPES)
    simulation = estimate(op, shape, chip=chip, timing_mode="resource")
    assert isclose(simulation.latency_us, schedule["model_us"])
    future = estimate(op, shape, chip="Ascend970DT", timing_mode="resource")
    gap = lambda i: predicted[i]["Read"][0] - observed[i]["Read"][0]
    return dict(model_name=name, op=op, shape=shape, chip=chip, model_version=__version__,
                batch=8, sequence=seq, active_aiv=plan["active_aiv"], ca_tile_elements=counts, model_tile_elements=expected,
                ca_core=schedule["core"], model_us=simulation.latency_us, ca_us=schedule["ca_us"],
                projection_970dt_us=future.latency_us, ca_wall_s=row["cost"]["seconds"],
                kernel=receipt.get("kernel", receipt.get("name")), model_tiles=predicted, ca_tiles=observed,
                read_start_gap_us={str(i+1):gap(i) for i in (0, n//2, n-1)},
                origin="first instruction on any core; positive gap means model is late",
                role="archived CA replay with current resource model; 970DT is projection only",
                model_semantics=schedule["phase"], trace_sha256=schedule["trace_sha256"],
                receipt=str(record_path.relative_to(ROOT)), intervals=str(obs_path.relative_to(ROOT)),
                intervals_sha256=sha256(obs_path.read_bytes()).hexdigest())

def draw(d, output):
    total = len(d["ca_tile_elements"])
    selected = sorted({0, 1, total//2, total-2, total-1})
    error = 100 * (d["model_us"] / d["ca_us"] - 1)
    short = d["chip"].split("_")[0].replace("Ascend","")
    cast_case = d["op"].startswith("cast")
    opname = "Cast BF16 → FP32" if cast_case else "FP32 Sigmoid"
    fig, axes = plt.subplots(2, 1, figsize=(11.3, 7.7), sharex=True)
    fig.subplots_adjust(left=.193, right=.96, top=.77, bottom=.137, hspace=.44)
    fig.text(.193,.946,f"{short}  {opname}  |  {d['shape'][0]:,} × {d['shape'][1]}",size=21,weight="bold",color=INK)
    fig.text(.193,.902,f"{d['model_name']}   ·   8 × {d['sequence']//1024}k prefill   ·   {d['active_aiv']} AIVs   ·   {total} UB blocks on the last core",size=12,color=MUTED)
    labels = ("MTE2  /  GM → UB", "Vector  /  conversion" if cast_case else "Vector  /  sigmoid", "MTE3  /  UB → GM")
    fig.legend([Patch(color=c) for c in PIPES.values()], labels, loc="upper left",
               bbox_to_anchor=(.184,.884), ncol=3, frameon=False, labelcolor=INK)
    for idx, (ax, tiles, end) in enumerate(zip(axes, (d["model_tiles"],d["ca_tiles"]), (d["model_us"],d["ca_us"]))):
        counts = d["ca_tile_elements" if idx else "model_tile_elements"]
        rows = [f"UB block {i+1}\n{counts[i]:,} elements" for i in selected] + ["All read issues" if idx else "All read starts"]
        title = ("Resource model  ·  rate-equivalent schedule" if cast_case else "Resource model  ·  two-buffer schedule") if idx==0 else f"CA trace  ·  {d['ca_core']} (last to finish)"
        ax.set_title(title,loc="left",fontsize=13,weight="bold",color=INK,pad=9)
        ax.set_title(f"{end:.3f} µs",loc="right",fontsize=14,weight="bold",color=INK,pad=9)
        for j,i in enumerate(selected):
            for k,(p,color) in enumerate(PIPES.items()):
                a,b = tiles[i][p]
                y=j+(k-1)*.18
                ax.barh(y,b-a,left=a,height=.14,color=color,alpha=.65 if idx==0 and cast_case else 1,zorder=3)
                ax.vlines(a,y-.10,y+.10,color=color,lw=.65,zorder=4)
            if j and i-selected[j-1]>1:
                ax.axhline(j-.5,color="#d8e2e7",lw=.9)
        ax.vlines([b["Read"][0] for b in tiles],len(selected)-.18,len(selected)+.18,color=PIPES["Read"],lw=.9)
        ax.axvline(end,color="#546b78",lw=1,ls=(0,(3,3)))
        ax.set_yticks(range(len(rows)),rows,color=INK,fontsize=9)
        ax.set_ylim(len(rows)-.45,-.57)
        ax.set_xlim(0,max(d["ca_us"],d["model_us"])*1.035)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=11))
        ax.tick_params(length=0,pad=6,labelcolor=MUTED)
        ax.grid(axis="x",color="#dbe4e8",lw=.7)
        ax.set_axisbelow(True)
        ax.spines[:].set_visible(False)
    axes[-1].set_xlabel("Device time from first instruction on any core (µs)",color=INK,labelpad=10)
    foot = ("Model intervals allocate exposed service by tile bytes. CA shows DMA descriptors and conversion intervals."
            if cast_case else "Model intervals follow the UB-slot dependency schedule. CA Vector spans first RV_VLD through last RV_VST.")
    fig.text(.193,.045,foot,fontsize=9,color=MUTED)
    fig.text(.193,.018,f"All-read ticks include every block on this core. Whole-kernel error {error:+.1f}% (model − CA).",fontsize=9,color=MUTED)
    fig.savefig(output,dpi=150,facecolor="white")
    plt.close(fig)

def main():
    cases = {f"{case[0]}-{case[2]//1024}ki-{sku}": (case, sku)
             for case in CASES for sku in ("dt", "pr")}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", dest="case_id", choices=cases, help="Draw a single model, length and SKU.")
    parser.add_argument("--replot", action="store_true", help="Use the published intervals without rerunning the predictor.")
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    for slug in ([args.case_id] if args.case_id else cases):
        path = OUT / (slug + ".json")
        if args.replot:
            d = read(path)
        else:
            d = compare(*cases[slug])
            path.write_text(json.dumps(d, indent=2, ensure_ascii=False)+"\n")
        draw(d, FIG / ("prefill-"+slug+".png"))
        print(slug, d["model_us"], d["ca_us"], d["projection_970dt_us"], d["read_start_gap_us"])

if __name__=="__main__":
    main()
