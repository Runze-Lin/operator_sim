"""Freeze and replay FA Vector prices against the released CA windows, without sampling CA."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from statistics import median
from time import perf_counter_ns

from operator_model import estimate, load_chip
from operator_model.operators.attention.vector_classes import vector_class_tile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/v2.1"
WINDOWS = ROOT / "calibration/reference/windows.jsonl"
FA = ROOT / "calibration/examples/fa-950dt.json"
MODES = ("baseline", "instruction_classes")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    package = ROOT / "src/operator_model"
    paths = sorted([*package.rglob("*.py"), *package.rglob("*.json"), WINDOWS, FA, OUT/"fa-vector-opcodes.json", Path(__file__).resolve()])
    fingerprint = hashlib.sha256()
    for p in paths:
        fingerprint.update(f"{p.relative_to(ROOT)} {digest(p)}\n".encode())
    return {"sha256": fingerprint.hexdigest(), "files": len(paths),
            "policy": "All archived op=fa windows, paired with their v2.0 prices. Development replay. No new CA.",
            "subgroups": "D=64/128 and KV<=1024 or KV>1024, separately by SKU. Warm process, 3 alternating calls/mode/window."}


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def query(shape, chip, mode, dtype="bf16", port_model=None):
    attrs = {} if mode == "baseline" else {"vector_model": mode}
    if port_model:
        attrs["fa_port_model"] = port_model
    start = perf_counter_ns()
    r = estimate("fa", shape, attrs=attrs, chip=chip, dtype=dtype)
    elapsed = (perf_counter_ns() - start) / 1e6
    if r.latency_us is None:
        raise ValueError((shape, mode, r.reason))
    return r, elapsed


def measured(shape, chip, line):
    results = {}
    samples = {mode: [] for mode in MODES}
    for rep in range(3):
        for mode in (MODES if (rep + line) % 2 else MODES[::-1]):
            results[mode], ms = query(shape, chip, mode)
            samples[mode].append(ms)
    return results, samples


def summary(rows):
    out = {"windows": len(rows), "shapes": len({r["shape"] for r in rows})}
    for mode in MODES:
        errors = [abs(r[mode+"_us"] / r["ca_us"] - 1)*100 for r in rows]
        wall = [v for r in rows for v in r[mode+"_wall_ms"]]
        out[mode] = {
            "wape_pct": 100*sum(abs(r[mode+"_us"] - r["ca_us"]) for r in rows)/sum(r["ca_us"] for r in rows),
            "max_ape_pct": max(errors),
            "worst_line": rows[errors.index(max(errors))]["line"],
            "cpu_median_ms": median(wall), "cpu_p95_ms": sorted(wall)[math.ceil(.95*len(wall))-1],
            "cpu_max_ms": max(wall)}
    return out


def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 5.8), constrained_layout=True)
    for col, chip in enumerate(sorted({r["chip"] for r in rows})):
        rs = [r for r in rows if r["chip"] == chip]
        for mode, marker, color in zip(MODES, ["o", "+"], ["#196968", "#b45527"]):
            axes[0,col].scatter([r["ca_us"] for r in rs],
                 [100*(r[mode+"_us"]/r["ca_us"]-1) for r in rs], label=mode,
                 marker=marker, color=color, s=18, alpha=.85)
        axes[0,col].axhline(0,color="grey",lw=.8)
        axes[0,col].set_xscale("log")
        axes[0,col].set_title(chip)
        axes[0,col].set_xlabel("CA device time (µs)")
        for phase,color in zip(("first_tile", "steady", "drain"),("#6479a8","#b45527","#196968")):
            selected = sorted([r for r in rs if r["shape"][0]==9 and r["shape"][2:]==(64,1,1)],key=lambda r:r["shape"][1])
            axes[1,col].plot([r["shape"][1] for r in selected],
                 [r["instruction_classes_"+phase+"_us"]-r["baseline_"+phase+"_us"] for r in selected],
                 marker=".",lw=1, label=phase, color=color)
        axes[1,col].set_xscale("log",base=2)
        axes[1,col].set_xlabel("Q=9, D=64, one head: KV length")
        for ax in axes[:,col]:
            ax.grid(alpha=.15)
    axes[0,0].set_ylabel("Signed device-time error (%)")
    axes[1,0].set_ylabel("Class model minus baseline (µs)")
    axes[0,0].legend(fontsize=8)
    axes[1,0].legend(fontsize=8)
    fig.savefig(ROOT/"docs/figures/fa-vector-models.png",dpi=180)
    plt.close(fig)


def audit_trace(path):
    evidence = json.loads(FA.read_text())
    if digest(path) != evidence["steady_db_sha256"]:
        raise ValueError("The trace does not match the archived DT 32Ki reference.")
    with sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True) as con:
        records = con.execute("""
            SELECT CoreTypeId,ExecInstrName,COUNT(*) FROM ExecutedInstructions
            WHERE CoreTypeId IN (2,3) AND ExecInstrCoreId=0 AND ExecInstrName GLOB 'RV_*'
            AND ExecInstrIsDurationExcluded=0 GROUP BY CoreTypeId,ExecInstrName
            ORDER BY CoreTypeId,ExecInstrName""").fetchall()
        opcode_union = {}
        for core in (2,3):
            total = previous_end = 0
            intervals = con.execute("""
                SELECT ExecInstrTickStart,ExecInstrTickEnd FROM ExecutedInstructions
                WHERE CoreTypeId=? AND ExecInstrCoreId=0 AND ExecInstrName GLOB 'RV_*'
                AND ExecInstrIsDurationExcluded=0 ORDER BY ExecInstrTickStart""", (core,))
            for start,end in intervals:
                total += max(0, end-max(previous_end,start))
                previous_end = max(previous_end,end)
            opcode_union[str(core)] = total
    dump(OUT/"fa-vector-opcodes.json", {"source_sha256":digest(path),
         "shape":evidence["steady_shape"], "filter":"Physical core 0, AIV core types 2/3, RV_* with valid duration.",
         "opcode_union_cycles_by_aiv_type":opcode_union,
         "counts_by_aiv_type":{str(core):{name:count for i,name,count in records if i==core} for core in (2,3)},
         "measured_busy_cycles_max_aiv":evidence["queue_union_ticks"]["SIMD"]})


def vector_audit():
    """Compare a selected busy AIV's logical program with observed opcodes (no fitting)."""
    evidence = json.loads(FA.read_text())
    measured = json.loads((OUT/"fa-vector-opcodes.json").read_text())
    if measured["source_sha256"] != evidence["steady_db_sha256"]:
        raise ValueError("Opcode archive source mismatch")
    chip = load_chip(evidence["chip"])
    q,kv,d,b,h = evidence["steady_shape"]
    pages = kv//evidence["page_keys"]
    if (q,b,h,pages) != (9,1,1,128):
        raise ValueError("Busy-lane identity must be re-established")
    totals = {}
    cycles = 0
    for first,last,number in ((True,False,1),(False,False,pages-2),(False,True,1)):
        tile = vector_class_tile(5,256,d,first=first,last=last,dtype="bf16",chip=chip)
        cycles += number*(tile["vec1"]+tile["vec2"])
        for phase in tile["phases"].values():
            for key,count in phase["counts"].items():
                totals[key] = totals.get(key,0) + number*count
    raw = measured["counts_by_aiv_type"]["2"]
    # 5 VCMAX per page confirms AIV type 2 carries the five-row share.
    assert raw["RV_VCMAX"] == pages*5
    return {"shape": [q,kv,d,b,h], "pages":pages, "aiv_type":2,
            "ca_valid_opcode_subset":{"exp":raw["RV_VEXPDIF"],"divide":raw["RV_VDIV"],
                                      "convert":raw["RV_VCVT_F2F"],"reduce":raw["RV_VCMAX"]+raw["RV_VCADD"]},
            "predicted_counts":totals, "predicted_vf_service_cycles":cycles,
            "ca_type2_rv_only_union_cycles":measured["opcode_union_cycles_by_aiv_type"]["2"],
            "ca_max_aiv_simd_union_cycles":evidence["queue_union_ticks"]["SIMD"],
            "interpretation":"The CA reference is an interval union. Modeled service includes dependency and port constraints; equality is not required."}

def compare_new_precision():
    """Exercise the uncalibrated FP16 design contract without assigning CA error."""
    data=[]
    for sku in ("Ascend950DT_9582","Ascend950PR_957b","Ascend970DT"):
        chip=load_chip(sku)
        for shape in ((9,32768,64,1,1),(128,128,64,1,1)):
            auto,_=query(shape,chip,"baseline",dtype="fp16")
            classes,_=query(shape,chip,"instruction_classes",dtype="fp16")
            assert auto.latency_us==classes.latency_us
            data.append({"shape":shape,"dtype":"fp16","sku":sku,"status":auto.status,
                         "auto_vector_model":auto.projection["timing_diagnostics"]["vector_model"],
                         "predicted_us":auto.latency_us,"ca_us":None})
    dump(OUT/"fa-vector-fp16.json",data)


def compare_port_arms(rows):
    """Keep Vector method and port assumptions orthogonal on matched geometries."""
    dt,pr = ("Ascend950DT_9582", "Ascend950PR_957b")
    results=[]
    for kv in (8191,32768,131072):
        shape = (9,kv,64,1,1)
        paired={r["chip"]:r for r in rows if r["shape"]==shape}
        ca_ratio = paired[dt]["ca_us"]/paired[pr]["ca_us"]
        for mode in MODES:
            for port in ("default","probe"):
                preds={}
                for sku in (dt,pr):
                    pred,_=query(shape,load_chip(sku),mode,port_model=port)
                    preds[sku]=pred.latency_us
                ratio=preds[dt]/preds[pr]
                results.append(dict(shape=shape,vector_model=mode,port_model=port,
                    dt_us=preds[dt],pr_us=preds[pr],dt_ca_us=paired[dt]["ca_us"],pr_ca_us=paired[pr]["ca_us"],
                    speedup_dt_over_pr=ratio,ca_speedup_dt_over_pr=ca_ratio,
                    speedup_error_pct=100*(ratio/ca_ratio-1)))
    dump(OUT/"fa-vector-port-ablation.json",results)

def evaluate():
    rows = []
    future = {}
    chip970 = load_chip("Ascend970DT")
    query((9,32768,64,1,1),chip970,"baseline")
    query((9,32768,64,1,1),chip970,"instruction_classes")
    for line,raw in enumerate(WINDOWS.read_text().splitlines(),1):
        original = json.loads(raw)
        if original["op"] != "fa":
            continue
        shape = tuple(original["shape"])
        chip = load_chip(original["chip"])
        states,timings = measured(shape,chip,line)
        base = states["baseline"].latency_us
        if not math.isclose(base,original["predicted_us"],rel_tol=1e-10):
            raise ValueError(f"default estimate changed on window {line}")
        if shape not in future:
            future[shape] = {m:query(shape,chip970,m)[0].latency_us for m in MODES}
        row = dict(line=line, shape=shape, chip=original["chip"], ca_us=original["ca_us"],role=original["data_role"])
        for mode,r in states.items():
            diag = r.projection["timing_diagnostics"]
            wave = diag["wave_types"][0]
            t = wave["first"]
            fields = dict(us=r.latency_us,signed_error_pct=100*(r.latency_us/original["ca_us"]-1),wall_ms=timings[mode],extrapolated_970dt_us=future[shape][mode],
                          first_page_vec1_cycles=t["vec1"],first_page_vec2_cycles=t["vec2"],
                          first_wave_first_page_bound=t["steady_bottleneck"],
                          first_wave_repeat_bound=(wave.get("repeat") or {}).get("steady_bottleneck"),
                          first_wave_repeat_vec_cycles=(wave.get("repeat") or {}).get("demands_cycles",{}).get("vector"),
                          **{k+"_us":v for k,v in diag["stages_us"].items()})
            row.update({mode+"_"+k:v for k,v in fields.items()})
        rows.append(row)
    with (OUT/"fa-vector-windows.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(tuple,list)) else v for k,v in row.items()})
    metrics={}
    for sku in sorted({r["chip"] for r in rows}):
        group = [r for r in rows if r["chip"]==sku]
        for key,rs in (("all",group),("D64",[r for r in group if r["shape"][2]==64]),
                       ("D128",[r for r in group if r["shape"][2]==128]),
                       ("KV<=1024",[r for r in group if r["shape"][1]<=1024]),
                       ("KV>1024",[r for r in group if r["shape"][1]>1024])):
            metrics[sku+" "+key]=summary(rs)
    metrics["DT_32Ki_stages"] = vector_audit()
    dump(OUT/"fa-vector-metrics.json",metrics)
    compare_new_precision()
    compare_port_arms(rows)
    make_figure(rows)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("freeze","replay","audit"))
    parser.add_argument("--db",type=Path)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    manifest=OUT/"fa-vector-freeze.json"
    if args.action=="freeze":
        dump(manifest,freeze())
    elif args.action=="audit":
        if not args.db: parser.error("--db is required for audit")
        audit_trace(args.db)
    else:
        if json.loads(manifest.read_text()) != freeze(): raise SystemExit("Frozen inputs changed")
        evaluate()
