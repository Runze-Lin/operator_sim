"""Illustrative same-clock schedule and physical DMA for the first registered DT endpoint."""
from pathlib import Path
import hashlib
import json
import math
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/v0.34/attn"
PNG = ROOT / "docs/examples/figures/attn-res-kimi-prefill-65-dt.png"
DT = "Ascend950DT_9582"
CLOCK = 1650.0


def prepare(prefix):
    sys.path.insert(0, str(prefix))
    from operator_model import estimate

    reg = json.loads((OUT / "freeze.json").read_bytes())
    job = next(j for j in reg["jobs"] if j["shape"] == [65, 7168, 8, 1])
    r = estimate("attn_res_fwd", job["shape"], job["attrs"], chip=DT)
    if not math.isclose(r.latency_us, job["prices"][DT]["us"], abs_tol=1e-8):
        raise ValueError("changed frozen price")
    diag = r.projection["timing_diagnostics"]
    q, st = diag["per_core_cycles"], r.projection["stages_us"]
    model = []
    def bar(row, start, duration, resource):
        model.append([row, start/CLOCK, duration/CLOCK, resource])

    dispatch = st["dispatch"] * CLOCK
    bar(0, 0, dispatch, "launch")
    bar(0, dispatch, q["weight_setup"]+q["gamma_setup"], "mixed")
    t = (st["dispatch"] + st["setup_weights"]) * CLOCK
    ready = t
    for row in (1, 2):
        bar(row, t, q["prefix"], "mixed")
        prefix_end = max(ready, t + q["prefix"])
        bar(row, prefix_end, q["store"], "write")
        bulk = q["score_softmax_mix"] + q["output_norm"]
        bar(row, prefix_end, bulk, "vector")
        t = prefix_end + max(bulk, q["store"])
        bar(row, t, q["store"], "write")
        ready = t + q["store"]
    if abs((ready+q["tail"])/CLOCK-r.latency_us)>1e-8:
        raise ValueError("rendering does not conserve complete critical window")

    file = OUT / "65-1-dt.json"
    actual = None
    if file.exists():
        receipt = json.loads(file.read_bytes())
        v = receipt["verdict"]
        if v.get("qualified") is True:
            if receipt["freeze_sha256"] != hashlib.sha256((OUT/"freeze.json").read_bytes()).hexdigest():
                raise ValueError("unregistered window")
            sys.path.insert(0, str(ROOT/"development/validation/coverage33"))
            from rms_archive import moves, endpoint

            core = max(v["core_ticks"], key=lambda k: v["core_ticks"][k][1])
            shard = (ROOT/v["trace"]).parent/core/"trace.json"
            marks = endpoint(shard)
            # Raw core times use a physical origin; align both panes to the earliest system instruction.
            offset = max(z["retire"] for z in marks.values())-v["core_ticks"][core][1]+min(a for a,b in v["core_ticks"].values())
            events = {"MTE2": [], "MTE3": []}
            for e in moves(shard):
                d = json.loads(e["args"]["detail"])
                if d["instr_type"] in events and d["processed_bytes"]==14336 and not any(d.get(k) for k in ("is_wait","is_sync","is_duration_excluded")):
                    events[d["instr_type"]].append([e["ts"]-offset/CLOCK, e["dur"]])
            for es in events.values():
                es.sort()
            n = len(events["MTE3"])//2
            bars = []
            mapped = (n in (1,2) and len(events["MTE3"])==2*n and len(events["MTE2"])==3+3*n)
            if mapped:
                for i,(a,d) in enumerate(events["MTE2"]):
                    bars.append([0 if i<3 else 1+(i-3)//3, a, d, "read"])
                for i,(a,d) in enumerate(events["MTE3"]):
                    bars.append([1+i//2, a, d, "write"])
            actual = dict(core=core, us=v["instruction_us"], trace_sha256=v["trace_sha256"],
                          mapped=mapped, bars=bars, read_count=len(events["MTE2"]), write_count=len(events["MTE3"]))
    return dict(chip=DT, shape=job["shape"], model_us=r.latency_us, model_bars=model, ca=actual,
                clock_ticks_per_us=CLOCK, row_order=["setup", "first token", "second token"],
                origin="earliest device instruction", meaning="Model bars mix local compute, DMA and synchronization. CA uses physical row-sized descriptors on the last core.")


def render(prefix):
    d = prepare(prefix)
    target = OUT / "timeline-65-dt.json"
    payload = json.dumps(d, indent=2, allow_nan=False)+"\n"
    if target.is_file() and target.read_text()==payload and PNG.is_file():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    colors = dict(launch="#d1dbe2",mixed="#697d96",vector="#8562ac",read="#16848a",write="#b9532d")
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9,4.7), sharex=True)
    fig.subplots_adjust(left=.145,right=.98,bottom=.18,top=.72,hspace=.75)
    ca = d["ca"]
    hi = max(d["model_us"], ca["us"] if ca else 0)
    def pane(a, bars, title, us):
        a.set_title(title,loc="left",fontsize=11,fontweight="bold",color="#203540")
        a.set_yticks([0,1,2],["Setup", "Token 1", "Token 2"],fontsize=9)
        a.set_ylim(2.5,-.5)
        a.set_xlim(0,math.ceil(hi*1.04))
        a.grid(axis="x",color="#dfebed",lw=.7)
        a.set_axisbelow(True)
        for side in ("top","right","left"):
            a.spines[side].set_visible(False)
        for row,start,span,name in bars:
            top=.16 if name=="write" else -.12
            a.barh(row+top, span, left=start, height=.18, color=colors[name],edgecolor="none")
        if us is not None:
            a.axvline(us, color="#354b57",ls="--",lw=1)
            a.set_title(f"{us:.3f} µs",loc="right",fontsize=11,fontweight="bold")
    pane(ax,d["model_bars"],"Resource model · busiest AIV (two tokens)",d["model_us"])
    pane(bx,ca["bars"] if ca else [],"CA · "+(ca["core"]+" (last to finish)" if ca else "registered 40021 window"),ca["us"] if ca else None)
    if not ca or not ca["mapped"]:
        bx.text(.5,.48,"Pending full CA window" if not ca else "CA window qualified; row mapping unresolved",
                ha="center",va="center",transform=bx.transAxes,color="#495d65",fontsize=11,
                bbox=dict(facecolor="white",edgecolor="none",alpha=.9))
    fig.suptitle("950DT  Kimi AttnResFwd  |  65 × 7168, one history row",x=.145,ha="left",fontsize=16,fontweight="bold",color="#203540")
    labels={"launch":"Dispatch","mixed":"Mixed setup / prefix","vector":"Score / mix / norm","read":"MTE2 read","write":"MTE3 write"}
    fig.legend([Patch(color=c) for c in colors.values()],[labels[k] for k in colors],ncol=5,bbox_to_anchor=(.14,.915),loc="upper left",frameon=False,fontsize=8,handlelength=1.4)
    bx.set_xlabel("Device time from first instruction (µs)")
    fig.text(.145,.035,"Model colors denote exposed phases. CA bars include row-sized DMA descriptors only. Dashed lines mark complete windows.",fontsize=8,color="#526870")
    temp=PNG.with_suffix(".tmp.png")
    fig.savefig(temp,dpi=145);plt.close(fig)
    temp.replace(PNG)
    temp=target.with_suffix(".tmp");temp.write_text(payload);temp.replace(target)

if __name__=="__main__":
    render(Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT)
