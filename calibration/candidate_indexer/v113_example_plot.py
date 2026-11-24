"""Retain the slowest Cube team's native MMAD starts, then redraw the Q64 example."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ("shared","moving","rotated")
FREQ = 1650


def collect(raw):
    from calibration.mx_matmul.collect import trace_events
    frozen = json.loads((HERE/"v113b_frozen.json").read_text())["prices"]
    rows = []
    for arm in ARMS:
        receipt = json.loads((HERE/f"v113_control_{arm}.json").read_text())
        point = next(x["known_layout"] for x in frozen if x["case"]=="control_"+arm
                     and x["sku"]==receipt["sku"])
        plan = point["stages"]
        service = plan["tile_service_cycles"]
        first = plan["stages_us"]["startup"] + sum(service[i] for i in ("gather","publish","cube_read"))/FREQ
        gap1 = plan["first_fragment_interval_cycles"]/FREQ
        gap2 = plan["carried_fragment_interval_cycles"]/FREQ
        next_first = first + 31*gap1 + plan["same_document_handoff_cycles"]/FREQ
        predicted = [first+i*gap1 for i in range(32)] + [next_first+i*gap2 for i in range(32)]
        traces = {}
        for path in (raw/arm/"profile").glob("OPPROF_*/simulator/core*.cubecore0/trace.json"):
            pairs = sorted((round(e["ts"]*FREQ),round((e["ts"]+e["dur"])*FREQ))
                           for e in trace_events(path) if e.get("ph")=="X" and e["name"]=="MMAD_MX")
            if len(pairs)!=64:
                raise ValueError(f"missing Q64 MMAD on {path}")
            traces[path.parent.name]=pairs
        if len(traces)!=32:
            raise ValueError("missing physical Cube cores")
        key = max(traces,key=lambda k:traces[k][-1][1])
        origin = receipt["first_tick"]
        rows.append(dict(arm=arm,ca_core=key,model_window=point["ca_window_us"],
                         ca_window=receipt["device_us"],model_issue_us=predicted,
                         ca_issue_us=[(start-origin)/FREQ for start,_ in traces[key]],
                         ca_retire_us=[(end-origin)/FREQ for _,end in traces[key]]))
    return rows


def draw(rows, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(3,1,figsize=(10,7.6),sharex=True)
    colors=["#147989","#7055a4"]
    for ax,row in zip(axes,rows):
        for lane,(prefix,caption) in enumerate([("ca","CA"),("model","Model")]):
            stamps=row[prefix+"_issue_us"]
            for i,color in enumerate(colors):
                xs=stamps[i*32:(i+1)*32]
                ax.plot([xs[0],xs[-1]],[lane,lane],lw=5,color=color,alpha=.28,solid_capstyle="butt")
                ax.vlines(xs,lane-.12,lane+.12,linewidth=.8,color=color)
            ax.plot(stamps[31:33],[lane,lane],ls=":",lw=1.5,color="#4f6472")
            window=row[prefix+"_window"]
            ax.vlines(window,lane-.24,lane+.24,color="#172e3c",lw=1.5)
            ax.text(window+1,lane+.16,f"{window:.2f} µs",fontsize=8,color="#172e3c")
        ax.set(yticks=[0,1],yticklabels=["CA","Model"],ylim=(-.55,1.55))
        ax.set_title(f"{row['arm'].capitalize()}   ·   {row['ca_core']}",loc="left",fontweight="bold",fontsize=11)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.grid(axis="x",alpha=.21)
    axes[-1].set_xlabel("Device time since earliest kernel instruction (µs)")
    axes[-1].set_xlim(0,max(max(r["model_window"],r["ca_window"]) for r in rows)*1.15)
    fig.suptitle("Flash candidate indexer on 950DT   |   64 queries, 1 Mi context",
                 x=.08,y=.975,ha="left",fontweight="bold",fontsize=16)
    handles=[Line2D([0],[0],color=c,marker="|",label=l) for c,l in zip(colors,["First row MMAD issues","Second row MMAD issues"])]
    handles.append(Line2D([0],[0],color="#172e3c",marker="|",lw=0,label="Full device window"))
    fig.legend(handles=handles,ncol=3,loc="upper left",bbox_to_anchor=(.073,.942),frameon=False,fontsize=9)
    fig.subplots_adjust(top=.84,bottom=.105,left=.09,right=.98,hspace=.64)
    fig.savefig(target,dpi=155,facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,help="extract once from the three complete CA runs")
    args=parser.parse_args()
    data=HERE/"v113_example_issues.json"
    if args.raw:
        data.write_text(json.dumps(collect(args.raw),indent=2)+"\n")
    draw(json.loads(data.read_text()),ROOT/"docs/figures/flash-qsli-q64-layout.png")
