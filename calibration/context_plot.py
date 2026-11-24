"""Plot phase times at fixed parallelism and the 147-round Kimi shape transition."""
import argparse
import csv
from pathlib import Path


def render(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"v4-pro":"#176a75","v41-flash":"#ad4b28","kimi-k31":"#654c9b"}
    names = {"v4-pro":"DeepSeek-V4 Pro","v41-flash":"DeepSeek-V4.1 Flash","kimi-k31":"Kimi K3.1 Pro"}
    with (data/"requests.csv").open() as f:
        requests=list(csv.DictReader(f))
    with (data/"graphs.csv").open() as f:
        graphs=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.0),layout="constrained")
    for ax,phase,label in zip(axes,("prefill","decode"),("Prefill","Speculative decode")):
        for model,color in colors.items():
            rows=sorted((r for r in requests if r["model"]==model and r["batch"]=="1"),key=lambda r:int(r["prompt"]))
            ys=[float(r[phase+"_full_s"]) for r in rows]
            limit=4 if phase=="prefill" else 3
            ax.plot(range(limit),ys[:limit],color=color,marker="o",label=names[model])
            if limit==3:
                ax.plot([2,3],ys[2:],color=color,ls="--")
                ax.scatter([3],ys[3:],facecolor="white",edgecolor=color,zorder=3)
        ax.set(yscale="log",title=label,ylabel="Phase time (s)",xlabel="Input per request",
               xticks=range(4),xticklabels=["32Ki","128Ki","512Ki","1Mi"])
        ax.grid(axis="y",which="both",alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].legend(fontsize=8,frameon=False)
    fig.suptitle("950DT system simulation   ·   B=1   ·   TP8 / EP32   ·   1024 output tokens",fontsize=12)
    output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/"context-phase-times.png",dpi=160)
    plt.close(fig)

    target={}
    for g in graphs:
        if (g["model"],g["prompt"],g["batch"],g["phase"])==("kimi-k31","131072","32","decode"):
            i=(int(g["kv_history"])-131072)//7
            target[i]=target.get(i,0)+float(g["full_us"])/1e3
    keys=sorted(target)
    if keys!=list(range(147)):
        raise ValueError("Expected one complete evaluation for every decode round")
    fig,ax=plt.subplots(figsize=(8.8,3.25),layout="constrained")
    ax.scatter(keys,[target[k] for k in keys],s=11,color="#205c72",label="Target + draft, per-round DES")
    anchor=[0,73,146]
    ax.plot(anchor,[target[k] for k in anchor],color="#b35c32",ls="--",marker="o",ms=5,label="Three-point interpolation")
    ax.set(title="Kimi K3.1 Pro   ·   128Ki input   ·   B=32   ·   TP8 / EP64",
           xlabel="Speculative decode round (zero-based)",ylabel="Round time (ms)")
    ax.grid(axis="y",alpha=.2)
    ax.legend(frameon=False,fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    fig.savefig(output/"context-kimi-rounds.png",dpi=160)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("data",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    render(a.data,a.output)
