"""Plot relative changes in predicted decode-round time for the 27 configurations."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]


def render():
    with (ROOT/"calibration/reference/system-full-comparison.csv").open() as source:
        rows={(r["model"],r["layout"],int(r["batch"])):r for r in csv.DictReader(source)}
    layouts=("TP8/EP8","TP8/EP32","DEP64")
    batches=(1,16,64)
    norm=TwoSlopeNorm(vmin=-100,vcenter=0,vmax=175)
    fig,axes=plt.subplots(1,3,figsize=(11.3,3.2),layout="constrained")
    for i,(ax,(model,label)) in enumerate(zip(axes,(("v4-pro","DeepSeek-V4 Pro"),
                          ("v41-flash","DeepSeek-V4.1 Flash"),("kimi-k31","Kimi K3.1 Pro")))):
        matrix=[[float(rows[(model,g,b)]["delta_pct"]) for b in batches] for g in layouts]
        im=ax.imshow(matrix,aspect="auto",cmap="RdBu_r",norm=norm)
        for y,layout in enumerate(layouts):
            for x,b in enumerate(batches):
                value=matrix[y][x]; is_oversize=rows[(model,layout,b)]["fits_144gb"]=="False"
                color="white" if norm(value)>.8 or norm(value)<.2 else "#16212c"
                ax.text(x,y,f"{value:+.1f}%"+("*" if is_oversize else ""),ha="center",va="center",
                        color=color,fontsize=10)
        ax.set_title(label,fontsize=11)
        ax.set_xticks(range(3),batches);ax.set_xlabel("Local batch size")
        ax.set_yticks(range(3),layouts if i==0 else [""]*3)
        ax.tick_params(length=0)
        for spine in ax.spines.values():spine.set_visible(False)
    fig.suptitle("Predicted decode-round time change vs. original Roofline")
    bar=fig.colorbar(im,ax=axes,shrink=.83)
    bar.set_label("Change (%)");bar.set_ticks([-100,-50,0,50,100,150])
    fig.savefig(ROOT/"docs/figures/system-full-decode.png",dpi=155)
    plt.close(fig)


if __name__=="__main__":
    render()
