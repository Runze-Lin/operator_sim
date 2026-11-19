"""Render full device time, per-core cadence and first/middle/tail close-ups."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "results/v0.21/wide-rms/tiles.json"
COLORS = {"x":"#127889","reduce":"#594798","normalize":"#a48ecb","rstd":"#cc8b28","y":"#af472e"}
NAMES = {"x":"Read x","reduce":"Reduce / rsqrt","normalize":"Normalize","rstd":"Write rstd","y":"Write y"}
INK, DIM = "#1a303e", "#496274"
ORDER = list(COLORS)


def window(tile, q):
    t = tile.get(q)
    if isinstance(t, dict):
        return [t["begin_us"], t["end_us"]]
    return t


def bar(ax, level, a, b, color, height=.48):
    ax.barh(level, b-a, left=a, height=height, color=color, linewidth=0,
            zorder=3)


def style(ax):
    ax.spines[:].set_visible(False)
    ax.grid(axis="x", lw=.7, color="#d8e1e6", zorder=0)
    ax.tick_params(axis="both", labelcolor=INK)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3,color=DIM)
    ax.set_axisbelow(True)


def draw(side, data):
    m,c = data["model"],data["ca"]
    tiles = m["tiles"]
    count,rows = len(tiles),m["rows_per_core"]
    selected = (0,count//2,count-1)
    fig = plt.figure(figsize=(14.2,10.2),facecolor="white")
    gs = fig.add_gridspec(3,2,width_ratios=(1.58,1),height_ratios=(.85,1.54,1.54),
                          left=.145,right=.974,top=.813,bottom=.153,hspace=.58,wspace=.25)
    whole = fig.add_subplot(gs[0,:])
    fig.text(.145,.951,f"950{side.upper()} RMSNorm  |  32,768 × 1,280",color=INK,fontsize=23,weight="bold")
    fig.text(.145,.915,f"BF16   ·   {m['active_aiv']} AIVs   ·   {rows} rows on the last core   ·   "
             f"{count-1} full tiles × 24 rows + {tiles[-1]['rows']} tail rows",color=DIM,fontsize=12.5)
    fig.legend([Patch(color=v) for v in COLORS.values()],list(NAMES.values()),frameon=False,
               ncol=5,loc="upper left",bbox_to_anchor=(.137,.896),labelcolor=INK,fontsize=10.8)
    whole.set_title("Full kernel window",loc="left",pad=10,color=INK,weight="bold",fontsize=14)
    phases = m["stages_us"]
    tm,ca = m["ca_independent_total_us"],c["ca_us"]
    simd = phases["SIMD"]
    local = m["local_us"]
    setup = local - simd - phases["MTE3"]
    for a,b,color in [(0,setup,"#8fa3b0"),(setup,setup+simd,"#7462a3"),(setup+simd,local,"#af472e"),
                      (local,tm,"#477a82")]:
        if b>a:bar(whole,1,a,b,color,.5)
    whole.text(setup+simd/2,1,"SIMD budget",ha="center",va="center",color="white",fontsize=9)
    if tm-local > 5:
        whole.text((local+tm)/2,1,"exposed shared I/O",ha="center",va="center",color="white",fontsize=9)
    bar(whole,0,0,c["core_begin_us"],"#bdcbd1",.5)
    bar(whole,0,c["core_begin_us"],ca,"#354c5a",.5)
    whole.text((ca+c["core_begin_us"])/2,0,f"{c['core']}  ·  last core",ha="center",va="center",
               color="white",fontsize=9)
    for y,t in [(0,ca),(1,tm)]:
        whole.text(t+.012*ca,y,f"{t:.3f} µs",ha="left",va="center",fontsize=10,color=INK,weight="bold")
    whole.set(yticks=[0,1],yticklabels=["CA","v0.21"],xlim=(0,ca*1.12),ylim=(-.47,1.47),
              xlabel="Time since the first instruction across all cores (µs)")
    style(whole)

    axes = [[],[]]
    leftmax=0; rightmin=rightmax=0
    for j,(model_or_ca,heading) in enumerate([(m,"Resource model"),(c,f"CA trace  ·  {c['core']}")]):
        all_ax,detail_ax = fig.add_subplot(gs[j+1,0]),fig.add_subplot(gs[j+1,1])
        axes[0].append(all_ax);axes[1].append(detail_ax)
        ts = model_or_ca["tiles"]
        base = window(ts[0],"rstd")[0]
        for i,t in enumerate(ts):
            for qidx,q in enumerate(ORDER):
                w=window(t,q)
                if w is None:
                    continue
                a,b=w[0]-base,w[1]-base
                bar(all_ax,4-qidx,a,b,COLORS[q],.55)
                if q=="x":
                    all_ax.vlines(a,4-qidx-.275,4-qidx+.275,color="white",lw=.6,zorder=4)
                leftmax=max(leftmax,b)
                if i in selected:
                    z=window(t,"rstd")[0]
                    a,b=w[0]-z,w[1]-z
                    y=2-selected.index(i)+(.34-qidx*.17)
                    bar(detail_ax,y,a,b,COLORS[q],.139)
                    rightmin=min(rightmin,a);rightmax=max(rightmax,b)
        if j==0:
            all_ax.text(.015,4,"Per-tile read issue unavailable",transform=all_ax.get_yaxis_transform(),
                        va="center",fontsize=8.5,color=DIM,style="italic")
        all_ax.set_title(heading+"  ·  tile cadence",loc="left",color=INK,weight="bold",fontsize=14,pad=12)
        detail_ax.set_title("First, middle and tail tile",loc="left",color=INK,weight="bold",fontsize=12,pad=12)
        all_ax.set(yticks=range(5),yticklabels=[NAMES[q] for q in ORDER[::-1]],ylim=(-.48,4.48))
        detail_ax.set(yticks=[2,1,0],yticklabels=[f"#{i+1}  ·  {ts[i]['rows']} rows" for i in selected],ylim=(-.58,2.58))
        detail_ax.axvline(0,color=DIM,linestyle="--",lw=.8)
        for y in [.5,1.5]:
            detail_ax.axhline(y,color="#e6ebee",lw=.7)
        for ax in (all_ax,detail_ax):style(ax)
    for ax in axes[0]:
        ax.set_xlim(-3.5,1.025*leftmax)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=9,steps=[1,2,5,10]))
    for ax in axes[1]:ax.set_xlim(rightmin-.28,rightmax+.23)
    axes[0][1].set_xlabel("Time from the first tile’s rstd issue (µs)")
    axes[1][1].set_xlabel("Time from this tile’s rstd issue (µs)")
    fig.text(.145,.085,"Upper and lower tile panels use the same origins. The model specifies relative store cadence. "
             "Absolute first rstd is observed only in CA.",color=DIM,fontsize=10)
    fig.text(.145,.060,"CA bands show DMA and vector instruction windows. White ticks mark x reads. "
             "Shared I/O adds time only above the local budget.",color=DIM,fontsize=10)
    fig.savefig(ROOT/f"docs/examples/figures/rms-wide-tiles-{side}.png",dpi=150)
    plt.close(fig)


if __name__=="__main__":
    for side,row in json.loads(DATA.read_text()).items():
        draw(side,row)
