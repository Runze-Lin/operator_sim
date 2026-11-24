"""Draw the 950DT one-team diagnostic using only the retained native receipt."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def draw(receipt, output):
    data = json.loads(receipt.read_text())
    issue = data["mmad_issue_us"]
    clock = 1650  # cycles per microsecond for this measured 950DT profile
    gaps = [(b-a)*clock for a,b in zip(issue,issue[1:])]
    fig, ax = plt.subplots(figsize=(9,3.8),layout="constrained")
    ax.bar(range(1,len(gaps)+1), gaps, color="#267282", width=.72, label="CA · time between MMAD_MX issues")
    ax.axhline(data["frozen_predicted_ii_cycles"],color="#a74e2d",lw=1.8,ls="--",label="Original resource estimate")
    ax.axhline(data["mean_issue_interval_cycles"],color="#6a4d94",lw=1.8,label="Full-row mean · model parameter")
    ax.set(xlim=(.25,31.75),ylim=(0,max(gaps)*1.25),xlabel="Interval between adjacent 512-key tiles",
           ylabel="Cycles",title="950DT QSLI  ·  one team scores 16,384 candidate keys")
    ax.legend(loc="upper right",frameon=False,fontsize=9)
    ax.spines[["right","top"]].set_visible(False)
    ax.grid(axis="y",alpha=.15)
    fig.savefig(output,dpi=150)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    draw(a.receipt,a.output)
