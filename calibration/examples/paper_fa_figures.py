"""Redraw the FA paper figures from versioned data; optionally audit the original DB."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from replay_fa import issue_slope

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/figures"
COLORS = {"K": "#127c80", "V": "#b25b2e", "QK": "#7452a1", "PV": "#167d83"}


def raw_check(path, evidence):
    if hashlib.sha256(path.read_bytes()).hexdigest() != evidence["steady_db_sha256"]:
        raise ValueError("Wrong source database")
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        zero, = db.execute("SELECT MIN(ExecInstrTickStart) FROM ExecutedInstructions").fetchone()
        records = []
        for start, end, details in db.execute(
            """SELECT ExecInstrTickStart, ExecInstrTickEnd, ExtendParams FROM ExecutedInstructions
               WHERE CoreTypeId=1 AND ExecInstrCoreId=0 AND ExecInstrIsDurationExcluded=0
                 AND ExecInstrName='MOV_OUT_TO_L1_MULTI_ND2NZ'"""
        ):
            arg = json.loads(details)
            if arg["processed_bytes"] == 32768 and arg["src_mem"] == "OUT" and arg["dst_mem"] == "L1":
                records.append((arg["src_addr"],start,end))
        records.sort()
        for stream, group in (("k",records[:128]), ("v",records[128:])):
            ticks = [s-zero for _,s,_ in group]
            if ticks != evidence["page_" + stream + "_issue_ticks"]:
                raise ValueError("Original ticks differ from release")
            print(stream, "zero",zero,"selected (page, absolute issue, retirement)",
                  [(i,*group[i][1:]) for i in [0,8,19,27,119]])


def fit_plot(evidence):
    fig, axes = plt.subplots(2, 1, figsize=(8.6,4.8), sharex=True,
                             gridspec_kw={"height_ratios":[2,1]},layout="constrained")
    xs = np.arange(128)
    for tag in ("K","V"):
        ticks = np.array(evidence[f"page_{tag.lower()}_issue_ticks"])
        slope = issue_slope(ticks)
        pred = np.mean(ticks[8:120]) + (xs-63.5)*slope
        color = COLORS[tag]
        axes[0].plot(xs,ticks/1000,".",color=color,ms=2.8)
        axes[0].plot(xs[8:120],pred[8:120]/1000,color=color,lw=.85,
                     label=f"{tag}  {slope:.2f} cycles/page")
        axes[1].plot(xs,ticks-pred,color=color,lw=.8)
    for ax in axes:
        ax.axvspan(-.5,7.5,color="#d4dce2",alpha=.55)
        ax.axvspan(119.5,127.5,color="#d4dce2",alpha=.55)
        ax.grid(alpha=.16)
    axes[0].legend(loc="upper left",fontsize=9)
    axes[0].set_ylabel("Issue (thousand cycles)")
    axes[1].axhline(0,color="black",lw=.6)
    axes[1].set_ylabel("Residual (cycles)")
    axes[1].set_xlabel("Page index (zero based)")
    fig.savefig(OUT/"paper-fa-issue-fit.png",dpi=175)
    plt.close(fig)


def timing_plot(asset):
    fig, axes = plt.subplots(2,1,figsize=(8.6,4.4),sharex=True,layout="constrained")
    for ax,source,label in zip(axes,("model","ca"),("Resource schedule","CA instruction intervals")):
        for j,op in enumerate(("QK","PV")):
            for page in (0,1):
                s = asset[source][op][page]
                intervals = [s] if source=="model" else s
                y = 1-page + (.14 if j==0 else -.14)
                for begin,end in intervals:
                    ax.broken_barh([(begin,end-begin)],(y-.085,.17),facecolors=COLORS[op])
        ax.set_yticks([1,0],["Page 0","Page 1"])
        ax.set_ylim(-.4,1.5)
        ax.set_title(label,loc="left",fontsize=11)
        ax.grid(axis="x",alpha=.18)
    axes[-1].set_xlabel("Device time since earliest instruction on any participating core (µs)")
    axes[0].legend(handles=[plt.Line2D([0],[0],color=c,lw=6,label=k)
                            for k,c in COLORS.items() if k in ("QK","PV")],loc="upper right",ncols=2)
    fig.savefig(OUT/"paper-fa-two-pages.png",dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db",type=Path)
    args = ap.parse_args()
    data = json.loads((ROOT/"calibration/examples/fa-950dt.json").read_text())
    if args.db:
        raw_check(args.db,data)
    fit_plot(data)
    timing_plot(json.loads((ROOT/"docs/examples/assets/fa.json").read_text()))
