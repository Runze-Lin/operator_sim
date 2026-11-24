"""Illustrate the one-row split schedule using saved prices and device landmarks."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent


def predictions(filename):
    prices = json.loads((HERE / filename).read_text())["prices"]
    return {r["sku"]: r["latency_us"] for r in prices if r["case"] == "split_row"}


def draw(output):
    records = json.loads((HERE / "long-row-results.json").read_text())["records"]
    old = predictions("v112_baseline.json")
    current = predictions("v112_frozen.json")
    diagnostics = json.loads((HERE / "v112_merge.json").read_text())["records"]
    colors = ["#7da7ad", "#665295", "#e4ba87", "#be713a"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
    for ax, row in zip(axes, records):
        sku, origin = row["chip"], row["first_tick"]
        clock = row["cycles"] / row["ca_us"]
        stops = [(row[key]-origin)/clock for key in ("first_tick", "first_mmad_tick",
                                                      "last_mmad_finish_tick", "last_retire_tick")]
        merged = diagnostics[sku + "-ipc"]["passes"]
        stops.insert(-1, (merged[0]["start"] - origin)/clock)
        for left, right, color in zip(stops, stops[1:], colors):
            ax.barh(0, right-left, left=left, height=.4, color=color)
            ax.text((left+right)/2, -.34, f"{right-left:.2f}", ha="center", fontsize=9)
        for price, style, color in [(old[sku], "--", "#354457"), (current[sku], "-", "#943631")]:
            ax.axvline(price, ymin=.30, ymax=.86, color=color, ls=style, lw=1.6)
        ax.text(stops[-1]+.3, -.04, f"{stops[-1]:.2f} µs", va="center", fontsize=9)
        ax.set_title(f"{sku.split('_')[0].replace('Ascend','')}   ·   "
                     f"{row['metadata_final_merge_parts']} partial lists", loc="left", fontsize=11)
        for step in merged[1:]:
            ax.vlines((step["start"]-origin)/clock, -.2, .2, color="white", lw=1)
        ax.set_ylim(-.62,.68)
        ax.set_yticks([])
        ax.spines[["top", "left", "right"]].set_visible(False)
    axes[-1].set_xlim(0,max(max(current.values()),max(r["ca_us"] for r in records))+3)
    axes[-1].set_xlabel("Time from first device instruction (µs)")
    handles = [Patch(color=c, label=l) for c, l in zip(colors,
               ["Before first MMAD", "First issue to last MMAD retire", "Local completion and handoff", "Final merge and retirement"])]
    handles += [Line2D([0], [0], color="#354457", ls="--", label="v1.11 total"),
                Line2D([0], [0], color="#943631", label="Frozen v1.12 total")]
    fig.suptitle("Flash candidate Indexer at 1Mi  ·  one query split across cores", x=.08, ha="left", fontsize=14)
    fig.legend(handles=handles, loc="lower center", frameon=False, ncol=3, fontsize=9)
    fig.subplots_adjust(left=.08, right=.96, top=.80, bottom=.25, hspace=.56)
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    draw(parser.parse_args().output)
