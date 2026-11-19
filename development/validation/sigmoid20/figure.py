"""Render a frozen analytical pipeline and its all-core-origin, terminal-core CA trace."""

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/operator-sim-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from development.validation.sigmoid20.campaign import DEST, DT, ROOT, read, write
from campaign_collect import events
from operator_model import estimate


def merge(spans):
    out = []
    for begin, end in sorted(spans):
        if out and begin <= out[-1][1] + 1e-8:
            out[-1][1] = max(end, out[-1][1])
        else:
            out.append([begin, end])
    return out


def data(row):
    k = row["kernel"]
    core = max(k["windows"], key=lambda c: k["windows"][c][1])
    ca = k["ca_us"]
    shard = ROOT / Path(k["trace"]).parent / core / "trace.json"
    lo, hi = float("inf"), -float("inf")
    spans = {"Read": [], "Vector": [], "Write": []}
    descriptors = {"Read": [], "Write": []}
    loads = []
    stores = []
    for e in events(shard):
        if e.get("ph") != "X":
            continue
        start = e["ts"]
        end = start + e["dur"]
        lo = min(lo, start)
        hi = max(hi, end)
        info = json.loads(e["args"]["detail"])
        q = info["queue_type"]
        ins = e["name"]
        if e["dur"] <= 0 or any(
            info.get(v) for v in ("is_wait", "is_sync", "is_duration_excluded")
        ):
            continue
        if q in ("MTE2", "MTE3") and ins.startswith("MOV_SRC_TO_DST"):
            pipe = "Read" if q == "MTE2" else "Write"
            spans[pipe].append([start, end])
            descriptors[pipe].append((start, end, info["processed_bytes"]))
        if ins == "RV_VLD":
            loads.append(start)
        if ins == "RV_VST":
            stores.append(end)
        if (
            q == "SIMD"
            and info["instr_type"] in ("RVECEX", "RVECST", "RVECLD")
            and ins.startswith(("RV_V", "RV_LD", "RV_ST"))
        ):
            spans["Vector"].append([start, end])
    start, stop = k["windows"][core]
    assert abs(hi - lo - (stop - start)) < 0.03
    origin = lo - start
    reads, outputs = sorted(descriptors["Read"]), sorted(descriptors["Write"])
    # The metadata descriptor precedes the tensor reads; distinguish it even
    # when actual ping-pong tensor descriptors overlap and their union merges.
    assert reads[0][2] == 64 and len(reads) == len(outputs) + 1
    spans["Metadata"] = [reads[0][:2]]
    spans["Read"] = [v[:2] for v in reads[1:]]
    lanes = {
        q: [[a - origin, b - origin] for a, b in merge(v)] for q, v in spans.items()
    }
    loads.sort()
    stores.sort()
    off = 0
    ca_tiles = []
    for (a, b, size), (c, d, wsize) in zip(reads[1:], outputs):
        assert size == wsize
        groups = (size + 255) // 256
        ca_tiles.append(
            dict(
                bytes=size,
                vector_groups=groups,
                Read=[a - origin, b - origin],
                Vector=[loads[off] - origin, stores[off + groups - 1] - origin],
                Write=[c - origin, d - origin],
            )
        )
        off += groups
    assert off == len(loads) == len(stores)
    r = estimate("sigmoid_fp32", row["shape"], dtype="fp32", chip=row["sku"])
    d = r.projection["timing_diagnostics"]
    p = d["target_plan"]
    s = d["schedule_us"]
    tiles = p["critical_core_tiles"]
    model = {q: [] for q in ("Read", "Vector", "Write")}
    for i in range(tiles):
        if i == tiles - 1:
            chunks = [
                (s["last_read_at"], s["last_read"]),
                (s["last_vector_at"], s["last_vector"]),
                (s["last_write_at"], s["last_write"]),
            ]
        else:
            a = s["first_read_at"] + s["interval"] * i
            b = a + s["first_read"]
            z = b + s["first_vector"] + s["write_dependency"]
            chunks = [
                (a, s["first_read"]),
                (b, s["first_vector"]),
                (z, s["first_write"]),
            ]
        for q, (a, w) in zip(model, chunks):
            model[q].append([a, a + w])
    model["Front"] = [[0, s["first_read_at"]]]
    return dict(
        shape=row["shape"],
        chip=row["sku"],
        terminal_core=core,
        origin="earliest instruction on any core",
        ca_us=ca,
        terminal_start_us=start,
        ca_windows=lanes,
        ca_tiles=ca_tiles,
        trace_sha256=k["trace_sha256"],
        model_us=r.latency_us,
        model_windows=model,
        plan=p,
        event_filter="positive-duration non-wait/non-sync/non-excluded payload MTE2/MTE3 and RVEC arithmetic/UB load/store; times relative to all-core origin",
    )


def render(d, slug):
    colors = dict(
        Front="#d9e4e8",
        Metadata="#95abb4",
        Read="#0d7c86",
        Vector="#7054a3",
        Write="#b64b2a",
    )
    lanes = ["Read", "Vector", "Write"]
    pos = {q: i for i, q in enumerate(lanes[::-1])}
    fig, axs = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
    fig.subplots_adjust(left=0.16, right=0.95, top=0.77, bottom=0.19, hspace=0.65)
    rows, heads = d["shape"]
    chip = d["chip"].split("_")[0].replace("Ascend", "")
    fig.text(
        0.16,
        0.95,
        f"{chip} FP32 Sigmoid  |  {rows:,} tokens × {heads} local heads",
        size=19,
        weight="bold",
        color="#173044",
    )
    fig.text(
        0.16,
        0.898,
        f"{d['plan']['active_aiv']} modeled AIVs   ·   critical core {d['plan']['critical_core_elements']:,} elements   ·   {d['plan']['critical_core_tiles']} modeled UB tile(s)",
        size=11,
        color="#385869",
    )
    labels = {
        "Front": "Launch & metadata",
        "Metadata": "CA metadata read",
        "Read": "MTE2 read",
        "Vector": "Vector",
        "Write": "MTE3 write",
    }
    fig.legend(
        [Patch(facecolor=col) for col in colors.values()],
        [labels[q] for q in colors],
        loc="center left",
        bbox_to_anchor=(0.153, 0.841),
        ncol=5,
        frameon=False,
        columnspacing=1,
    )
    for ax, title, windows, total in zip(
        axs,
        [
            "Analytical schedule · critical block",
            f"CA trace · {d['terminal_core']} (last to finish)",
        ],
        [d["model_windows"], d["ca_windows"]],
        [d["model_us"], d["ca_us"]],
    ):
        ax.set_title(
            title, loc="left", fontsize=13, fontweight="bold", color="#173044", pad=10
        )
        ax.set_title(
            f"{total:.3f} µs",
            loc="right",
            fontsize=13,
            fontweight="bold",
            color="#173044",
            pad=10,
        )
        for q, segments in windows.items():
            y = pos.get(q, 2)
            ax.broken_barh(
                [(a, b - a) for a, b in segments],
                (y - 0.23, 0.46),
                facecolors=colors[q],
                alpha=0.96,
            )
        ax.axvline(total, color="#4a6470", linestyle=(0, (3, 3)), linewidth=1.1)
        ax.set_yticks(list(pos.values()), ["MTE3 write", "SIMD", "MTE2 read"])
        ax.set_ylim(-0.64, 2.65)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#d2dfe4", lw=0.75)
    axs[-1].set_xlim(0, max(d["ca_us"], d["model_us"]) * 1.035)
    axs[-1].set_xlabel("Device time from earliest core instruction (µs)")
    axs[-1].xaxis.set_major_locator(MaxNLocator(nbins=11))
    fig.text(
        0.16,
        0.070,
        "Model bars reconstruct service windows from resources and dependencies. CA bars show recorded descriptors and RVEC interval unions.",
        fontsize=9,
        color="#385869",
    )
    fig.text(
        0.16,
        0.035,
        "Blank = no selected event on that lane. Dashed line = all-core end, including control and synchronization.",
        fontsize=9,
        color="#385869",
    )
    if (
        1 < len(d["ca_tiles"]) <= 6
        and len(d["ca_tiles"]) == d["plan"]["critical_core_tiles"]
    ):
        count = len(d["ca_tiles"])
        for ax, actual in zip(axs, [False, True]):
            for artist in list(ax.collections):
                artist.remove()
            for i, tile in enumerate(d["ca_tiles"]):
                y = count - 1 - i
                for q, offset in [("Read", 0.18), ("Vector", 0.0), ("Write", -0.18)]:
                    a, b = tile[q] if actual else d["model_windows"][q][i]
                    ax.broken_barh(
                        [(a, b - a)], (y + offset - 0.085, 0.17), facecolors=colors[q]
                    )
            if actual:
                for a, b in d["ca_windows"]["Metadata"]:
                    ax.broken_barh(
                        [(a, b - a)], (count - 0.9, 0.17), facecolors=colors["Metadata"]
                    )
            else:
                a, b = d["model_windows"]["Front"][0]
                ax.broken_barh(
                    [(a, b - a)], (count - 0.9, 0.17), facecolors=colors["Front"]
                )
            ax.set_yticks(
                list(range(count))[::-1], [f"UB tile {n + 1}" for n in range(count)]
            )
            ax.set_ylim(-0.48, count - 0.43)
        fig.texts[-2].set_text(
            "Per tile: read / SIMD / write. CA SIMD spans first RV_VLD to last RV_VST. DMA descriptors can overlap in time."
        )
    fig.savefig(
        ROOT / "docs/examples/figures" / f"sigmoid-{slug}.png",
        dpi=160,
        facecolor="white",
    )
    plt.close(fig)


def main(stage, rows, heads, slug, chip=DT):
    row = next(
        x
        for x in read(DEST / (stage + ".json"))
        if x["qualified"] and x["sku"] == chip and x["shape"] == [rows, heads]
    )
    d = data(row)
    write(DEST / f"timeline-{slug}.json", d)
    render(d, slug)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], *(sys.argv[5:]))
