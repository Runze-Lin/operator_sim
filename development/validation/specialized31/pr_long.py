"""Compare the frozen B-resident supply schedule to one qualified long PR critical shard."""
import json
from pathlib import Path
from statistics import mean, median
import sys
from collections import defaultdict
from hashlib import sha256

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "development/validation/sku17"))
from campaign_collect import events

PR = "Ascend950PR_957b"

OUT = ROOT / "results/v0.31/matmul/long-pr"
SHAPE = [1048576, 128, 512]


def critical_core(receipt):
    stage, = receipt["stages"]
    core = max(stage["windows"], key=lambda c: stage["windows"][c][1])
    path = Path(stage["trace"]).parent / core / "trace.json"
    ticks = 1650
    origin = receipt["terminal"][core]["first_tick"] - round(stage["windows"][core][0]*ticks)
    queues = defaultdict(list)
    for e in events(ROOT / path):
        name = e["name"]
        if e.get("ph") != "X" or name not in {"MMAD", "FIX_L0C_TO_DST", "MOV_OUT_TO_L1_MULTI_ND2NZ"}:
            continue
        d = json.loads(e["args"]["detail"])
        if any(d.get(k) for k in ("is_wait", "is_sync", "is_duration_excluded")):
            continue
        if name == "MMAD":
            q, rows = "mmads", d["m"]
            assert (d["n"], d["k"]) == (128, 32)
        elif name == "FIX_L0C_TO_DST":
            q, rows = "write", d["m"]
            assert d["n"] == 128 and d["pre_quant_mode"].startswith("F322BF16")
        else:
            rows = d["n"]
            assert d["d"] == 128
            q = "b" if rows == SHAPE[2] else "a_pages"
        queues[q].append((round(e["ts"]*ticks)-origin, round((e["ts"]+e["dur"])*ticks)-origin,
                          rows, d.get("src_addr")))
    for s in queues.values():
        s.sort()
    n = len(queues["write"])
    assert len(queues["b"]) == 1 and len(queues["a_pages"]) == 4*n and len(queues["mmads"]) == 16*n
    tiles = []
    for i, w in enumerate(queues["write"]):
        pages = queues["a_pages"][4*i:4*i+4]
        cubes = queues["mmads"][16*i:16*i+16]
        assert all(e[2] == w[2] for e in pages + cubes)
        assert [e[3]-pages[0][3] for e in pages] == [0,256,512,768]
        tiles.append(dict(rows=w[2], write=w[:2], mmads=[e[:2] for e in cubes], a_pages=[e[:2] for e in pages]))
    with (ROOT / path).open("rb") as stream:
        digest = sha256()
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    return core, str(path), digest.hexdigest(), ticks, tiles


def main():
    ledger = ROOT / "results/v0.28/matmul/long-prefill/131072-pr.json"
    receipt = json.loads(ledger.read_bytes())
    assert receipt["qualified"] and receipt["shape"] == SHAPE and receipt["chip"] == PR
    core, shard, fingerprint, ticks, tiles = critical_core(receipt)
    reg = json.loads((ROOT / "results/v0.30/matmul/freeze.json").read_bytes())
    row, = [v for v in reg["rows"] if v["shape"] == SHAPE and v["chip"] == PR]
    p = row["candidate"]
    observed = [t["mmads"][0][0] for t in tiles]
    gaps = [b - a for a, b in zip(observed, observed[1:])]
    plan = [s["cycles_per_wave"] for s in p["segments"]
            for _ in range(s["first_wave"], s["exclusive_end_wave"])]
    assert len(plan) == p["waves"] and len(tiles) <= len(plan)
    ca = receipt["ca_us"]
    dt = json.loads((ledger.parent / "131072-dt.json").read_bytes())["ca_us"]
    spans = []
    for quartile in range(4):
        lo, end = quartile*len(tiles)//4, (quartile+1)*len(tiles)//4
        hi = min(end, len(gaps))
        spans.append(dict(first=lo + 1, last=hi, observed=mean(gaps[lo:hi]),
                          model=mean(plan[lo:hi]), a_response_median=median(b-a for t in tiles[lo:end] for a,b in t["a_pages"])))
    data = dict(shape=SHAPE, chip=PR, ca_us=ca, dt_ca_us=dt,
                predictions_us=dict(public=row["older_public_us"], resident=row["v029_us"],
                                    supply=p["latency_us"], **row["ablation_us"]),
                actual_pr_over_dt=ca/dt, global_cyc_per_us=ticks, acq_minutes=receipt["cost"]["seconds"]/60,
                elapsed_raw_gib=receipt["cost"]["peak_bytes"]/2**30, numerical=receipt["correctness"],
                active=len(receipt["terminal"]), core=core, core_tiles=len(tiles), last_rows=tiles[-1]["rows"],
                all_core_mmad=receipt["stages"][0]["mmad"], frozen_mmad=p["mmad_total"],
                core_mmad=16*len(tiles), kernel=receipt["stages"][0]["kernel"],
                trace=shard, trace_sha256=fingerprint, receipt_sha256=sha256(ledger.read_bytes()).hexdigest(),
                first_mmad_us=observed[0]/ticks, predicted_first_us=p["first_cube_cycles"]/ticks,
                last_mmad_us=tiles[-1]["mmads"][-1][0]/ticks, segments=p["segments"],
                observed_first_mmad_cycles=observed, quartiles=spans)
    late_gap = sum((q["model"]-q["observed"])*(q["last"]+1-q["first"]) for q in spans[2:])/ticks
    data["last_two_quartiles_excess_us"] = late_gap
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data.json").write_text(json.dumps(data, indent=2) + "\n")

    fig, (top, lower, interval) = plt.subplots(3, 1, figsize=(12, 8.2), gridspec_kw={"height_ratios":[.8,1.1,1.15]})
    lower.sharex(top)
    top.tick_params(axis="x", labelbottom=False)
    fig.subplots_adjust(left=.14, right=.97, top=.83, bottom=.09, hspace=.72)
    fig.text(.065, .948, "950PR MatMulV3  |  1,048,576 × 128 × 512", size=20, weight="bold")
    fig.text(.065, .89, f"Flash indexer.wk   ·   B8 × 128 Ki   ·   {data['active']} AICs   ·   {len(tiles)} output tiles on the last core")
    first = p["first_cube_cycles"]/ticks
    top.barh(0, first, left=0, color="#c5d2d6", label="First operand")
    for s in p["segments"]:
        n = s["exclusive_end_wave"]-s["first_wave"]
        width = n*s["cycles_per_wave"]/ticks
        color = "#337f8c" if s["mixed_cycles"] == 0 else "#695699"
        top.barh(0, width, left=first, color=color, height=.45)
        first += width
    top.set_title(f"Frozen capacity schedule   ·   {p['latency_us']:.3f} µs", loc="left", weight="bold")
    top.set_yticks([0], ["Output waves"])
    top.text(.012, .88, "Teal: reads and resident output     Purple: read + write supply", transform=top.transAxes, size=9)
    top.set_ylim(-.58, .75)
    top.axvline(p["latency_us"], ls="--", lw=1, color="#213840")
    lower.set_title(f"CA trace   ·   {core}   ·   {ca:.3f} µs", loc="left", weight="bold")
    for i, (key, label, color) in enumerate([("a_pages", "MTE2 / A", "#267c87"),
                                              ("mmads", "Cube / MMAD", "#695699"),
                                              ("write", "FIXP / C", "#b85c32")]):
        windows = [span for t in tiles for span in ([t[key]] if key=="write" else t[key])]
        lower.broken_barh([(lo/ticks, (hi-lo)/ticks) for lo,hi in windows], (i-.18,.36), facecolor=color)
    lower.set_yticks([0,1,2], ["MTE2 / A", "Cube / MMAD", "FIXP / C"])
    lower.set_ylim(2.55,-.55)
    lower.axvline(ca, ls="--", lw=1, color="#213840")
    lower.set_xlabel("Device time since the earliest participating instruction (µs)")
    interval.plot(range(1,len(tiles)), gaps, color="#546f7c", linewidth=1, label="CA issue interval")
    interval.step(range(1,len(tiles)), plan[:len(gaps)], where="mid", color="#9c4825", linewidth=1.7,
                  label="Frozen supply interval")
    interval.set_title("Cycles between the first MMADs of adjacent output tiles", loc="left", weight="bold")
    interval.set_xlim(1,len(tiles)-1)
    interval.set_xlabel("Output tile on the critical AIC")
    interval.set_ylabel("Device cycles")
    interval.legend(loc="upper left", ncol=2, frameon=False, fontsize=9)
    for ax in (top,lower,interval):
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#dce2e5")
        ax.spines[["top","left","right"]].set_visible(False)
    fig.savefig(OUT / "pipeline.png", dpi=150)
    plt.close(fig)
    err = lambda t: 100*(t/ca-1)
    lines = ["# Flash 整批 B8 × 128Ki 的 950PR Matmul", "",
             f"TP8 的 indexer.wk 将 1,048,576 行、宽 512 的 BF16 激活投影到 128 通道。"
             f"输入 A 为 1 GiB，权重 B 为 128 KiB，输出 C 为 256 MiB；计算量为 137.439 GFLOP。"
             f"实调 aclnnMatmul，安装包 kernel 为 `{data['kernel']}`。全部 134,217,728 个结果通过黄金值核验。", "",
             "| 排程 | 950PR 预测 µs | 相对全窗偏差 |","|---|---:|---:|"]
    for name, value in [("公开 estimate",row["older_public_us"]),("B 常驻 L1",row["v029_us"]),
                        ("B 常驻并计算容量和读写供给",p["latency_us"]),
                        ("冻结敏感性：读写混合效率取 1",row["ablation_us"]["unity_mixed_efficiency"])]:
        lines.append(f"| {name} | {value:.3f} | {err(value):+.2f}% |")
    lines += ["", f"950DT / 950PR 设备全窗为 **{dt:.3f} / {ca:.3f} µs**，实测 PR/DT 为 {ca/dt:.3f}。"
              f"PR 采集墙钟 {data['acq_minutes']:.2f} 分钟，原始输出峰值 {data['elapsed_raw_gib']:.2f} GiB。"
              f"按物理 END 和异步指令最终退休核对全部 {data['active']} 个 AIC。", "",
              "![同轴资源排程和 CA 末核指令](pipeline.png)", "",
              f"末核 `{core}` 执行 {len(tiles)} 个输出块。每块读四页 A，发起 16 条 MMAD，写一次 BF16 C。"
              f"满块为 256×128×32，最后一个块只有 {tiles[-1]['rows']} 行。全片 CA / 容量候选的 MMAD 条数为 "
              f"{data['all_core_mmad']:,} / {data['frozen_mmad']:,}。"
              f"首条 MMAD 的预测 / 实测时刻为 {data['predicted_first_us']:.3f} / {data['first_mmad_us']:.3f} µs。"
              f"相邻输出块首条 MMAD 的实测中位间隔为 {median(gaps):.0f} 周期。", "",
              "| 末核块序号 | 预测平均发起间隔，周期 | CA 平均发起间隔，周期 | A 页响应中位数，周期 |",
              "|---|---:|---:|---:|"]
    lines.extend(f"| {q['first']}–{q['last']} | {q['model']:.0f} | {q['observed']:.0f} | {q['a_response_median']:.0f} |" for q in spans)
    lines += ["", "块序号从 1 开始；间隔是该块首条 MMAD 到下一块首条 MMAD 的间距，最后一块只用于测结束。"
              f"后两个区间累计多计 {late_gap:.3f} µs，占容量候选整窗高估的 {100*late_gap/(p['latency_us']-ca):.1f}%。"
              "模型用 L2 脏输出容量估计并发的 A 读与 C 写，再用共享带宽和混合读写效率定价。"
              "列中的 A 页响应包含排队及依赖，和跨块发起间隔可以重叠。"
              "末核数据可定位时间差发生在哪一段；区分 HBM 总线效率、L2 命中和写回仍需内存控制器计数。", "",
              "[数据、原始 shard 与指纹](data.json) · [冻结条件与敏感性](../../../v0.30/matmul/freeze.json) · "
              "[全核与数值回执](../../../v0.28/matmul/long-prefill/131072-pr.json)", ""]
    (OUT / "report.md").write_text("\n".join(lines))
    print(ca, spans)


if __name__ == "__main__":
    main()
