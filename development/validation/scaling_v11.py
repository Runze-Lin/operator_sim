"""Summarize existing full CA measurements, without invoking the simulator."""
import argparse
import csv
import json
import os
import re
import sqlite3
from collections import defaultdict
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/v0.11"
MIB = 2 ** 20
OPS = {"rmsnorm": ("rms", "RMSNorm"), "mhc_post": ("post", "mHC Post")}
# Fixed by the collection plan before timing the new cases.
WIDTH = {"rmsnorm": 4096, "mhc_post": 1024}


def read(path):
    return json.loads(path.read_text())


def extent(intervals):
    """The sum of non-overlapping address intervals and span of their bounding range."""
    if not intervals:
        return 0, 0
    intervals.sort()
    start, end = intervals[0]
    total=0
    for a,b in intervals[1:]:
        if a>end:
            total+=end-start
            start,end=a,b
        else:
            end=max(end,b)
    return total+end-start, end-intervals[0][0]


def audit_db(op, shape, db_path):
    helper=(db_path.parent/"cannsim_helper.log").read_text()
    launch=int(db_path.name.split("_",1)[0])
    flush,=re.findall(fr"start launch: id={launch} .*?instruction handler flush: ([^\n]*)",helper,re.S)
    metrics=dict(re.findall(r"(\w+)=(\w+)",flush))
    t,d=shape[0],shape[-1]
    rms=op=="rmsnorm"
    main=t*d*2*(1 if rms else shape[1])
    aux=4*t if rms else 0
    sql="""SELECT ExecInstrCoreId,CoreTypeId,ExecInstrTickStart,ExecInstrTickEnd,ExtendParams
           FROM ExecutedInstructions WHERE ExecInstrName='MOV_SRC_TO_DST_ALIGNv2'
           AND json_extract(ExtendParams,'$.dst_mem')='OUT' ORDER BY ExecInstrTickStart"""
    core_work="""SELECT COUNT(*),SUM(ExecInstrName GLOB 'RV_*'),
                        SUM(ExecInstrTickEnd < ExecInstrTickStart)
                 FROM ExecutedInstructions GROUP BY ExecInstrCoreId,CoreTypeId"""
    ranges=[[],[]]
    copies=[]
    raw=[0,0]
    with sqlite3.connect(db_path.resolve().as_uri()+"?mode=ro",uri=True) as db:
        core=list(db.execute(core_work))
        for c,u,a,b,params in db.execute(sql):
            p=json.loads(params)
            k=int(rms and p["burst_len"]<d*2)
            addr=p["dst_addr"]
            for i in range(p["burst_number"]):
                start=addr+i*p["dst_stride"]
                ranges[k].append((start,start+p["burst_len"]))
            raw[k]+=p["processed_bytes"]
            copies.append((c,u,a,b,addr,p["processed_bytes"]))
    size,span=extent(ranges[0])
    aux_size,aux_span=extent(ranges[1])
    main_ok=main==raw[0]==size==span
    aux_ok=aux==raw[1]==aux_size==aux_span
    n=sum(w[0] for w in core)
    writes_ok=(metrics["status"]=="ok" and all(int(metrics[n])==0 for n in ("invalid","mismatched","write_failed"))
               and int(metrics["written"])==n and n>0 and not sum(w[2] for w in core))
    example=[(a,b,hex(addr),nbytes) for c,u,a,b,addr,nbytes in copies if (c,u)==(0,2)]
    return dict(op=op, shape=str(shape), db=str(db_path.relative_to(ROOT)), launch=launch,
                flush=metrics["status"], **{k:int(metrics[k]) for k in ("invalid","mismatched","write_failed","written",
                                                                  "start_overwrites","end_overwrites","start_only","end_only")},
                db_rows=n, aiv=len(core), reversed_intervals=sum(w[2] for w in core),
                per_core_rv_min=min((w[1] for w in core),default=None), per_core_rv_max=max((w[1] for w in core),default=None),
                n_out_copies=len(copies), n_out_addresses=len({c[4] for c in copies}),
                main_expected=main, main_raw=raw[0], main_unique=size, main_span=span,
                aux_expected=aux, aux_raw=raw[1], aux_unique=aux_size, aux_span=aux_span,
                core0_aiv0_out_tick_start_end_addr_bytes=example, passed=writes_ok and main_ok and aux_ok)


def audits():
    records={}
    for file in sorted((BASE/"rms").glob("rmsnorm-*.json")):
        for r in read(file):
            records[r["source"]]=("rmsnorm",r["shape"],"original")
    original=read(BASE/"post"/"original-attempt.json")
    for r in original["completed"]+original["timing_failed"]:
        records[r["source"]]=("mhc_post",r["shape"],"original")
    for file in sorted((BASE/"post").glob("retry-v11_mhc_post_*.json")):
        data=read(file)
        for i,r in enumerate(data["completed"]):
            records[r["source"]]=("mhc_post",r["shape"],"recovery" if i else "conditioning_control")
    return [dict(role=role,**audit_db(op,shape,ROOT/path)) for path,(op,shape,role) in records.items()]


def save_table(path, rows):
    with path.open("w", newline="") as table:
        writer=csv.DictWriter(table, lineterminator="\n", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def record(row, role):
    op, shape = row["op"], row["shape"]
    t, d = shape[0], shape[-1]
    if op == "rmsnorm":
        n = 1
        inputs, main, extra = 2*t*d + 2*d, 2*t*d, 4*t
    else:
        n = shape[1]
        inputs, main, extra = 2*t*n*d + 2*t*d + 4*t*n*(n+1), 2*t*n*d, 0
    active = row["active_aiv"]
    before = row.get("prior_aiv")
    state = ("unknown" if before is None else "first" if before == 0 else
             "expand" if before < active else "reuse")
    return dict(op=op, role=role, id=row["id"], shape=str(shape),
                shape_group="few_wide" if t <= 4 and d >= 2048 else "controls",
                tokens=t, streams=n, width=d, kernel=row["kernel"],
                active_aiv=active, prior_aiv=before, state=state,
                rows_per_busiest_aiv=ceil(t / 64),
                input_bytes=inputs, main_output_bytes=main,
                aux_output_bytes=extra, logical_io_bytes=inputs+main+extra,
                exposed_mte3_us=(row.get("phase_us") or {}).get("MTE3"),
                exposed_simd_us=(row.get("phase_us") or {}).get("SIMD"),
                critical_simd_events=(row.get("counts") or {}).get("SIMD"),
                ca_us=row["reference_us"], source=row.get("db_source") or row.get("source"),
                batch=row.get("batch"), launch=row.get("launch"))


def series(rows):
    # Screen on shape and measured operating conditions only (never on latency).
    groups = defaultdict(list)
    for r in rows:
        if (r["role"] == "v0.11_independent" and r["width"] == WIDTH[r["op"]]
                and r["tokens"] >= 64 and r["state"] != "unknown"):
            key = (r["streams"], r["kernel"], r["active_aiv"], r["prior_aiv"])
            groups[key].append(r)
    if not groups:
        return []
    largest = max(groups.values(), key=lambda group: (len(group), max(r["main_output_bytes"] for r in group)))
    return sorted(largest, key=lambda r: r["tokens"]) if len(largest) >= 2 else []


def fit(points):
    if len(points) < 2:
        return None
    x = [r["main_output_bytes"] / MIB for r in points]
    y = [r["ca_us"] for r in points]
    mean_x, mean_y = sum(x)/len(x), sum(y)/len(y)
    slope = sum((a-mean_x)*(b-mean_y) for a, b in zip(x,y)) / sum((a-mean_x)**2 for a in x)
    intercept = mean_y - slope*mean_x
    origin = sum(a*b for a,b in zip(x,y))/sum(a*a for a in x)
    def wape(fixed, gradient):
        return sum(abs(b-fixed-gradient*a) for a,b in zip(x,y))/sum(y)
    adjacent = []
    for i in range(1, len(x)):
        adjacent.append(dict(t_from=points[i-1]["tokens"], t_to=points[i]["tokens"],
                             bytes_ratio=x[i]/x[i-1], time_ratio=y[i]/y[i-1],
                             slope_us_per_mib=(y[i]-y[i-1])/(x[i]-x[i-1])))
    result = dict(origin_us_per_mib=origin, origin_wape=wape(0,origin),
                affine_fixed_us=intercept, affine_us_per_mib=slope, affine_wape=wape(intercept,slope),
                adjacent=adjacent)
    if len(points) >= 3:
        # One extrapolation over the next interval, independent of the operator model.
        earlier = fit(points[:-1])
        result["from_smaller_to_largest_us"] = {
            "origin": earlier["origin_us_per_mib"] * x[-1],
            "affine": earlier["affine_fixed_us"] + earlier["affine_us_per_mib"] * x[-1],
            "observed": y[-1],
        }
    return result


def save_plot(rows, groups, output):
    with TemporaryDirectory() as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4), layout="constrained")
        for i, (op, (_, title)) in enumerate(OPS.items()):
            left, right = axes[i]
            left.set(title=title+"  |  coverage", xlabel="Logical main output (MiB), log scale",
                     ylabel="Full CA latency (µs)", xscale="log")
            by_op = [r for r in rows if r["op"]==op]
            for role, c, m, name in [("development_replay","#65798c","o","Earlier CA"),
                                       ("v0.11_independent","#006d77","D","New CA")]:
                group = [r for r in by_op if r["role"]==role]
                left.scatter([r["main_output_bytes"]/MIB for r in group], [r["ca_us"] for r in group],
                             marker=m, color=c, alpha=.70, s=24, label=name)
            left.legend(loc="best", fontsize=8)
            g = groups[op]
            right.set(title=title+"  |  fixed-width series", xlabel="Logical main output (MiB)",
                      ylabel="Full CA latency (µs)")
            if g:
                right.plot([r["main_output_bytes"]/MIB for r in g],[r["ca_us"] for r in g],
                           color="#006d77", marker="o", label=f"D={g[0]['width']}, {g[0]['active_aiv']} AIV, {g[0]['state']}")
                for r in g:
                    right.annotate(f"T={r['tokens']}", (r["main_output_bytes"]/MIB,r["ca_us"]),
                                   xytext=(0,6),textcoords="offset points",fontsize=8,ha="center")
                for field, color, label in [
                    ("exposed_simd_us", "#d08024", "Exposed SIMD"),
                    ("exposed_mte3_us", "#5061aa", "Exposed MTE3"),
                ]:
                    right.plot([r["main_output_bytes"]/MIB for r in g], [r[field] for r in g],
                               marker=".", color=color, linestyle="--", label=label)
                right.margins(x=.14,y=.20)
                right.legend(loc="best", fontsize=8)
            for ax in (left,right):
                ax.grid(True, alpha=.2)
        fig.savefig(output, dpi=160)
        plt.close(fig)


def coverage(rows):
    largest = max(rows, key=lambda r:r["main_output_bytes"])
    return (f"{len(rows)} | {min(r['tokens'] for r in rows)}–{max(r['tokens'] for r in rows)} | "
            f"{min(r['width'] for r in rows)}–{max(r['width'] for r in rows)} | {largest['shape']} | "
            f"{largest['input_bytes']/MIB:.4f} | {largest['main_output_bytes']/MIB:.4f} | "
            f"{largest['logical_io_bytes']/MIB:.4f} | {largest['ca_us']:.3f}")


def write_report(rows, groups, fits, output):
    lines = ["# 数据量增加时，两类 Vector 算子的 CA 耗时如何变化", "",
             "## 实测范围", "",
             "| 算子 | 数据角色 | shape 数 | token 范围 | 通道范围 | 主输出最大的 shape | 该点输入 MiB | 主输出 MiB | 总读写 MiB | CA µs |",
             "|---|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for op, (_, title) in OPS.items():
        for role,label in [("development_replay","历史开发回放"),("v0.11_independent","本轮冻结独立")]:
            group = [r for r in rows if r["op"]==op and r["role"]==role]
            lines.append(f"| {title} | {label} | {coverage(group)} |")
    lines += ["", "每个案例按张量元素计算逻辑数据量，MiB＝2²⁰ 字节。T 是 token 数，D 是通道宽度，N 是 mHC 流数。CA 时间取完整 kernel 的跨核指令区间。",
              "RMSNorm 的输入是 BF16 x 和 gamma，合计 2TD＋2D 字节。主输出 y 占 2TD 字节，输出 rstd 另占 4T 字节。"
              "mHC Post 的 BF16 x、h 和 FP32 res、post 共占 2TND＋2TD＋4TN²＋4TN 字节，主输出占 2TND 字节。"
              "总读写是上述各输入与输出字节的和。缓存复用、多核重复读取及搬运对齐会改变物理流量，本表没有 HBM 计数。",
              "", "RMSNorm 每行需要处理 D 个通道的平方归约和归一化。Post 每个输出元素需要 N＋1 个乘法和 N 个加法，N=4 时逻辑乘加工作随 TND 增长。两个算子在固定 D 与 N 时，加 token 会同时增加访存和计算工作。",
              "", "![全量 CA 覆盖及控制条件的多轮序列](scaling.png)", "",
              "## 固定宽度的多轮区间", "",
              "冻结计划使用 RMSNorm D=4096 和 mHC Post D=1024 的 T≥64 序列。按 N、安装包 kernel、实测活跃 AIV 数、启动前高水位分组，只有同组内计算斜率。"
              "状态由同进程此前启动的 AIV 高水位与本次活跃数划分，reuse 表示当前核数未超过已有高水位。该字段不测量缓存命中。"
              "每核行数取分核规则 ⌈T/64⌉，表示最忙 AIV 分配的 token 行数；每轮 UB 能处理的行数仍取决于库实现。"
              "暴露 MTE3 取最后完成核的区间分解，SIMD 与 MTE3 同时活动的时间归到 SIMD。它是可见的写流水时间，并非原始指令时长之和。", ""]
    for op, (_,title) in OPS.items():
        g, f = groups[op],fits[op]
        lines += [f"### {title}", ""]
        if not g:
            lines += ["相同条件的冻结样本不足两点。", ""]
            continue
        lines += [f"每点均取新留出 CA。固定 D={g[0]['width']}，N={g[0]['streams']}，包内 kernel `{g[0]['kernel']}`。",
                  "", "| T | 主输出 MiB | 总读写 MiB | AIV 高水位→活跃 | 每核行数 | CA µs | 暴露 SIMD / MTE3 µs | SIMD 事件数 | 数据量/上点 | 耗时/上点 | 相邻增量 µs/MiB |",
                  "|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|"]
        for i,r in enumerate(g):
            a = f["adjacent"][i-1] if i else None
            tail = f"{a['bytes_ratio']:.2f}× | {a['time_ratio']:.2f}× | {a['slope_us_per_mib']:.3f}" if a else "— | — | —"
            lines.append(f"| {r['tokens']} | {r['main_output_bytes']/MIB:g} | {r['logical_io_bytes']/MIB:.4f} | "
                         f"{r['prior_aiv']}→{r['active_aiv']} | {r['rows_per_busiest_aiv']} | {r['ca_us']:.3f} | "
                         f"{r['exposed_simd_us']:.3f} / {r['exposed_mte3_us']:.3f} | {r['critical_simd_events']} | {tail} |")
        for r in rows:
            if (r["op"]==op and r["role"]=="v0.11_independent" and r["width"]==WIDTH[op]
                    and r["tokens"]>=64 and r not in g):
                lines += ["", f"{r['shape']} 同宽度点 CA {r['ca_us']:.3f} µs，启动前高水位 "
                          f"{r['prior_aiv']}，活跃核 {r['active_aiv']}。该点计入全量覆盖，与上表分开统计。"]
        lines += ["",f"用相同 {len(g)} 点描述时间随主输出 B（MiB）的变化。过原点的最小二乘解为 "
                  f"t={f['origin_us_per_mib']:.3f}B，组内 WAPE {f['origin_wape']:.1%}。"
                  f"允许固定项后的解为 t={f['affine_fixed_us']:.3f}{f['affine_us_per_mib']:+.3f}B，组内 WAPE {f['affine_wape']:.1%}。"
                  "t 的单位为 µs。两项 WAPE 均为线性趋势描述的同点残差，不代表新模型对留出集的预测误差。固定项是外推到零数据量的截距。",
                  ""]
        if len({r["batch"] for r in g})>1:
            lines += ["本组结合不同 CA 进程，同一高水位只控制了此前启用的核数。不同的热身操作、缓存及运行间差异仍会影响相邻时间差，表中斜率含这些差异。每点进程及启动编号见逐点数据。", ""]
        ext = f.get("from_smaller_to_largest_us")
        if ext:
            lines += [f"只用前 {len(g)-1} 点拟合，将最大一点留给趋势检验。过原点给出 {ext['origin']:.3f} µs，"
                      f"带截距给出 {ext['affine']:.3f} µs，该点 CA 为 {ext['observed']:.3f} µs。", ""]
        if op == "rmsnorm" and [r["tokens"] for r in g] == [128,256,512]:
            a,b,c=g
            lines += [f"从 T=128 到 256，关键核 SIMD 事件从 {a['critical_simd_events']} 增至 {b['critical_simd_events']}，"
                      f"其暴露时间增加 {b['exposed_simd_us']-a['exposed_simd_us']:.3f} µs；同期暴露写回减少 "
                      f"{a['exposed_mte3_us']-b['exposed_mte3_us']:.3f} µs，整核完成时间反而缩短。"
                      f"再增至 T=512 时，SIMD 事件达到 {c['critical_simd_events']}，两段相邻总时延的斜率异号。"
                      "计算工作随行数增长已经可见，总时延还没有表现出稳定的单位字节增量。", ""]
        if op=="mhc_post" and [r["tokens"] for r in g]==[64,128,192,256]:
            increments=f["adjacent"]
            difference=abs(increments[-1]["slope_us_per_mib"]/increments[-2]["slope_us_per_mib"]-1)
            lines += [f"主输出 1 到 2 MiB 的三点来自相同热身协议，两个相邻增量斜率相差 {difference:.1%}，"
                      f"关键核每新增 64 个 token 的 SIMD 事件增量依次为 {g[2]['critical_simd_events']-g[1]['critical_simd_events']}、"
                      f"{g[3]['critical_simd_events']-g[2]['critical_simd_events']}。"
                      "这些点支持用固定开销加单位工作开销描述这一个区间。起始半 MiB 区间的斜率较高，"
                      "首点在含其它算子的原批次中执行，无法由此区分状态差异和服务节奏变化。", ""]
    lines += ["## 解释边界", "",
              "相邻增量比较额外 1 MiB 主输出对应的时间。若同一实现已进入稳态，核数和每核服务节奏保持稳定时，连续多个区间的增量应接近。"
              "两点总能确定直线；引入固定项也会降低拟合同点残差。这些单次 CA 观测缺少同 shape 重复测量，不能只凭较小的组内误差认定已经达到稳态或定位拐点。"
              "对于不同 D、通道切分、核数或启动状态，图左保留单点，连成统一直线会混合不同执行条件。"
              "图右连线只连接相邻实测点，没有把趋势外推到更大 shape 或其它代际。斜率同时含计算、发起与写回等开销，不能直接换算 HBM 峰值利用率。",
              "", "## 原始记录归属与重采", "",
              "对本轮每个 launch，将 helper 的 flush 状态、invalid、mismatched、write_failed、written 与 SQLite 行数、逆序区间核对。"
              "helper 有 start/end_overwrites、start_only 与 end_only 回调计数，没有独立的 DROP 总计；这些计数与落库指令行分别记录。"
              "再从 MOV_SRC_TO_DST_ALIGNv2 的 OUT 写提取目的地址、burst 次数、长度与步长，每个 burst 展开为一个半开字节区间。"
              "对每个输出要求实际写入字节数、地址去重并集和连续跨度都与期望张量字节数相同。RMSNorm 分别核对 y、rstd。"
              "检查的是 CA 输出地址记录，每个张量分配的数值正确性另由 CPU 参考验证。全量审计见 [逐 launch 地址与计数](trace-audit.csv)。",
              "", "RMSNorm 原十条均通过。固定 D=4096 时，T=64/128/256 每核各发一笔主输出 copy，"
              "每笔覆盖 1/2/4 行；T=512 每核把 8 行拆成 7 行和 1 行分两笔写出，rstd 对应拆分。"
              "后者来自 OUT 参数中每核的 burst 个数与长度，证明写回组织改变，不能仅按每核 token 行数推定 UB 轮数。",
              "", "Post 原前七条通过，原 T=128、192 各有 1144、1163 个 invalid 区间，落库 0 条。"
              "原 T=256 虽然 flush 成功，单个 2 MiB 输出却记录 408 笔 8192-B OUT 写和 311 个唯一行地址，"
              "其写入总量 3342336 B、并集 2547712 B、跨度 2744320 B，全超过预期 2097152 B。"
              "在物理核 0 的 AIV0，前段写地址 0x9154d4f800 与 0x9154d51800，后段的四笔从 0x9154ded800 开始连续。"
              "相邻 AIV 前段按三行偏移，后段按四行偏移，符合前一 T=192 与当前 T=256 的每核任务量。"
              "三个原点均剔除。三个隔离进程各先用 [4,4,6144] 激活 64 个 AIV；只计第二次启动的目标 shape。"
              "控制样本和原失败点均未重复计入留出集。", "",
              "每个有效点的角色、逻辑字节、核数、CA 与原始记录路径见 [逐点数据](points.csv)。"]
    output.write_text("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE/"scaling")
    parser.add_argument("--audit-only", action="store_true", help="Check original and any available recovered DBs only")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    audited=audits()
    save_table(args.output/"trace-audit.csv",audited)
    if args.audit_only:
        return
    accepted={r["db"] for r in audited if r["passed"]}
    old=read(ROOT/"results/v0.10/evaluation.json")["rows"]
    rows=[record(r,"development_replay") for r in old if r["op"] in OPS]
    for op,(directory, _) in OPS.items():
        fresh=read(BASE/directory/"rows.json")
        if isinstance(fresh,dict):
            fresh=fresh["rows"]
        fresh=[r for r in fresh if r["role"] == "validation"]
        if len(fresh)!=10 or any(r["op"]!=op for r in fresh):
            raise ValueError(f"Expected 10 complete independent {op} cases")
        highwater={}
        if op=="mhc_post":
            for file in sorted((BASE/"post").glob("retry-v11_mhc_post_*.json")):
                control=read(file)["completed"][0]
                highwater[control["batch"]]=control["active_aiv"]
        for r in sorted(fresh, key=lambda r:(r["batch"], r["launch"])):
            before=highwater.get(r["batch"],0)
            if r["source"] not in accepted:
                raise ValueError(f"Output/flush audit failed for {r['source']}")
            if r["prior_aiv"]!=before:
                raise ValueError(f"Unexpected prior AIV for {r['id']}")
            highwater[r["batch"]]=max(before,r["active_aiv"])
        rows.extend(record(r,"v0.11_independent") for r in fresh)
    ids={(r["op"],r["shape"]) for r in rows}
    if len(ids)!=len(rows):
        raise ValueError("Duplicate shape across development and validation")
    groups={op:series([r for r in rows if r["op"]==op]) for op in OPS}
    fits={op:fit(group) for op,group in groups.items()}
    save_table(args.output/"points.csv",rows)
    detail={op:dict(ids=[r["id"] for r in groups[op]],fit=fits[op]) for op in OPS}
    (args.output/"trends.json").write_text(json.dumps(detail,indent=2)+"\n")
    save_plot(rows, groups, args.output/"scaling.png")
    write_report(rows, groups, fits, args.output/"summary.md")


if __name__=="__main__":
    main()
