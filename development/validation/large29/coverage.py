"""One large-shape evidence index; inspect full-device windows and wrapper endpoints separately."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate
OUT=ROOT/"results/v0.29/large"
DT,PR,FUT="Ascend950DT_9582","Ascend950PR_957b","Ascend970DT"
LINKS={"fusion":"fusion/report.md","vector":"vector/report.md","cube":"cube/report.md",
       "matmul":"../matmul/full-load/score.md","fa":"fa/report.md","kda_v1":"../kda/report.md"}


def load(p):
    return json.loads(p.read_bytes())


def render():
    rows=[]
    def add(op,shape,group,role,sku,price,ca,status,evidence,protocol="full",conditional=None):
        rows.append(dict(op=op,shape=shape,group=group,role=role,sku=sku,model_us=price,
            conditional_us=conditional,ca_us=ca,signed_pct=100*(price/ca-1) if ca and price else None,
            qualification=("extrapolation_only" if sku==FUT else status),protocol=protocol,source=evidence))
    for r in load(OUT.parent/"matmul/full-load/score.json")["rows"]:
        if r["shape"][0]<262144:continue
        add("matmul",r["shape"],"matmul","Flash B8 整批",r["chip"],r["legacy_us"],r.get("ca_us"),
            ("qualified" if r.get("ca_us") else r.get("qualification",r.get("status","pending"))),r.get("receipt") or "../matmul/full-load/freeze.json",conditional=r["candidate_us"])
    fa=load(OUT/"fa/freeze.json")
    for r in fa["archive"]:
        add("fa",r["shape"],"fa","dense 单头长上下文",r["chip"],r["current_us"],r["ca_us"],"qualified",
            "fa/freeze.json",protocol="compact" if r["kernel"].startswith("entry_kernel") else "full")
    for shape in {tuple(r["shape"]) for r in fa["archive"]}:
        f=next(r["future_970dt_extrapolated_us"] for r in fa["archive"] if tuple(r["shape"])==shape)
        add("fa",list(shape),"fa","dense 单头长上下文",FUT,f,None,"extrapolated","fa/freeze.json")
    for j in fa["jobs"]:
        chip=j["chip"];path=OUT/"fa"/(j["id"]+".json");raw=load(path) if path.exists() else {}
        add("fa",j["shape"],"fa","dense 单头长上下文",chip,j["predictions"][chip]["us"],raw.get("ca_us"),
            "qualified" if raw.get("qualified") else ("unqualified" if raw else "pending"),str(path.relative_to(OUT)) if raw else "fa/freeze.json")
        if not any(r["op"]=="fa" and r["sku"]==FUT and r["shape"]==j["shape"] for r in rows):
            add("fa",j["shape"],"fa","dense 单头长上下文",FUT,j["predictions"][FUT]["us"],None,
                "extrapolated","fa/freeze.json")
    pre={ (x["t"],x["chip"]):(x.get("candidate") or {}).get("us")
          for x in load(OUT/"fusion/stage-freeze.json")["pre"] }
    for r in load(OUT/"fusion/score.json")["rows"]:
        for chip,v in r["target"].items():
            add(r["op"],r["shape"],"fusion",r["case"],chip,v["model_us"],v.get("ca_us"),
                v.get("qualification"),v.get("ca_receipt") or "fusion/freeze.json",
                conditional=pre.get((r["shape"][0],chip)) if r["op"]=="mhc_pre" else None)
    cube=load(OUT/"cube/freeze.json")
    for j in cube["jobs"]:
        for chip,tag in ((DT,"dt"),(PR,"pr"),(FUT,None)):
            path=OUT/"cube"/(j["id"]+"-"+str(tag)+".json")
            obs=load(path) if path.exists() else {}
            for name in j["ops"]:
                op="fla_kda_v2" if name=="kda_chunk" else name
                forecast=j["predictions"][name][chip]
                m=next((x for x in obs.get("operators",[]) if x["op"]==name),{})
                c=m.get("ca_us")
                add(op,j["shape"],"cube",j["id"],chip,forecast["us"],c,
                    "qualified" if c else ("unqualified" if obs else "pending"),
                    str(path.relative_to(OUT)) if obs else "cube/freeze.json",
                    conditional=forecast.get("independent_state_scenario",{}).get("scenario_us"))
    for chip in (DT,PR,FUT):
        add("kda_chunk_fused_v1",[1,16384,12,12,128,128],"kda_v1","Kimi B8×2048，融合 V1",
            chip,None,None,"small-scale synchronization failure, large dispatch not timed","../kda/report.md")
    vector=load(OUT/"vector/freeze.json")
    for j in vector["jobs"]:
        if j["shape"]==[32,4096]:continue  # Independent check of native transport, not a large scenario.
        chip=j["chip"];p=OUT/"vector"/(j["case"]+".json");d=load(p) if p.exists() else {}
        v=j["prices"][chip]
        ca=d.get("instruction_us",d.get("ca_us")) if d.get("qualified") else None
        if j["protocol"]=="wrapper" and d.get("qualified"):
            ca=d.get("wrapper_us",d.get("core_envelope_us",ca))
        add(j["op"],j["shape"],"vector",j["protocol"],chip,v["public_us"],ca,
            "qualified" if ca else ("unqualified" if d else "pending"),
            str(p.relative_to(OUT)) if d else "vector/freeze.json",protocol=j["protocol"],
            conditional=(v.get("ideal_service") or {}).get("conditional_schedule_us"))
        if not any(r["op"]==j["op"] and r["sku"]==FUT and r["shape"]==j["shape"] for r in rows):
            future=j["prices"][FUT]
            add(j["op"],j["shape"],"vector",j["protocol"],FUT,future["public_us"],None,"extrapolated",
                "vector/freeze.json",protocol=j["protocol"],
                conditional=(future.get("ideal_service") or {}).get("conditional_schedule_us"))
    for r in load(OUT/"vector/archive.json"):
        if r["protocol"]!="wrapper":continue
        add(r["op"],r["shape"],"vector","整批 full hidden",r["chip"],r["current_us"],r["ca_us"],
            "qualified",r["source"],protocol="wrapper")
    # Existing large measured model-shaped tables; compare current public prices.
    for model,op in (("kimi","sigmoid_fp32"),("pro","cast_bf16_fp32"),("flash","cast_bf16_fp32")):
        dirs=(ROOT/"results/v0.23/prefill",ROOT/"results/v0.23/pro-prefill")
        for folder in dirs:
            for path in sorted(folder.glob("*"+model+"*ki-*.json")):
                r=load(path)
                if not all(k in r for k in ("shape","chip","ca_us","trace_sha256")) or r.get("op")!=op or not r.get("ca_us"):
                    continue
                chip=r["chip"];add(op,r["shape"],"vector",model+" 整批 indexer / gate",chip,
                    estimate(op,r["shape"],chip=chip).latency_us,r["ca_us"],"qualified",
                    str(path.relative_to(ROOT)),protocol="full")
    # Some qualified large rows are in the main replay, e.g. Pro B8×256Ki Cast.
    for r in load(OUT.parent/"replay.json")["rows"]:
        if r["op"] not in ("cast_bf16_fp32","sigmoid_fp32") or r["shape"][0]<262144:continue
        chip=r["chip"]
        if not any(x["op"]==r["op"] and x["shape"]==r["shape"] and x["sku"]==chip and x["ca_us"]==r["ca_us"] for x in rows):
            add(r["op"],r["shape"],"vector","归档完整窗",chip,r["current_us"],r["ca_us"],"qualified","../replay.json")
    targets={j["op"] for j in rows}
    assert len(targets)==22, sorted(targets)
    bykey=defaultdict(list)
    for row in rows:bykey[row["op"]].append(row)
    def display(item):
        return f"{item['ca_us']:.3f} µs {item['shape']}" if item else "待合格窗"
    lines=["# 每个算子的大 shape 验证","",
           "单卡 B8×32Ki 整批交付 262,144 个 token。逐 token 算子沿行增加，FA 沿 KV 页推进，MoE 按每专家实际接收行数，KDA recurrent 按独立状态链。这里逐项展示本轮选择的长轴或多轮工作、已有完整窗和正在验证的范围。DT/PR 的数字均为 CA 设备窗口，970DT 只列在各算子的条件预测中。","",
           "| 算子 | 本轮代表大形，按接口轴顺序 | 950DT 已有的代表完整 CA | 950PR 已有的代表完整 CA | 口径与逐形来源 |",
           "|---|---|---|---|---|"]
    for name,vs in sorted(bykey.items()):
        group=vs[0]["group"]
        shapes=[]
        for x in vs:
            if x["shape"] not in shapes:shapes.append(x["shape"])
        def pick(chip):
            ready=[x for x in vs if x["sku"]==chip and x["ca_us"] is not None and x["protocol"]!="wrapper"]
            # This is an illustrative complete window, not an independent summary or claim that
            # unmatched shapes have been timed. Larger leading dimensions win at unchanged geometry.
            return max(ready,key=lambda x:(x["shape"][1] if name=="fa" else x["shape"][0],x["ca_us"]),default=None)
        label={
            "cube":"工作量、库合同及全输出核验","fusion":"几何等价性、链及各段","vector":"全指令窗和原生包络分列",
            "fa":"dense 单头 SBH 长循环","matmul":"B 常驻 L1 的窄输出投影","kda_v1":"重复 SET；大形只有 Finalize 工作量",
        }[group]
        show=shapes[:3]
        if group=="vector" and name in ("cast_bf16_fp32","sigmoid_fp32"):
            show=sorted(shapes,key=lambda s:s[0]*s[1],reverse=True)[:2]
        vals=[]
        for chip in (DT,PR):
            full=pick(chip);wr=next((x for x in vs if x["sku"]==chip and x["protocol"]=="wrapper" and x["ca_us"]),None)
            vals.append(display(full) if full else (f"原生包络 {display(wr)}" if wr else "—"))
        lines.append(f"| {name} | {'、'.join(map(str,show))} | {vals[0]} | {vals[1]} | [{label}]({LINKS[group]}) |")
    lines+=["","原生包络覆盖所有参与核的硬件启动到完成，指令窗口以全片第一条到末核最后退休指令为准。形状覆盖须同时检查 dtype、布局、核实现和完整输出，表中 — 表示该范围尚缺合格大形。MoE 的 4096 行按单个 attention 来源、均衡路由计算。若有 R 个同等 DP 来源汇到同一个 EP 目标，本专家的均衡行数为 4096R。MoE 的 MXFP8 替代合同、独立稠密 FA 与实际部署的不同精度及稀疏路线在各分表标明。两个大端点都返回时才能评价同形跨卡比值。[所有大形模型/实测值和状态](coverage.json) · [小形与全量误差](../../../docs/error-analysis.md)"]
    complete_pairs=0
    paired={}
    for op,vs in sorted(bykey.items()):
        observed={c:{tuple(x["shape"]) for x in vs if x["sku"]==c and x["ca_us"] is not None
                                    and x["protocol"]!="wrapper"} for c in (DT,PR)}
        common=sorted(observed[DT]&observed[PR])
        paired[op]=common
        complete_pairs+=bool(common)
    lines[2]+=f" 目前 {len(bykey)} 条路线中有 {complete_pairs} 条已至少拿到一个本表大几何的 DT/PR 同形完整指令窗；其余仍在采集、等待同形或缺合格实现。"
    lines+=["","### 大形同集的公开点预测误差","",
            "分别按芯片、算子和窗口口径计分。相同几何、相同指标及相同 CA 数值被多个来源引用时只计一次。候选价格与下界见各分表。少量大形只描述已返回结果；单形 WAPE 等于该形的绝对相对误差。","",
            "| 算子 · 窗口口径 | DT 已有点预测的窗数 · WAPE / 最大 | PR 已有点预测的窗数 · WAPE / 最大 |",
            "|---|---:|---:|"]
    metrics={}
    for op,vs in sorted(bykey.items()):
        for kind in sorted({"wrapper" if row["protocol"]=="wrapper" else "完整指令" for row in vs}):
            def agg(target):
                vals={(tuple(row["shape"]),row["ca_us"]):row["model_us"] for row in vs if row["sku"]==target
                         and row["ca_us"] and row["model_us"] and
                         (row["protocol"]=="wrapper")== (kind=="wrapper")}
                if not vals:return {"n":0,"wape":None,"maximum":None}
                return {"n":len(vals),"wape":100*sum(abs(v-y) for (_,y),v in vals.items())/sum(y for _,y in vals),
                        "maximum":100*max(abs(v/y-1) for (_,y),v in vals.items())}
            def metric(d):
                return f"{d['n']} · {d['wape']:.2f}% / {d['maximum']:.2f}%" if d["n"] else "0 · N/A"
            m={c:agg(c) for c in (DT,PR)};metrics[op,kind]=m
            lines.append(f"| {op} · {kind} | {metric(m[DT])} | {metric(m[PR])} |")
    (OUT/"report.md").write_text("\n".join(lines)+"\n")
    (OUT/"coverage.json").write_text(json.dumps(dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),operations=len(targets),complete_same_shape_pairs_by_op=paired,
        large_metrics=[dict(op=op,protocol=kind,metrics=v) for (op,kind),v in metrics.items()],rows=rows),ensure_ascii=False,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":render()
