"""Summarise the reproducible release inputs, keeping measurement protocols separate."""
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.matmul29.workloads import render as render_matmul_workloads
OUT = ROOT/"results/v0.29"
OLD = ROOT/"results/v0.28"
DT, PR = "Ascend950DT_9582", "Ascend950PR_957b"


def read(p):
    return json.loads(p.read_bytes())


def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def summarise(rows):
    if not rows:
        return dict(n=0, shapes=0, wape=None, maximum=None, within_limits=None)
    errors = [abs(p-ca) for shape, p, ca in rows]
    wape = 100*sum(errors)/sum(ca for shape,p,ca in rows)
    maximum = 100*max(e/ca for e, (_,p,ca) in zip(errors,rows))
    return dict(n=len(rows), shapes=len({tuple(s) for s,p,ca in rows}), wape=wape,
                maximum=maximum, within_limits=wape<=25 and maximum<=50)


def snapshot():
    protocols = defaultdict(lambda: defaultdict(list))
    seen = set()
    references = {}
    without_public_price = []

    def add(label, chip, shape, prediction, ca, identity, source):
        key = (label,chip,tuple(identity) if isinstance(identity,list) else identity)
        if key in seen:
            return
        if prediction is None or ca is None:
            raise ValueError(f"Not a timed observation: {source}")
        seen.add(key)
        protocols[label][chip].append((shape,prediction,ca))
        references.setdefault(label,set()).add(source)

    for r in read(OUT/"replay.json")["rows"]:
        if r["op"] not in ("matmul","mhc_sinkhorn") and r["current_us"] is not None:
            add(r["op"],r["chip"],r["shape"],r["current_us"],r["ca_us"],r["trace_sha256"],"replay.json")
    # Enriched Matmul archive includes the same old receipts and all returned application geometries.
    for r in read(OUT/"matmul/score.json")["rows"]:
        add("matmul", r["chip"], r["shape"], r["old_us"], r["ca_us"], r["raw_sha256"], "matmul/score.json")
    for pair in read(OUT/"matmul/decode/score.json")["unique_rows"].values():
        for chip,r in pair.items():
            if r.get("qualified"):
                add("matmul", chip, r["shape"], r["old_us"], r["ca_us"], [r["trace_sha256"]], "matmul/decode/score.json")
    retry = OUT/"matmul/decode/retry/receipt.json"
    if retry.exists():
        r = read(retry)
        if r["qualified"]:
            stage, = r["stages"]
            registered = read(OUT/"matmul/decode/score.json")["unique_rows"][str(r["shape"][0])][r["chip"]]
            assert r["shape"] == registered["shape"]
            add("matmul",r["chip"],r["shape"],registered["old_us"],r["ca_us"],
                [stage["trace_sha256"]],"matmul/decode/retry/receipt.json")
    for r in read(OUT/"post/score.json")["rows"]:
        for label,field in [("mhc_post · shape","stateless_us"),("mhc_post · visible stream","ordered_us")]:
            add(label,r["chip"],r["shape"],r[field],r["ca_us"],r["sha256"],"post/score.json")
    rms = read(OLD/"rms/stream-replay.json")
    # Archived DB dialects have different protocols. The three paired experiments all use msopprof.
    for r in rms["archived"]:
        label="rmsnorm · compact CA" if r["provider"]=="npusim compact" else "rmsnorm · msopprof"
        add(label,r["chip"],r["shape"],r["unknown_us"],r["ca_us"],r["trace"],"../v0.28/rms/stream-replay.json")
    for r in rms["pairs"]:
        add("rmsnorm · msopprof",r["chip"],r["shape"],r["unknown_us"],r["ca_us"],r["trace"],"../v0.28/rms/stream-replay.json")
        add("rmsnorm · visible stream",r["chip"],r["shape"],r["stream_us"],r["ca_us"],r["trace"],"../v0.28/rms/stream-replay.json")
    for r in read(OUT/"replay.json")["rows"]:
        if r["op"]=="mhc_sinkhorn":
            add("mhc_sinkhorn",r["chip"],r["shape"],r["current_us"],r["ca_us"],r["trace_sha256"],"replay.json")
    for fn in ("replay","independent"):
        src=f"../v0.28/fusion/{fn}.json"
        for r in read(OUT/src)["rows"]:
            if r["old_unknown_us"] is None:
                without_public_price.append(dict(op="mhc_sinkhorn",chip=r["sku"],shape=[r["t"],4],
                                                 iterations=r["iters"],ca_us=r["ca_us"],trace=r["id"],source=src))
                continue
            add("mhc_sinkhorn",r["sku"],[r["t"],4],r["old_unknown_us"],r["ca_us"],[r["id"]],src)
    cast = read(OLD/"cast/score.json")
    prices = read(OLD/"cast/register.json")["prices"]
    for chip,r in cast["heldout"].items():
        if r["qualified"]:
            raw = read(ROOT/r["receipt"])
            add("cast_bf16_fp32",chip,raw["shape"],prices[chip]["pressure"]["us"],raw["ca_us"],
                [raw["stage"]["trace_sha256"]],"../v0.28/cast/score.json")
    table = {name:{c:summarise(groups[c]) for c in (DT,PR)} for name,groups in sorted(protocols.items())}
    table["kda_chunk · fused V1"] = {c:summarise([]) for c in (DT,PR)}
    references["kda_chunk · fused V1"]={"kda/report.md"}
    utc=datetime.now(timezone.utc).isoformat(timespec="seconds")
    registers=[p for p in OUT.rglob("*.json") if p.name.endswith(("freeze.json","register.json")) or p.name=="registration.json"]
    checks = list(sorted((ROOT/"operator_model").glob("*.py"))) + list(sorted((ROOT/"operator_model/extensions").glob("*.py")))
    assets=[ROOT/line.split(" ",1)[1] for line in (ROOT/"MANIFEST.in").read_text().splitlines() if line.startswith(("include results/","include configs/"))]
    frozen={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sorted(set(checks+assets+registers))}
    model=dict(version="0.29.0", baseline="0.28.0", inputs_sha256=frozen,
               contracts=[str(p.relative_to(ROOT)) for p in sorted(registers)],
               default="Installed public estimate; Post tracking opt-in, full-load Matmul/placement and quant memory candidates separate.",
               unknown_target="970DT is a scenario. Never included in measured errors.")
    write(OUT/"model.json",model)
    sources={name:sorted(paths) for name,paths in references.items()}
    write(OUT/"evaluation.json",dict(snapshot_utc=utc,
          model_sha256=sha256((OUT/"model.json").read_bytes()).hexdigest(),limits=dict(wape=25,maximum=50),
          data_role="Broad deduplicated replay of named archives. Mix of earlier development and registered inputs. Each separate protocol has its own denominator. Model-specific forward tests are linked separately.",
          summary=table, sources=sources, measured_without_public_price=without_public_price,
          wheel_check=read(OUT/"wheel-check.json"),
          exclusions=["V1 repeated SET: no qualified latency", "970DT extrapolated", "partial traces",
                      "native wrapper envelopes reported separately", "host workspace-only probes"]))
    lines=["# 以当前公开调用重新核对两端误差","",
        "下表汇总各来源中已归档且按该口径可计时的样本，包含开发回放。每格是“不同几何数 / 窗数，WAPE / 最大绝对误差”。同一时序的不同实现、同形的前序、不同计时提供者按表内协议处理。单独的新形验收、分组尾差及原生 wrapper 在总误差分析列出。","","| 算子及输入合同 | 950DT | 950PR | 双端是否通过此表两项限值 | 数据 |","|---|---:|---:|---|---|"]
    for name,v in sorted(table.items()):
        def fmt(x):
            if x["n"]==0: return "0 · N/A"
            return f"{x['shapes']} / {x['n']} · {x['wape']:.2f}% / {x['maximum']:.2f}%"
        ok=v[DT]["within_limits"] and v[PR]["within_limits"]
        status={True:"通过",False:"有超限",None:"缺合格时延"}[ok]
        links=" · ".join(f"[{Path(p).parent.name+'/'+Path(p).name if '/' in p else p}]({p})" for p in sources[name])
        lines.append(f"| {name} | {fmt(v[DT])} | {fmt(v[PR])} | {status} | {links} |")
    lines+=["","Matmul 此处用公开路由。RMSNorm 和内置 Post 的有序行使用从真实顺序可重建的历史，compact 和 msopprof 指令口径分列。KDA V2 三段和扩展 HcPost 各有单独的实现合同。[全部独立验收和未满足的分组](../../docs/error-analysis.md) · [各协议来源和未舍入数值](evaluation.json)"]
    (OUT/"acceptance.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    snapshot()
    render_matmul_workloads()
