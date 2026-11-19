"""Publish qualified observations separately from candidate studies and background jobs."""
from datetime import datetime, timezone
from hashlib import sha1, sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import __version__, estimate

OUT = ROOT/"results/v0.23"
DT, PR, FUT = ("Ascend950DT_9582","Ascend950PR_957b","Ascend970DT")
def read(p):
    return json.loads(p.read_text())
def put(name,v):
    (OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n")
def digest(p):
    return sha256(p.read_bytes()).hexdigest()
def main():
    reg=read(OUT/"hc-post/development.json")
    tree=read(ROOT/"runs/workloads19/sources/revision-tree.json")
    assert reg["vllm_revision"]==tree["sha"] and not tree["truncated"]
    items={r["path"]:r["sha"] for r in tree["tree"]}
    upstream=ROOT/"runs/workloads19/sources/full"
    proofs={}
    for path,sha in reg["hashes"].items():
        fp=ROOT/path
        if fp.is_relative_to(upstream):
            data=fp.read_bytes();short=str(fp.relative_to(upstream))
            git_blob=sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
            assert git_blob==items[short] and sha256(data).hexdigest()==sha,path
            proofs[short]=git_blob
    put("hc-post/source.json",dict(revision=tree["sha"],files=proofs))
    rms=read(ROOT/"results/rms-unknown23/independent.json")
    kda=read(ROOT/"results/kda-v23/holdout-comparison.json")
    hcs=read(OUT/"hc-timing/comparison.json")
    rows=[]
    for fp in sorted((OUT/"prefill").glob("*.json")):
        obs=read(fp)
        event=estimate(obs["op"],obs["shape"],chip=obs["chip"],timing_mode="resource")
        assert abs(event.latency_us-obs["model_us"])<1e-9
        rows.append(dict(op=obs["op"],shape=obs["shape"],chip=obs["chip"],
                         us=event.latency_us,ca_us=obs["ca_us"],future_us=obs["projection_970dt_us"],
                         role="archived full prefill replay",receipt=str(fp.relative_to(ROOT))))
    for r in kda["rows"]:
        if not r["qualified"]:continue
        attrs={"implementation":"fla_26_9_1_v2"}
        same=estimate("kda_chunk",r["shape"],attrs,chip=r["chip"])
        assert abs(same.latency_us-r["prediction_us"])<1e-9
        rows.append(dict(op="kda_chunk",implementation=attrs["implementation"],shape=r["shape"],chip=r["chip"],us=same.latency_us,
                         ca_us=r["ca_us"],future_us=estimate("kda_chunk",r["shape"],attrs,chip=FUT).latency_us,
                         receipt="results/kda-v23/"+r['receipt'],selection=r.get("selection","initial"),role="frozen independent"))
    hc_attrs={"implementation":"vllm_hc_post"}
    for r in hcs["rows"]:
        receipt=read(ROOT/r["raw_receipt"])
        assert receipt["qualified"]
        measured=estimate("mhc_post",r["shape"],hc_attrs,chip=r["sku"])
        assert abs(measured.latency_us-r["prediction_us"])<1e-8
        rows.append(dict(op="mhc_post",implementation="vllm_hc_post",shape=r["shape"],chip=r["sku"],us=measured.latency_us,
                         ca_us=r["ca_us"],future_us=estimate("mhc_post",r["shape"],hc_attrs,chip=FUT).latency_us,
                         role=r["role"],receipt=r["raw_receipt"]))
    # Every qualified RMS row retains both target and future conditional prices.
    future_rows=read(ROOT/"results/rms-unknown23/freeze.json")["rows"]
    for r in rms["rows"]:
        if r["role"] not in ("entry","immediate_repeat"):continue
        measured=r["ca_us"]
        attrs={"resident_aiv":r["prior_aiv"]}
        p=estimate("rmsnorm",r["shape"],attrs,chip=r["chip"])
        assert abs(p.latency_us-r["known_us"])<1e-9
        future_fleet=estimate("rmsnorm",r["shape"],chip=FUT)
        # A high-water mark in lanes belongs to the target stream; the registered
        # 970 sequence contains its own prior_AIV for the same ordinal.
        future,=[x for x in future_rows if x["chip"]==FUT and x["process"]==r["process"] and x["i"]==r["i"]]
        rows.append(dict(op="rmsnorm",shape=r["shape"],chip=r["chip"],prior_aiv=r["prior_aiv"],us=r["previous_us"],known_us=p.latency_us,
                         ca_us=measured,future_us=future_fleet.latency_us,future_prior_aiv=future["prior_aiv"],future_known_us=future["known_us"],
                         role=r["role"],protocol=r["process"],receipt="results/rms-unknown23/independent.json"))
    def paths(root):
        return (ROOT/path for path in root)
    hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(ROOT.joinpath('operator_model').rglob('*.py'))}
    hashes.update({str(p.relative_to(ROOT)):digest(p) for p in paths(["MANIFEST.in","pyproject.toml","configs/kda-fla-v2-26.9.1.json","configs/vllm-hc-post-arch35.json","configs/chips/Ascend950DT_9582.json","configs/chips/Ascend950PR_957b.json","configs/chips/Ascend970DT.json"])})
    put("model.json",dict(version=__version__,sha256=hashes,base_parameters="results/v0.22/model.json",
                          kda_prediction_freeze="results/kda-resource23/freeze.json",rms_freeze="results/rms-unknown23/freeze.json",
                          hc_prediction_freezes=["results/v0.23/hc-timing/holdout-register.json",
                                                "results/v0.23/hc-timing/supplement-register.json"]))
    put("evaluation.json",dict(version=__version__,as_of_utc=datetime.now(timezone.utc).isoformat(),
                               unchanged_contracts="results/v0.22/evaluation.json",rows=rows,
                               current_default_rms="previous_us in original candidate comparison",rms_groups=rms["groups"],
                               kda_groups=kda["by_chip"],hc_groups=hcs["summary"],hc_independent=hcs["accepted_independent_summary"],
                               hc_unqualified=hcs["failed"]+hcs["supplement_failed"]))
    table=["# 本版新增双端实测与外推","",
           "RMSNorm 默认列为只给 shape，H 列按目标芯片自己的流历史。KDA 计三个 kernel 的设备窗之和；HcPost 用显式 vLLM 实现的各片资源排程。970DT 全列为外推。单位均为微秒。开发回放与冻结独立数据按行标明。[既有全量逐形](../v0.22/shape-errors.md) · [本版机器可读数据](evaluation.json) · [HcPost 的数值失败和未覆盖形状](hc-timing/comparison.md)","",
           "| 算子 / 状态 | shape | 芯片 | 默认或显式模型 | 给 H 的模型 | CA | 默认误差 | 给 H 的误差 | 970DT 模型 / 给 H |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    fmt=lambda x:"N/A" if x is None else f"{x:.3f}"
    for r in rows:
        err=lambda x: "N/A" if x is None else f"{100*(x/r['ca_us']-1):+.2f}%"
        cost=r["us"]; known=r.get("known_us")
        label=r.get("implementation",r["op"])+" "+r["role"]
        if "prior_aiv" in r:label+=" H="+str(r["prior_aiv"])
        table.append(f"| {label} | {r['shape']} | {r['chip'].split('_')[0][6:]} | {fmt(cost)} | {fmt(known)} | {fmt(r['ca_us'])} | {err(cost)} | {err(known)} | {fmt(r['future_us'])} / {fmt(r.get('future_known_us'))} |")
    (OUT/"shape-errors.md").write_text("\n".join(table)+"\n")

if __name__=="__main__":
    main()
