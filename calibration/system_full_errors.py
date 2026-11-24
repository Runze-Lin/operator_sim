"""Publish contract-scoped, repeatable error summaries without assigning a proxy's CA to its parent."""
import csv
from collections import defaultdict
from pathlib import Path
import json
from calibration.system_full_report import ROOT, REF, DOC, load

CA=ROOT/"calibration"
DT,PR="Ascend950DT_9582","Ascend950PR_957b"
KEYS={
    "quant_mxfp4":"npu::npu_dynamic_mx_quant", "quant_token_fp8":"inference::dynamic_mx_quant_fp8_reference",
    "cache_load_mxfp4_antimx":"inference::cache_load","routed_grouped_matmul_mxfp4_v5":"npu::npu_grouped_matmul",
    "matmul_mx_native":"npu::npu_quant_matmul","sigmoid_fp32":"aten.sigmoid.default",
    "vllm_hc_post":"_C_ascend::npu_hc_post","rmsnorm · msopprof":"npu::npu_rms_norm",
    "cast_bf16_fp32":"aten._to_copy.default","moe_gating_top_k":"_C_ascend::moe_gating_top_k",
}
VECTOR={"add":"aten.add.Tensor","mul":"aten.mul.Tensor","swiglu":"npu::npu_swiglu",
        "cache_load_fp8":"inference::cache_load","rope":"_C_ascend::inplace_partial_rotary_mul",
        "mhc_pre_sinkhorn":"_C_ascend::npu_hc_pre_v2","moe_gating_top_k_hash":"_C_ascend::moe_gating_top_k_hash"}
ATT={"conv":"vllm_ascend::causal_conv1d_update_reference","fia":"npu::npu_fused_infer_attention_score"}


def doc_json(dir,name):
    return json.loads((CA/dir/name).read_text())


def samples():
    rows=[]
    def add(raw,scope,sku,role,shape,pred,actual,ref):
        rows.append(dict(raw_name=raw,checked_contract=scope,chip=sku,role=role,fixture=json.dumps(shape),
                         predicted_us=float(pred),ca_us=float(actual),source=ref))
    for line in (REF/"windows.jsonl").read_text().splitlines():
        r=json.loads(line);c=r["contract"]
        if c in KEYS and r["predicted_us"] is not None:
            add(KEYS[c],c,r["chip"],"independent" if r["data_role"]=="independent" else "replay",
                r["shape"],r["predicted_us"],r["ca_us"],"reference/windows.jsonl")
    for r in csv.DictReader((CA/"system_vector_ca/primitives.csv").open()):
        role = "replay" if int(r["rows"])==9 else "size_holdout"
        add(r["raw_name"],r["api"],r["chip"],role,r["input_shapes"],r["predicted_us"],r["ca_us"],"system_vector_ca/primitives.csv")
    for r in doc_json("system_vector_ca","windows.json"):
        add(VECTOR[r["op"]],r["op"],r["chip"],"independent" if r["sample_role"]=="postfreeze_native_check" else "replay",
            r["shape"],r["predicted_us"],r["measured_us"],"system_vector_ca/windows.json")
    freeze=doc_json("system_cube_ca","frozen.json")
    runs=[("measured.json",freeze["development"],"replay"),("independent.json",freeze["independent"],"independent")]
    router=doc_json("system_cube_ca","router-current.json")
    for fname,key,role in (("router.json","development","replay"),("router-independent.json","independent","independent")):
        if (CA/"system_cube_ca"/fname).is_file():
            runs.append((fname,{k:[r] for k,r in router[key].items()},role))
    for fname,table,role in runs:
        data=doc_json("system_cube_ca",fname)
        for sku,results in data.items():
            for r in results["cases"]:
                shape=r["shape_MNK"]
                pred,=[p for p in table[sku] if p["shape_MNK"]==shape]
                raw="inference::matmul_rhs_transposed" if fname.startswith("router") else ("aten.addmm.default" if r["api"]=="aclnnAddmm" else "aten.mm.default")
                add(raw,r["api"]+" "+r["dtype"],sku,role,shape,pred["predicted_us"],r["ca_us"],"system_cube_ca/"+fname)
    for r in doc_json("system_attention_ca","development.json"):
        add(ATT[r["operator"]],r["operator"]+" BF16",r["sku"],"replay",r["shape"],
            r["model_us"],r["device_us"],"system_attention_ca/development.json")
    fp=doc_json("system_attention_ca","frozen.json")["cases"]
    for r in doc_json("system_attention_ca","independent.json"):
        p,=[k["model_us"] for k in fp if k["sku"]==r["sku"] and k["operator"]==r["operator"]]
        add(ATT[r["operator"]],r["operator"]+" BF16",r["sku"],"independent",r["shape"],p,r["device_us"],"system_attention_ca/independent.json")
    return rows


def metrics(data,sku):
    s=[r for r in data if r["chip"]==sku]
    if not s:return "N/A"
    abs_errors=[abs(r["ca_us"]-r["predicted_us"]) for r in s]
    return f"{100*sum(abs_errors)/sum(r['ca_us'] for r in s):.1f}% / {100*max(a/r['ca_us'] for a,r in zip(abs_errors,s)):.1f}%"


def main():
    records=samples()
    with (REF/"system-full-ca.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator="\n");w.writeheader();w.writerows(records)
    groups=defaultdict(list)
    for r in records: groups[r["raw_name"]].append(r)
    appendix=["## Summary · 27 组系统图中的完整调用","",
        "原始名可以对应多条实现。下表的误差只属于已列合同及其代表 shape。列数为双端合计窗口数；先选冻结独立结果，无该组则报告预留尺寸回放，再无则给开发回放。两个芯片均使用同一数据角色。未配对的库组件、设计和结构零成本用 N/A。普通算子独立留出仍见上表；这里也列出本轮新增的矩阵、卷积、路由和基础操作。",
        "","| 原始调用 | 可计分的代表合同 | 回放 / 留尺寸 / 冻结独立窗 | 本表计分集 | 950DT WAPE / 最大 | 950PR WAPE / 最大 |",
        "|---|---|---:|---|---:|---:|"]
    for op in load("system-full-operators.csv"):
        name=op["raw_name"];data=groups[name]
        roles=["replay","size_holdout","independent"]
        sub=[[r for r in data if r["role"]==role] for role in roles]
        scored=next((s for s in reversed(sub) if s),[])
        role_label={"replay":"开发/复用回放","size_holdout":"留尺寸","independent":"冻结后"}
        summary="、".join(sorted({r["checked_contract"] for r in scored})) or "见路径证据，整窗 N/A"
        label=role_label[scored[0]["role"]] if scored else "N/A"
        appendix.append(f'| `{name}` | {summary} | {" / ".join(str(len(s)) for s in sub)} | {label} | {metrics(scored,DT)} | {metrics(scored,PR)} |')
    appendix+=["","同一名称的多种精度、布局及组合计时边界见逐合同例页。小样本的一次冻结命中不能建立整个 dtype 或长上下文的精度保证。预留尺寸一列汇总 144 行同族验证；其中一部分采集发生在参数冻结之前，其余在冻结后补采。CA 为安装包与列明探针的完整设备窗，系统 950DT-system 与 970DT 的配置变化依然属于外推。",
        "","[逐 shape 的模型时间和 CA 真值 CSV](../calibration/reference/system-full-ca.csv) · [80 个调用的精度、证据及最大查询墙钟](system-full-evidence.md) · [全部图 shape 定价](../calibration/reference/system-full-shapes.csv.gz) · [全量端到端对比](system-full-comparison.md)","",""]
    p=DOC/"error-analysis.md";s=p.read_text()
    start=s.find("## Summary · 27 组系统图中的完整调用")
    stop=s.index("## 设备窗与指标")
    p.write_text(s[:start if start!=-1 else stop]+"\n".join(appendix)+s[stop:])


if __name__=="__main__":
    main()
