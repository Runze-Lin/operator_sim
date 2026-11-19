"""Pair equal public RMS inputs, preserving provider, generated values and kernel identity."""
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate

OUT=ROOT/"results/v0.22/rms-unknown"
DT,PR="Ascend950DT_9582","Ascend950PR_957b"
REFERENCE=Path("toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_nn/rms_norm/RmsNorm_normal_all_bf16_high_performance.o")
HOSTS={"build/vector-v8/NORM":"period17","build/smoke/extension_norm_addmm":"period17",
       "build/smoke/large_vector":"sign","build/rms-host-recovery/large_vector_rms16":"sign"}
CASE=re.compile(r"case=rmsnorm api=aclnnRmsNorm R=(\d+) C=(\d+) dtype=BF16 epsilon=([\d.eE+-]+)")


def read(path):return json.loads(path.read_text())


@lru_cache(None)
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024**2),b""):h.update(b)
    return h.hexdigest()


@lru_cache(None)
def cases(log):
    return [(int(t),int(d),float(eps)) for t,d,eps in CASE.findall(log.read_text(errors="replace"))]


def scored(sku,s,h,ca):
    u=estimate("rmsnorm",s,chip=sku,dtype="bf16").latency_us
    known=estimate("rmsnorm",s,{"resident_aiv":h},chip=sku,dtype="bf16").latency_us
    return dict(chip=sku,shape=s,prior_aiv=h,ca_us=ca,unknown_us=u,known_us=known,
                unknown_error_pct=100*(u/ca-1),known_error_pct=100*(known/ca-1))


def official(root):
    hosts={sha(root/p):(p,g) for p,g in HOSTS.items()}
    repaired=read(ROOT/"results/v0.21/rms/host-recovery.json")
    for src,digest in repaired["files"].items():
        assert sha(root/src)==digest
    original_text=(root/"smoke/large_vector.cpp").read_text()
    old_gate='c<=0 || c>7168 || c%64'
    new_gate='c<=0 || (op=="rmsnorm" ? (c>8192 || c%16) : (c>7168 || c%64))'
    assert original_text.replace(old_gate,new_gate)==(root/"build/rms-host-recovery/large_vector_rms16.cpp").read_text()
    assert '#include "extension_norm_addmm.cpp"' in (root/"smoke/vector_v8.cpp").read_text()
    elf=sha(root/REFERENCE)
    rows=[]
    def add(s,sku,h,st,correct,runner,source,provider):
        assert correct and len(correct)==2 and all("bad=0" in c for c in correct)
        assert [int(re.search(r"count=(\d+)",c)[1]) for c in correct]==[s[0]*s[1],s[0]]
        assert st["kernel"]=="RmsNorm_normal_all_bf16_high_performance_5000" and st["binary_sha256"]==elf
        assert st["active_aic"]==0 and 0 < st["active_aiv"]
        lo=min(v[0] for v in st["windows"].values());hi=max(v[1] for v in st["windows"].values())
        assert abs((hi-lo)-st["ca_us"])<.00001
        trace=Path(st["trace"]);folder=Path(*trace.parts[:trace.parts.index("profile")])
        assert (*s,1.e-6) in cases(root/folder/"run.log")
        host,g=hosts[runner]
        rows.append(dict(**scored(sku,s,h,st["ca_us"]),provider=provider,generator=g,elf=elf,
             kernel=st["kernel"],trace=str(trace),trace_id=st["trace_sha256"],active_aiv=st["active_aiv"],
             source=str(source),host=host,host_sha256=runner))
    rs=Path("results")
    files=[rs/ver/sub/"records.json" for ver,sub in [("v0.17","sku"),("v0.18","sku"),("v0.19","sku"),
                          ("v0.19","rms-confirmation"),("v0.20","rms")]]
    for p in files:
        for r in read(ROOT/p):
            if r["op"]!="rmsnorm" or not r.get("stages"):continue
            assert r["scope"]=="one all-core window" and r.get("attrs",{}) in ({},{"dtype":"bf16"})
            stage,=r["stages"]
            command=r["cost"].get("command",[])
            assert not command or command[-3:]==["rmsnorm",*map(str,r["shape"])]
            add(r["shape"],r["chip"],0,stage,r["correctness"],r["runner_sha256"],p,"msopprof")
    original={j["id"]:j for j in read(ROOT/"results/v0.21/rms/freeze.json")["jobs"]}
    for sub in ("receipts","retries"):
        for p in sorted((ROOT/"results/v0.21/rms"/sub).rglob("rmsnorm_*.json")):
            r=read(p)
            if not r["qualified"]:continue
            o=r["observation"];st,=o["stages"]
            runner=o.get("runner_sha256") or original[r["id"]]["runner_sha256"]
            add(r["shape"],r["chip"],0,st,o["correctness"],runner,p.relative_to(ROOT),"msopprof")
    base=rs/"rms-dispatch-audit"
    runner=read(ROOT/base/"freeze.json")["files"]["build/vector-v8/NORM"]
    for sku in (DT,PR):
        for p in (base/("verified-ordered-"+sku+".json"),base/"prefix-recovery"/("verified-published-"+sku+".json")):
            d=read(ROOT/p)
            if "controls" in d:assert d["qualified"] and d["matched_prefix_identity"]
            seq=d["controls"]+d["scored_rows"] if "controls" in d else d["rows"]
            for r in seq:
                if not r["timing_qualified"]:continue
                add(r["shape"],sku,r["prior_aiv"],r["stage"],r["correctness"],runner,p,"msopprof")
    return rows


@lru_cache(None)
def dbinfo(path):
    with sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True) as db:
        return dict(db.execute("select * from KernelInfo"))


def compact(root):
    src=Path("results/rms-dispatch-audit/core-windows.json")
    get=[]
    for r in read(ROOT/src)["rows"]:
        get.append((r["shape"],r["source"],r["prior"],r["window"]/1650,src))
    src=Path("results/v0.10/wide/paired-controls.json")
    get.extend((r["shape"],r["db_source"],r["prior_aiv"],r["reference_us"],src)
               for r in read(ROOT/src) if r["op"]=="rmsnorm")
    for p in sorted((ROOT/"results/v0.12/rms").glob("recovered-*.json")):
        for r in read(p)["controls"]:
            assert int(r["audit"]["launch"])==0 and r["audit"]["passed"] and all("bad=0" in x for x in r["correctness"])
            get.append((r["shape"],r["source"],0,r["reference_us"],p.relative_to(ROOT)))
    rows=[]
    for s,rel,h,ca,p in get:
        path=root/rel;meta=dbinfo(path);ordinal=int(meta["launch_id"])
        folder=path.parents[3]
        ordered=cases(folder/"run.log")
        assert ordered[ordinal]==(*s,1.e-6)
        previous=sorted(path.parent.glob("*.db"))[:ordinal]
        assert max([0]+[int(dbinfo(x)["ai_core_active"]) for x in previous])==h
        assert meta["name"]=="entry_kernel_1388" and meta["frequency_ghz"]==1.65
        # Compare the raw min/max and actual record count with the completed
        # DB metadata. Full duration alone cannot identify an ELF image.
        with sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True) as db:
            n,a,b=db.execute("select count(*),min(ExecInstrTickStart),max(ExecInstrTickEnd) from ExecutedInstructions").fetchone()
        assert n==meta["instrs_exec_num"] and b-a==meta["exec_time_ticks"] and abs((b-a)/1650-ca)<.00001
        rows.append(dict(**scored(DT,s,h,ca),provider="npusim compact",generator="period17",
            elf=None,object_path=meta["object_path"],kernel=meta["name"],trace=rel,trace_id=sha(path),
            active_aiv=meta["ai_core_active"],source=str(p),launch=ordinal))
    return rows


def pairs(rows,keys):
    groups=defaultdict(list)
    for r in rows:
        key=tuple(r[k] if k!="shape" else tuple(r[k]) for k in keys)
        groups[key].append(r)
    out=[]
    for a in groups.values():
        if len(a)<2:continue
        a.sort(key=lambda r:r["ca_us"])
        l,h=a[0],a[-1]
        out.append(dict(chip=l["chip"],shape=l["shape"],n=len(a),endpoints=[dict(
            h=r["prior_aiv"],ca_us=r["ca_us"],provider=r["provider"],gen=r["generator"],
            elf=r["elf"],trace=r["trace"],trace_id=r["trace_id"]) for r in (l,h)],
            same_extreme_H=h["prior_aiv"]==l["prior_aiv"],
            different_H_in_set=len({r["prior_aiv"] for r in a})>1,
            opposite20_intervals_disjoint=.8*h["ca_us"]>1.2*l["ca_us"],
            opposite40_intervals_disjoint=.6*h["ca_us"]>1.4*l["ca_us"],
            minimax_abs_relative_pct=100*(h["ca_us"]-l["ca_us"])/(h["ca_us"]+l["ca_us"])))
    return sorted(out,key=lambda r:-r["minimax_abs_relative_pct"])


def summary(rows):
    def val(k):
        return dict(wape_pct=100*sum(abs(r[k]-r["ca_us"]) for r in rows)/sum(r["ca_us"] for r in rows),
                    max_abs_pct=max(100*abs(r[k]/r["ca_us"]-1) for r in rows))
    return dict(n=len(rows),unknown=val("unknown_us"),known=val("known_us"))


def main(root):
    rows={}
    for r in official(root)+compact(root):
        index=(r["chip"],r["trace_id"])
        if index in rows:
            old=rows[index]
            assert (old["ca_us"],old["prior_aiv"],old["shape"])==(r["ca_us"],r["prior_aiv"],r["shape"])
        else:rows[index]=r
    rows=list(rows.values())
    key=("chip","shape","generator","provider","kernel","elf")
    strong=pairs([r for r in rows if r["elf"]],key)
    weak=pairs([r for r in rows if not r["elf"]],key)
    cross=[g for g in pairs(rows,("chip","shape","generator")) if
           g["endpoints"][0]["provider"]!=g["endpoints"][1]["provider"]]
    over=[r for r in rows if abs(r["unknown_error_pct"])>40]
    cohorts={c+" "+p:summary([r for r in rows if r["chip"]==c and r["provider"]==p])
             for c,p in sorted({(r["chip"],r["provider"]) for r in rows})}
    data=dict(inputs=dict(layout="contiguous ND",x="BF16",gamma="BF16 [D]",y="BF16",rstd="FP32 [T,1]",
                          epsilon=1e-6,axis="last/full row",
                          generators={"period17":"x[i]=((i%17)-8)/16; gamma[j]=(((j+3)%17)-8)/16",
                                      "sign":"x[i]=(-1)^i; gamma[j]=1"}),
            evidence="Historical observations; no CA or parameter changes. Different providers are compared only as weak evidence.",
            rows=rows,strong_pairs=strong,compact_only_pairs=weak,cross_provider_pairs=cross,cohorts=cohorts,
            unknown_over_40=over,over_40_summary=summary(over))
    (OUT/"audit.json").write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--raw-root",type=Path,required=True)
    main(p.parse_args().raw_root.resolve())
