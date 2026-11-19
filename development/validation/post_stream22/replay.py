"""Historical, ordered mHC split-host replay. No training, simulator or new measurement."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import OperatorStream, estimate

OUT=ROOT/"results/v0.22/post-stream"
REG=Path("results/v0.17/sku/records.json")
COMPACT=Path("results/v0.12/evaluation.json")
DT,PR="Ascend950DT_9582","Ascend950PR_957b"
PRE,SINK,POST="mhc_pre","mhc_sinkhorn","mhc_post"
POST_ATTRS={"dtype":"bf16","has_h_res":True}


def read(path):
    return json.loads(path.read_text())


def metrics(rows,model,observed):
    if not rows:return None
    worst=max(rows,key=lambda r:abs(r[model]/r[observed]-1))
    return dict(n=len(rows),shapes=len({tuple(r["shape"]) for r in rows}),
        wape_pct=100*sum(abs(r[model]-r[observed]) for r in rows)/sum(r[observed] for r in rows),
        max_abs_pct=100*abs(worst[model]/worst[observed]-1),worst_shape=worst["shape"])


def ordered(shape,chip,stages):
    stream=OperatorStream(chip)
    t,n,d=shape
    spec=[(PRE,shape,{"dtype":"bf16","has_gamma":False,"alpha_count":3,"return_intermediates":True}),
          (SINK,[t,n],{"dtype":"fp32","num_iters":20,"return_intermediates":False}),
          (POST,shape,POST_ATTRS)]
    assert len(stages)==len(spec)
    result=[]
    for (op,s,attrs),raw in zip(spec,stages):
        r=stream.advance(op,s,attrs)
        if r.latency_us is None:raise ValueError((op,s,r.reason))
        result.append(dict(op=op,shape=s,attrs=attrs,us=r.latency_us,predicted_aiv=r.active_aiv,**raw))
    us=r.latency_us
    diag=r.projection["timing_diagnostics"]
    auto=estimate(POST,shape,POST_ATTRS,chip=chip)
    assert diag["context"]=="unknown" and us==auto.latency_us
    sink_count=stages[1]["active_aiv"]
    carry=estimate(POST,shape,{**POST_ATTRS,"resident_aiv":sink_count},chip=chip)
    return dict(chip=chip,shape=shape,public_context=diag["context"],post_stream_us=us,
                post_shape_only_us=auto.latency_us,post_ca_us=stages[-1]["ca_us"],
                explicit_sink_count_sensitivity_us=carry.latency_us,
                explicit_context=carry.projection["timing_diagnostics"]["context"],
                chain_stream_us=sum(s["us"] for s in result),
                chain_ca_us=sum(s["ca_us"] for s in stages),stages=result)


def audit_log(path,shape):
    log=path.read_text(errors="replace")
    t,n,d=shape
    assert f"case=mhc api=aclnnMhcPre T={t} N={n} D={d}" in log and "SMOKE_PASS" in log
    for output in ("chain_sinkhorn","chain_post","chain_end_to_end"):
        assert any("correctness name="+output+" " in l and "bad=0" in l for l in log.splitlines())


def main(raw):
    chains=[];requalified=[];isolated=[];repeated=[]
    for r in read(ROOT/REG):
        if r["op"] not in ("mhc",POST) or not r.get("stages"):continue
        assert r["host_batch_row"] is None
        s=r["shape"]
        assert r["host_args"]==[r["op"],*map(str,s)] and all("bad=0" in k for k in r["correctness"])
        if r["op"]==POST:
            st,=r["stages"]
            isolated.append(dict(chip=r["chip"],backend="msopprof",shape=s,
                us=OperatorStream(r["chip"]).advance(POST,s,POST_ATTRS).latency_us,
                ca_us=st["ca_us"],active_aiv=st["active_aiv"],kernel=st["kernel"],binary=st["binary_sha256"],
                trace=st["trace_sha256"],source=str(REG)))
            continue
        assert len(r["correctness"])==9
        assert r["api"]=="aclnnMhcPre + aclnnMhcSinkhorn + aclnnMhcPost"
        audit_log(raw/r["run_log"],s)
        stages=[{k:x[k] for k in ("ca_us","kernel","active_aiv","active_aic","binary_sha256","trace_sha256")}
                for x in r["stages"]]
        assert [x["kernel"].split("_")[0] for x in stages]==["MhcPre","MhcSinkhorn","MhcPost"]
        assert math.isclose(sum(x["ca_us"] for x in stages),r["ca_us"])
        chains.append(dict(**ordered(s,r["chip"],stages),backend="msopprof",
            source=str(REG),host_log=r["run_log"],runner_sha256=r["runner_sha256"]))
    for r in read(ROOT/COMPACT)["rows"]:
        if r["op"]!="mhc":continue
        p=raw/r["source"];run=read(p);s=r["shape"]
        assert list(map(str,run["case"]["arguments"]))==["mhc",*map(str,s)]
        if not run.get("measurement"):
            # The final v0.8 publisher recovered two new Pre entry IDs by
            # checking the actual ordered host fences. Verify the same raw
            # marks and each archived kernel DB digest before including them.
            assert run.get("error")=="unverified Pre stage identity"
            official_log,=p.parent.glob("profile/npusim_*/npusim.log")
            sections=re.split(r"correctness name=(?:hin|chain_sinkhorn|chain_post) count=",official_log.read_text())
            tasks=[set(re.findall(r"\[TASK_DONE\].*stream_id:(\d+), task_id:(\d+) TASK_DONE",v)) for v in sections]
            assert len(tasks)==4 and all(len(group)==1 for group in tasks[:3]) and not tasks[3]
            launches=set.union(*tasks[:3])
            assert len(launches)==3  # Provider task IDs can differ across ACLNN launches.
            dbs=sorted(p.parent.glob("profile/npusim_*/record/*.db"))
            assert len(dbs)==len(r["db_sha256"])==3
            for db,digest in zip(dbs,r["db_sha256"]):
                actual=hashlib.sha256()
                with db.open("rb") as f:
                    for part in iter(lambda:f.read(1024**2),b""):
                        actual.update(part)
                assert actual.hexdigest()==digest
            requalified.append(dict(shape=s,source=r["source"],initial_reason=run["error"],
                                    correction="scripts/fusion-v8-report.py",
                                    raw_fences=[next(iter(t)) for t in tasks[:3]],db_sha256=r["db_sha256"]))
        audit_log(p.parent/"run.log",s)
        assert len(r["correctness"])==9 and all("bad=0" in k for k in r["correctness"])
        assert [x["launch_id"] for x in r["kernel_records"]]==[0,1,2]
        assert len(r["db_sha256"]) == len(r["kernel_records"]) == 3
        stages=[dict(ca_us=x["reference_us"],active_aiv=x["active_aiv"],active_aic=x["active_aic"],
                    kernel=x["kernel"],trace_sha256=digest,binary_sha256=None)
                for x,digest in zip(r["kernel_records"],r["db_sha256"])]
        assert math.isclose(sum(x["ca_us"] for x in stages),r["reference_us"])
        chains.append(dict(**ordered(s,DT,stages),backend="npusim compact",source=str(COMPACT),
                           host_log=str((p.parent/"run.log").relative_to(raw))))
    assert len({(r["chip"],r["stages"][-1]["trace_sha256"]) for r in chains})==len(chains)

    # First-launch compact controls have an audited original process high-water
    # mark of zero. Their short entry_kernel labels cannot prove binary identity.
    phase=Path("results/post-candidate/phase-evidence.json")
    for r in read(ROOT/phase):
        if (r["backend"],r["state"])!=("npusim compact","first"): continue
        assert r["prior_aiv"]==0
        isolated.append(dict(chip=r["chip"],backend=r["backend"],shape=r["shape"],
            us=OperatorStream(r["chip"]).advance(POST,r["shape"],POST_ATTRS).latency_us,ca_us=r["us"],
            active_aiv=r["active_aiv"],kernel=r["kernel"],binary=None,trace=r["trace"],source=str(phase)))
    matches=[]
    for ch in chains:
        s=ch["stages"][-1]
        for cold in isolated:
            if (cold["shape"],cold["backend"],cold["chip"],cold["active_aiv"],cold["kernel"],cold["binary"])!=(
                    ch["shape"],ch["backend"],ch["chip"],s["active_aiv"],s["kernel"],s["binary_sha256"]):continue
            matches.append(dict(chip=ch["chip"],backend=ch["backend"],binary_match=bool(cold["binary"]),
                shape=ch["shape"],chain_post_us=ch["post_stream_us"],chain_post_ca_us=ch["post_ca_us"],
                cold_post_us=cold["us"],cold_post_ca_us=cold["ca_us"],cold_trace=cold["trace"],
                chain_trace=s["trace_sha256"],binary=cold["binary"]))
    for chip in (DT,PR):
        file=Path("results/post-candidate")/("paired-"+chip+".json");rec=read(ROOT/file)
        shape=rec["shape"];stream=OperatorStream(chip)
        order=(raw/"runs/post-candidate-repeat"/chip/"cases.txt").read_text().splitlines()
        assert [line.split() for line in order]==[[POST,str(i),*map(str,shape)] for i in (0,1)]
        assert rec["qualified"] and len(rec["rows"])==2
        for i,row in enumerate(rec["rows"]):
            assert i==row["ordinal"] and "bad=0" in row["correctness"]
            p=stream.advance(POST,shape,POST_ATTRS);state=p.projection["timing_diagnostics"]["context"]
            assert state==row["condition"]
            repeated.append(dict(chip=chip,shape=shape,ordinal=i,state=state,us=p.latency_us,
                ca_us=row["stage"]["ca_us"],source=str(file),kernel=row["stage"]["kernel"],
                trace_sha256=row["stage"]["trace_sha256"],binary_sha256=row["stage"]["binary_sha256"]))

    summary=[]
    for chip,back in sorted({(j["chip"],j["backend"]) for j in chains}):
        j=[r for r in chains if (r["chip"],r["backend"])==(chip,back)]
        matched=[r for r in matches if (r["chip"],r["backend"])==(chip,back)]
        summary.append(dict(chip=chip,backend=back,post=metrics(j,"post_stream_us","post_ca_us"),
                            chain_sum=metrics(j,"chain_stream_us","chain_ca_us"),
                            force_last_sink_aiv=metrics(j,"explicit_sink_count_sensitivity_us","post_ca_us"),
                            matched_chain_post=metrics(matched,"chain_post_us","chain_post_ca_us"),
                            matched_isolated_post=metrics(matched,"cold_post_us","cold_post_ca_us")))
    scope="Historical diagnostics only. Each chain uses a fresh public stream, three measured launches, " \
          "physical per-launch all-core windows and no host gaps. Exact inter-family residency is unobserved."
    payload=dict(scope=scope,summary=summary,chains=chains,matched_protocols=matches,
                 requalified_compact_receipts=requalified,same_family_controls=repeated,
                 same_family_by_sku={c:metrics([r for r in repeated if r["chip"]==c],"us","ca_us") for c in (DT,PR)},
                 source_sha256={str(p):hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                    [Path("smoke/extension_indexer_mhc.cpp"),COMPACT,REG,phase,Path("operator_model/stream.py")]})
    (OUT/"replay.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--raw-root",type=Path,default=ROOT)
    main(p.parse_args().raw_root.resolve())
