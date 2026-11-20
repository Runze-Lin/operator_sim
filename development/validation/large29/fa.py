"""Long KV verification of the shipped dense FA, one shared CA slot and no duplicate 128Ki DT."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/"results/v0.29/large/fa"
RAW = ROOT/"runs/large29"
REG = OUT/"freeze.json"
sys.path.insert(0,str(ROOT))
from operator_model import estimate
from development.validation.coverage25 import fa_holdout
sys.path.insert(0,str(Path(fa_holdout.__file__).parent))
from development.validation.coverage25.fa_exit_proof import verify

ca=fa_holdout.ca
DT,PR,FUT=fa_holdout.DT,fa_holdout.PR,fa_holdout.FUT


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    temp=path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def load(p):
    return json.loads(p.read_bytes())


def freeze():
    if REG.exists() or (RAW/"fa").exists():
        raise FileExistsError("The long-KV jobs already have a registration")
    archive=load(ROOT/"results/v0.29/replay.json")["rows"]
    old=[r for r in archive if r["op"]=="fa" and r["shape"][1]>=32768 and r["shape"][2:]==[64,1,1]]
    keys={(r["chip"],tuple(r["shape"])) for r in old}
    jobs=[]
    for kv in (131072,262144,524288):
        shape=[9,kv,64,1,1]
        predictions={}
        for c in (DT,PR,FUT):
            p=estimate("fa",shape,chip=c)
            predictions[c]=dict(us=p.latency_us,aic=p.active_aic,aiv=p.active_aiv,work=p.work,
                                pages=(kv+255)//256,expected_mmad=p.projection["timing_diagnostics"]["mmads_total"])
        for chip in (PR,DT):
            if (chip,tuple(shape)) in keys:
                continue
            jobs.append(dict(id=f"fa-{kv}-{chip}",shape=shape,chip=chip,predictions=predictions,
                wall_s=(kv//131072)*14400,raw_gib=(kv//131072)*12))
    assert len(jobs)==5
    inputs=[Path(__file__),Path(ca.__file__),ROOT/"smoke/three_ops.cpp",ROOT/"build/smoke/three_ops",
            ROOT/"scripts/cann-run.sh",ROOT/"operator_model/fa_resource.py",ROOT/"operator_model/fa_transfer_v15.py",
            ROOT/"results/v0.16/fa.json"]+[ROOT/"configs/chips"/(c+".json") for c in (DT,PR,FUT)]
    with REG.open("x") as f:
        json.dump(dict(utc=now(), scope="BF16 independent noncausal SBH, Q9, KV axis, D64 B1 H1. "
            "Not the deployed sparse/MLA path. DT 128Ki already qualified; never repeat it. "
            "One prospective official profiling attempt per listed endpoint, all output golden, "
            "all core windows and physical ENDs, exact SKU and installed kernel ELF. "
            "Future target is extrapolation only. Case-level physical shared slot.",archive=old,jobs=jobs,
            inputs_sha256={str(p.relative_to(ROOT)):ca.sha(p) for p in inputs}),f,indent=2);f.write("\n")


def capture(j,d):
    cap=j["wall_s"]
    args=["/usr/bin/time","-f","%U %S","-o",str(d/"cpu.txt"),"bash",str(ROOT/"scripts/cann-run.sh"),
          "msopprof","simulator","--soc-version="+j["chip"],"--dump=false","--launch-count=1",
          "--timeout="+str((cap+59)//60),"--output="+str(d/"profile"),str(ROOT/"build/smoke/three_ops"),
          "fa",*map(str,j["shape"])]
    start=time.monotonic();stamp=now();peak=0;stop=None
    with (d/"run.log").open("w") as stdout:
        with subprocess.Popen(args,cwd=d,stdout=stdout,stderr=subprocess.STDOUT,start_new_session=True) as proc:
            try:
                while proc.poll() is None:
                    peak=max(peak,ca.nbytes(d))
                    if time.monotonic()-start>cap or peak>j["raw_gib"]*2**30:
                        stop="wall cap" if time.monotonic()-start>cap else "raw cap"
                        os.killpg(proc.pid,signal.SIGKILL)
                        break
                    time.sleep(5)
            finally:
                if not stop and proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                proc.wait()
    save(d/"cost.json", dict(seconds=time.monotonic()-start,started_utc=stamp,finished_utc=now(),
         peak_bytes=max(peak,ca.nbytes(d)),stop_reason=stop,exit_code=proc.returncode,command=args[5:]))


def collect():
    reg=load(REG)
    for path,sha in reg["inputs_sha256"].items():
        if ca.sha(ROOT/path)!=sha:
            raise ValueError(f"Frozen input changed: {path}")
    here=RAW/"fa";here.mkdir(parents=True,exist_ok=True)
    os.nice(12);os.environ.update(OMP_WAIT_POLICY="PASSIVE",GOMP_SPINCOUNT="0")
    def state(message,**fields):
        save(OUT/"state.json",dict(utc=now(),phase=message,pid=os.getpid(),**fields))
    with (here/"worker.lock").open("a") as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for j in reg["jobs"]:
            receipt=OUT/(j["id"]+".json");d=here/j["id"];chip=j["chip"]
            if receipt.exists():continue
            state("waiting for shared CA slot",case=j["id"])
            with (RAW/"shared.lock").open("a") as slot:
                fcntl.flock(slot,fcntl.LOCK_EX)
                if not (d/"cost.json").exists():
                    if d.exists():
                        state("interrupted raw retained",case=j["id"]);return
                    if shutil.disk_usage(ROOT).free < 2**30*(j["raw_gib"]+16):
                        state("insufficient raw disk budget",case=j["id"]);return
                    d.mkdir();state("acquiring",case=j["id"])
                    capture(j,d)
                try:
                    cost=load(d/"cost.json");assert cost["started_utc"]>reg["utc"]
                    stage,checks=ca.stages(dict(nchecks=1,kernels=1),chip,d);s,=stage
                    assert "elements=576 " in checks[0] and s["kernel"].startswith("FlashAttentionScore_") and s["mmad"]>0
                    import re
                    name=re.fullmatch(r"(FlashAttentionScore_[0-9a-f]+)_\d+_mix_aic",s["kernel"])
                    if not name:raise ValueError(s["kernel"])
                    package=ROOT/"toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/flash_attention_score_apt"
                    if ca.sha(package/(name[1]+".o"))!=s["binary_sha256"]:
                        raise ValueError("No matching installed kernel ELF for "+name[1])
                    item=dict(case_id=j["id"],qualified=True,chip=chip,shape=j["shape"],stages=stage,
                              ca_us=s["ca_us"],correctness=checks,cost=cost,freeze_sha256=ca.sha(REG))
                    temp=d/"qualified-initial.json";save(temp,item)
                    proof=verify(temp,{(s["kernel"],s["binary_sha256"])})
                    item["physical_core_count"]=len(proof["cores"])
                    item["signed_pct"]=100*(j["predictions"][chip]["us"]/s["ca_us"]-1)
                except (AssertionError,ValueError,KeyError,OSError,IndexError,TypeError) as exc:
                    item=dict(qualified=False,ca_us=None,reason=repr(exc),cost=load(d/"cost.json"),chip=chip,
                              shape=j["shape"],freeze_sha256=ca.sha(REG))
                save(receipt,item);render()
            time.sleep(2)
        state("finished")


def render():
    reg=load(REG)
    rows=["# Dense FlashAttentionScoreV2 的长 KV 验证","",
          "SBH 的输入轴为 [Q,KV,D,B,H]。该独立库合同测一次 MTP8 校验的 Q=9、本卡单头 D=64。每 256 个 KV 位置一页，32Ki、128Ki、256Ki、512Ki 分别推进 128、512、1024、2048 页。这些窗口验证长循环，实际模型的压缩稀疏注意力和 MLA 各自需要对应实现。","",
          "| KV | 芯片 | 页数 | 模型 / CA µs | 有符号误差 | 970DT 外推 µs | 窗口与角色 |",
          "|---:|---|---:|---:|---:|---:|---|"]
    for r in reg["archive"]:
        p=r["current_us"];y=r["ca_us"];kv=r["shape"][1]
        rows.append(f"| {kv:,} | {r['chip']} | {(kv+255)//256} | {p:.3f} / {y:.3f} | {100*(p/y-1):+.2f}% | {r['future_970dt_extrapolated_us']:.3f} | 已有完整窗，{r['kernel']} |")
    for j in reg["jobs"]:
        target=j["chip"];path=OUT/(j["id"]+".json")
        q=load(path) if path.exists() else {}
        p=j["predictions"][target];y=q.get("ca_us")
        measured=f"{y:.3f}" if y else "—"
        err=f"{100*(p['us']/y-1):+.2f}%" if y else "—"
        status=("完整新窗" if y else "未合格") if path.exists() else "冻结后待测"
        rows.append(f"| {j['shape'][1]:,} | {target} | {p['pages']} | {p['us']:.3f} / {measured} | {err} | {j['predictions'][FUT]['us']:.3f} | {status} |")
    rows+=["","每端点单次采集。128Ki、256Ki、512Ki 的墙钟/原始盘上限分别为 4h/12GiB、8h/24GiB、16h/48GiB。已返回的 DT 128Ki 采用原 compact CA 指令数据库与全输出检查。新端点只接受精确 SKU、正式安装包 ELF、全输出 golden、全核物理 END 与完整指令窗。预测不读取目标 trace。[采前冻结](freeze.json) · [后台进度](state.json) · [小形的逐页流水](../../../../docs/examples/fa.md)"]
    (OUT/"report.md").write_text("\n".join(rows)+"\n")


if __name__=="__main__":
    {"freeze":freeze,"collect":collect,"report":render}[sys.argv[1]]()
