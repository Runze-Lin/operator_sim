"""Pre-register and verify bounded development captures of the pinned vLLM extension."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/"development/validation/sku17"),str(ROOT)]
from campaign_collect import nbytes, stages
from development.validation.rms_ordered_verify.verify import text_bytes
sys.path.insert(0,str(ROOT/"development/validation/chunk_recovery/fla2691/v2"))
from verify_chrome import content

OUT = ROOT/"results/v0.23/hc-post"
BUILD = ROOT/"build/hcpost23"
VENDOR = BUILD/"vendors/hcpost"
RAW = ROOT/"runs/hcpost23"
CHIPS = ("Ascend950DT_9582","Ascend950PR_957b")
CASES = ((3,5120),(1,1024),(5,7168))
def dump(path, content):
    path.write_text(json.dumps(content,indent=2,ensure_ascii=False)+"\n")
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():
    return datetime.now(timezone.utc).isoformat()
def freeze():
    path = OUT/"development.json"
    if path.exists():
        raise FileExistsError(path)
    source = ROOT/"runs/workloads19/sources/full/csrc"
    kernel, = (BUILD/"kernel").glob("*.o")
    files = [f for f in (source/"moe/hc_post").rglob("*") if f.is_file()]
    files += [source/"torch_binding.cpp",BUILD/"host_probe",kernel,Path(__file__).with_name("host.cpp")]
    files += [p for p in VENDOR.rglob("*") if p.is_file() and not p.is_symlink()]
    dump(path,dict(utc=utc(),role="source geometry and development CA only",vllm_revision="c09d3e20b9c45c80b34ae3781440d90904a44023",
        kernel=kernel.name,hashes={str(p.relative_to(ROOT)):digest(p) for p in files},
        wall_s=360,raw_mib=1800,golden="BF16 bit equality to independent FP64 output rounded nearest-even, signed zero equivalent",
        cases=[dict(tokens=t,hidden=d,blocks=[4096]*(d//4096)+([d%4096] if d%4096 else []),aiv_dt=min(t,64),aiv_pr=min(t,56)) for t,d in CASES]))
def qualify(info, reg):
    t,d=info["shape"][1],info["shape"][-1]
    chip=info["chip"];raw=ROOT/info["raw"]
    try:
        st,checks=stages(dict(kernels=1,nchecks=1),chip,raw)
        k,=st
        assert k["kernel"].startswith(reg["kernel"][:-2]) and k["active_aiv"]==min(t,64 if chip==CHIPS[0] else 56) and k["active_aic"]==0, "kernel/active core mismatch"
        assert f"count={4*t*d} bad=0" in checks[0], "incomplete golden"
        base=(ROOT/k["trace"]).parent
        published=text_bytes(BUILD/"kernel"/reg["kernel"])
        assert text_bytes(base.parent/"dump/aicore_binary.o")==published,"captured .text differs"
        # dladdr may return the executable's AArch64 PLT entry. Resolve BOTH
        # undefined symbols in the dynamic loader with the same search path.
        launcher='export LD_LIBRARY_PATH="$1/op_api/lib:$LD_LIBRARY_PATH"; shift; exec env LD_DEBUG=bindings LD_BIND_NOW=1 "$@"'
        probe=subprocess.run(["bash",str(ROOT/"scripts/cann-run.sh"),"bash","-c",launcher,"verify",str(VENDOR),str(BUILD/"host_probe")],capture_output=True,text=True)
        bindings=[line.strip() for line in probe.stderr.splitlines() if "binding file "+str(BUILD/"host_probe")+" " in line and str(VENDOR/"op_api/lib/libcust_opapi.so") in line and "aclnnHcPost" in line]
        assert len(bindings)==2,"loader did not resolve both public calls to pinned vendor"
        merged=content(base/"trace.json",True)
        assert set(merged)==set(k["windows"]),"incomplete core roster"
        for core,group in merged.items():
            shard=content(base/core/"trace.json",False)[core]
            assert all(shard[field]==group[field] for field in ("first","last","by_instr","sync","ends","labels")), "shard content "+core
            assert len(shard["ends"])==len(shard["labels"])==1, "physical END and profiler label "+core
        all_ticks=max(g["last"] for g in merged.values())-min(g["first"] for g in merged.values())
        assert abs(k["ca_us"]-all_ticks/1650)<.001,"incomplete aggregate window"
        info.pop("reason",None)
        info.update(qualified=True,stage=k,checks=checks,ca_us=k["ca_us"],text_sha256=hashlib.sha256(published).hexdigest(),bindings=bindings,all_shards_end=True,chrome_sync_pairs=sum(g["be_count"] for g in merged.values()))
    except (ValueError,AssertionError,KeyError) as exc:
        info["reason"]=repr(exc)
    return info

def capture():
    reg=json.loads((OUT/"development.json").read_text())
    for name, expected in reg["hashes"].items():
        assert digest(ROOT/name)==expected, name
    launch='export ASCEND_CUSTOM_OPP_PATH="$1"; export LD_LIBRARY_PATH="$1/op_api/lib:$LD_LIBRARY_PATH"; shift; exec "$@"'
    for case in reg["cases"]:
        t,d=case["tokens"],case["hidden"]
        for chip in CHIPS:
            name=f"{chip}-{t}-{d}"
            result=OUT/(name+".json")
            if result.exists():
                previous=json.loads(result.read_text())
                if not previous["qualified"] and previous["cost"]["exit_code"]==0:
                    dump(result,qualify(previous,reg))
                if previous["qualified"]:
                    continue
                raise RuntimeError("Inspect previous failed capture: "+str(previous))

            raw=RAW/(name+"-"+datetime.now().strftime("%H%M%S"))
            raw.mkdir(parents=True)
            cmd=["bash",str(ROOT/"scripts/cann-run.sh"),"bash","-c",launch,"hcpost",str(VENDOR),"msopprof","simulator",
                 "--soc-version="+chip,"--dump=false","--launch-count=1","--timeout=6","--output="+str(raw/"profile"),str(BUILD/"host_probe"),str(t),str(d)]
            start=time.monotonic(); begin=utc(); reason=None
            with (raw/"run.log").open("w") as log:
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=raw,start_new_session=True)
                while proc.poll() is None:
                    if time.monotonic()-start>reg["wall_s"] or nbytes(raw)>reg["raw_mib"]*2**20:
                        reason="capture budget";os.killpg(proc.pid,signal.SIGKILL);break
                    time.sleep(2)
                code=proc.wait()
            cost=dict(started_utc=begin,finished_utc=utc(),seconds=time.monotonic()-start,exit_code=code,stop_reason=reason,raw_bytes=nbytes(raw))
            dump(raw/"cost.json",cost)
            info=dict(shape=[1,t,4,d],chip=chip,api="aclnnHcPost",qualified=False,role=reg["role"],cost=cost,raw=str(raw.relative_to(ROOT)))
            qualify(info,reg)
            dump(result,info)
            print(name,info.get("ca_us"),info.get("reason",""),flush=True)
            if not info["qualified"]:
                return 1
    return 0
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("freeze","capture"))
    args=parser.parse_args()
    sys.exit(freeze() if args.mode=="freeze" else capture())
