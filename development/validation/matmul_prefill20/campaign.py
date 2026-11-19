"""Freeze first, then capture all launches in one single-dispatch executable."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate
from development.validation.sku19.collect import nbytes
from campaign_collect import stages
OUT=ROOT/"results/v0.20/matmul-prefill"
RAW=ROOT/"runs/matmul_prefill20"
BIN=ROOT/"build/smoke/matmul_prefill20"
CHIPS=("Ascend950DT_9582","Ascend950PR_957b","Ascend970DT")
def now():
    return datetime.now(timezone.utc).isoformat()
def write(p,d):
    p.write_text(json.dumps(d,indent=2)+"\n")
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def freeze():
    p=OUT/"freeze.json"
    if p.exists():raise FileExistsError(p)
    projections=[]
    for t in (2048,262144,1048576,2097152):
        shape=[t,32,5120]
        for chip in CHIPS:
            start=time.perf_counter()
            e=estimate("matmul",shape,chip=chip,timing_mode="resource",dtype="bf16")
            ms=(time.perf_counter()-start)*1000
            projections.append(dict(shape=shape,chip=chip,us=e.latency_us,query_ms=ms,status=e.status,
                work=e.work,projection=e.projection))
    write(p,dict(utc=now(),runner_sha256=digest(BIN),source_sha256=digest(Path(__file__).with_name("matmul.cpp")),
        prediction_sources={f:digest(ROOT/"operator_model"/f) for f in ("matmul_resource.py","matmul_waves.py","matmul_v15.py")},
        projections=projections,measured_shape=[2048,32,5120],measured_chip=CHIPS[0],max_processes=1,
        wall_s=1800,raw_mib=4096,launch_filter=None,reference="BF16 ND X[M,K] and contiguous RHS[K,N]. Synthetic dense ±1, exact ±1280 BF16 output. Logical indexer.weights_proj geometry; weight transform and deployed runtime kernel not observed."))

def register_pr():
    prior=OUT/"freeze.json"
    registration=OUT/"pr-registration.json"
    if registration.exists():
        raise FileExistsError(registration)
    old=json.loads(prior.read_text())
    dt=json.loads((OUT/"ca.json").read_text())
    assert dt["qualified"] and digest(BIN)==old["runner_sha256"]
    row=next(r for r in old["projections"] if r["shape"]==old["measured_shape"] and r["chip"]==CHIPS[1])
    write(registration,dict(utc=now(),measured_chip=CHIPS[1],measured_shape=row["shape"],prediction_us=row["us"],
        original_predictions_utc=old["utc"],original_freeze_sha256=digest(prior),
        dt_observation_sha256=digest(OUT/"ca.json"),runner_sha256=old["runner_sha256"],source_sha256=old["source_sha256"],
        role="Cross-SKU paired validation elected after observing this geometry on DT. PR prediction frozen before DT/PR measurements; no PR observation used to choose timing constants.",
        max_processes=1,wall_s=2400,raw_mib=4096,launch_filter=None))

def collect(chip=CHIPS[0], registration="freeze.json", result_name="ca.json"):
    reg=json.loads((OUT/registration).read_text())
    assert digest(BIN)==reg["runner_sha256"]
    RAW.mkdir(exist_ok=False)
    cmd=["/usr/bin/time","-f","%U %S","-o",str(RAW/"cpu.txt"),"bash",str(ROOT/"scripts/cann-run.sh"),
         "msopprof","simulator","--soc-version="+chip,"--dump=false","--timeout="+str((reg["wall_s"]+59)//60),"--output="+str(RAW/"profile"),str(BIN)]
    start=time.perf_counter()
    env={**os.environ,"OMP_WAIT_POLICY":"PASSIVE","GOMP_SPINCOUNT":"0"}
    cost=dict(started_utc=now(),command=cmd[5:],stop_reason=None)
    peak=0
    with (RAW/"run.log").open("w") as log:
        proc=subprocess.Popen(cmd,cwd=RAW,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while proc.poll() is None:
                peak=max(peak,nbytes(RAW))
                if time.perf_counter()-start>reg["wall_s"] or peak>reg["raw_mib"]*2**20:
                    cost["stop_reason"]="wall or raw limit";os.killpg(proc.pid,signal.SIGKILL);break
                time.sleep(2)
        finally:
            if proc.poll() is None and not cost["stop_reason"]:
                os.killpg(proc.pid,signal.SIGTERM)
            proc.wait()
    cpu=(RAW/"cpu.txt").read_text().splitlines()
    cpu=cpu[-1].split() if cpu else []
    cost.update(seconds=time.perf_counter()-start,exit_code=proc.returncode,raw_bytes=nbytes(RAW),
        peak_bytes=max(nbytes(RAW),peak),cpu_seconds=sum(map(float,cpu)) if len(cpu)==2 else None,finished_utc=now())
    write(RAW/"cost.json",cost)
    qualify(cost,chip,result_name)


def qualify(cost,chip=CHIPS[0],result_name="ca.json"):
    job=dict(kernels=1,nchecks=1,id="matmul_prefill",op="matmul",shape=[2048,32,5120])
    result=dict(qualified=False,cost=cost)
    try:
        ss,correct=stages(job,chip,RAW)
        assert len(ss)==1 and ss[0]["kernel"].lower().startswith("matmul"), ss
        assert "correctness count=65536 bad=0" in (RAW/"run.log").read_text()
        result.update(qualified=True,stage=ss[0],correctness=correct)
    except (AssertionError,ValueError,KeyError,IndexError,OSError) as exc:
        result["reason"]=repr(exc)
    write(OUT/result_name,result)
    print("qualified",result['qualified'],result.get("reason"),flush=True)

if __name__=="__main__":
    if sys.argv[1]=="collect-pr":
        RAW=ROOT/"runs/matmul_prefill20-pr"
        collect(CHIPS[1],"pr-registration.json","ca-pr.json")
    elif sys.argv[1]=="register-pr":
        register_pr()
    elif sys.argv[1] == "resume-approved":
        # The CA and /usr/bin/time have their own process group and continue if
        # the interactive supervisor exits. Reattach to the same on-disk run.
        RAW=ROOT/"runs/matmul_prefill20-approved"
        cpu=RAW/"cpu.txt"
        reg=json.loads((OUT/"freeze.json").read_text())
        birth=subprocess.check_output(["stat","-c","%W",str(RAW/"run.log")],text=True)
        started=float(birth)
        if started<=0:
            raise RuntimeError("Missing original capture start time")
        peak=nbytes(RAW)
        while not cpu.stat().st_size:
            peak=max(peak,nbytes(RAW))
            if time.time()-started>reg["wall_s"] or peak>reg["raw_mib"]*2**20:
                raise RuntimeError("Existing capture reached budget; stop its registered process group")
            time.sleep(2)
        words=cpu.read_text().splitlines()
        codes=[v for v in words[:-1] if "status" in v or "signal" in v]
        done=cpu.stat().st_mtime
        cost=dict(started_utc=datetime.fromtimestamp(started,timezone.utc).isoformat(),
            finished_utc=datetime.fromtimestamp(done,timezone.utc).isoformat(),
            seconds=done-started, raw_bytes=nbytes(RAW),peak_bytes=max(nbytes(RAW),peak),
            cpu_seconds=sum(float(v) for v in words[-1].split()),exit_code=1 if codes else 0,
            stop_reason="; ".join(codes) or None,
            qualification="Recovered wall time from initial run.log file creation and GNU time completion (start precision 1 s); peak is observed sampling plus final directory size. No new launch.")
        write(RAW/"cost.json",cost)
        qualify(cost)
    elif sys.argv[1] == "collect-approved":
        old_cost=RAW/"cost.json"
        if not old_cost.exists():
            raise RuntimeError("Wait for the previous process to finish")
        first=OUT/"ca.json"
        if first.exists():
            if json.loads(first.read_text())["qualified"]:
                raise RuntimeError("A qualified run already exists")
            first.rename(OUT/"sandbox-attempt.json")
        RAW=ROOT/"runs/matmul_prefill20-approved"
        collect()
    else:
        {"freeze":freeze,"collect":collect}[sys.argv[1]]()
