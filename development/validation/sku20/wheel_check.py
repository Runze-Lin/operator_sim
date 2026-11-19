"""Verify the installed public API and time one query per distinct dispatch.

Run from outside the checkout using the wheel environment's python -I.
The checkout supplies only immutable test inputs and the output location.
"""
import gzip
from importlib.metadata import distribution as installed_distribution, version
import json
import math
from pathlib import Path
import statistics
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
NEW = RESULTS / "v0.20"
OLD = RESULTS / "v0.19"
CHIPS = {"dt":"Ascend950DT_9582", "pr":"Ascend950PR_957b", "future":"Ascend970DT"}
ANY_POSITIVE = object()


def read(path):
    return json.loads(path.read_text())


def distribution(values):
    ordered = sorted(values)
    return {"n": len(values), "median_ms": statistics.median(values),
            "p95_ms": ordered[math.ceil(.95*len(values))-1], "max_ms": ordered[-1]}


def main():
    if not sys.flags.isolated or Path.cwd().is_relative_to(ROOT):
        raise RuntimeError("Invoke the installed wheel with python -I from outside the checkout")
    start = perf_counter_ns()
    import operator_model
    from operator_model import estimate
    import_ms = (perf_counter_ns()-start)/1e6
    package = Path(operator_model.__file__).resolve()
    if package.is_relative_to(ROOT) or version("ascend-operator-model") != "0.20.0" or operator_model.__version__ != "0.20.0":
        raise RuntimeError(f"Expected the independently installed 0.20.0 release, imported {package}")

    seen, groups, by_operator = {}, {}, {}
    qualifications = {"real_prefill":0, "frozen_rms":0, "frozen_sigmoid":0}

    def check(op, shape, attrs, chip, mode, expected, group, *, status=None, aiv=None, dtype=None):
        key = (op, tuple(shape), json.dumps(attrs, sort_keys=True), chip, mode, dtype)
        if key not in seen:
            t = perf_counter_ns()
            result = estimate(op,shape,attrs,chip=chip,timing_mode=mode,dtype=dtype)
            ms = (perf_counter_ns()-t)/1e6
            seen[key] = (result,ms)
            by_operator.setdefault(op,[]).append(ms)
        result, ms = seen[key]
        groups.setdefault(group,set()).add(key)
        if status is not None and result.status != status:
            raise AssertionError((key, status, result.status, result.reason))
        if expected is None:
            assert result.latency_us is None and result.status in ("unsupported","unavailable"), (key,result)
        else:
            assert result.latency_us is not None and math.isfinite(result.latency_us) and result.latency_us > 0, (key,result)
            assert result.status in ("estimated","extrapolated") and result.work is not None, (key,result)
            if expected is not ANY_POSITIVE:
                assert math.isclose(expected,result.latency_us,rel_tol=1e-11,abs_tol=1e-13), (key,expected,result.latency_us)
        if aiv is not None:
            assert aiv == result.active_aiv, (key,aiv,result.active_aiv)
        return result

    # Preserve numeric compatibility on unchanged historical families. RMS, Quant,
    # Cast and new Sigmoid have their own dedicated fixtures.
    for row in read(RESULTS/"v0.18/evaluation.json")["rows"]:
        op = row["op"]
        if op in ("rmsnorm","quant"):
            continue
        mode = "resource" if op in ("matmul","fa") else "auto"
        for alias,chip in CHIPS.items():
            check(op,row["shape"],row["attrs"],chip,mode,row[alias+"_us"],"legacy")
    for job in read(OLD/"sku/freeze.json")["jobs"]:
        o = job["outputs"][0]
        if o["op"] in ("quant","rmsnorm"):
            for chip in CHIPS.values():
                check(o["op"],o["shape"],o["attrs"],chip,"resource",ANY_POSITIVE,"legacy_changed")
    for family in ("router","cast"):
        for row in read(OLD/"workloads"/(family+"-holdout-freeze.json"))["jobs"]:
            for chip,us in row["predictions"].items():
                check(row["op"],row["shape"],{},chip,"resource",us if family=="router" else ANY_POSITIVE,"legacy")
    for filename in ("attn-res-holdout-freeze.json","attn-res-extension-freeze.json"):
        for row in read(OLD/"workloads"/filename)["scenarios"]:
            attrs={"addend":True,"materialize":True,"write_idx":row["valid"]}
            shape=[row["tokens"],row["hidden"],row["slots"],row["valid"]]
            check("attn_res_fwd",shape,attrs,row["chip"],"resource",row["latency_us"],"legacy")
    for job in read(NEW/"rms/freeze.json")["jobs"]:
        for output in job["outputs"]:
            for alias,chip in CHIPS.items():
                check(output["op"],output["shape"],output["attrs"],chip,"resource",output[alias+"_us"],
                      "frozen_rms",status=output[alias+"_status"],aiv=output[alias+"_aiv"])
                qualifications["frozen_rms"] += 1
    for job in read(NEW/"newop/holdout-freeze.json")["jobs"]:
        for chip,r in job["predictions"].items():
            check(job["op"],job["shape"],{},chip,"resource",r["us"],"frozen_sigmoid",aiv=r["aiv"])
            qualifications["frozen_sigmoid"] += 1
    with gzip.open(NEW/"workloads/contracts.jsonl.gz","rt") as src:
        for line in src:
            for r in json.loads(line)["contracts"]:
                c = r.get("standalone_call")
                if c is None:
                    continue
                for chip,p in r["standalone_prediction"].items():
                    check(c["op"],c["shape"],c["attrs"],chip,"resource",p["us"],"real_prefill",status=p["status"])
                    qualifications["real_prefill"] += 1
    # BF16 projection component resource probes retain their dtype and op scope.
    for chip in CHIPS.values():
        for m in (262144,1048576,2097152):
            for op,shape in (("matmul",[m,32,5120]),("matmul",[m,2304,1536]),("quant",[m,7168])):
                for mode in ("resource","auto"):
                    check(op,shape,{},chip,mode,ANY_POSITIVE,"large_standalone_probes",dtype="bf16")
        for op,shape,attrs,dtype in (("unknown_op",[1],{},None),
                ("quant",[262144,17],{},None),("quant",[True,7168],{},None),
                ("matmul",[2097152,32,5120],{},"fp32"),("rmsnorm",[0,7168],{},None),
                ("rmsnorm",[64,7168],{"resident_aiv":100000},None),
                ("sigmoid_fp32",[262144,12],{},"bf16")):
            check(op,shape,attrs,chip,"resource",None,"invalid",status="unsupported",dtype=dtype)

    vals=[ms for _,ms in seen.values()]
    origin = json.loads(installed_distribution("ascend-operator-model").read_text("direct_url.json") or "{}")
    output = {"version":operator_model.__version__, "package_path":str(package),
        "installed_wheel_sha256":origin.get("archive_info",{}).get("hashes",{}).get("sha256"),
        "import_ms":import_ms, "protocol":"Installed noneditable wheel, python -I outside checkout, elapsed in-process wall per distinct public estimate call; input and archive IO are outside timers. Per-group measurements may overlap. Includes first parameter-file loads.",
        "overall":distribution(vals), "per_op":{op:distribution(v) for op,v in sorted(by_operator.items()) if op!="unknown_op"},
        "by_group":{name:distribution([seen[k][1] for k in keys]) for name,keys in groups.items()},
        "validated_entries_including_repeats":qualifications,
        "slowest":[{"op":k[0],"shape":k[1],"chip":k[3],"mode":k[4],"wall_ms":seen[k][1]}
            for k in sorted(seen,key=lambda k:seen[k][1],reverse=True)[:5]]}
    assert len(output["per_op"]) == 20, list(output["per_op"])
    (NEW/"wheel-smoke.json").write_text(json.dumps(output,indent=2)+"\n")
    print(output["overall"])
    assert output["overall"]["max_ms"] < 100, output["slowest"]


if __name__ == "__main__":
    main()
