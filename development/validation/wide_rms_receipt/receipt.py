"""Audit the completed v0.20 pair using its published receipts and small host logs."""
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model import __version__, estimate

CAMPAIGN = ROOT / "results/v0.20/rms"
OUT = ROOT / "results/v0.21/wide-rms/receipt.json"
SHAPE = [32768, 1280]


def read(path):
    return json.loads(path.read_text())


def digest_small(path):
    assert path.stat().st_size < 16 * 1024**2
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed_pct(pred, measured):
    return 100 * (pred / measured - 1)


def error_summary(measured, predicted):
    residuals = [abs(x - y) for x, y in zip(measured, predicted)]
    return dict(n=len(measured), wape_pct=100 * sum(residuals) / sum(measured),
                max_abs_pct=max(100 * r / x for r, x in zip(residuals, measured)))


def main():
    freeze_path = CAMPAIGN / "freeze.json"
    freeze = read(freeze_path)
    pairs = read(CAMPAIGN / "pairs.json")
    records = read(CAMPAIGN / "records.json")
    job, = [j for j in freeze["jobs"] if j["shape"] == SHAPE]
    pair, = [p for p in pairs if p["id"] == job["id"]]
    assert len(records) == 2 * len(pairs) == 2 * len(freeze["jobs"])
    assert len({tuple(p["shape"]) for p in pairs}) == len(pairs)
    assert job["nchecks"] == 2 and job["role"] == "holdout"
    assert __version__ == "0.21.0"
    endpoints = {}
    cohort = {}
    for side in ("dt", "pr"):
        r = pair[side]
        assert r in records and r["scope"] == "one all-core window"
        raw = ROOT / "runs/sku20-rms" / (job["id"] + "_" + r["chip"])
        qualified = raw / "qualified.json"
        assert read(qualified) == r
        outcome = read(raw / "outcome.json")
        assert outcome["eligible"] and outcome["exit_code"] == 0 and not outcome["stop_reason"]
        assert freeze["utc"] < outcome["started_utc"] < outcome["finished_utc"]
        log = (raw / "run.log").read_text()
        assert r["correctness"] == ["correctness count=41943040 bad=0", "correctness count=32768 bad=0"]
        assert all(x in log for x in [*r["correctness"], "SMOKE_PASS", "All task success.",
                   "Profiling data parse finished.", "runtime_soc=" + r["chip"],
                   "hbm3_8000M.ini" if side == "dt" else "lpddr5_6400M_bailu.ini"])
        stage, = r["stages"]
        trace = ROOT / stage["trace"]
        assert r["trace_sha256"] == [stage["trace_sha256"]]
        assert re.fullmatch(r"[a-f0-9]{64}", stage["trace_sha256"])
        assert r["runner_sha256"] == job["runner_sha256"] == digest_small(ROOT / job["runner"])
        binary = trace.parents[1] / "dump/aicore_binary.o"
        assert stage["binary_sha256"] == digest_small(binary)
        vendor = dict(re.findall(r"^(core\d+\.veccore\d+)\s+([.\d]+)\s+[.\d]+\s*$", log, re.M))
        wins = stage["windows"]
        assert set(wins) == set(vendor) and len(wins) == r["active_aiv"] and r["active_aic"] == 0
        for core, (begin, end) in wins.items():
            assert abs(float(vendor[core]) - (end-begin)) <= max(.03, .02*(end-begin))
        first = min(w[0] for w in wins.values())
        last_core = max(wins, key=lambda c: wins[c][1])
        assert first == 0 and math.isclose(wins[last_core][1], pair[side+"_ca_us"], abs_tol=1e-9)
        assert sum(stage["queue_events"].values()) == stage["total_events"]
        predictions = dict(v019_frozen=job["outputs"][0]["old_"+side+"_us"],
                           v020_frozen=job["outputs"][0][side+"_us"],
                           v021_replay=estimate("rmsnorm", SHAPE, dtype="bf16", chip=r["chip"]).latency_us)
        endpoints[side] = dict(chip=r["chip"], ca_us=r["ca_us"], prediction_us=predictions,
            signed_error_pct={k:signed_pct(v,r["ca_us"]) for k,v in predictions.items()},
            active_aiv=r["active_aiv"],active_aic=r["active_aic"],last_core=last_core,
            x_events=stage["total_events"], queue_events=stage["queue_events"],trace=stage["trace"],
            trace_bytes=trace.stat().st_size,trace_sha256=stage["trace_sha256"],binary_sha256=stage["binary_sha256"],
            qualified=str(qualified.relative_to(ROOT)),qualified_sha256=digest_small(qualified),
            correctness=r["correctness"],cost={k:outcome[k] for k in
                ("started_utc","finished_utc","seconds","cpu_seconds","raw_bytes","peak_bytes","peak_is_lower_bound")})
        measured = [p[side+"_ca_us"] for p in pairs]
        cohort[side] = {
            "v019_frozen":error_summary(measured,[p["old_"+side+"_us"] for p in pairs]),
            "v020_frozen":error_summary(measured,[p[side+"_us"] for p in pairs]),
            "v021_replay":error_summary(measured,[estimate("rmsnorm",p["shape"],dtype="bf16",chip=r["chip"]).latency_us for p in pairs]),
        }
    dt, pr = endpoints["dt"], endpoints["pr"]
    ratios = {"ca": dt["ca_us"]/pr["ca_us"]}
    ratios.update({key:value/pr["prediction_us"][key] for key,value in dt["prediction_us"].items()})
    future = estimate("rmsnorm",SHAPE,dtype="bf16",chip="Ascend970DT").latency_us
    result = dict(shape=SHAPE,dtype="bf16",role="independent for original v0.20; v0.21 post-result replay",
        api=pair["dt"]["api"],kernel=pair["dt"]["kernel"],freeze=str(freeze_path.relative_to(ROOT)),
        freeze_utc=freeze["utc"],freeze_sha256=digest_small(freeze_path),runner_sha256=job["runner_sha256"],
        endpoints=endpoints, dt_over_pr=ratios,
        future_us=dict(status="extrapolation",v019=pair["old_future_us"],v020=pair["future_us"],v021=future),
        all_ten_pairs_errors=cohort,
        trace_audit="Original qualifier streamed all complete events and the raw SHA256. "
                    "This receipt reuses those results and checks the published endpoint against the host log, "
                    "per-core windows, raw file size, binary digest, runner and preregistration.")
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
