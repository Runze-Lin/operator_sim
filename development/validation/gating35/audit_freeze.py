"""Audit the immutable 1536-row registration against its source, callers and score."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/v0.35/gating"
REG = OUT / "freeze.json"
ORIGINAL_SHA = "e6d46cd411c932b065269157c6b344202c3fc3998b10dc4a9d37301edba230a1"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subscripts(path, function, obj):
    module = ast.parse(path.read_text())
    body = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == function)
    return sorted({n.slice.value for n in ast.walk(body)
                   if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                   and n.value.id == obj and isinstance(n.slice, ast.Constant)})


def audit():
    assert digest(REG) == ORIGINAL_SHA
    frozen = read(REG)
    job = frozen["job"]
    shape = [1536, 896, 16]
    assert job["shape"] == shape and job["args"] == list(map(str, shape))
    old_file = ROOT / frozen["source"]
    assert digest(old_file) == frozen["previous_sha"]
    old = next(j for j in read(old_file)["jobs"]
               if j["op"] == job["op"] and j["shape"] == [4096,896,16] and j["chip"] == job["chip"])
    assert job["prices"] == old["prices"] and job["scenario"] == old["scenario"]
    assert "B8×192" in frozen["scenario"] and "B8 × 512" in job["scenario"]
    assert [x["shape"] for x in frozen["cases"]].count(shape) == 1 and frozen["cases"][0]["shape"] == shape

    capture_dir = ROOT / "development/validation/large29/vector"
    consumers = {}
    for filename, fn, var in (("acquire.py", "capture", "j"), ("verify.py", "qualify", "job"),
                             ("verify.py", "numerical", "job")):
        source = capture_dir / filename
        assert digest(source) == frozen["hashes"][str(source.relative_to(ROOT))]
        keys = subscripts(source, fn, var)
        assert not {"prices", "scenario"}.intersection(keys)
        consumers[fn] = keys

    scoreboard = read(OUT / "score.json")
    assert scoreboard["freeze_sha256"] == ORIGINAL_SHA
    forward = {(tuple(r["shape"]), r["chip"]): r for r in scoreboard["forward"]}
    assert len(forward) == len(scoreboard["forward"])
    matched = 0
    for case in frozen["cases"]:
        for sku, forecast in case["forecasts"].items():
            row = forward[tuple(case["shape"]), sku]
            assert (row["public"], row["candidate"], row["active_aiv"], row["critical_rows"]) == (
                forecast["public_us"], forecast["us"], forecast["active_aiv"], forecast["rows_per_core"])
            matched += 1
    assert matched == len(forward)

    commands = {}
    for tag, sku in (("dt", "Ascend950DT_9582"), ("pr", "Ascend950PR_957b")):
        expected_suffix = [str(ROOT / job["runner"]), *job["args"]]
        expected_sku = "--soc-version=" + sku
        output = ROOT / "runs/gating35" / tag
        expected_command = ["bash", str(ROOT / "scripts/cann-run.sh"), "msopprof", "simulator",
                            expected_sku, "--dump=false", "--launch-count=1",
                            "--timeout=" + str((job["wall_s"] + 59) // 60),
                            "--output=" + str(output / "profile"), *expected_suffix]
        cost = output / "cost.json"
        if cost.exists():
            command = read(cost)["command"]
            assert command == expected_command
        receipt = OUT / (tag + ".json")
        if receipt.exists():
            r = read(receipt)
            assert cost.exists() and r["cost"] == read(cost)
            assert r["shape"] == shape and r["chip"] == sku and r["freeze_sha256"] == ORIGINAL_SHA
            if r["verdict"]["qualified"]:
                assert r["verdict"]["correctness"] == [
                    "correctness ids count=24576 bad=0", "correctness weights count=24576 bad=0"]
        commands[sku] = dict(expected_command=expected_command, actual_command_observed=cost.exists(),
                             qualified_receipt=read(receipt)["verdict"]["qualified"] if receipt.exists() else None)
    return dict(original_freeze_sha256=ORIGINAL_SHA, canonical_scenario=frozen["scenario"],
                shape=shape, projections_match=matched, consumed_job_fields=consumers, endpoints=commands)


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, ensure_ascii=False))
