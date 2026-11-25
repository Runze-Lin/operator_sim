"""Capture real rank-local tensor contracts for the preview's text model catalogue."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
import tomllib
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[2]
PREVIEW = ROOT / "system-sim-workspace/preview-0.1.0-preview.1"
CHIP = PREVIEW / "sim-workspace/sim-platform/hardware/chip/950DT.jsonc"


def tasks():
    registry = json.loads((PREVIEW / "inference-models.json").read_text())["models"]
    for model in registry:
        if not model["family"].startswith("text/"):
            continue
        for preset in model["presets"]:
            if preset in ("debug", "official"):
                continue
            longest = (
                131072
                if model["name"] in ("deepseek_v3", "deepseek_v32")
                else 262144
                if model["name"].startswith("qwen")
                else 1048576
            )
            grid = (
                ("decode", 32768, 1, 1, "mixed"),
                ("prefill", 32768 - 4096, 4096, 4, "mixed"),
                ("decode", 131072 - 16, 1, 16, "bf16"),
                ("prefill", 0, 512, 1, "bf16"),
                ("verify", 32768, 9, 1, "mixed"),
                ("verify", 65536, 5, 4, "fp8"),
                ("decode", 65536, 1, 32, "fp8"),
                ("decode", longest - 1024, 1, 4, "fp4"),
                ("prefill", longest - 2048, 2048, 8, "fp4"),
                ("prefill", 131072 - 1024, 1024, 16, "fp8"),
            )
            if model["name"] == "block_diffusion_lm":
                grid = [
                    (phase, h, q, b, "source-fp32")
                    for phase, h, q, b in (
                        ("prefill", 0, 128, 1),
                        ("prefill", 0, 512, 4),
                        ("denoise", 512, 8, 1),
                        ("commit", 512, 8, 1),
                        ("denoise", 2048, 16, 4),
                        ("commit", 2048, 16, 4),
                    )
                ]
            # Additional native calls target the least frequent geometries in the first pass.
            # In this producer BF16 V2 dense attention and transpose BMM occur in Kimi K3/V3.
            extras = {
                "deepseek_v3": [("decode", b, 1) for b in (2, 3, 5, 9)]
                + [
                    ("prefill", b, q)
                    for b, q in ((1, 127), (2, 255), (3, 511), (4, 2047))
                ],
                "deepseek_v32": [("prefill", 1, 1023), ("decode", 9, 1)],
                "glm5_2": [
                    ("prefill", 2, 511),
                    ("prefill", 1, 2047),
                    ("decode", 11, 1),
                ],
                "glm5_3_flash": [
                    ("prefill", 1, 383),
                    ("prefill", 2, 767),
                    ("verify", 3, 7),
                ],
                "kimi_k3": [("prefill", 1, q) for q in (257, 513, 769)]
                + [("decode", b, 1) for b in (11, 17, 33)]
                + [("verify", b, q) for b, q in ((2, 3), (3, 7), (1, 11))],
                "kimi_k31": [
                    ("prefill", 2, 255),
                    ("prefill", 4, 513),
                    ("verify", 2, 7),
                    ("verify", 3, 11),
                ],
                "deepseek_v41": [
                    ("prefill", 2, 255),
                    ("prefill", 4, 511),
                    ("decode", 9, 1),
                ],
                "qwen3_8_flash_next": [
                    ("prefill", 2, 511),
                    ("prefill", 1, 1023),
                    ("verify", 3, 7),
                ],
            }
            extra = extras.get(model["name"], [])
            if preset.endswith("27B"):
                extra = (
                    [("prefill", 1, q) for q in (255, 513, 1023)]
                    + [
                        ("verify", b, q)
                        for b, q in ((2, 3), (3, 7), (1, 11), (2, 13), (1, 17))
                    ]
                    + [("decode", 9, 1)]
                )
            grid = list(grid) + [(phase, 32768, q, b, "bf16") for phase, b, q in extra]
            if preset.endswith("27B"):
                grid += [
                    ("prefill", 0, 512, b, "bf16") for b in (2, 3, 4, 8, 9, 16, 32)
                ]
            if model["name"] == "block_diffusion_lm":
                grid += [
                    ("prefill", 0, q, b, "source-fp32")
                    for b, q in (
                        (1, 31),
                        (2, 63),
                        (3, 127),
                        (4, 255),
                        (1, 1024),
                        (2, 2048),
                    )
                ]
            for phase, history, query, batch, precision in grid:
                yield dict(
                    name=model["name"],
                    preset=preset,
                    phase=phase,
                    history=history,
                    query=query,
                    batch=batch,
                    precision=precision,
                )


def case_id(task):
    return "-".join(
        str(task[k]).replace("/", "_")
        for k in ("preset", "phase", "history", "query", "batch", "precision")
    )


def task_text(spec):
    s = spec
    block = s["name"] == "block_diffusion_lm"
    dense = s["preset"].endswith("27B") or block
    weight = "fp8" if s["precision"] == "fp8" else "fp4"
    activation = "fp4" if s["precision"] == "fp4" else "fp8"
    return (
        f"""[inference.model]
name="{s["name"]}"
preset="{s["preset"]}"
[inference.runtime]
phase="{s["phase"]}"
input_seq_len={s["query"]}
kv_seq_len={s["history"]}
kv_capacity={s["history"] + s["query"]}
batch_size={s["batch"]}
tp={1 if block else 8}
dp={1 if dense else 4}
enable_expert_parallel={str(not dense).lower()}
"""
        + ("" if dense else 'ep_dispatch_backend="mc2_shape"\n')
        + (
            'attention_output_layout="standard"\n'
            if block
            else 'precision="bf16"\n'
            if s["precision"] == "bf16"
            else f"""[inference.runtime.precision_config]
attention_projections={{weight_precision="{weight}",activation_precision="{activation}"}}
attention_compute="{activation}"
moe_routed="{weight}"
moe_shared="{weight}"
vector_compute="bf16"
"""
        )
        + f"""[inference.capture]
rank=0
rank_mode="exact"
[platform]
hardware={json.dumps(str(CHIP))}
[output]
directory="run"
[logging]
console_level="ERROR"
"""
    )


def write_packet(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(
        gzip.compress(
            json.dumps(obj, separators=(",", ":"), allow_nan=False).encode(), mtime=0
        )
    )
    tmp.replace(path)


def capture(spec, cache):
    from sim_platform.api.inference import run_inference
    from sim_platform.config.inference import load_inference_task
    from sim_platform.backend.prediction import convert_node_features

    key = case_id(spec)
    case = cache / key
    case.mkdir(exist_ok=True, parents=True)
    task = case / "task.toml"
    raw = task_text(spec)
    output = cache / (key + ".json.gz")
    if output.exists():
        packet = json.loads(gzip.decompress(output.read_bytes()))
        if packet["task"] == spec and packet["task_toml"] == raw.replace(
            str(CHIP), "950DT.jsonc"
        ):
            return key
    task.write_text(raw)
    config = load_inference_task(task, overrides={"exports.enabled": False})
    workload = run_inference(replace(config, options=None)).workload
    stages = []
    for graph, facts in zip(workload.graphs, workload.captures, strict=True):
        counts = Counter(convert_node_features(n) for n in graph.nodes)
        sigs = [dict(features=asdict(f), count=n) for f, n in counts.items()]
        stages.append(dict(stage=facts.stage, nodes=len(graph.nodes), signatures=sigs))
    output_data = dict(
        task=spec, task_toml=raw.replace(str(CHIP), "950DT.jsonc"), stages=stages
    )
    write_packet(output, output_data)
    return key


def publish(cache, output):
    """Merge counts across stages while keeping each original feature and case recoverable."""
    records = {}
    index = []
    for spec in tasks():
        case = case_id(spec)
        packet = json.loads(gzip.decompress((cache / (case + ".json.gz")).read_bytes()))
        if packet["task"] != spec:
            raise ValueError(f"Task mismatch in {case}")
        stages = []
        for stage in packet["stages"]:
            weights = []
            for row in stage["signatures"]:
                feature = row["features"]
                encoded = json.dumps(
                    feature, sort_keys=True, separators=(",", ":")
                ).encode()
                key = hashlib.sha256(encoded).hexdigest()
                records[key] = feature
                weights.append([key, row["count"]])
            if sum(count for _, count in weights) != stage["nodes"]:
                raise ValueError(f"Node count mismatch in {case}")
            stages.append(
                dict(stage=stage["stage"], nodes=stage["nodes"], signatures=weights)
            )
        index.append(dict(task=spec, task_toml=packet["task_toml"], stages=stages))
    output.mkdir(exist_ok=True, parents=True)
    write_packet(output / "features.json.gz", records)
    write_packet(output / "cases.json.gz", index)
    from inference_models.request import build_request

    resolved = {}
    for row in index:
        spec = row["task"]
        if spec["preset"] in resolved:
            continue
        req = build_request(tomllib.loads(row["task_toml"])["inference"])
        resolved[spec["preset"]] = req.effective_config()["model"]
    write_packet(output / "model-configs.json.gz", resolved)
    manifest = dict(
        workload="registered text families; rank 0 exact shape capture; one forward per task",
        preview=PREVIEW.name,
        versions={
            name: version(name)
            for name in (
                "inference-models-simulator",
                "sim-platform",
                "operator-performance-models",
                "torch",
            )
        },
        presets=len(resolved),
        cases=len(index),
        unique_signatures=len(records),
        nodes=sum(stage["nodes"] for row in index for stage in row["stages"]),
        operators=sorted({f["raw_name"] for f in records.values()}),
        digests={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                output / "features.json.gz",
                output / "cases.json.gz",
                output / "model-configs.json.gz",
                CHIP,
                PREVIEW / "inference-models.json",
            )
        },
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cache", type=Path)
    cli.add_argument("--workers", type=int, default=6)
    cli.add_argument("--filter", default="")
    cli.add_argument("--publish", type=Path)
    args = cli.parse_args()
    if args.publish:
        publish(args.cache, args.publish)
        return
    items = [s for s in tasks() if args.filter in case_id(s)]
    failures = {}
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(capture, s, args.cache): s for s in items}
        for job in as_completed(jobs):
            key = case_id(jobs[job])
            try:
                print(job.result(), flush=True)
            except Exception as e:
                failures[key] = f"{type(e).__name__}: {e}"
                print(key, failures[key], flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
