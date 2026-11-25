"""Recover post-Vector tails from qualified v2.4 development and former holdout traces."""

import argparse
from hashlib import sha256
import json
from math import ceil
from pathlib import Path

from calibration.catalog_ca.campaign import ROOT, forecast
from operator_model import load_chip


def evidence(roots):
    records = []
    for filename in ("add-rms-development.json", "add-rms-independent.json"):
        for r in json.loads((ROOT / "results/v2.4" / filename).read_text()):
            sku = r["chip"]
            rows, width = r["shape"]
            core = r["last_core"]
            case = f"{sku}-{rows}-{width}"
            paths = [
                p
                for base in roots
                for p in (base / case / "profile").glob(
                    f"OPPROF_*/simulator/{core}/trace.json"
                )
            ]
            path = next((p for p in paths if p.is_file()), None)
            if path is None:
                raise ValueError(f"missing last-core source for {case}")
            events = json.loads(path.read_bytes())["traceEvents"]
            names = {
                e["pid"]: e["args"]["name"]
                for e in events
                if e["name"] == "process_name"
            }

            def end(e):
                return round(1650 * (e["ts"] + e["dur"]))

            instructions = [e for e in events if e["ph"] == "X"]
            final = max(end(e) for e in instructions)
            if abs(final - r["last_tick"]) > 1:
                raise ValueError("Selected retry does not match the archived timing")
            vector = [e for e in instructions if e["name"].startswith("RV_")]
            first = min(
                round(e["ts"] * 1650) for e in vector if e["name"].startswith("RV_VLD")
            )
            last = max(map(end, vector))
            writes = []
            for e in instructions:
                if names[e["pid"]] == "MTE3" and e["name"].startswith("MOV_SRC_TO_DST"):
                    detail = json.loads(e["args"]["detail"])
                    writes.append(
                        [
                            detail["processed_bytes"],
                            round(e["ts"] * 1650) - r["first_tick"],
                            round(e["dur"] * 1650),
                        ]
                    )
            chip = load_chip(sku)
            local = ceil(rows / chip.vector_cores)
            active = ceil(rows / local)
            owners = min(active, ceil(128 / (4 * local)))
            if active != r["active_aiv"]:
                raise ValueError(
                    "Uniform contiguous row partition disagrees with roster"
                )
            old = r.get("auto_us")
            if old is None:
                from calibration.catalog_ca.add_rms_refinement import prior_forecast

                old = prior_forecast(sku, (rows, width))["auto_us"]
            records.append(
                dict(
                    source=filename,
                    chip=sku,
                    shape=[rows, width],
                    busy_rows=local,
                    active_aiv=active,
                    sharing_owners=owners,
                    ca_us=r["measured_us"],
                    v24_us=old,
                    v25_us=forecast(sku, (rows, width))["auto_us"],
                    first_vload=first - r["first_tick"],
                    vector_span=last - first,
                    post_vector=final - last,
                    writes_bytes_start_duration=writes,
                    trace_sha256=sha256(path.read_bytes()).hexdigest(),
                )
            )
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path)
    args = ap.parse_args()
    records = evidence(args.roots)
    result = {
        "status": "development replay including former v2.4 holdout",
        "windows": records,
    }
    out = ROOT / "results/v2.5/add-rms-development.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
