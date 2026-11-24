"""Freeze layout-conditioned prices and source/input identities before the CA controls."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import operator_model
from operator_model import estimate
from v112_prices import CASES, DT, PR, FUTURE

HERE = Path(__file__).resolve().parent
ARMS = ("shared", "moving", "rotated")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quote(shape, attrs, sku):
    def price(memory_policy):
        return estimate("candidate_filtered_indexer", shape, attrs, chip=sku, dtype="mxfp4",
                        memory_policy=memory_policy)
    ca_window, cold = price("ca_window"), price("cold_io")
    return dict(ca_window_us=ca_window.latency_us, cold_io_us=cold.latency_us,
                active_aic=ca_window.active_aic, active_aiv=ca_window.active_aiv,
                cold_U_pct=100*cold.rates["roofline"]["utilization"],
                window_time_factor=ca_window.rates["logical_roofline"]["time_factor"],
                stages=ca_window.projection)


def record():
    package = Path(operator_model.__file__).resolve().parent
    model_files = [package / "backend.py", package / "sim_backend.py",
                   *(package / "operators/attention" / name for name in
                     ("flash_sparse.py", "indexer.py", "candidate_indexer.py", "candidate_layout.py")),
                   *(package / "data/parameters" / name for name in
                     ("flash_qsli.json", "attention_resources.json")),
                   *(package / "data/chips" / f"{sku}.json" for sku in (DT, PR, FUTURE))]
    inputs = [HERE / name for name in ("v113_control_probe.py", "v113_control_capture.py",
                                        "v113_mechanism.json", "v113_prices.py")]
    cases = []
    for label, q, length, cap, width, _ in CASES:
        # Identity is verified from the frozen v1.12 fixtures. Production 4096
        # query IDs are unavailable, so expose sensitivity separately below.
        overlap = {"shared_wave": 1., "varied_two_waves": 0.}.get(label)
        cases.append((label, (1,q,32,128,length,2048,512), cap, width, overlap))
    for arm in ARMS:
        src = HERE / f"v113_control_{arm}_fixture.json"
        inputs.append(src)
        fixture = json.loads(src.read_text())
        cases.append(("control_"+arm, tuple(fixture["shape"]), 2048, fixture["table_pages"],
                      fixture["all_unordered_row_pairs_same_rank_fraction"]))
    prices = []
    for label, shape, cap, width, overlap in cases:
        options = {"implementation":"a5_qsli_projection", "candidate_limit": cap,
                   "pa_block_size":128, "table_pages":width}
        for sku in (DT, PR, FUTURE):
            unknown = quote(shape, options, sku)
            known = quote(shape, {**options, "candidate_order_overlap":overlap}, sku) if overlap is not None else None
            prices.append(dict(case=label, shape=shape, sku=sku, attrs=options,
                               candidate_order_overlap=overlap, shape_only=unknown, known_layout=known))
    return dict(version=operator_model.__version__, frozen_utc=datetime.now(timezone.utc).isoformat(),
                model_sha256={str(p.relative_to(package)):digest(p) for p in model_files},
                input_sha256={p.name:digest(p) for p in inputs}, prices=prices,
                protocol=dict(native="950DT, released QSLI MXFP4 PA_BBND ratio=1, one launch per arm",
                              independent="one tensor shape, three independent of model fitting; two controlled contrasts",
                              gates="full golden, schedule, inputs, MMAD, addresses and complete physical-core windows",
                              budget="3 launches, watchdog 42 minutes to golden + 20 minutes to export per launch",
                              system="33 unchanged graphs; original, previous, unknown layout, declared overlap and CA replay",
                              extrapolation="970DT, production candidate placement, Q4096 and metadata-inclusive deployment"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    snapshot = record()
    with args.output.open("x") as stream:
        json.dump(snapshot, stream, indent=2)
        stream.write("\n")
