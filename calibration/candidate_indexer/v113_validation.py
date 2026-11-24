"""Reproduce v1.13 errors from frozen estimates and qualified device receipts."""
import argparse
import json
from hashlib import sha256
from pathlib import Path
from v113_prices import ARMS, DT, PR, FUTURE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def summarize(rows, key):
    return dict(windows=len(rows), distinct_shapes=len({tuple(r["shape"]) for r in rows}),
                wape_pct=100*sum(abs(r[key]-r["ca_us"]) for r in rows)/sum(r["ca_us"] for r in rows),
                max_abs_pct=max(abs(100*(r[key]/r["ca_us"]-1)) for r in rows))


def evaluate(initial=False):
    frozen_file = HERE / ("v113_frozen.json" if initial else "v113b_frozen.json")
    freeze_sha = sha256(frozen_file.read_bytes()).hexdigest()
    snapshot = read(frozen_file)
    prices = {(r["case"],r["sku"]): r for r in snapshot["prices"]}
    previous = read(ROOT/"results/v1.12/evaluation.json")["rows"]
    evidence = [(r["case"],r["sku"],r["ca_us"],r["v112_us"],r["source"],"development replay")
                for r in previous]
    old_dt64 = next(r["v112_us"] for r in previous if r["case"]=="varied_two_waves" and r["sku"]==DT)
    for arm in ARMS:
        name = f"v113_control_{arm}.json"
        measurement = read(HERE / name)
        if measurement["freeze_sha256"] != (freeze_sha if initial else snapshot["initial_freeze_sha256"]) or measurement["status"] != "qualified full device window":
            raise ValueError(f"incomplete or different freeze: {arm}")
        role = "frozen address control" if initial or arm=="moving" else "endpoint mechanism review"
        evidence.append(("control_"+arm,DT,measurement["device_us"],old_dt64,name,role))
    rows = []
    for label, sku, ca, old, src, role in evidence:
        point = prices[label,sku]
        known = point["known_layout"] or point["shape_only"]
        future = prices[label,FUTURE]
        shape_only = point["shape_only"]["ca_window_us"]
        current = known["ca_window_us"]
        rows.append(dict(case=label,sku=sku,shape=point["shape"],role=role,source=src,ca_us=ca,
                         v112_us=old,shape_only_us=shape_only,known_layout_us=current,
                         signed_v112_pct=100*(old/ca-1),signed_shape_only_pct=100*(shape_only/ca-1),
                         signed_known_pct=100*(current/ca-1),candidate_order_overlap=point["candidate_order_overlap"],
                         window_time_factor=known["window_time_factor"],cold_U_pct=known["cold_U_pct"],
                         shape_only_cold_U_pct=point["shape_only"]["cold_U_pct"],
                         future_970dt_us=(future["known_layout"] or future["shape_only"])["ca_window_us"],
                         future_970dt_status="extrapolated"))
    groups = {}
    for sku in (DT, PR):
        for role in ("development replay", "endpoint mechanism review", "frozen address control", "all three layouts"):

            group = [r for r in rows if r["sku"]==sku and (r["role"]==role or (role=="all three layouts" and r["case"].startswith("control_")))]
            groups[f"{sku}/{role}"] = {key:summarize(group,key) for key in (
                "v112_us", "shape_only_us", "known_layout_us")} if group else None
    contrasts = []
    shared, moving, rotated = rows[-3:]
    for other in (moving,rotated):
        contrasts.append(dict(change=f"shared to {other['case'][8:]}",**{
            key:other[key]-shared[key] for key in ("ca_us","v112_us","shape_only_us","known_layout_us")}))
    return dict(version=snapshot["version"],freeze_sha256=freeze_sha,rows=rows,groups=groups,
                paired_differences_us=contrasts, qualification="native QSLI device only, fixed metadata; control arms share one shape")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--initial",action="store_true")
    args = parser.parse_args()
    args.output.write_text(json.dumps(evaluate(args.initial),indent=2)+"\n")
