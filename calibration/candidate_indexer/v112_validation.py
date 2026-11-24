"""Join original CA receipts to immutable release prices; keep data roles separate."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DT = "Ascend950DT_9582"
PR = "Ascend950PR_957b"
FUTURE = "Ascend970DT"


def load(name):
    return json.loads((HERE / name).read_text())


def join():
    old, new = ({(r["case"], r["sku"]): r for r in load(f)["prices"]}
                for f in ("v112_baseline.json", "v112_frozen.json"))
    windows = []
    for short in load("fused-qsli.json"):
        case = "small" if short["table_columns"] == 8 else "small_wide"
        windows.append((case, short["sku"], short["exact_native_kernel_window_us"], "development", "fused-qsli.json"))
    for row in load("long-row-results.json")["records"]:
        windows.append(("split_row", row["chip"], row["ca_us"], "development", "long-row-results.json"))
    for case, sku, role, filename in [
        ("shared_wave", DT, "v1.11 holdout replay", "prefill-wave.json"),
        ("shared_wave", PR, "independent", "v112_wave_pr.json"),
        ("varied_two_waves", DT, "independent", "v112_wave_dt.json"),
    ]:
        receipt = load(filename)
        if receipt["sku"] != sku or list(receipt["shape"]) != new[case, sku]["shape"]:
            raise ValueError(f"CA receipt does not match its prediction: {filename}")
        windows.append((case, sku, receipt["device_us"], role, filename))
    rows = []
    for case, sku, ca, role, filename in windows:
        v111, v112 = old[case, sku]["latency_us"], new[case, sku]["latency_us"]
        rows.append(dict(case=case, sku=sku, shape=new[case, sku]["shape"], role=role, source=filename,
                         ca_us=ca, v111_us=v111, v112_us=v112, signed_v111_pct=100*(v111/ca-1),
                         signed_v112_pct=100*(v112/ca-1), cold_U_pct=new[case, sku]["cold_U_pct"],
                         extrapolated_970dt_us=new[case, FUTURE]["latency_us"]))
    return rows


def score(rows):
    return {key: dict(wape_pct=100*sum(abs(r[key]-r["ca_us"]) for r in rows)/sum(r["ca_us"] for r in rows),
                      max_abs_pct=max(100*abs(r[key]/r["ca_us"]-1) for r in rows))
            for key in ("v111_us", "v112_us")}


def aggregate(rows):
    idx = {(r["case"], r["sku"]): r for r in rows}
    pairs = {}
    for case in ("small", "small_wide", "split_row", "shared_wave"):
        pair = [idx[case, sku] for sku in (DT, PR)]
        ratios = {key: pair[0][key]/pair[1][key] for key in ("ca_us", "v111_us", "v112_us")}
        ratios["v112_error_pct"] = 100*(ratios["v112_us"]/ratios["ca_us"]-1)
        pairs[case] = ratios
    groups = {}
    for sku in (DT, PR):
        for role in ("development", "v1.11 holdout replay", "independent"):
            group = [r for r in rows if r["role"] == role and r["sku"] == sku]
            if group:
                groups[f"{sku}:{role}"] = dict(windows=len(group), **score(group))
    return dict(rows=rows, groups=groups, dt_time_over_pr_time=pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(aggregate(join()), indent=2) + "\n")
