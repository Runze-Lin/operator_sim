"""Recompute the FA reference prices from archived 950DT development observations."""
import json
from collections import defaultdict
from math import ceil, isclose
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "src/operator_model/data"


def issue_slope(ticks):
    """Least-squares slope in ticks/page, omitting the fill and drain."""
    pages = range(8, 120)
    center = mean(pages)
    return sum((i - center) * ticks[i] for i in pages) / sum((i - center) ** 2 for i in pages)


def steady_prices(evidence, chip):
    q, kv, d, _, _ = evidence["steady_shape"]
    n = evidence["page_keys"]
    pages = kv // n
    rows = ceil(q / 2)  # two cooperating AIVs per AIC in this DT trace
    m16 = 16 * ceil(q / 16)
    k, v = evidence["page_k_issue_ticks"], evidence["page_v_issue_ticks"]
    if pages != len(k) or pages != len(v):
        raise ValueError("Incomplete K/V issue series")
    # Both streams must advance by eight pages. Take the fastest interior window.
    burst_ticks = min(max(k[i + 8] - k[i], v[i + 8] - v[i]) for i in range(8, 112))
    union = evidence["queue_union_ticks"]
    clock = chip["frequency_ghz"]
    return {
        "dt_cube_peak_ops_per_cycle": 1000 * chip["cube_tops"]["16"] / (clock * chip["cube_cores"]),
        "dt_vector_peak_ops_per_cycle": 1000 * chip["vector_tops"]["32"] / (clock * chip["vector_cores"]),
        "mte_bytes_per_cycle": 8 * 4 * n * d / burst_ticks,
        "issue_cycles_256": max(issue_slope(k), issue_slope(v)),
        "vector_ops_per_cycle": pages * rows * (5 * n + 3 * d) / union["SIMD"],
        "mte1_bytes_per_cycle": pages * 2 * (2 * n * d + m16 * (d + n)) / union["MTE1"],
        "fix_bytes_per_cycle": pages * 4 * m16 * (n + d) / union["FIXP"],
    }


def route_prices(windows, prices):
    """Use control gaps and residual service ticks, one median per route/field."""
    series = defaultdict(list)
    read_rate = prices["mte_bytes_per_cycle"]
    vector_rate = prices["vector_ops_per_cycle"]
    for w in windows:
        q, _, d, _, _ = w["shape"]
        n = w["kv_tile"]
        parts = ceil(n / 128)
        cube_ops = 2 * (16 * ceil(q / 16)) * (16 * ceil(d / 16)) * (16 * ceil(n / 16))
        ideal = cube_ops / prices["dt_cube_peak_ops_per_cycle"]
        for key in ("startup", "k_lag", "v_lag", "l0_ready"):
            series[key].append(w[key])
        for latency, payload in w["reads"]:
            series["read_intercept"].append(max(0, latency - payload / read_rate))
        for span, key in (("first_qk_span", "qk_overhead"), ("pv_span", "pv_overhead")):
            series[key].append(max(0, w[span] - ideal) / parts)
        q_read, k_read, v_read = (r[0] for r in w["reads"])
        qk_end = max(q_read, w["k_lag"] + k_read) + w["l0_ready"] + w["first_qk_span"]
        v_end = w["k_lag"] + w["v_lag"] + v_read
        rows = ceil(q / 2)
        vec1 = 5 * rows * n / vector_rate
        visible_vec = max(qk_end + vec1, v_end) - max(qk_end, v_end)
        series["handoff_control"].append(max(0, w["handoff"] - visible_vec))
        output_bytes = 2 * q * d + 64 * q
        series["drain_control"].append(max(0, w["drain"] - 3 * rows * d / vector_rate - output_bytes / read_rate))
    return {key: median(values) for key, values in series.items()}


def main():
    evidence = json.loads((HERE / "fa-950dt.json").read_text())
    chip = json.loads((DATA / "chips" / (evidence["chip"] + ".json")).read_text())
    prices = steady_prices(evidence, chip)
    rows = evidence["window_rows"]
    # Only the comparison reads the published prices.
    frozen = json.loads((DATA / "parameters/attention_resources.json").read_text())
    for label, values, expected in [
        ("steady", prices, {k: frozen[k] for k in prices}),
        ("short", route_prices([r for r in rows if r["kv_tile"] <= 128], prices), frozen["routes"]["short"]),
        ("wide", route_prices([r for r in rows if r["kv_tile"] == 256], prices), frozen["routes"]["wide"]),
    ]:
        if expected.keys() != values.keys():
            raise ValueError(f"{label}: parameter names differ")
        for key, value in values.items():
            if not isclose(value, expected[key], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"{label}.{key}: recomputed {value}, published {expected[key]}")
            print(f"{label:6} {key:28} {value:14.8f}")


if __name__ == "__main__":
    main()
