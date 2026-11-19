"""Post-hoc retention scenarios; these are diagnostics, never the frozen holdout score."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from development.validation.sigmoid20.campaign import DEST, DT, PR, FUTURE, write
from operator_model import load_chip, estimate


def scenario(shape, chip, retained):
    r = estimate("sigmoid_fp32", shape, chip=chip, dtype="fp32")
    d = r.projection["timing_diagnostics"]
    s = d["schedule_us"]
    p = d["target_plan"]
    f = chip.frequency_ghz
    A = p["active_aiv"]
    local = p["critical_core_elements"]
    cap = p["ub_tile_elements"]
    # Capacity scenario: keep the whole input live and allow the remaining L2
    # to absorb output. A future SKU may have explicit aggregate L2 bandwidth.
    per_side = 4 * p["elements"]
    l2 = chip.details.get("l2_bytes") or 0
    rho = min(retained, max(0.0, (l2 - per_side) / per_side))
    limit = chip.details.get("l2_bandwidth_gbs") or float("inf")

    def tile(x):
        b = 4 * x
        rd = b * f / d["port_read_gbs"]
        wr = b * f / d["port_write_gbs"]
        shared = b * A * f / d["shared_gbs"]
        R = 1159 + max(rd, shared)
        V = (79 + 4 * ((x + 63) // 64 - 1)) * d["special_function_rate_scale"]
        W = 159 + max(wr, (1 - rho) * shared, b * A * f / limit)
        I = max(
            rd,
            wr,
            (2 - rho) * shared,
            2 * b * A * f / limit,
            V,
            (R + V) / 2,
            (V + W) / 2,
        )
        return R, V, W, I

    R, V, W, I = tile(min(local, cap))
    r1, v1, w1, _ = tile(p["last_tile_elements"])
    n = p["critical_core_tiles"]
    fx = 1000 * f
    last = s["first_read_at"] * fx + (n - 1) * I
    rv = last + r1
    wr = rv + v1 + 17
    if n > 1:
        prev = last - I
        rv = max(rv, prev + R + V)
        wr = max(rv + v1 + 17, prev + R + V + 17 + W)
    return (wr + w1 + 10) / fx, rho


if __name__ == "__main__":
    cases = []
    for shape in ((262144, 12), (1048576, 12), (2097152, 12)):
        row = {"shape": shape}
        for name in (DT, PR, FUTURE):
            chip = load_chip(name)
            frozen = estimate("sigmoid_fp32", shape, chip=chip).latency_us
            baseline, _ = scenario(shape, chip, 0.0)
            assert abs(baseline - frozen) < 1e-8
            alt, rho = scenario(shape, chip, 1.0)
            row[name] = dict(frozen_us=frozen, capacity_retention_us=alt, fraction=rho)
        cases.append(row)
    write(
        DEST / "l2-sensitivity.json",
        dict(
            role="after-freeze diagnostic, excluded from holdout accuracy and API",
            assumption="All input remains live, spare L2 can absorb output through local MTE3 or documented L2 bandwidth; external fraction competes with input. Reuses the exact frozen control, tile and special-function prices.",
            cast_reference="results/v0.20/large/cast_262144_32_native-dram.json",
            cases=cases,
        ),
    )
    for r in cases:
        print(r)
