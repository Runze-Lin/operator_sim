"""Estimate elapsed RMSNorm time including correlated per-row statistic stores."""
from functools import lru_cache
import json
from pathlib import Path

from .wide_v10 import predict as predict_v10


@lru_cache(maxsize=1)
def parameters():
    return json.loads((Path(__file__).resolve().parents[1] / "results/v0.11/rms/profile.json").read_text())


def pressure(active, rows, window_bytes):
    """Concurrent neighbouring rstd writers per effective window, scaled by occupancy."""
    capacity = window_bytes / 4
    neighbours = max(0, min(active, capacity / rows) - 1)
    return neighbours * min(1, active / capacity)


def predict(op, shape, attrs=None):
    if op != "rmsnorm":
        return {"predicted_us": None, "status": "unsupported", "reason": "Expected rmsnorm"}
    prediction = predict_v10(op, shape, attrs)
    if prediction["predicted_us"] is None:
        return prediction
    t, d = shape
    profile = parameters()
    nearby = pressure(prediction["predicted_active_aiv"], prediction["rows_per_core"], profile["window_bytes"])
    write_kib = t * (d * 2 + 4) / 1024
    write_us = (profile["write_start_us"] + profile["effective_us_per_kib"] * write_kib
                + profile["neighbour_us"] * nearby)
    prediction["stages_us"]["MTE3"] = write_us
    prediction["predicted_us"] = sum(prediction["stages_us"].values())
    return prediction
