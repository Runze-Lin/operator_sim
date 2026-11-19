"""Full-row RMSNorm: schedule the two output copies against per-tile vector work."""
from functools import lru_cache
import json
from pathlib import Path
from .rms_v11 import pressure
from .wide_v10 import predict as base

@lru_cache(maxsize=1)
def parameters():
    return json.loads((Path(__file__).resolve().parents[1] / 'results/v0.12/rms/profile.json').read_text())

def tile_capacity(d, ub_bytes=253952):
    """BF16 full-row UB budget from operator_model.extensions.norm_addmm._rmsnorm."""
    aligned = (d + 15) // 16 * 16
    fold = 1 << (aligned - 1).bit_length() - 1
    aux = ((fold + 63) // 64 + 31) // 32 * 32
    cap = (ub_bytes - 1024 - 2048 - 2 * aligned) // (8 * aligned + 12 + aux)
    if cap < 1:
        raise ValueError("Target UB cannot hold a full RMSNorm row on this kernel route")
    return cap


def tile_rows(rows, d, ub_bytes=253952):
    cap = tile_capacity(d, ub_bytes)
    (full, tail) = divmod(rows, cap)
    return [cap] * full + ([tail] if tail else [])

def schedule_writes(d, rows, active, tiles, p, neighbours=None):
    """Build the union of two pending stores per tile in CA ticks; mask it by useful SIMD windows."""
    lanes = (d + 127) // 128
    pending = []
    cursor = 0.0
    last_y = 0.0
    near = pressure(active, rows, p['window_bytes']) if neighbours is None else neighbours
    for (i, n) in enumerate(tiles):
        work = n * lanes
        if i:
            cursor += p['reduce'][0] + p['reduce'][1] * work
        rs = cursor
        rs_end = rs + p['rstd'][0] + p['rstd'][1] * near
        cursor += p['normalize'][0] + p['normalize'][1] * work
        last_y = cursor
        y_end = max(rs_end, cursor + p['y'][0] + p['y'][1] * (2 * d * n / 1024))
        pending.extend([(rs, rs_end), (cursor, y_end)])
    merged = []
    for (a, b) in sorted(pending):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    raw = sum((b - a for (a, b) in merged))
    hidden = sum((max(0, min(b, last_y) - a) for (a, b) in merged))
    return (raw, hidden)

def predict(op, shape, attrs=None):
    if op != 'rmsnorm':
        return dict(predicted_us=None, status='unsupported', reason='Expected rmsnorm')
    ans = base(op, shape, attrs)
    if ans['predicted_us'] is None:
        return ans
    (t, d) = shape
    rows = ans['rows_per_core']
    active = ans['predicted_active_aiv']
    tiles = tile_rows(rows, d)
    p = parameters()
    weights = ans['context_weights']
    phases = ans['stages_us']
    simd = sum((weights[k] * p['simd'][i] for (i, k) in enumerate(('first', 'expand', 'reuse')))) + p['simd'][3] * rows * ((d + 127) // 128)
    (raw, hidden) = schedule_writes(d, rows, active, tiles, p)
    hidden = min(simd, hidden)
    phases['SIMD'] = simd / p['ticks_per_us']
    phases['MTE3'] = (raw - hidden) / p['ticks_per_us']
    ans.update(predicted_us=sum(phases.values()), ub_tile_rows=tiles, write_union_ticks=raw, write_simd_overlap_ticks=hidden)
    return ans
