"""Descriptor dependencies for the FLA 26.9.1 GK recurrent state kernel.

All times are on the target core clock. The three deadlines crossing a chunk
boundary are the H producer, the W slot release and the U slot release.
"""
from math import inf


FREEZE_SHA256 = "22237d064eb5eddc1b63d23d8af32227d09194b8ec1db68897998f33ba4f674c"

# Observed on the matching AIC/AIV workgroup in archived complete device
# windows, relative to local producers. These are control cycles, not bytes/s.
FIRST_W_ISSUE_CYCLES = 334.5
FIRST_U_ISSUE_CYCLES = 3
NEXT_U_ISSUE_CYCLES = 6


def _prices(rows, tile):
    payload = 256 * rows  # BF16 [rows,128]
    padded_rows = 16 * ((rows + 15) // 16)
    return {
        "w": tile.dma("OUT->L1", payload),
        "h": tile.dma("OUT->L1", 32768),
        "u": tile.dma("OUT->UB", payload),
        "l0": max(tile.dma("L1->L0A", 256 * padded_rows), tile.dma("L1->L0B", 32768)),
        "mm0": tile.calc([(padded_rows, 128, 128, "bf16")]),
        "fix0": tile.dma("L0C->UB", 2 * payload),
        "vf1": tile.vec(1, rows),
        "right_write": tile.dma("UB->OUT", payload, payload),
        "k": tile.dma("OUT->L1", payload),
        "right": tile.dma("OUT->L1", payload),
        "l2": max(tile.dma("L1->L0A", 256 * padded_rows), tile.dma("L1->L0B", 256 * padded_rows)),
        "mm2": tile.calc([(128, 128, padded_rows, "bf16")]),
        "fix2": tile.dma("L0C->UB", 65536),
        "state_and_gate": tile.dma("OUT->UB", 65536, 512),
        "gate": tile.dma("OUT->UB", 512),
        "vf3": tile.vec(2, rows),
        "write_h": tile.dma("UB->OUT", 32768),
        "write_vk": tile.dma("UB->OUT", 65536),
        "initial": tile.dma("OUT->UB", 65536) + tile.vec(0, rows) + tile.dma("UB->OUT", 32768),
    }


def _advance(deadlines, costs, first=False, last=False):
    state_ready, w_free, u_free = deadlines
    if first:
        w_free = state_ready + FIRST_W_ISSUE_CYCLES
        state_ready += costs["initial"]
        u_free = state_ready + FIRST_U_ISSUE_CYCLES
    w_end = w_free + costs["w"]
    h_end = state_ready + costs["h"]
    mm0 = max(w_end, h_end) + costs["l0"]
    product = mm0 + costs["mm0"] + costs["fix0"]
    u_end = u_free + costs["u"]
    vf1 = max(product, u_end)
    right_write = vf1 + costs["vf1"]
    right = right_write + costs["right_write"]
    k_end = mm0 + costs["k"]
    right_end = right + costs["right"]
    mm2 = max(k_end, right_end) + costs["l2"]
    state_input = right + (costs["state_and_gate"] if first else costs["gate"])
    vf3 = max(mm2 + costs["mm2"] + costs["fix2"], state_input)
    state_write = vf3 + costs["vf3"]
    state_end = state_write + (costs["write_vk"] if last else costs["write_h"])
    events = {
        "W": [w_free, w_end], "H": [state_ready, h_end], "K": [mm0, k_end],
        "right": [right, right_end], "U": [u_free, u_end],
        "mm0": [mm0, mm0 + costs["mm0"]], "mm2": [mm2, mm2 + costs["mm2"]],
        "corrected": [vf1, right_write], "write_right": [right_write, right],
        "state": [vf3, state_write], "write_state": [state_write, state_end],
    }
    # W and U can be resident before H is ready. MMAD2 retirement safely frees
    # the W slot even when a real machine releases it during MMAD2 execution.
    return (state_end, mm2 + costs["mm2"], right + NEXT_U_ISSUE_CYCLES), events


def _repeat(state, costs, count):
    """Skip constant-size interior chunks without a sequence-length loop."""
    following, _ = _advance(state, costs)
    shifts = [x - y for x, y in zip(following, state)]
    if max(shifts) - min(shifts) < 1e-6:
        return tuple(x + count * shifts[0] for x in state)
    # An unusual target may still be in its transient. A chunk only adds fixed
    # resource services and takes maxima. Probe its three input basis deadlines
    # to recover the exact max-plus transition, then square it in O(log count).
    def apply(columns, vector):
        return tuple(max(column[i] + x for column, x in zip(columns, vector)) for i in range(3))

    matrix = [
        _advance(tuple(0.0 if i == j else -inf for i in range(3)), costs)[0]
        for j in range(3)
    ]
    while count:
        if count % 2:
            state = apply(matrix, state)
        matrix = [apply(matrix, col) for col in matrix]
        count //= 2
    return state


def schedule_chain(sequence, startup, tile):
    """Return critical cycles and first interior / tail windows for one chain."""
    chunks = (sequence + 63) // 64
    full = _prices(64, tile) if chunks > 1 else None
    tail_rows = sequence - 64 * (chunks - 1)
    tail = full if full and tail_rows == 64 else _prices(tail_rows, tile)
    deadlines = (startup, 0.0, 0.0)
    timeline = []
    for index in range(min(chunks - 1, 3)):
        deadlines, sample = _advance(deadlines, full, first=index == 0)
        timeline.append(sample)
    if chunks > 4:
        deadlines = _repeat(deadlines, full, chunks - 4)
    deadlines, sample = _advance(deadlines, tail, first=chunks == 1, last=True)
    timeline.append(sample)
    return deadlines[0], timeline
