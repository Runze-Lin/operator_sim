"""A uniform per-core tile stream with independent issue ports and finite buffers."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from itertools import repeat


@dataclass(frozen=True)
class TransferProfile:
    opcode: str
    direction: str
    layout: str
    dtype: str
    bytes: int
    issue_cycles: float
    response_cycles: float
    source: str

    def __post_init__(self):
        for name in ("opcode", "direction", "layout", "dtype", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"transfer profile requires {name}")
        if type(self.bytes) is not int or self.bytes <= 0:
            raise ValueError("transfer profile requires a positive descriptor byte count")
        for value in (self.issue_cycles, self.response_cycles):
            if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                raise ValueError("transfer issue and response cycles must be positive and finite")


def stream_bound(chip, program):
    """Lower bound for a declared critical worker, in device cycles, O(1) per query."""
    required = {"read_profile", "write_profile", "tiles", "slots", "compute_cycles"}
    if not isinstance(program, Mapping) or set(program) != required:
        raise ValueError(f"transfer_pipeline requires {sorted(required)}")
    n, slots, compute = (program[k] for k in ("tiles", "slots", "compute_cycles"))
    if any(type(i) is not int or i <= 0 for i in (n, slots)):
        raise ValueError("transfer tiles and slots must be positive integers")
    if type(compute) not in (int, float) or not isfinite(compute) or compute < 0:
        raise ValueError("compute_cycles must be finite and nonnegative")
    read = chip.transfer_profiles.get(program["read_profile"])
    write = chip.transfer_profiles.get(program["write_profile"])
    if read is None or write is None:
        raise ValueError("missing exact transfer profile on selected chip")
    if read.direction == write.direction:
        raise ValueError("transfer pipeline assumes independent read and write ports")
    interval = max(read.issue_cycles, compute, write.issue_cycles)
    drain = read.response_cycles + compute + write.response_cycles
    waits = (n - 1) // slots
    cycles = drain + (n - 1) * interval + waits * max(0.0, drain - slots * interval)
    return dict(
        **program,
        resource_interval_cycles=interval,
        drain_cycles=drain,
        cycles=cycles,
        read_source=read.source,
        write_source=write.source,
        bound_us=cycles / (1000 * chip.frequency_ghz),
    )


def replay_tiles(read, write, compute_cycles, tiles, slots):
    """Offline audit. Yield each occupancy interval on the three engines."""
    if any(type(i) is not int or i <= 0 for i in (tiles, slots)):
        raise ValueError("tiles and slots must be positive integers")
    yield from replay_program(repeat((read, write, compute_cycles), tiles), slots)


def replay_program(program, slots):
    """Offline variable-geometry replay, including a partial head or tail tile."""
    if type(slots) is not int or slots < 1:
        raise ValueError("slots must be positive")
    free = {}
    read_port = compute_port = write_port = 0.0
    for i, (read, write, compute_cycles) in enumerate(program):
        if type(compute_cycles) not in (int, float) or not isfinite(compute_cycles) or compute_cycles < 0:
            raise ValueError("compute_cycles must be finite and nonnegative")
        issue = max(free.get(i % slots, 0.0), read_port)
        ready = issue + read.response_cycles
        compute = max(ready, compute_port)
        store = max(compute + compute_cycles, write_port)
        end = store + write.response_cycles
        read_port = issue + read.issue_cycles
        compute_port = compute + compute_cycles
        write_port = store + write.issue_cycles
        free[i % slots] = end
        yield dict(
            tile=i,
            read_issue=issue,
            read_ready=ready,
            compute_start=compute,
            compute_end=compute_port,
            write_issue=store,
            write_retire=end,
        )
