"""Verify the plotted tiles and every CA interval against the original MatMulV3 trace."""
import argparse
from collections import defaultdict
import hashlib
import json
from math import isclose
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RECEIPT = ROOT / "results/v0.20/examples/matmul-tile-pages.json"
ENGINES = {
    "MMAD": "MMAD",
    "MOV_OUT_TO_L1_MULTI_ND2NZ": "MTE2",
    "LOAD_2Dv2": "MTE1",
    "FIX_L0C_TO_DST": "FIXP",
}


def extract(path, receipt):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["trace_sha256"]:
        raise ValueError("The source trace differs from the recorded measurement")
    events = [e for e in json.loads(raw)["traceEvents"] if e["ph"] == "X"]
    origin = min(e["ts"] for e in events)
    terminal = max(events, key=lambda e: e["ts"] + e["dur"])
    by_core = defaultdict(lambda: defaultdict(list))
    for e in events:
        if e["name"] in ENGINES:
            by_core[e["pid"]][ENGINES[e["name"]]].append((
                e["ts"] - origin, e["ts"] + e["dur"] - origin,
                e["args"]["pc_addr"], json.loads(e["args"]["detail"]),
            ))
    _, n, _ = receipt["shape"]
    tiles_by_core = {}
    for core, queues in by_core.items():
        for group in queues.values():
            group.sort(key=lambda event: event[0])
        geometry = receipt["per_core_tiles"][core]
        assert len(queues["FIXP"]) == len(geometry)
        assert len(queues["MMAD"]) == sum(t["mmad"] for t in geometry)
        assert len(queues["MTE2"]) == sum(2 * t["pages"] for t in geometry)
        assert len(queues["MTE1"]) == 2 * len(queues["MMAD"])
        tiles = []
        pages_seen = macs_seen = 0
        for t, tile in enumerate(geometry):
            side, count, pages = tile["size"], tile["mmad"], tile["pages"]
            mmads = queues["MMAD"][macs_seen:macs_seen + count]
            k_step = mmads[0][3]["k"]
            k_page = k_step * (count // pages)
            assert all((e[3]["m"], e[3]["n"], e[3]["k"]) == (side, side, k_step) for e in mmads)
            assert [e[3]["matrix_c_init_val"] for e in mmads] == [1] + [0] * (count - 1)
            assert [e[3]["hw_sets_unit_flag"] for e in mmads] == [0] * (count - 1) + [1]
            write = queues["FIXP"][t]
            assert (write[3]["m"], write[3]["n"]) == (side, side)
            windows = []
            previous = None
            for p in range(pages):
                gm = queues["MTE2"][2*(pages_seen+p):2*(pages_seen+p+1)]
                assert sum(e[3]["processed_bytes"] for e in gm) == tile["read_bytes_per_page"]
                if previous:
                    # A traverses contiguous K. B advances by K rows in the ND matrix.
                    assert [e[2] for e in gm] == [e[2] for e in previous]
                    assert [e[3]["src_addr"] - prev[3]["src_addr"] for e, prev in zip(gm,previous)] == [2*k_page, 2*k_page*n]
                previous = gm
                start = macs_seen + p * count // pages
                stop = start + count // pages
                loads = queues["MTE1"][2*start:2*stop]
                assert [e[3]["dst_mem"] for e in loads] == ["L0A", "L0B"] * (count // pages)
                for e in loads:
                    descriptor = gm[0 if e[3]["dst_mem"] == "L0A" else 1][3]
                    assert descriptor["dst_addr"] <= e[3]["src_addr"] < descriptor["dst_addr"] + descriptor["processed_bytes"]
                    assert e[3]["processed_bytes"] == 2 * side * k_step
                windows.append({key: [list(e[:2]) for e in seq] for key, seq in (
                    ("MTE2", gm), ("MTE1", loads), ("MMAD", queues["MMAD"][start:stop]))})
            tiles.append({"pages": windows, "write": list(write[:2])})
            pages_seen += pages
            macs_seen += count
        tiles_by_core[core] = tiles
    assert set(by_core) == set(receipt["per_core_tiles"])
    assert sum(len(q["MMAD"]) for q in by_core.values()) == receipt["counts"]["CUBE"]
    assert isclose(terminal["ts"] + terminal["dur"] - origin, receipt["ca_us"], abs_tol=1e-6)
    assert terminal["pid"] == receipt["core"]
    assert tiles_by_core[terminal["pid"]] == receipt["ca_tiles"]
    return len(by_core), terminal["pid"]


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("trace", type=Path, nargs="?", help="Original msOpProf merged trace.json")
    receipt = json.loads(RECEIPT.read_text())
    path = args.parse_args().trace or ROOT / receipt["stage_timeline_source"]["ca"]
    cores, last = extract(path, receipt)
    print(f"Verified {cores} AICs, {receipt['counts']['CUBE']} MMADs and all plotted spans on {last}.")
