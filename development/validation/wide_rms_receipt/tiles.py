"""Recover each critical AIV's tensor regions and the v0.21 RMS write cadence."""
from bisect import bisect_right
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import io
import json
from math import ceil, isclose
from pathlib import Path
import sys

from receipt import ROOT, OUT, SHAPE, read
from operator_model import estimate
from operator_model.rms_v12 import parameters, schedule_writes
from operator_model.chips import load_chip

sys.path.insert(0, str(ROOT / "development/validation/sku17"))
from campaign_collect import events


class HashedTrace:
    """Expose the original chunked event reader while hashing the same byte stream."""
    def __init__(self, path):
        self.path = path
        self.hash = hashlib.sha256()
        self.nbytes = 0

    @contextmanager
    def open(self):
        owner = self
        with self.path.open("rb") as raw:
            class Reader(io.RawIOBase):
                def readable(self):
                    return True

                def readinto(self, buf):
                    n = raw.readinto(buf)
                    owner.hash.update(memoryview(buf)[:n])
                    owner.nbytes += n
                    return n

            with io.TextIOWrapper(io.BufferedReader(Reader()), encoding="utf-8") as stream:
                yield stream
                while stream.read(1024**2):
                    pass


def model_tiles(chip_name):
    chip = load_chip(chip_name)
    a = estimate("rmsnorm", SHAPE, dtype="bf16", chip=chip_name)
    d = a.projection["timing_diagnostics"]
    plan = d["target_plan"]
    p = dict(parameters())
    for q in ("reduce", "normalize"):
        fixed, per_group = p[q]
        p[q] = [fixed, per_group * d["vector_price_cycles_per_reference_cycle"]]
    from operator_model.rms_resource import STAT_SOLO, STAT_PER_NEIGHBOUR
    p["y"] = [p["y"][0], d["write_cycles_per_kib"]]
    p["rstd"] = [STAT_SOLO, STAT_PER_NEIGHBOUR]
    neighbours = d["statistic_neighbours"]
    clock = 1000 * chip.frequency_ghz
    width = SHAPE[1]
    tiles = plan["ub_tile_rows"]
    raw, hidden = schedule_writes(width, plan["rows_per_core"], plan["active_aiv"], tiles, p, neighbours)
    assert isclose(raw, d["store_union_cycles"]) and isclose(hidden, d["store_eligible_overlap_cycles"])
    rows = []
    cursor = 0.
    for i, n in enumerate(tiles):
        groups = n * ceil(width/128)
        reduce = None
        if i:
            begin = cursor
            cursor += p["reduce"][0] + p["reduce"][1] * groups
            reduce = [begin/clock, cursor/clock]
        rs = cursor
        rs_end = rs + p["rstd"][0] + p["rstd"][1] * neighbours
        cursor += p["normalize"][0] + p["normalize"][1] * groups
        y_end = max(rs_end, cursor + p["y"][0] + p["y"][1] * (2 * width * n / 1024))
        rows.append(dict(rows=n, reduce=reduce, normalize=[rs/clock,cursor/clock],
                         rstd=[rs/clock,rs_end/clock], y=[cursor/clock,y_end/clock]))
    return dict(ca_independent_total_us=a.latency_us, stages_us=d["stages_us"],
        local_us=sum(t for q,t in d["stages_us"].items() if q!="shared_io"), tiles=rows,
        active_aiv=a.active_aiv,rows_per_core=plan["rows_per_core"],frequency_ghz=chip.frequency_ghz,
        external_floor_us=d["external_floor_cycles"]/clock,retained_write_bytes=d["assumed_retained_write_bytes"],
        external_bytes=d["modeled_external_bytes"], prices=p, vector_groups_per_row=ceil(width/128))


def read_core(receipt):
    stage, = read(ROOT / receipt["qualified"])["stages"]
    core = receipt["last_core"]
    path = (ROOT / stage["trace"]).parent / core / "trace.json"
    source = HashedTrace(path)
    xmin, xmax = float("inf"), -float("inf")
    dma = defaultdict(list)
    vf = defaultdict(list)
    # Bound PC identities to the measured program binary. The three vector
    # sequences contain VCADD, VSQRT and VMUL×2 + VCVT + VST respectively.
    expected_bin = "0926fe478e5f718ef49241c08ef00b665797690f0f7f22c67fdc25044e0cb54d"
    assert stage["binary_sha256"] == expected_bin
    functions = {
        "0x9000d12300":("reduce_begin","RV_VAG"), "0x9000d12678":("reduce_end","RV_SEND"),
        "0x9000d12700":("norm_begin","RV_VAG"), "0x9000d12744":("norm_end","RV_SEND"),
    }
    n = 0
    for event in events(source):
        if event["ph"] != "X":
            continue
        n += 1
        a = event["ts"]
        b = a + event["dur"]
        xmin, xmax = min(xmin, a), max(xmax, b)
        pc = event["args"]["pc_addr"]
        if pc in functions:
            kind, name = functions[pc]
            assert event["name"] == name
            vf[kind].append((a,b))
        if event["name"] != "MOV_SRC_TO_DST_ALIGNv2":
            continue
        d = json.loads(event["args"]["detail"])
        assert d["processed_bytes"] == d["burst_len"] * d["burst_number"]
        queue = d["instr_type"]
        assert (queue,d["src_mem"],d["dst_mem"]) in {("MTE2","OUT","UB"),("MTE3","UB","OUT")}
        addr, ub = ((d["src_addr"], d["dst_addr"]) if queue=="MTE2" else (d["dst_addr"], d["src_addr"]))
        dma[pc].append(dict(bytes=d["processed_bytes"],gm_addr=addr,ub_addr=ub,
                            queue=queue,begin_us=a,end_us=b))
    assert source.nbytes == path.stat().st_size
    lo, hi = stage["windows"][core]
    assert abs((xmax-xmin) - (hi-lo)) < .0001 and hi == receipt["ca_us"]
    origin = xmin - lo
    for rows in dma.values():
        for r in rows:
            r["begin_us"] -= origin
            r["end_us"] -= origin
    return dict(path=str(path.relative_to(ROOT)),sha256=source.hash.hexdigest(),bytes=source.nbytes,
                core=core,ca_us=hi,core_begin_us=lo,raw_origin_us=origin,x_events=n), dma, vf


def vector_windows(vf, name, n, origin):
    starts = sorted(a for a,_ in vf[name+"_begin"])
    assert len(starts) == n
    ends = [[] for _ in starts]
    for t,end in vf[name+"_end"]:
        group = bisect_right(starts,t)-1
        assert group >= 0
        ends[group].append(end)
    # Each programmed SEND appears on four issue/execute sublanes in this export.
    assert all(len(ts)==4 for ts in ends)
    return [[start-origin,max(last)-origin] for start,last in zip(starts,ends)]


def actual_tiles(receipt):
    meta, dma, vf = read_core(receipt)
    width = SHAPE[1]
    # GM is contiguous ND. Independent PC families, lengths and contiguous
    # GM increments identify x, rstd and y even when two UB slots alternate.
    pc = dict(tiling="0x9000d10e9c",gamma="0x9000d11878",x="0x9000d11a48",
              rstd="0x9000d11f38",y="0x9000d12128")
    assert set(dma) == set(pc.values())
    families = {k:sorted(dma[a],key=lambda r:r["gm_addr"]) for k,a in pc.items()}
    tiling, = families["tiling"]
    gamma, = families["gamma"]
    assert tiling["bytes"] == 64 and gamma["bytes"] == width*2
    n = len(families["x"])
    for k in ("x","rstd","y"):
        f = families[k]
        assert len(f)==n and all(a["gm_addr"]+a["bytes"]==b["gm_addr"] for a,b in zip(f,f[1:]))
        assert sorted(f,key=lambda r:r["begin_us"]) == f
        assert len({r["ub_addr"] for r in f}) == 2
        assert all(r["ub_addr"] == f[i%2]["ub_addr"] for i,r in enumerate(f))
    reduce = vector_windows(vf,"reduce",n,meta["raw_origin_us"])
    norm = vector_windows(vf,"norm",n,meta["raw_origin_us"])
    tiles = []
    for i in range(n):
        x,rs,y = (families[k][i] for k in ("x","rstd","y"))
        count, rem = divmod(x["bytes"], 2*width)
        assert rem == 0 and y["bytes"] == x["bytes"] and rs["bytes"] == count*4
        assert reduce[i][0] >= x["end_us"] and norm[i][0] >= reduce[i][0]
        assert rs["begin_us"] >= reduce[i][0] and y["begin_us"] >= norm[i][0]
        tiles.append(dict(rows=count,x=x,rstd=rs,y=y,reduce=reduce[i],normalize=norm[i]))
    return dict(**meta,pc=pc,setup=dict(tiling=tiling,gamma=gamma),tiles=tiles)


def main():
    receipt = read(OUT)
    data = {}
    for side, r in receipt["endpoints"].items():
        model,ca = model_tiles(r["chip"]),actual_tiles(r)
        assert [t["rows"] for t in model["tiles"]] == [t["rows"] for t in ca["tiles"]]
        data[side] = dict(model=model,ca=ca)
    (OUT.parent/"tiles.json").write_text(json.dumps(data, indent=1)+"\n")


if __name__ == "__main__":
    main()
