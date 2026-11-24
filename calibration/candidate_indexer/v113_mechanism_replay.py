"""Replay only pre-freeze controls, then benchmark public calls on unique shapes."""
import json
from math import ceil
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from operator_model import estimate, load_chip
from operator_model.operators.attention.flash_sparse import IMPLEMENTATION

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE/(name+".json")).read_text())


def price(shape, chip, attrs):
    return estimate("candidate_filtered_indexer",shape,dict(implementation=IMPLEMENTATION,**attrs),
                    chip=chip,dtype="mxfp4")


def main():
    prefill = []
    for name,known in (("prefill-wave",1),("v112_wave_pr",1),("v112_wave_dt",0)):
        rec = read(name)
        ca = rec["device_us"]
        quotes = []
        for value in (None,known):
            opts = {"table_pages":8201}
            if value is not None:
                opts["candidate_order_overlap"] = value
            e = price(rec["shape"],rec["sku"],opts)
            quotes.append(dict(order_overlap_input=value,prediction_us=e.latency_us,signed_pct=100*(e.latency_us/ca-1),
                               team_phase=e.projection["team_phase_pair_fraction"],
                               stage_us=e.projection["stages_us"]))
        prefill.append(dict(source=name+".json",shape=rec["shape"],sku=rec["sku"],ca_us=ca,quotes=quotes))
    single = []
    for rec in read("v112_merge_replay")["records"]:
        shape = (1,1,32,128,rec["L"],2048,512)
        e = price(shape,rec["sku"],
                  dict(candidate_limit=rec["candidate_count"],table_pages=rec["table_width"]))
        assert abs(e.latency_us-rec["v112_us"]) < 1e-9
        single.append(dict(shape=list(shape),candidate_limit=rec["candidate_count"],table_pages=rec["table_width"],
                           sku=rec["sku"],v112_equals_v113_us=e.latency_us,ca_us=rec["ca_us"]))
    timings = {}
    for sku in ("Ascend950DT_9582","Ascend950PR_957b","Ascend970DT"):
        chip = load_chip(sku)
        samples = []
        for i in range(250):
            # Each call has a unique geometry. Reuse only parsed chip and static rate files.
            shape = (1+i%11,(1,32,64,4096,1<<17)[i%5],32,128,(1<<20)+i,2048,512)
            start = perf_counter_ns()
            result = price(shape,chip,{})
            samples.append((perf_counter_ns()-start)/1e6)
            assert result.latency_us > 0
        ms = sorted(samples)
        timings[sku] = dict(n=len(ms),p50_ms=median(ms),p95_ms=ms[ceil(len(ms)*.95)-1],
                            p99_ms=ms[ceil(len(ms)*.99)-1],max_ms=max(ms))
    out = dict(role="pre-v1.13 absorbed development timing evidence; earlier samples are replays",
               prefill=prefill,single_query=single,
               wall_protocol="unique geometry per call; preloaded chip and static rate files; public estimate; one process; ms",
               query_wall=timings)
    (HERE/"v113_mechanism_replay.json").write_text(json.dumps(out,indent=2)+"\n")


if __name__ == "__main__":
    main()
