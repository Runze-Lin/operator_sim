"""Post-freeze diagnostic. Only three explicitly named, pre-v1.13 raw directories are opened."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from pathlib import Path
from statistics import median
import json

from calibration.candidate_indexer.v113_control_report import FREQ, coincidence, folded_instructions

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUTS = {"dt32_shared": ("candidate-v20-wave","prefill-wave"),
          "pr32_shared": ("candidate-v112-wave/pr","v112_wave_pr"),
          "dt64_varied": ("candidate-v112-wave/dt","v112_wave_dt")}
BIN_CYCLES = (128,512,2048)


def load(name):
    return json.loads((HERE/(name+".json")).read_text())


def summarize(issues,origin):
    lo = min(t for t, _, _ in issues)
    hi = max(t for t, _, _ in issues)
    n = len(issues)
    active_cycles = hi-lo+1
    density = []
    for width in BIN_CYCLES:
        bins = Counter((t-origin)//width for t, _, _ in issues)
        counts = sorted(bins[i] for i in range(min(bins),max(bins)+1))
        density.append(dict(bin_cycles=width,active_envelope_bin_count=len(counts),
                            p50_records=median(counts),p90_records=counts[ceil(.9*len(counts))-1],
                            max_records=max(counts)))
    by_team = defaultdict(list)
    for t,team,_ in issues:
        by_team[team].append(t)
    rate = n/active_cycles
    per_team = [len(times)/(max(times)-min(times)+1) for times in by_team.values()]
    return dict(records=n,distinct_records=len({addr for _,_,addr in issues}),first_issue_cycle=lo-origin,
                last_issue_cycle=hi-origin,record_issue_envelope_us=active_cycles/FREQ,records_per_cycle=rate,
                logical_operand_GBs=rate*FREQ*544/1000,
                median_team_records_per_cycle=median(per_team),density=density,
                actual_same_address=[coincidence(issues,origin,w) for w in BIN_CYCLES])


def run(tag,pool):
    raw,receipt=INPUTS[tag]
    rec=load(receipt)
    folder, = (ROOT/"runs"/raw/"profile").glob("OPPROF_*/simulator")
    if tag=="dt32_shared":
        # Reference row-window ticks are relative to device zero. Recover it
        # from the same physical AIC's first absolute tick in mechanism.json.
        zero=load("v113_mechanism")[tag]["teams"][0]["first_issue"]-rec["rows"][0]["first_issue_cycles"]
    else:
        zero=rec["first_tick"]
    stats={r["core"].split(".")[0]:r for r in load("v113_mechanism")[tag]["teams"]}
    by_tag=defaultdict(list)
    bursts_by_scope=defaultdict(Counter)
    lane_handoff=[]
    shards=sorted(folder.glob("core*.veccore*/trace.json"))
    for path,descriptors in zip(shards,pool.map(folded_instructions,shards),strict=True):
        name=path.parent.name.split(".")[0]
        team=int(name[4:])
        expected=stats[name]
        seen=0
        first_row_end=32-expected["start_tile_mod_32"]
        previous=None
        previous_tick=None
        for tick, _, end, addrs in descriptors:
            label_at_start = "first_fragment" if seen//32 < first_row_end else "following_fragments"
            bursts_by_scope[label_at_start][len(addrs)]+=1
            bursts_by_scope["all"][len(addrs)]+=1
            for addr in addrs:
                # Each paired AIV produces 32 of 64 folded operands per tile.
                label="first_fragment" if seen//32 < first_row_end else "following_fragments"
                assert label==label_at_start, "folded descriptor crosses row-fragment boundary"
                by_tag[label].append((tick,team,addr))
                by_tag["all"].append((tick,team,addr))
                if previous=="first_fragment" and previous!=label:
                    lane_handoff.append(tick-previous_tick)
                previous,previous_tick=label,tick
                seen+=1
        assert seen==32*(expected["first_count"]+expected["carry_count"]+len(expected["handoff"])+1)
    compiled={k:summarize(v,zero) for k,v in by_tag.items()}
    for scope,values in compiled.items():
        counts=bursts_by_scope[scope]
        values["descriptors_by_burst_count"]=dict(counts)
        values["records_per_descriptor"]=values["records"]/sum(counts.values())
    assert compiled["all"]["distinct_records"]*544==load(
        "v112_wave_timeline_"+{"dt32_shared":"dt32","pr32_shared":"pr32","dt64_varied":"dt64"}[tag])["distinct_folded_source_bytes"]
    return dict(raw=raw,receipt=receipt+".json",frozen_trace_set_sha256=rec["trace_set_sha256"],
                origin=zero,physical_aiv_count=len(shards),descriptors_by_burst_count=bursts_by_scope["all"],
                median_lane_last_to_next_first_record_issue_cycles=median(lane_handoff) if lane_handoff else None,
                scopes=compiled)


def landmarks(tag,rec):
    """Reconstruct the frozen v1.13 model's binding team from the old metadata."""
    from operator_model import estimate,load_chip
    from operator_model.operators.attention.flash_sparse import IMPLEMENTATION,_native_rates
    from operator_model.operators.attention.indexer import _prices

    old=load(INPUTS[tag][1])
    chip=load_chip(old["sku"])
    overlap={"dt32_shared":1,"pr32_shared":1,"dt64_varied":0}[tag]
    estimate_result=estimate("candidate_filtered_indexer",old["shape"],
                             {"implementation":IMPLEMENTATION,"candidate_order_overlap":overlap,"table_pages":8201},
                             dtype="mxfp4",chip=chip)
    plan=estimate_result.projection
    stage=plan["tile_service_cycles"]
    us=plan["stages_us"]
    fixed,group=_prices()["histogram_span"]
    vec=_prices()["vector_ops_cycle"]*2/(chip.vector_tops["16"]*1000/(chip.vector_cores*chip.frequency_ghz))
    fill=us["startup"]+sum(stage[key] for key in ("gather","publish","cube_read"))/FREQ
    one_tile=stage["fix"]+stage["vector"]+(fixed+2*group)*vec
    partial_extra=max(0,_native_rates()["split_row"]["one_tile_score_drain_cycles"]-one_tile)
    candidates=[]
    for t in load("v113_mechanism")[tag]["teams"]:
        n=t["first_count"]+t["carry_count"]+len(t["handoff"])+1
        first=t["start_tile_mod_32"]
        end_rank=(first+n-1)%32
        final_tiles=min(n,end_rank+1)
        partial=end_rank<31 or (not t["handoff"] and first>0)
        tail=(stage["fix"]+stage["vector"]+(fixed+2*final_tiles*group)*vec+partial*partial_extra)/FREQ+us["merge"]
        span=(t["first_count"]*plan["first_fragment_interval_cycles"]
              +t["carry_count"]*plan["carried_fragment_interval_cycles"]
              +len(t["handoff"])*plan["same_document_handoff_cycles"]+stage["cube"])/FREQ
        candidates.append((fill+span+tail,span,tail))
    total,span,tail=max(candidates)
    assert abs(total-estimate_result.latency_us)<1e-8
    trace=load("v112_wave_timeline_"+{"dt32_shared":"dt32","pr32_shared":"pr32","dt64_varied":"dt64"}[tag])
    actual=(trace["first_mmad_us"],trace["last_mmad_retire_us"]-trace["first_mmad_us"],trace["last_mmad_to_full_end_us"])
    rec["landmark_us"]={k:{"ca":a,"model":m,"ca_minus_model":a-m} for k,a,m in
                        zip(("start_to_first_issue","first_issue_to_last_retire","last_retire_to_exit"),
                            actual,(fill,span,tail),strict=True)}


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--jobs",type=int,default=2)
    parser.add_argument("--reuse-traffic",action="store_true",help="refresh frozen-model landmarks on existing traffic census")
    args=parser.parse_args()
    out=HERE/"v113_mechanism_traffic.json"
    if args.reuse_traffic:
        data=json.loads(out.read_text())
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            data={key:run(key,pool) for key in INPUTS}
    for tag,rec in data.items():
        assert rec["frozen_trace_set_sha256"]==load(INPUTS[tag][1])["trace_set_sha256"]
        landmarks(tag,rec)
    out.write_text(json.dumps(data,indent=2)+"\n")
    for tag,rec in data.items():
        for scope,vals in rec["scopes"].items():
            print(tag,scope,vals["records"],round(vals["records_per_cycle"],4),
                  [round(x["request_fraction_with_other_team_same_address_in_bin"],6) for x in vals["actual_same_address"]])
