"""Extend the observed second-row issue spans on each AIC to a 128-row team."""
import argparse
import json
from pathlib import Path

from calibration.mx_matmul.collect import trace_events

HERE = Path(__file__).resolve().parent


def extrapolate(arm, raw):
    receipt = json.loads((HERE/f"v113_control_{arm}.json").read_text())
    teams = []
    for path in sorted((raw/arm/"profile").glob("OPPROF_*/simulator/core*.cubecore0/trace.json")):
        stamps = sorted((round(e["ts"]*1650), round((e["ts"]+e["dur"])*1650))
                        for e in trace_events(path) if e.get("ph")=="X" and e["name"]=="MMAD_MX")
        if len(stamps)!=64:
            raise ValueError(f"missing native Q64 events in {path}")
        join = stamps[32][0]-stamps[31][0]
        span = stamps[63][0]-stamps[32][0]
        teams.append(dict(core=path.parent.name,second_row_span=span,row_transition=join,
                          final_retire=stamps[-1][1],projected_retire=stamps[-1][1]+126*(span+join)))
    if len(teams)!=32:
        raise ValueError("need all 32 physical AICs")
    post_mmad = receipt["last_tick"]-max(t["final_retire"] for t in teams)
    period = max(teams,key=lambda t:t["projected_retire"])
    return dict(arm=arm,observed_q64_us=receipt["device_us"],source_trace=receipt["trace_set_sha256"],
                origin=receipt["first_tick"],observed_tail_cycles=post_mmad,critical_aic=period["core"],
                projected_q4096_us=(period["projected_retire"]-receipt["first_tick"]+post_mmad)/1650,
                per_team_cycles=teams)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw",type=Path)
    args=parser.parse_args()
    data=dict(scope="950DT only. Source kernel Q64 per team = 2 full rows. Up to Q4096 assume the same 32 teams, row counts, per-team intervals, handoffs and final drain. A consistently slow team remains slow in this scenario; not an actual Q4096 window.",
              arms=[extrapolate(arm,args.raw) for arm in ("shared","moving","rotated")])
    (HERE/"v113_warm_rows.json").write_text(json.dumps(data,indent=2)+"\n")
