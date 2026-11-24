"""Summarize same-AIC MMAD issue cycles using only archived through-v1.12 runs.

Invoke from the repository root. Input windows are named explicitly so new
v1.13 holdouts cannot accidentally enter the calibration.
"""
import json
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = {"dt32_shared": "candidate-v20-wave", "pr32_shared": "candidate-v112-wave/pr",
        "dt64_varied": "candidate-v112-wave/dt"}


def read(name):
    return json.loads((HERE / name).read_text())


def summarize(tag, run):
    fixture = {"dt64_varied": "dt", "pr32_shared": "pr"}.get(tag)
    meta, match = {}, {}
    if fixture:
        meta = read(f"v112_wave_{fixture}_fixture.json")["metadata_records"]
        match = {r["physical"]: r["matching_logical_teams"]
                 for r in read(f"v112_wave_{fixture}.json")["physical_teams"]}
    sim, = (ROOT / "runs" / run / "profile").glob("OPPROF_*/simulator")
    rows = []
    for path in sorted(sim.glob("core*.cubecore0/trace.json")):
        events = json.loads(path.read_text())["traceEvents"]
        issues = sorted(round(e["ts"] * 1650) for e in events if e.get("name") == "MMAD_MX")
        if not issues:
            continue
        offsets = {meta[str(i)][3] for i in match[path.parent.name]} if fixture else {0}
        assert len(offsets) == 1, (path, offsets)
        offset = offsets.pop()
        within_first, within_carry, handoff = [], [], []
        for i in range(1, len(issues)):
            gap = issues[i]-issues[i-1]
            rank = offset+i
            dest = handoff if rank%32 == 0 else within_first if rank<32 else within_carry
            dest.append(gap)
        rows.append(dict(core=path.parent.name, start_tile_mod_32=offset,
                         first_count=len(within_first), first_mean=mean(within_first) if within_first else None,
                         carry_count=len(within_carry), carry_mean=mean(within_carry) if within_carry else None,
                         handoff=handoff, final_ring_mean=mean([b-a for a,b in zip(issues[-5:],issues[-4:])]), first_issue=issues[0], last_issue=issues[-1]))
    # Median across workers of the elapsed cycles per useful *interval* on
    # that worker, including bubbles. All weights/counts are kept for audit.
    def column(col):
        return median(r[col] for r in rows if r[col] is not None)
    cuts = [v for r in rows for v in r["handoff"]]
    return dict(run=run, core_count=len(rows), issues=sum(r["first_count"]+r["carry_count"]+len(r["handoff"])+1 for r in rows),
                first_median_team_mean_cycles=column("first_mean"),
                carried_median_team_mean_cycles=column("carry_mean") if cuts else None,
                handoff_median_cycles=median(cuts) if cuts else None,
                handoff_count=len(cuts), final_ring_median_team_mean_cycles=column("final_ring_mean"), teams=rows)


if __name__ == "__main__":
    result = {tag: summarize(tag, run) for tag, run in RUNS.items()}
    (HERE/"v113_mechanism.json").write_text(json.dumps(result, indent=2)+"\n")
    for tag, value in result.items():
        print(tag, {k:v for k,v in value.items() if k!="teams"})
