"""Break down QSLI gather, score and merge windows from physical-core traces."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

from calibration.__main__ import require
from calibration.candidate_indexer.v112_merge_trace import MERGE_PART_READ, MERGE_OUT
from calibration.candidate_indexer.v112_wave_report import FREQ, intervals_peak, scan
from calibration.msopprof import CORE


def observed(raw, log, rows=32):
    prof,=raw.glob('profile/OPPROF_*')
    roster=CORE.findall(log.read_text())
    require('All task success.' in log.read_text() and roster,'Missing official window')
    parts={n:scan(prof/'simulator'/n/'trace.json',float(t)) for n,t in roster}
    origin=min(r['first'] for r in parts.values())
    end=max(r['last'] for r in parts.values())
    issuers={n:r for n,r in parts.items() if r['mmads']}
    mmads=[t for r in issuers.values() for t in r['mmads']]
    require(len(mmads)==rows*32,'Unexpected score volume')
    gaps=sorted(b[0]-a[0] for r in issuers.values() for a,b in zip(r['mmads'],r['mmads'][1:]))
    first=min(a for a,b in mmads)
    last=max(b for a,b in mmads)
    gathers=[t for r in parts.values() for t in r['reads']]
    starts=[a for a,b in gathers]
    stops=[b for a,b in gathers]
    addresses=Counter()
    for r in parts.values():addresses.update(r['blocks'])
    fd_starts=[a for r in parts.values() for a,b in r['merge'].get(MERGE_PART_READ[0],[])]
    fd_ends=[b for r in parts.values() for a,b in r['merge'].get(MERGE_OUT[-1],[])]
    tile_counts=Counter(len(r['mmads']) for r in issuers.values())
    issuing_window_us=[(r['mmads'][-1][1]-r['mmads'][0][0])/FREQ for r in issuers.values()]
    result=dict(device_us=(end-origin)/FREQ, first_mmad_us=(first-origin)/FREQ, last_mmad_retire_us=(last-origin)/FREQ,
                first_to_last_mmad_us=(last-first)/FREQ,last_mmad_to_full_end_us=(end-last)/FREQ,
                same_core_issue_gap_p10_p50_p90_cycles=[gaps[len(gaps)//10],median(gaps),gaps[len(gaps)*9//10]],
                per_issuing_core_mmad_window_median_us=median(issuing_window_us),observed_mmad_tiles_per_core=dict(tile_counts),
                first_gather_descriptor_us=(min(starts)-origin)/FREQ,last_gather_retirement_us=(max(stops)-origin)/FREQ,
                peak_concurrent_gather_aivs=intervals_peak([r['reads'] for r in parts.values()]),
                peak_concurrent_mmad_aics=intervals_peak([r['mmads'] for r in issuers.values()]),
                paid_folded_gather_bytes=sum(addresses.values())*544,distinct_folded_source_bytes=len(addresses)*544,
                fd_first_partial_read_us=(min(fd_starts)-origin)/FREQ if fd_starts else None,
                fd_last_output_write_us=(max(fd_ends)-origin)/FREQ if fd_ends else None,
                fd_read_to_write_window_us=(max(fd_ends)-min(fd_starts))/FREQ if fd_starts and fd_ends else None,
                last_finishing_physical=max(parts,key=lambda n:parts[n]['last']),
                last_mmad_finishing_physical=max(issuers,key=lambda n:issuers[n]['mmads'][-1][1]))
    if rows == 64:
        require(tile_counts == {64:32}, 'Cross-row steps require paired full rows on every team')
        streams = [r['mmads'] for r in issuers.values()]
        handoff = [times[32][0]-times[31][0] for times in streams]
        # Each real host-metadata team owns exactly two complete consecutive rows.
        result['two_row_team_slices'] = dict(
            row0_median_first_issue_to_last_retire_us=median((times[31][1]-times[0][0])/FREQ for times in streams),
            row1_median_first_issue_to_last_retire_us=median((times[63][1]-times[32][0])/FREQ for times in streams),
            row0_last_issue_to_row1_first_issue_min_median_max_cycles=[min(handoff),median(handoff),max(handoff)])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw',type=Path)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--rows',type=int,choices=(32,64),default=32)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.write_text(json.dumps(observed(args.raw,args.log,args.rows),indent=2)+'\n')
