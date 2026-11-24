"""Build expected per-team folded addresses from frozen inputs and native metadata."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from v112_wave_probe import ROWS, WORKERS, checksum, schedule, team_ranges, varied_inputs, wave_inputs


def fingerprint(values):
    packed = bytearray()
    for block, count in sorted(Counter(map(int,values)).items()):
        packed.extend(struct.pack('<II',block,count))
    return hashlib.sha256(packed).hexdigest()


def components(case):
    inputs, table, _ = wave_inputs() if case == 'pr' else varied_inputs()
    frozen = json.loads(Path(__file__).with_name(f'v112_wave_{case}_fixture.json').read_text())
    ids = inputs[-1][:,0]
    assert frozen['input_sha256']['ids'] == checksum(inputs[-1])
    assert frozen['input_sha256']['table'] == checksum(table)
    physical = table[0,ids//16]*16 + ids%16
    md = schedule(ROWS[case],WORKERS[case])
    assert checksum(md) == frozen['metadata_sha256']
    tokens = physical.reshape(-1)
    teams=[]
    within_team_actual_part_intersections=[]
    for budget in team_ranges(md, ROWS[case]):
        lo,hi=budget['first_tile']*64,budget['end_tile_exclusive']*64
        row_parts=[]
        for row in range(budget['first_row'],budget['last_row']+1):
            row_start=row*2048
            data=set(map(int,tokens[max(lo,row_start):min(hi,row_start+2048)]))
            pages={block//16 for block in data}
            if row_parts:
                prev_blocks,prev_pages=row_parts[-1]
                within_team_actual_part_intersections.append(dict(team=budget['logical_team'],row_pair=[row-1,row],
                    first_pages=len(prev_pages),second_pages=len(pages),shared_pages=len(prev_pages&pages),
                    first_blocks=len(prev_blocks),second_blocks=len(data),shared_blocks=len(prev_blocks&data)))
            row_parts.append((data,pages))
        teams.append(dict(logical_team=budget['logical_team'], expected_tiles=budget['tile_count'],
                          folded_physical_block_multiset_sha256=fingerprint(tokens[lo:hi])))
    return dict(shape_tag=case,min_block=int(tokens.min()),max_block=int(tokens.max()),
                distinct_source_blocks=int(np.unique(tokens).size),metadata_sha256=checksum(md),expected=teams,
                within_team_actual_part_intersections=within_team_actual_part_intersections)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.write_text(json.dumps({case:components(case) for case in ROWS},indent=2)+'\n')
