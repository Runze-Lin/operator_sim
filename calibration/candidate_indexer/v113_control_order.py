"""Form per-lane expected source issue sequences from the pre-CA matched inputs."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from v113_control_probe import ROWS, WORKERS, checksum, make_fixture, schedule, team_ranges


def main():
    _, table, all_ids = make_fixture()
    base = Path(__file__).parent
    md = schedule(ROWS,WORKERS)
    output = {}
    for arm, ids in all_ids.items():
        frozen = json.loads((base/f'v113_control_{arm}_fixture.json').read_text())
        assert checksum(ids) == frozen['input_sha256']['ids'] and checksum(table)==frozen['input_sha256']['table']
        assert checksum(md) == frozen['metadata_sha256']
        physical = table[0,ids[:,0]//16]*16 + ids[:,0]%16
        tiled = physical.reshape(-1,64)
        digests = []
        for team in team_ranges(md,ROWS):
            shards = tiled[team['first_tile']:team['end_tile_exclusive']]
            digests.append([sha256(shards[:,start:start+32].flatten().astype('<u4').tobytes()).hexdigest() for start in (0,32)])
        # Native _copy_pa_pair accepts forward gaps whose encoded spacing fits 39 bits.
        pairs = physical.reshape(-1,2).astype('int64')
        gap = (pairs[:,1]-pairs[:,0])*544
        double = int(np.count_nonzero((gap>=544) & (gap-544<=2**39-1)))
        output[arm] = dict(candidate_sha256=checksum(ids),ordered_lane_sha256_by_logical_team=digests,
                           source_expected_descriptors_by_burst_count={'1':physical.size-2*double,'2':double})
    (base/'v113_control_order.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    main()
