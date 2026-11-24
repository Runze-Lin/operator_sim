"""Frozen QSLI matched inputs. Only the candidate ID tensor differs between arms."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import struct

import numpy as np
import torch
from fused_qsli_probe import invoke, npu
from long_row_probe import CONTEXT, PAGE, SELECTED, TABLE_PAGES, TOPK, checksum
from prefill_wave_probe import set_nibble
from v112_wave_probe import schedule, team_ranges

ARMS = ('shared', 'moving', 'rotated')
ROWS = 64
WORKERS = 32
BLOCK_BYTES = 544
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])


def ids_for(arm, common, pool):
    ids = np.empty((ROWS, 1, SELECTED), np.int32)
    for row in range(ROWS):
        shift = 48*row if arm == 'moving' else 0
        pages = np.r_[common, pool[(shift + np.arange(128)) % pool.size]]
        if arm == 'rotated':
            # Anchor the causal tail at slot 127 so visibility is also invariant.
            ring = np.r_[np.arange(127), np.arange(128,256)]
            pages[ring] = np.roll(pages[ring], 13*row)
        # Every arm/row uses identical issue-position permutations and offsets.
        blocks = (pages[:,None]*16 + 2*np.arange(8) + 1).reshape(-1)
        ids[row,0] = blocks[(37*np.arange(SELECTED)) % SELECTED]
    return ids


def make_fixture():
    n_pages = CONTEXT // PAGE
    physical = (4093*np.arange(n_pages) + 17) % n_pages
    table = np.zeros((1, TABLE_PAGES), np.int32)
    table[0,:n_pages] = physical
    common = np.linspace(0, n_pages-1, 128, dtype=np.int32)
    rest = np.setdiff1d(np.arange(n_pages), common)
    pool = rest[np.linspace(0, rest.size-1, 2048, dtype=np.int32)]
    ids = {arm:ids_for(arm, common, pool) for arm in ARMS}
    q = np.zeros((ROWS,32,64), np.uint8)
    qscale = np.full((ROWS,32,2,2), 127, np.uint8)
    qscale[:,0,0,:] = 126
    weights = np.zeros((ROWS,32), np.float32)
    weights[:,:2] = (.5, 2.)
    keys = np.full((CONTEXT,64), 0x77, np.uint8)
    # All arms use one identical allocation covering the union of relevant keys.
    all_pages = np.r_[common, pool]
    offsets = np.arange(PAGE)
    positions = (all_pages[:,None]*PAGE + offsets).reshape(-1)
    key_rows = physical[positions//PAGE]*PAGE + positions%PAGE
    for row in range(ROWS):
        winners = np.isin(positions%PAGE, (8+row%8, 24+row//8))
        future = positions >= CONTEXT-ROWS+row+1
        odd = positions%2 != 0
        for head, dim in enumerate((row, 64+row)):
            q[row,head,dim//2] = 2 << (4*(dim%2))
        set_nibble(keys, key_rows, row, np.where(future,7,np.where(winners,np.where(odd,7,10),5)))
        set_nibble(keys, key_rows, 64+row, np.where(future,7,np.where(winners,np.where(odd,10,1),10)))
    folded = np.empty((n_pages,PAGE//8,BLOCK_BYTES), np.uint8)
    folded[:,:,:512] = keys.reshape(n_pages,PAGE//8,512)
    kscales = folded[:,:,512:].reshape(-1,8,4)
    kscales[:,:,:2] = 127
    kscales[:,:,2:] = 128
    return (q,folded,weights,qscale), table, ids


def reference(inputs, ids, table):
    """Independently decode actual nibbles/scales and take strict masked top-k."""
    q, folded, weights, qscale = inputs
    keys = folded[:,:,:512].reshape(-1,64)
    kscale = folded[:,:,512:].reshape(-1,4)
    winners = []
    scores_hash, mask_hash = sha256(), sha256()
    for row, blocks in enumerate(ids[:,0]):
        positions = (8*blocks[:,None] + np.arange(8)).reshape(-1)
        physical = table[0,positions//PAGE]*PAGE + positions%PAGE
        visible = positions < CONTEXT-ROWS+row+1
        scores = np.zeros(positions.size, np.float64)
        for head in np.flatnonzero(weights[row]):
            qbytes = q[row,head]
            qvalues = np.stack((qbytes&15,qbytes>>4),axis=-1).reshape(-1)
            dot = np.zeros(positions.size, np.float64)
            for dim in np.flatnonzero(FP4[qvalues]):
                nibbles = (keys[physical,dim//2] >> (4*(dim%2))) & 15
                exp = int(qscale[row,head].reshape(-1)[dim//32]) + kscale[physical,dim//32].astype(int) - 254
                dot += np.ldexp(FP4[qvalues[dim]]*FP4[nibbles], exp)
            scores += weights[row,head]*np.maximum(dot,0)
        threshold = np.partition(scores[visible], -TOPK)[-TOPK]
        winning = positions[visible & (scores >= threshold)]
        if winning.size != TOPK or np.any(scores[~visible] <= scores[visible].max()):
            raise ValueError('Ambiguous top-k or missing masked distractor')
        winners.append(np.sort(winning))
        scores_hash.update(scores.astype('<f4').tobytes())
        mask_hash.update(visible.tobytes())
    expected = np.asarray(winners,np.int32)
    if len(np.unique(expected,axis=0)) != ROWS:
        raise ValueError('Per-row goldens are not distinct')
    return expected, scores_hash.hexdigest(), mask_hash.hexdigest()


def facts(arm, inputs, table, ids, md, expected, score_hash, mask_hash):
    physical = table[0,ids[:,0]//16]*16 + ids[:,0]%16
    page_sets = [set(row//16) for row in physical]
    adjacent = [len(page_sets[r]&page_sets[r+1]) for r in range(0,ROWS,2)]
    ranges = team_ranges(md,ROWS)
    fingerprints = []
    for team in ranges:
        lo,hi = team['first_tile']*64, team['end_tile_exclusive']*64
        counts = Counter(map(int,physical.reshape(-1)[lo:hi]))
        packed = b''.join(struct.pack('<II',k,n) for k,n in sorted(counts.items()))
        fingerprints.append(sha256(packed).hexdigest())
    rank_intersections = []
    for phase in range(2):
        workers = physical[phase::2]
        counts = 1 + np.count_nonzero(np.diff(np.sort(workers,axis=0),axis=0),axis=0)
        matching_pairs = sum(np.count_nonzero(workers[i]==workers[j]) for i in range(WORKERS) for j in range(i))
        rank_intersections.append(dict(row_ordinal_in_team=phase, ranks=SELECTED,
            all_workers_same_fraction=float(np.mean(counts==1)), mean_unique_addresses_per_rank=float(counts.mean()),
            matching_worker_pair_fraction=matching_pairs/(SELECTED*WORKERS*(WORKERS-1)//2)))
    all_row_pair_matches = sum(np.count_nonzero(physical[i]==physical[j]) for i in range(ROWS) for j in range(i))
    return dict(arm=arm,shape=[1,64,32,128,CONTEXT,SELECTED,TOPK],table_pages=TABLE_PAGES,pa_block_size=PAGE,
        source_group=8,ratio=1,causal_mask=3,teams=len(ranges),tiles_per_team=[t['tile_count'] for t in ranges],
        split_merge_records=int(np.count_nonzero(md[36:108,0])),metadata_sha256=checksum(md),
        input_sha256={k:checksum(v) for k,v in zip(('q','folded_k','weights','qscale','ids','table','golden'),(*inputs,ids,table,expected))},
        score_at_each_candidate_including_future_sha256=score_hash,visibility_at_each_candidate_sha256=mask_hash,
        per_slot_offset_within_page_sha256=checksum(ids%16),backing_bytes=inputs[1].nbytes,
        pages_per_row=sorted({len(s) for s in page_sets}),within_team_page_intersection_range=[min(adjacent),max(adjacent)],
        accessed_page_union=len(set.union(*page_sets)),folded_gather_bytes=int(physical.size)*BLOCK_BYTES,
        folded_distinct_bytes=len(np.unique(physical))*BLOCK_BYTES,min_physical_block=int(physical.min()),
        max_physical_block=int(physical.max()),logical_team_address_multiset_sha256=fingerprints,
        same_logical_rank_across_32_workers=rank_intersections,
        all_unordered_row_pairs_same_rank_fraction=all_row_pair_matches/(SELECTED*ROWS*(ROWS-1)//2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm',choices=ARMS)
    parser.add_argument('--cpu-only',type=Path)
    args = parser.parse_args()
    inputs,table,ids_all = make_fixture()
    md = schedule(ROWS,WORKERS)
    expected,scores,masks = reference(inputs,ids_all[args.arm],table)
    data = facts(args.arm,inputs,table,ids_all[args.arm],md,expected,scores,masks)
    if args.cpu_only:
        args.cpu_only.write_text(json.dumps(data,indent=2)+'\n')
        return
    if int(torch.npu.get_device_properties(0).cube_core_num) != WORKERS:
        raise ValueError('Requires a 32-team 950DT')
    print('V113_FIXTURE',json.dumps(data,separators=(',',':')),flush=True)
    tensors = tuple(npu(a) for a in (*inputs,ids_all[args.arm]))
    actual = invoke(tensors,npu(np.full((ROWS,1),SELECTED,np.int32)),npu(table),npu(md.reshape(-1)),
                    npu(np.array([CONTEXT],np.int32)),npu(np.array([0,ROWS],np.int32)),None,1)
    actual = np.sort(actual.reshape(ROWS,TOPK),axis=1)
    np.testing.assert_array_equal(actual,expected)
    print('V113_CONTROL_GOLDEN_PASS',actual.size,checksum(actual),flush=True)


if __name__ == '__main__':
    main()
