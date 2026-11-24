"""Frozen 950PR shared-page and 950DT moving-page native QSLI wave probes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ops.quant_sparse_lightning_indexer_metadata_dsl import choose_splits, metadata_kernel

from fused_qsli_probe import invoke, npu
from long_row_probe import CONTEXT, PAGE, SELECTED, TABLE_PAGES, TOPK, checksum
from prefill_wave_probe import set_nibble, wave_inputs

ROWS = {'pr': 32, 'dt': 64}
WORKERS = {'pr': 28, 'dt': 32}
TILES = SELECTED * 8 // 512


def varied_inputs():
    """64 distinct views of a common 1Mi-token physical cache, with adjacent reuse."""
    rows = ROWS['dt']
    page_count = CONTEXT // PAGE
    # All real logical pages have separate backing, in a fixed physical permutation.
    physical_pages = (4093 * np.arange(page_count) + 17) % page_count
    table = np.zeros((1, TABLE_PAGES), np.int32)
    table[0, :page_count] = physical_pages
    common = np.linspace(0, page_count-1, 128, dtype=np.int32)
    rest = np.setdiff1d(np.arange(page_count), common)
    pool = rest[np.linspace(0, rest.size-1, 2048, dtype=np.int32)]
    query = np.zeros((rows, 32, 64), np.uint8)
    qscale = np.full((rows, 32, 2, 2), 127, np.uint8)
    qscale[:, 0, 0, :] = 126
    weights = np.zeros((rows, 32), np.float32)
    weights[:, :2] = (.5, 2.)
    candidates = np.empty((rows, 1, SELECTED), np.int32)
    golden = np.empty((rows, TOPK), np.int32)
    keys = np.full((CONTEXT, 64), 0x77, np.uint8)
    for row in range(rows):
        pages = np.r_[common, pool[(48*row + np.arange(128)) % pool.size]]
        # Stable, page-dependent eight-of-sixteen blocks include the final block
        # of logical page 8191. The issue sequence is independently rotated per row.
        blocks = (pages[:, None]*16 + 2*np.arange(8) + (pages[:, None] % 2)).reshape(-1)
        blocks = blocks[(37*np.arange(SELECTED) + 19*row) % SELECTED]
        candidates[row, 0] = blocks
        positions = (8*blocks[:, None] + np.arange(8)).reshape(-1)
        physical = physical_pages[positions//PAGE]*PAGE + positions%PAGE
        visible = positions < CONTEXT - rows + row + 1
        live = np.flatnonzero(visible)
        order = (live*73 + 397*row + 11) % positions.size
        chosen = live[np.argsort(order)[:TOPK]]
        wanted = np.zeros(positions.size, dtype=bool)
        wanted[chosen] = True
        odd = positions % 2 != 0
        # Exact dyadic scores use only two heads, distinct coordinates per row.
        # Future entries are planted as strict higher-scoring invalid distractors.
        for head, dim in enumerate((row, 64+row)):
            query[row, head, dim//2] = 2 << (4*(dim%2))
        set_nibble(keys, physical, row, np.where(~visible, 7, np.where(wanted, np.where(odd, 7, 10), 5)))
        set_nibble(keys, physical, row+64, np.where(~visible, 7, np.where(wanted, np.where(odd, 10, 1), 10)))
        golden[row] = np.sort(positions[chosen])
    folded = np.empty((CONTEXT//8, 544), np.uint8)
    folded[:, :512] = keys.reshape(-1,512)
    scales = folded[:, 512:].reshape(-1,8,4)
    scales[:, :, :2] = 127
    scales[:, :, 2:] = 128
    folded = folded.reshape(page_count, PAGE//8, 544)
    verify_scores(keys, qscale, weights, table, candidates, golden)
    return (query, folded, weights, qscale, candidates), table, golden


def verify_scores(keys, qscale, weights, table, candidates, golden):
    """Decode actual nibbles at the mask boundary, checking the generated index set."""
    pos_codes = np.asarray([0, .5, 1, 1.5, 2, 3, 4, 6])
    fp4 = np.r_[pos_codes, -pos_codes]
    for row in range(len(golden)):
        positions = (8*candidates[row,0,:,None] + np.arange(8)).reshape(-1)
        src = table[0, positions//PAGE]*PAGE + positions%PAGE
        visible = positions < CONTEXT-len(golden)+row+1
        score = np.zeros(positions.size)
        for head, dim in enumerate((row,64+row)):
            nibble = (keys[src,dim//2] >> (4*(dim%2))) & 15
            q_exp = int(qscale[row,head].reshape(4)[dim//32])-127
            k_exp = int(dim>=64)
            score += np.maximum(0, fp4[nibble] * 2**(q_exp+k_exp)) * weights[row,head]
        candidates_at_cut = score[visible]
        cutoff = np.partition(candidates_at_cut, -TOPK)[-TOPK]
        selected = np.sort(positions[visible & (score>=cutoff)])
        np.testing.assert_array_equal(selected, golden[row])
        if (~visible).any():
            assert np.all(score[~visible] > candidates_at_cut.max())
    assert np.unique(golden, axis=0).shape[0] == len(golden)


def schedule(rows, workers):
    splits = choose_splits(rows, 0, TOPK)
    result = [0]*1024
    code = metadata_kernel.run_host(
        cu=[0,rows], used_q=[], used_k=[CONTEXT], residual=[], candidate_length=[SELECTED]*rows,
        output_offset=[], output=result, scratch=[0]*(rows*(12+3*splits)), has_cu=1, has_q=0,
        has_k=1, has_residual=0, sparse=1, mask=3, ratio=1, total_q=rows, batch=1, max_tasks=1,
        capacity=-1, splits=splits, workers=workers, groups=rows, query_rows=6, heads=32,
        ld=1, has_offset=0,
    )
    if code:
        raise RuntimeError(f'Host metadata interpreter returned {code}')
    return np.array(result, dtype=np.int32).reshape(128,8)


def team_ranges(md, rows):
    ownership = []
    frontier = 0
    for logical_id, record in enumerate(md[:36]):
        if not record[0]:
            continue
        _, batch, row, tile, last_batch, last_row, last_tile, _ = map(int, record)
        first = (batch*rows+row)*TILES + tile
        end = (last_batch*rows+last_row)*TILES + last_tile
        if first != frontier or end<=first:
            raise ValueError('Metadata has a gap, overlap or empty LI team')
        ownership.append(dict(logical_team=logical_id, first_tile=first, end_tile_exclusive=end,
                              first_row=first//TILES, last_row=(end-1)//TILES, tile_count=end-first))
        frontier = end
    if frontier != rows*TILES:
        raise ValueError('Host metadata did not prove full matrix coverage')
    return ownership


def locality(inputs, table, golden, md):
    q, folded, weights, scales, ids = inputs
    rows = q.shape[0]
    blocks_per_page = PAGE//8
    pages = [set(map(int, np.unique(rr//blocks_per_page))) for rr in ids[:,0]]
    physical_blocks = [set(map(int, (table[0,rr//blocks_per_page]*blocks_per_page + rr%blocks_per_page))) for rr in ids[:,0]]
    teams = team_ranges(md, rows)
    adjacent_reuse = []
    for team in teams:
        for row in range(team['first_row'],team['last_row']):
            adjacent_reuse.append(dict(team=team['logical_team'],first_row=row,second_row=row+1,
                                       shared_pages=len(pages[row]&pages[row+1]),
                                       shared_blocks=len(physical_blocks[row]&physical_blocks[row+1])))
    union = set.union(*physical_blocks)
    logical_union_pages = set.union(*pages)
    all_pairs = [len(pages[r]&pages[s]) for r in range(rows) for s in range(r)]
    matrix = [len(pages[r]&pages[r+1]) for r in range(rows-1)]
    md_rows={i:record.tolist() for i,record in enumerate(md) if record[0]}
    return dict(rows=rows,context=CONTEXT,table_pages=TABLE_PAGES,pa_block_size=PAGE,
                input_sha256={name:checksum(x) for name,x in zip(('q','folded_k','weights','qscale','ids','table','golden'),(*inputs,table,golden))},
                selected_token_count_per_row=SELECTED*8, page_count_per_row=[len(s) for s in pages],
                adjacent_page_intersection_range=[min(matrix),max(matrix)],
                half_batch_page_intersection_range=[min(len(pages[i]&pages[i+rows//2]) for i in range(rows//2)),max(len(pages[i]&pages[i+rows//2]) for i in range(rows//2))],
                arbitrary_pair_page_intersection_range=[min(all_pairs),max(all_pairs)],
                union_of_selected_logical_pages=len(logical_union_pages),
                distinct_physical_pages_referenced=len({b//blocks_per_page for b in union}),
                unique_physical_page_count_in_active_table=len(np.unique(table[0,:CONTEXT//PAGE])),
                folded_backing_pages=folded.shape[0],folded_backing_bytes=folded.nbytes,physical_table_bytes=table.nbytes,
                unfused_candidate_id_bytes=ids.nbytes,
                candidate_gather_demand_bytes=rows*SELECTED*544,
                distinct_folded_source_bytes=len(union)*544,
                byte_scope='Source data for selected eight-token blocks including E8M0. Excludes staging rereads and transfers. Cache hits and actual HBM bytes unavailable.',
                same_team_adjacent_full_row_or_partial_reuse=adjacent_reuse,
                metadata_workers_arg=max(t['logical_team'] for t in teams)+1,
                metadata_records=md_rows,metadata_sha256=checksum(md),assigned_teams=teams,
                fd_records=sum(int(record[0]!=0) for record in md[36:108]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sku',choices=ROWS)
    parser.add_argument('--cpu-only',action='store_true')
    parser.add_argument('--facts',type=Path)
    args=parser.parse_args()
    case=args.sku
    inputs,table,expected=wave_inputs() if case=='pr' else varied_inputs()
    rows=ROWS[case]
    workers=WORKERS[case] if args.cpu_only else int(torch.npu.get_device_properties(0).cube_core_num)
    md=schedule(rows,workers)
    facts=locality(inputs, table, expected, md)
    if case=='pr':
        old=json.loads(Path(__file__).with_name('prefill-wave.json').read_text())['fixture']
        assert facts['input_sha256']['golden']==old['golden_sha256']
        assert facts['input_sha256']['folded_k']==old['folded_sha256']
        assert facts['input_sha256']['table']==old['table_sha256']
    if args.cpu_only:
        args.facts.write_text(json.dumps(facts,indent=2)+'\n')
        return
    print('V112_FIXTURE',json.dumps(facts,separators=(',',':')),flush=True)
    tensors=tuple(npu(a) for a in inputs)
    actual=invoke(tensors,npu(np.full((rows,1),SELECTED,np.int32)),npu(table),npu(md.reshape(-1)),
                  npu(np.array([CONTEXT],np.int32)),npu(np.array([0,rows],np.int32)),None,1)
    actual=np.sort(actual.reshape(rows,TOPK),axis=1)
    np.testing.assert_array_equal(actual,expected)
    print('V112_WAVE_GOLDEN_PASS',actual.size,checksum(actual),flush=True)

if __name__=='__main__':
    main()
