"""One full-candidate MXFP4 QSLI row ending at 1Mi on Ascend 950."""
import hashlib
import json

import numpy as np
import torch
import torch_npu  # register the NPU backend
from ops.quant_sparse_lightning_indexer_dsl import quant_sparse_lightning_indexer
from ops.quant_sparse_lightning_indexer_metadata_dsl import choose_splits, metadata_kernel


CONTEXT = 1 << 20
PAGE = 128
TABLE_PAGES = 8201
SELECTED = 2048
TOPK = 512


def checksum(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def fixture():
    # Select half of the eight-token blocks from each of 256 widely spaced pages.
    page_rank = np.arange(256, dtype=np.int32)
    pages = page_rank * (CONTEXT // PAGE - 1) // (len(page_rank) - 1)
    local_blocks = (2 * np.arange(8)[None, :] + (page_rank % 2)[:, None])
    blocks = (pages[:, None] * (PAGE // 8) + local_blocks).reshape(-1)
    # All selected pages have distinct addressable data. Page zero is an
    # unselected distractor. Reverse pages to decouple logical and physical order.
    table = np.zeros((1, TABLE_PAGES), dtype=np.int32)
    table[0, pages] = 256 - page_rank
    # Permute the candidate stream so that wrong rank-local/global remapping fails.
    blocks = blocks[(np.arange(SELECTED) * 37) % SELECTED].astype(np.int32)
    positions = (8 * blocks[:, None] + np.arange(8)).reshape(-1)
    assert np.unique(positions).size == 16384 and positions.min() == 0 and positions.max() == CONTEXT - 1
    physical = table[0, positions // PAGE] * PAGE + positions % PAGE
    wanted_slots = (np.arange(16384) * 73 + 11) % 16384 < TOPK
    expected = np.sort(positions[wanted_slots]).astype(np.int32)

    query = np.zeros((1, 32, 64), dtype=np.uint8)
    query[0, 0, 0] = query[0, 1, 16] = query[0, 2, 32] = 2
    query_scale = np.full((1, 32, 2, 2), 127, dtype=np.uint8)
    query_scale[0, 0, 0, 0] = 126
    weights = np.zeros((1, 32), dtype=np.float32)
    weights[0, :2] = (0.5, 2.0)
    keys = np.zeros((257 * PAGE, 64), dtype=np.uint8)
    keys[:, [0, 16, 32]] = 7
    odd = positions % 2 != 0
    keys[physical, 0] = np.where(wanted_slots, np.where(odd, 7, 10), 5)
    keys[physical, 16] = np.where(wanted_slots, np.where(odd, 10, 1), 10)
    keys[physical, 32] = np.where(wanted_slots, 0, 7)
    scales = np.full((len(keys), 4), 127, dtype=np.uint8)
    scales[:, 1] = 128
    folded = np.concatenate((keys.reshape(-1, 512), scales.reshape(-1, 32)), axis=1)
    return (query, folded.reshape(257, PAGE // 8, 544), weights, query_scale,
            blocks.reshape(1, 1, -1), table, expected)


def npu(array):
    return torch.from_numpy(np.ascontiguousarray(array)).to('npu')


def main():
    query, folded, weights, scale, candidates, table, expected = fixture()
    q, k, w, s, ids, page_table = map(npu, (query, folded, weights, scale, candidates, table))
    lengths = npu(np.array([[SELECTED]], dtype=np.int32))
    cu = npu(np.array([0, 1], dtype=np.int32))
    used = npu(np.array([CONTEXT], dtype=np.int32))
    workers = int(torch.npu.get_device_properties(q.device).cube_core_num)
    splits = choose_splits(1, 0, TOPK)
    ref = [0] * 1024
    rc = metadata_kernel.run_host(
        cu=[0, 1], used_q=[], used_k=[CONTEXT], residual=[], candidate_length=[SELECTED],
        output_offset=[], output=ref, scratch=[0] * (12 + 3 * splits),
        has_cu=1, has_q=0, has_k=1, has_residual=0, sparse=1, mask=3, ratio=1, total_q=1,
        batch=1, max_tasks=1, capacity=-1, splits=splits, workers=workers, groups=1,
        query_rows=6, heads=32, ld=1, has_offset=0,
    )
    assert rc == 0
    metadata = np.asarray(ref, dtype=np.int32).reshape(128, 8)
    print('FIXTURE', json.dumps(dict(candidate_ids_sha256=checksum(candidates),
          expected_ids_sha256=checksum(expected), folded_sha256=checksum(folded), table_sha256=checksum(table),
          golden_min=int(expected.min()), golden_max=int(expected.max()), torch=torch.__version__,
          torch_npu=torch_npu.__version__, workers=workers, metadata_sha256=checksum(metadata),
          metadata_rows={i: row.tolist() for i, row in enumerate(metadata) if row[0]},
          splits=splits)), flush=True)
    md = npu(metadata.reshape(-1))
    try:
        idx, _ = quant_sparse_lightning_indexer(
            q, k, w, s, ids, lengths, cu_seqlens_q=cu, seqused_k=used,
            block_table=page_table, metadata=md, topk=TOPK, candidate_block_size=8,
            quant_mode=1, mask_mode=3, cmp_ratio=1, layout_q='TND', layout_k='PA_BBND')
    except RuntimeError as error:
        if str(error) != 'aclprofRangePop failed with status -1':
            raise
        tb = error.__traceback__
        while tb and tb.tb_frame.f_code is not quant_sparse_lightning_indexer.__code__:
            tb = tb.tb_next
        if tb is None:
            raise
        idx = tb.tb_frame.f_locals['final_sparse_indices']
        print('HOST_RANGE_POP_ERROR_AFTER_ENQUEUE', flush=True)
    torch.npu.synchronize()
    actual = np.sort(idx.cpu().numpy().reshape(-1))
    np.testing.assert_array_equal(actual, expected)
    print('FUSED_GOLDEN_PASS 512', checksum(actual), flush=True)


if __name__ == '__main__':
    main()
