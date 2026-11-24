"""Golden for the packaged QSLI PA_BBND AOT provider under msOpProf."""
import argparse
import json

import numpy as np
import torch
import torch_npu  # registers the NPU tensor and stream implementation
from ops.quant_sparse_lightning_indexer_dsl import quant_sparse_lightning_indexer
from ops.quant_sparse_lightning_indexer_metadata_dsl import (
    metadata_kernel,
    quant_sparse_lightning_indexer_metadata,
)


def npu(array):
    return torch.from_numpy(np.ascontiguousarray(array)).to('npu')


def folded_inputs(length, selected, page=128, topk=512):
    # Scores outside the candidate prefix are deliberately high. Two heads
    # and two non-unit E8M0 scales give distinct scores around the cutoff.
    blocks = np.arange(selected) * (length // 8 - 1) // (selected - 1)
    ids = (8 * blocks[:, None] + np.arange(8)).flatten()
    wanted = np.zeros(length, dtype=bool)
    wanted[ids[-topk:]] = True
    inside = np.zeros(length, dtype=bool)
    inside[ids] = True
    query = np.zeros((1, 32, 64), np.uint8)
    for head in range(3):
        query[0, head, 16 * head] = 2
    qscale = np.full((1, 32, 2, 2), 127, np.uint8)
    qscale[0, 0, 0, 0] = 126
    weights = np.zeros((1, 32), np.float32)
    weights[0, :2] = (.5, 2.)
    keys = np.zeros((length, 64), np.uint8)
    odd = np.arange(length) % 2 != 0
    keys[:, 0] = np.where(~inside, 7, np.where(wanted, np.where(odd, 7, 10), 5))
    keys[:, 16] = np.where(~inside, 7, np.where(wanted, np.where(odd, 10, 1), 10))
    keys[:, 32] = np.where(wanted, 0, 7)
    scales = np.full((length, 4), 127, np.uint8)
    scales[:, 1] = 128
    folded = np.concatenate((keys.reshape(-1, 512), scales.reshape(-1, 32)), axis=1)
    folded = folded.reshape(length // page, page // 8, 544)
    candidates = np.full((1, 1, 2048), -1, np.int32)
    candidates[0, 0, :selected] = blocks
    return tuple(map(npu, (query, folded, weights, qscale, candidates))), np.nonzero(wanted)[0]


def replay_metadata(workers, length, selected, ratio):
    rows = np.zeros((128, 8), np.int32)
    rows[0] = [1, 0, 0, 0, 0, 0, 1, 0]
    rows[1] = [1, 0, 0, 1, 1, 0, 0, 1]
    rows[36] = [1, 0, 0, 0, 2, 0, 1, 0]
    ref = [0] * 1024
    rc = metadata_kernel.run_host(
        cu=[0, 1], used_q=[], used_k=[length], residual=[0], candidate_length=[selected],
        output_offset=[], output=ref, scratch=[0] * 36, has_cu=1, has_q=0, has_k=1,
        has_residual=int(ratio != 1), sparse=1, mask=3, ratio=ratio, total_q=1,
        batch=1, max_tasks=1, capacity=-1, splits=8, workers=workers, groups=1,
        query_rows=6, heads=32, ld=1, has_offset=0,
    )
    if rc != 0:
        raise RuntimeError(f'Metadata host reference failed with {rc}')
    np.testing.assert_array_equal(ref, rows.reshape(-1))
    print('METADATA_HOST_REFERENCE_PASS 1024', flush=True)
    return npu(rows.reshape(-1))


def invoke(tensors, lens, table, metadata, used, cu, residual, ratio):
    try:
        indices, _ = quant_sparse_lightning_indexer(
            *tensors, lens, cu_seqlens_q=cu, seqused_k=used, cmp_residual_k=residual,
            block_table=table, metadata=metadata, topk=512, candidate_block_size=8,
            quant_mode=1, mask_mode=3, cmp_ratio=ratio, layout_q='TND', layout_k='PA_BBND',
        )
    except RuntimeError as error:
        if str(error) != 'aclprofRangePop failed with status -1':
            raise
        # The host range annotation fails after dispatch. Retain the published
        # wrapper's original output and require synchronization + full golden.
        frame = error.__traceback__
        while frame and frame.tb_frame.f_code is not quant_sparse_lightning_indexer.__code__:
            frame = frame.tb_next
        if frame is None:
            raise
        indices = frame.tb_frame.f_locals['final_sparse_indices']
        print('HOST_RANGE_POP_ERROR_AFTER_ENQUEUE', flush=True)
    torch.npu.synchronize()
    return indices.cpu().numpy().reshape(-1)


def trial(static, ratio, selected, table_pages):
    print('versions', torch.__version__, torch_npu.__version__, flush=True)
    length, page = 1024, 128
    tensors, wanted = folded_inputs(length, selected, page)
    cu = npu(np.array([0, 1], np.int32))
    used = npu(np.array([length], np.int32))
    residual = npu(np.array([0], np.int32)) if ratio == 2 else None
    lens = npu(np.array([[selected]], np.int32))
    table_host = np.zeros((1, table_pages), np.int32)
    table_host[0, :length // page] = np.arange(length // page)
    if static:
        workers = int(torch.npu.get_device_properties(tensors[0].device).cube_core_num)
        metadata = replay_metadata(workers, length, selected, ratio)
    else:
        metadata = quant_sparse_lightning_indexer_metadata(
            lens, cu_seqlens_q=cu, seqused_k=used, cmp_residual_k=residual,
            batch_size=1, max_seqlen_q=-1, max_seqlen_k=-1, num_heads_q=32,
            num_heads_k=1, head_dim=128, topk=512, quant_mode=1, candidate_block_size=8,
            mask_mode=3, cmp_ratio=ratio, layout_q='TND', layout_k='PA_BBND',
        )
    output = invoke(tensors, lens, npu(table_host), metadata, used, cu, residual, ratio)
    np.testing.assert_array_equal(np.sort(output), wanted)
    print('FUSED_GOLDEN_PASS 512', flush=True)
    rows = metadata.cpu().numpy().reshape(-1, 8)
    print(json.dumps({i: row.tolist() for i, row in enumerate(rows) if row[0]}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--static', action='store_true')
    parser.add_argument('--ratio', type=int, choices=(1, 2), default=2)
    parser.add_argument('--selected', type=int, choices=(81, 128), default=81)
    parser.add_argument('--table-pages', type=int, choices=(8, 8201), default=8)
    args = parser.parse_args()
    trial(args.static, args.ratio, args.selected, args.table_pages)
