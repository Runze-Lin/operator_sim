"""Full 32-AIC prefill wave with distinct causal golden sets for all QSLI rows."""
import json

import numpy as np
import torch
from ops.quant_sparse_lightning_indexer_metadata_dsl import choose_splits, metadata_kernel

from fused_qsli_probe import invoke, npu
from long_row_probe import CONTEXT, PAGE, SELECTED, TOPK, checksum, fixture

Q = 32


def set_nibble(storage, rows, dim, codes):
    shift = 4 * (dim % 2)
    col = dim // 2
    storage[rows, col] = (storage[rows, col] & (255 ^ (15 << shift))) | (codes << shift)


def wave_inputs():
    _, baseline_folded, weights, scale, candidate, table, _ = fixture()
    blocks = candidate.reshape(-1)
    positions = (8 * blocks[:, None] + np.arange(8)).reshape(-1)
    physical = table[0, positions // PAGE] * PAGE + positions % PAGE
    query = np.zeros((Q, 32, 64), dtype=np.uint8)
    k = np.full((baseline_folded.shape[0] * PAGE, 64), 0x77, dtype=np.uint8)
    winners = []
    for r in range(Q):
        visible = positions < CONTEXT - Q + r + 1
        live = np.flatnonzero(visible)
        # Distinct strict TopK sets and deliberately stronger values on future
        # positions. A missing causal mask or reused query row fails the golden.
        perm = (live * 73 + 397 * r + 11) % positions.size
        chosen = live[np.argsort(perm)[:TOPK]]
        winners.append(np.sort(positions[chosen]))
        wanted = np.zeros(positions.size, dtype=bool)
        wanted[chosen] = True
        odd = positions % 2 != 0
        for head in range(3):
            dim = head * 32 + r
            query[r, head, dim // 2] = 2 << (4 * (dim % 2))
        set_nibble(k, physical, r, np.where(~visible, 7, np.where(wanted, np.where(odd, 7, 10), 5)))
        set_nibble(k, physical, 32+r, np.where(~visible, 7, np.where(wanted, np.where(odd, 10, 1), 10)))
        set_nibble(k, physical, 64+r, np.where(wanted, 0, 7))
    expected = np.array(winners, dtype=np.int32)
    assert len(np.unique(expected, axis=0)) == Q
    assert all(row[-1] < CONTEXT-Q+r+1 for r, row in enumerate(expected))
    folded = baseline_folded.copy().reshape(-1, 544)
    folded[:, :512] = k.reshape(-1, 512)
    inputs = (query, folded.reshape(baseline_folded.shape), np.repeat(weights, Q, axis=0),
              np.repeat(scale, Q, axis=0), np.repeat(candidate, Q, axis=0))
    return inputs, table, expected


def main():
    inputs, table, expected = wave_inputs()
    tensors = tuple(npu(item) for item in inputs)
    workers = int(torch.npu.get_device_properties(tensors[0].device).cube_core_num)
    if workers != 32:
        raise RuntimeError(f'Expected 950DT 32 AICs, got {workers}')
    splits = choose_splits(Q, 0, TOPK)
    ref = [0] * 1024
    status = metadata_kernel.run_host(
        cu=[0,Q], used_q=[], used_k=[CONTEXT], residual=[], candidate_length=[SELECTED]*Q,
        output_offset=[], output=ref, scratch=[0] * (Q * (12 + 3*splits)),
        has_cu=1, has_q=0, has_k=1, has_residual=0, sparse=1, mask=3, ratio=1,
        total_q=Q, batch=1, max_tasks=1, capacity=-1, splits=splits, workers=workers,
        groups=Q, query_rows=6, heads=32, ld=1, has_offset=0,
    )
    if status:
        raise ValueError(f'Host metadata returned {status}')
    md = np.array(ref, dtype=np.int32).reshape(128,8)
    print('FIXTURE',json.dumps({'metadata':{i:r.tolist() for i,r in enumerate(md) if r[0]},
           'golden_sha256':checksum(expected), 'causal_visible_from_to':[CONTEXT-Q+1, CONTEXT],
           'folded_sha256':checksum(inputs[1]),'table_sha256':checksum(table)}),flush=True)
    output = invoke(tensors, npu(np.full((Q,1), SELECTED, np.int32)), npu(table), npu(md.reshape(-1)),
                    npu(np.array([CONTEXT],np.int32)), npu(np.array([0,Q],np.int32)), None, 1)
    actual = np.sort(output.reshape(Q,TOPK),axis=1)
    np.testing.assert_array_equal(actual,expected)
    print('WAVE_GOLDEN_PASS',actual.size,checksum(actual),flush=True)


if __name__ == '__main__':
    main()
