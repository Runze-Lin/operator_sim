"""Replay the A5 package's metadata algorithm for the final Flash prefill chunk.

Optional offline dependency: the arenanet and cannbotdsl wheels pinned in
source-audit.md. Only the partition and output digest are evidence here.
"""
import hashlib
import json
import struct
from ops.quant_sparse_lightning_indexer_metadata_dsl import metadata_kernel


def assignment(workers):
    rows, length = 4096, 1048576
    buf = [0]*1024
    metadata_kernel.run_host(
        cu=[0,rows], used_q=[], used_k=[length], residual=[],
        candidate_length=[2048]*rows, output_offset=[], output=buf,
        scratch=[0]*(rows*36), has_cu=1, has_q=0, has_k=1, has_residual=0,
        sparse=1, mask=3, ratio=1, total_q=rows, batch=1, max_tasks=rows,
        capacity=-1, splits=8, workers=workers, groups=rows, query_rows=6,
        heads=32, ld=1, has_offset=0)
    # LI entries encode (B,row,tile) start and exclusive end. The one
    # document ends at (1,0,0). Every row at this context uses 32 tiles.
    def position(batch,row,tile):
        return (batch*rows+row)*32+tile
    live = []
    frontier = 0
    for c in range(workers):
        enabled,b,r,t,end_b,end_r,end_t,slot = buf[c*8:c*8+8]
        if not enabled:
            continue
        start,stop = position(b,r,t),position(end_b,end_r,end_t)
        if start != frontier or not start < stop <= rows*32:
            raise ValueError(f"Unexpected native metadata at team {c}: {buf[c*8:c*8+8]}")
        live.append(dict(team=c,first_tile=start,stop_tile=stop,tiles=stop-start))
        frontier=stop
    if frontier != rows*32:
        raise ValueError("Native metadata did not assign every candidate tile")
    return dict(workers=workers,rows=rows,tiles_per_row=32,
                source_metadata_sha256=hashlib.sha256(struct.pack("<1024i",*buf)).hexdigest(),
                active_ld=sum(bool(buf[i*8]) for i in range(36,108)),teams=live)


if __name__ == "__main__":
    print(json.dumps([assignment(32),assignment(28)],indent=2))
