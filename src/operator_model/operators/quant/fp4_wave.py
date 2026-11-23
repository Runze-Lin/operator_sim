"""Visible AIV arrival and retirement of separate FP4 payload and scale stores."""
from collections import Counter
from math import ceil


def single_wave(rows, cols, plan, spread, service):
    """Price a single UB tile per active core. Partial owners may finish last."""
    rh, cw = plan['rows_per_core'], plan['cols_per_core']
    scale_pitch = 2*ceil(cols/64)
    requests = []
    data_owners=Counter()
    for r in range(0, rows, rh):
        h = min(rh, rows-r)
        for c in range(0, cols, cw):
            w = min(cw, cols-c)
            tile = service(h, w)
            arrival = spread*len(requests)/max(1, plan['active_aiv']-1)
            ready = arrival+tile['read_cycles']+tile['vector_cycles']
            # Successive scale rows have the full tensor pitch, including padding.
            # Count only externally shared boundary lines; interior lines generated
            # by this owner cannot introduce serialization with another AIV.
            edges=set(); data_edges=set()
            nbytes=2*ceil(w/64)
            for i in range(h):
                start=(r+i)*scale_pitch+c//32
                end=start+nbytes
                if start%32:
                    edges.add(start//32)
                if end%32:
                    edges.add((end-1)//32)
                ds=((r+i)*cols+c)//2
                de=ds+w//2
                if ds%32:
                    data_edges.add(ds//32)
                if de%32:
                    data_edges.add((de-1)//32)
            data_owners.update(data_edges)
            requests.append((ready,len(requests),h,w,arrival,tile,edges,data_edges))
    line_ready={}
    finish=0.
    critical={}
    for ready,owner,h,w,arrival,tile,edges,data_edges in sorted(requests):
        # Two partial output streams can contend at the writer. These are
        # effective descriptor intervals, not a specification of L2 atomics.
        concurrent_partial=any(data_owners[g]>1 for g in data_edges)
        grain_ii=310. if concurrent_partial else 172.
        scale_local=ready+tile['scale_write_cycles']+(310. if concurrent_partial else 0.)
        scale=scale_local
        for g in edges:
            end=max(scale_local,line_ready.get(g,-float('inf'))+grain_ii)
            line_ready[g]=end
            scale=max(scale,end)
        done=max(scale,ready+tile['data_write_cycles'])
        if done>finish:
            finish=done
            critical=dict(owner=owner,rows=h,cols=w,arrival_cycles=arrival,tile=tile,
                          scale_queue_cycles=scale-scale_local, shared_packed_grain=concurrent_partial)
    return finish,critical
