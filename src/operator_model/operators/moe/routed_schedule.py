"""Cycle schedule for the native 950 transposed M-group MX GMM route.

This is a last-wave subdivision model. Earlier expert tiles keep their shape
even when the last group's short tail recruits additional AICs.
"""

from operator_model.operators.matmul.mx_native import STREAM_BYTES_PER_CYCLE


def _div(x, y):
    return (x+y-1)//y


def schedule(rows, n, k, counts, work, chip):
    cores = chip.cube_cores
    ghz = chip.frequency_ghz
    peak = chip.cube_tops['4']
    if not rows or not peak:
        return None, {}
    # Reference counts are for groupType=0; K=2 is an ACLNN exclusion.
    # Outside this calibrated dispatch format a target resource estimate is meaningless.
    if k == 2 or k % 64:
        return None, {}
    details = chip.details
    bn = 256
    bm = max(16, min(128, (details['l0c_bytes']//(4*bn*16))*16))
    chunk_k = min(256, (details['l0b_bytes']//bn//64)*64)
    if chunk_k < 64 or 4*bm*bn > details['l0c_bytes'] or details['l1_bytes'] < (bm+bn)*chunk_k:
        return None, {}
    nc = _div(n,bn)
    ports = details.get('mte2_bandwidth_gbs')
    local_read = ports/(cores*ghz) if ports else STREAM_BYTES_PER_CYCLE
    write = (details.get('mte3_bandwidth_gbs') or cores*ghz*64)/(cores*ghz)
    cube = peak*1000/(cores*ghz)  # scalar ops / AIC / cycle
    cycles = [0.0]*cores
    per_core_tiles = [0]*cores
    slots = k//32
    ku = _div(k,chunk_k)
    base_tiles = 0
    total_encoded = 0
    max_splits = 1

    def assign(begin, count, m_tile, n_tile):
        nonlocal total_encoded
        if not count:
            return
        encoded=(m_tile+n_tile)*(k//2+slots)
        physical_ops=2*_div(m_tile,16)*16*_div(n_tile,16)*16*k
        cost=max(encoded/local_read, physical_ops/cube, ku*208, (2*m_tile*n_tile)/write)
        total_encoded+=encoded*count
        full,tail=divmod(count,cores)
        for offset in range(cores if full else tail):
            c=(begin+offset)%cores
            times=full+(offset<tail)
            cycles[c]+=times*cost
            per_core_tiles[c]+=times

    # Each expert contributes its own grid; all experts share one rotating block index.
    # Uniform-partition O(G*cores) accounting keeps query cost bounded for many M waves.
    last=max(i for i,v in enumerate(counts) if v)
    for index, m in enumerate(counts):
        if not m:
            continue
        mc=_div(m,bm)
        tiles=mc*nc
        height=_div(m,mc)
        width=_div(n,nc)
        tail_tiles=min(((base_tiles+tiles-1)%cores)+1,tiles) if index==last else 0
        whole=tiles-tail_tiles
        assign(base_tiles,whole,height,width)
        if tail_tiles:
            free=(cores-1-(base_tiles+tiles-1)%cores)//tail_tiles+1
            # The CA build splits even a two-row transposed-tail tile by individual
            # rows; its N slices are aligned to 16 columns. Empty geometric slices vanish.
            mt=min(height,free)
            nt=min(_div(width,16),free)
            while mt*nt>free:
                if mt>=nt: mt-=1
                else: nt-=1
            hs=_div(height,mt)
            ws=16*_div(_div(width,nt),16)
            max_splits=mt*nt
            for i in range(tail_tiles):
                begin=base_tiles+whole+i*max_splits
                for ns in range(_div(width,ws)):
                    for ms in range(_div(height,hs)):
                        assign(begin+ns*mt+ms,1,min(hs,height-ms*hs),min(ws,width-ns*ws))
        base_tiles+=tiles

    # Earliest payload issue, return latency and trailing retirement are clock
    # intervals. Throughput changes with the selected chip independently of clock.
    active=sum(x>0 for x in per_core_tiles)
    dispatch=6802.0
    pipe=1442.0
    drain=392.5
    link=chip.hbm_bandwidth_gbs/ghz*(details.get('matmul_hbm_efficiency') or 1.)
    cold_bytes=work['logical_read_bytes']+work['logical_write_bytes']
    cold=cold_bytes/link
    l2rate=details.get('l2_bandwidth_gbs')
    l2=(total_encoded+work['logical_write_bytes']) / (l2rate/ghz) if l2rate else 0.
    service=max(max(cycles),cold,l2)
    stages={'control_to_first_payload':dispatch, 'first_dependency':pipe,
            'overlapped_service':service,'retirement':drain}
    return sum(stages.values())/(ghz*1000), {
        'active_aic':active, 'active_aiv':0, 'issued_aic_assumption':cores,
        'base_m':bm,'base_n':bn,'fragment_k':chunk_k,'base_output_tiles':base_tiles,
        'tail_partition_slots':max_splits,'max_tiles_per_worker':max(per_core_tiles),
        'target_frequency_ghz':ghz,'per_core_read_bytes_per_cycle':local_read,
        'service_cycles':{'critical_core':max(cycles),'shared_hbm':cold,
                          'shared_l2':l2 if l2rate else None},'stage_cycles':stages,
    }
