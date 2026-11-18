"""Extract absolute cycle/service parameters from 950DT-only FA records."""
import hashlib
import json
import sqlite3
from collections import defaultdict
from math import ceil
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/v0.16'
CLOCK = 1650
CUBE_OPS_CYCLE = 432e6 / 32 / CLOCK


def rows(path, shape):
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
        start, end = db.execute('select min(ExecInstrTickStart), max(ExecInstrTickEnd) from ExecutedInstructions').fetchone()
        events = list(db.execute("""select ExecInstrName, SourceInstrAddr, ExecInstrTickStart, ExecInstrTickEnd,
                ExtendParams from ExecutedInstructions where ExecInstrCoreId=0 and CoreTypeId=1
                and ExecInstrName in ('MOV_OUT_TO_L1_MULTI_ND2NZ','MMAD') order by ExecInstrTickStart"""))
    reads, cube = defaultdict(list), defaultdict(list)
    for name, pc, ts, te, args in events:
        if name == 'MMAD':
            cube[pc].append((ts, te))
        else:
            reads[pc].append((ts, te, json.loads(args)['processed_bytes']))
    q, k, v = reads.values()  # encounter order: Q once, K, V; distinct PC per stream
    qk, pv = cube.values()    # first MMAD must be QK before P exists
    m, n, d, _, _ = shape
    assert len(q) == 1 and q[0][2] == 2*m*d and len(k) == len(v)
    assert sum(r[2] for r in k) == sum(r[2] for r in v) == n*d*2
    assert len(qk) == len(pv) == ceil(n/128)
    tile = k[0][2]//(d*2)
    nmm = ceil(tile/128)
    return dict(shape=shape, db=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                startup=q[0][0]-start, k_lag=k[0][0]-q[0][0], v_lag=v[0][0]-k[0][0],
                reads=[(t[1]-t[0], t[2]) for t in (q[0], k[0], v[0])],
                l0_ready=qk[0][0]-max(q[0][1],k[0][1]), first_qk_span=qk[nmm-1][1]-qk[0][0],
                handoff=pv[0][0]-max(qk[nmm-1][1],v[0][1]), pv_span=pv[nmm-1][1]-pv[0][0],
                drain=end-max(e for _,e in pv), kv_tile=tile)


def main():
    data = []
    for p in sorted((ROOT/'results/v0.8/core').glob('fa*.json')):
        meta = json.loads(p.read_text())
        if meta.get('status') != 'ok':
            continue
        assert 'OPERATOR_SIM_TARGET_SOC=Ascend950DT_9582' in meta['command']
        db = next((ROOT/'runs/v8core'/p.stem/'profile').glob('*/record/*.db'))
        data.append(rows(db,meta['shape']))
    long = json.loads((ROOT/'results/v0.5/trace-analysis.json').read_text())
    db = Path(long['db_path'])
    # The committed filename identifies this archive even on a different checkout.
    db = ROOT / 'runs' / str(db).split('/runs/', 1)[1]
    ref = rows(db,long['calibration_shape'])
    assert ref['sha256'] == long['db_sha256']
    data.append(ref)
    c = long['costs']
    # Local burst capacity: best sustained 8-issue window, excluding fill and drain.
    ks, vs = (long['read_streams'][x] for x in ('k','v'))
    duration = min(max(s[i+8]['start_us']-s[i]['start_us'] for s in (ks,vs)) for i in range(8,112))
    bpc = (2*32768*8)/(CLOCK*duration)
    base_vector_work = (5*256+3*64)*ceil(9/2)
    vops = base_vector_work / (c['vector_service_per_tile_us'] * CLOCK)
    params = dict(dt_cube_peak_ops_per_cycle=CUBE_OPS_CYCLE, mte_bytes_per_cycle=bpc, issue_cycles_256=CLOCK*max(c['k_initiation_interval_us'],c['v_initiation_interval_us']),
                  vector_ops_per_cycle=vops, dt_vector_peak_ops_per_cycle=27e6/64/CLOCK,
                  mte1_bytes_per_cycle=(4*256*64+2*16*(64+256))/(CLOCK*c['mte1_service_per_tile_us']),
                  fix_bytes_per_cycle=(4*16*(256+64))/(CLOCK*c['fix_service_per_tile_us']))
    routes = {}
    for name, group in (('short',[r for r in data if r['kv_tile']<=128]), ('wide',[r for r in data if r['kv_tile']==256])):
        stats = defaultdict(list)
        for r in group:
            q, _, d, _, _ = r['shape']
            n = r['kv_tile']
            parts = ceil(n/128)
            for field in ('startup','k_lag','v_lag','l0_ready'):
                stats[field].append(r[field])
            for duration, size in r['reads']:
                stats['read_intercept'].append(max(0,duration-size/bpc))
            work = 2*(16*ceil(q/16))*(16*ceil(n/16))*(16*ceil(d/16))
            for field,key in [('first_qk_span','qk_overhead'),('pv_span','pv_overhead')]:
                stats[key].append(max(0,r[field]-work/CUBE_OPS_CYCLE)/parts)
            # If V arrives late some Vec1 is already hidden. handoff is the
            # *remaining* exposed edge after both QK and V. Charge Vec1 on
            # QK's edge separately, and derive the remaining control gap.
            qdur, kdur, vdur = (x[0] for x in r['reads'])
            qk_end = max(qdur, r['k_lag']+kdur) + r['l0_ready'] + r['first_qk_span']
            v_end = r['k_lag']+r['v_lag']+vdur
            vec1 = (5*ceil(q/2)*n)/vops
            visible_vec = max(qk_end+vec1,v_end)-max(qk_end,v_end)
            stats['handoff_control'].append(max(0,r['handoff']-visible_vec))
            stats['drain_control'].append(max(0,r['drain']-3*ceil(q/2)*d/vops-(2*q*d+64*q)/bpc))
        routes[name] = {key:median(values) for key,values in stats.items()}
    out=dict(platform='Ascend950DT_9582',version=16, units='target cycles unless noted', **params, routes=routes)
    (OUT/'fa.json').write_text(json.dumps(out,indent=2)+'\n')
    (OUT/'fa-dt-evidence.json').write_text(json.dumps(dict(rows=data,steady_source='results/v0.5/trace-analysis.json'),indent=2)+'\n')


if __name__=='__main__':
    main()
