"""Recompute FA v2.2 development diagnostics from compact DBs and per-AIV msOpProf JSON."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median
import sqlite3

from operator_model import estimate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


GROUPS = {
    'control': {'RV_PLT','RV_SMOV','RV_SEND','RV_PSET','RV_VBR'},
    'packing': {'RV_VOR','RV_VSEL'},
    'fence': {'RV_SMEM_BAR'},
    'indexed_or_packed_io': {'RV_VLDI','RV_VSTI','RV_VSTAI','RV_VSTUI','RV_VSSTB'},
    'exp': {'RV_VEXPDIF'},
}


def union_cycles(intervals):
    end = total = 0
    for begin, finish, _, _ in intervals:
        total += max(0, finish - max(end, begin))
        end = max(end, finish)
    return total


def extract(events):
    lanes = defaultdict(list)
    for lane, op, pc, start, end in events:
        if op.startswith('RV_') and end > start:
            lanes[lane].append((start, end, op, pc))
    audit = {}
    for lane, intervals in lanes.items():
        counts = Counter()
        exp_pc = Counter()
        spans = defaultdict(list)
        intervals.sort()
        union = union_cycles(intervals)
        for start, end, op, pc in intervals:
            counts[op] += 1
            spans[op].append(end - start)
            if op == 'RV_VEXPDIF':
                exp_pc[pc] += 1
        exposure = {
            key: dict(group_union_cycles=union_cycles(e for e in intervals if e[2] in group),
                      exclusive_cycles=union-union_cycles(e for e in intervals if e[2] not in group))
            for key,group in GROUPS.items()
        }
        audit[lane] = dict(union_cycles=union, exp_by_pc=exp_pc, counts=counts, exposure=exposure,
                           median_cycles={k: median(v) for k,v in spans.items()})
    return audit


def db_data(path):
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        rows = db.execute("""SELECT ExecInstrCoreId,CoreTypeId,ExecInstrName,SourceInstrAddr,
                   ExecInstrTickStart,ExecInstrTickEnd FROM ExecutedInstructions
                   WHERE CoreTypeId IN (2,3) AND ExecInstrIsDurationExcluded=0""")
        return extract((f'{core}/{kind}',op,hex(pc),s,e) for core,kind,op,pc,s,e in rows)


def json_data(path):
    def events():
        for x in json.loads(path.read_bytes())['traceEvents']:
            if x.get('ph') == 'X' and x.get('name','').startswith('RV_') and x['dur'] > 0:
                if json.loads(x['args']['detail']).get('is_duration_excluded',0):
                    continue
                yield (x['pid'], x['name'], x['args']['pc_addr'], round(x['ts'] * 1650),
                       round((x['ts'] + x['dur']) * 1650))
    return extract(events())


def prices(shape, dtype, chip, method):
    result = estimate('fa', shape, {'vector_model': method}, dtype=dtype, chip=chip)
    if result.latency_us is None:
        return None
    diag = result.projection['timing_diagnostics']
    pages = diag['target_plan']['iterations']
    # These development windows have one wave with uniform work across active AICs.
    wave, = diag['wave_types']
    assert wave['waves'] == 1
    totals = Counter()
    cycles = 0
    for kind, count in [('first',1), ('repeat',max(0,pages-2)), ('last',int(pages>1))]:
        if count:
            tile = wave[kind]
            cycles += (tile['vec1'] + tile['vec2'])*count
            for phase in tile['vf'].get('phases',{}).values():
                for k,v in phase['counts'].items():
                    totals[k] += count*v
    return dict(kernel_us=result.latency_us,vf_cycles=cycles,class_counts=totals)


def main(args):
    rows = []
    def record(chip, shape, dtype, path, audit, expected):
        digest = sha(path)
        assert digest == expected
        lane = max(audit,key=lambda k:audit[k]['union_cycles'])
        rows.append(dict(chip=chip,shape=shape,dtype=dtype,raw_sha256=digest,
                         measured_busiest_lane=lane,rv=audit[lane],
                         calibrated=prices(shape,dtype,chip,'calibrated'),
                         classes=prices(shape,dtype,chip,'instruction_classes')))

    for r in json.loads(args.compact_manifest.read_bytes())['rows']:
        path = Path(r['db'])
        record('Ascend950DT_9582',r['shape'],'bf16',path,db_data(path),r['sha256'])
    for r in json.loads(args.receipts.read_bytes())['rows']:
        path, = args.raw.glob(r['chip']+'-'+r['dtype']+'/profile/*/simulator/trace.json')
        record(r['chip'],r['shape'],r['dtype'],path,json_data(path),r['trace_sha256'])
    print(json.dumps(rows,indent=2))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--compact-manifest',type=Path,default=Path('results/v0.16/fa-dt-evidence.json'))
    cli.add_argument('--receipts',type=Path,default=Path('results/v2.1/new-ca/receipts.json'))
    cli.add_argument('--raw',type=Path,default=Path('runs/fa-vector-v21-local-ipc'))
    main(cli.parse_args())
