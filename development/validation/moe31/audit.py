"""Audit closed MX traces and replay the public large-shape template, without running CA."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model.fusion_v8 import FIRST, SECOND, work

OUT = ROOT / 'results/v0.31/moe'
PAIRS = ROOT / 'results/v0.17/sku/pairs.json'
FREEZE = ROOT / 'results/v0.29/large/cube/freeze.json'
MODELS = ROOT / 'results/v0.8/fusion/model.json'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def database(path):
    return sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)


def read_closed_trace(record, freq):
    if record.get('stages'):
        stage, = record['stages']
        path = ROOT / stage['trace']
        assert digest(path) == stage['trace_sha256']
        issues, plain, shard_hashes = [], 0, {}
        for core, (start, _) in stage['windows'].items():
            if 'cubecore' not in core:
                continue
            shard = path.parent / core / 'trace.json'
            shard_hashes[core] = digest(shard)
            events = [e for e in read(shard)['traceEvents'] if e['ph'] == 'X']
            zero = min(e['ts'] for e in events) - start
            for e in events:
                if e['name'] == 'MMAD_MX':
                    issues.append((core, e['ts']-zero, json.loads(e['args']['detail'])))
                plain += e['name'] == 'MMAD'
        return path, stage['trace_sha256'], shard_hashes, plain, issues
    name, = record['trace']
    sha, = record['trace_sha256']
    path = ROOT / name
    assert digest(path) == sha
    with database(path) as db:
        start, end = db.execute('SELECT MIN(ExecInstrTickStart), MAX(ExecInstrTickEnd) FROM ExecutedInstructions').fetchone()
        assert math.isclose((end-start)/freq, record['ca_us'], abs_tol=1e-6)
        sql = """SELECT ExecInstrCoreId, ExecInstrTickStart, ExtendParams
                 FROM ExecutedInstructions WHERE ExecInstrName='MMAD_MX' AND CoreTypeId=1"""
        issues = [(str(c), (tick-start)/freq, json.loads(args)) for c, tick, args in db.execute(sql)]
        plain, = db.execute("SELECT count(*) FROM ExecutedInstructions WHERE ExecInstrName='MMAD'").fetchone()
    return path, sha, {}, plain, issues


def summarize_pair(pair, side):
    rec = pair[side]
    chip = read(ROOT / 'configs/chips' / (rec['chip']+'.json'))
    path, sha, shards, plain, issues = read_closed_trace(rec, chip['frequency_ghz']*1000)
    by_core = Counter(core for core, _, _ in issues)
    hist = Counter((args['m'], args['n'], args['k']) for _, _, args in issues)
    first = min(t for _, t, _ in issues)
    m, k, f, _ = pair['shape']
    logical = (4 if pair['op'] == FIRST else 2) * m*k*f
    budget = work(pair['op'], pair['shape'], pair['attrs'])
    return dict(op=pair['op'], chip=rec['chip'], shape=pair['shape'], attrs=pair['attrs'],
                api=rec['api'], kernel=rec['kernel'], role=rec['role'], ca_us=rec['ca_us'],
                public_us=pair[side+'_us'], launched_aic=rec['active_aic'], launched_aiv=rec['active_aiv'],
                mx_payload_aic=len(by_core), mx_issues=len(issues), plain_mmad=plain,
                mx_issues_per_core=dict(sorted(by_core.items())), first_mx_us=first,
                first_mx_fraction=first/rec['ca_us'], logical_cube_ops=logical,
                issued_cube_ops=sum(2*m*n*k*v for (m,n,k),v in hist.items()),
                dim_hist_mnk={f'{m}x{n}x{k}':v for (m,n,k),v in sorted(hist.items())},
                ref_work_units=budget, trace=str(path.relative_to(ROOT)), sha256=sha,
                cube_shard_sha256=shards)


def pre_issue_cost(db, lane, zero, first, rates):
    """Reconstruct target-priced cost in the reference interval preceding any MX issue."""
    sql = """SELECT ExecInstrTickStart, ExecInstrTickEnd, json_extract(ExtendParams,'$.queue_type')
             FROM ExecutedInstructions WHERE CoreTypeId=? AND ExecInstrCoreId=?
             AND ExecInstrIsDurationExcluded=0"""
    changes = defaultdict(Counter)
    for a,b,q in db.execute(sql, (lane['core_type_id'],lane['core_id'])):
        changes[a][q] += 1
        changes[b][q] -= 1
    changes[zero+lane['end_ticks']]
    changes[first]
    active = Counter()
    prev, total, prefix = zero, 0., 0.
    for tick, diff in sorted(changes.items()):
        queues = [q for q,n in active.items() if n] or ['IDLE']
        price = max(rates[q] for q in queues)
        total += (tick-prev)*price/1650
        if tick <= first:
            prefix += (tick-prev)*price/1650
        prev = tick
        active.update(diff)
    return total, prefix


def audit_templates(op, job, kernel_map):
    budget = work(op, job['shape'], job['attrs'])
    candidates = []
    for t in read(MODELS)['templates'][op]:
        digest_expected = t['db_sha256']
        kernel = kernel_map[digest_expected]
        path = Path(kernel['db_path'])
        assert digest(path) == digest_expected
        rates = {q:v/t['work'][q] for q,v in budget.items()}
        expected = max(sum(us*max(rates[q] for q in mask.split('+')) for mask,us in c.items()) for c in t['cores'])
        with database(path) as db:
            zero, = db.execute('SELECT MIN(ExecInstrTickStart) FROM ExecutedInstructions').fetchone()
            first, = db.execute("SELECT MIN(ExecInstrTickStart) FROM ExecutedInstructions WHERE ExecInstrName='MMAD_MX'").fetchone()
            lanes = []
            for kind in {c['core_type_id'] for c in kernel['cores']}:
                lane = max((c for c in kernel['cores'] if c['core_type_id'] == kind), key=lambda c:c['end_ticks'])
                total, prefix = pre_issue_cost(db, lane, zero, first, rates)
                lanes.append((total,prefix,lane))
            total,prefix,lane = max(lanes,key=lambda x:x[0])
        assert math.isclose(total,expected,abs_tol=1e-6)
        candidates.append(dict(template_shape=t['shape'], template_sha256=digest_expected,
                    template_db=str(path.relative_to(ROOT)), projected_us=total, projected_prefix_us=prefix,
                    original_prefix_us=(first-zero)/1650, prefix_fraction=prefix/total,
                    source_terminal_core=[lane['core_type_id'],lane['core_id']], work_multipliers=rates))
    candidates.sort(key=lambda c:c['projected_us'])
    middle = candidates[1:3]
    total = median(c['projected_us'] for c in candidates)
    assert math.isclose(total,job['predictions'][op]['Ascend950DT_9582']['us'],abs_tol=1e-6)
    projected_prefix = sum(c['projected_prefix_us'] for c in middle)/2
    return dict(dt_public_us=total, pre_first_mx_reference_phase_at_large_price_us=projected_prefix,
                pre_fraction=projected_prefix/total, candidates=candidates)


def main():
    pairs = read(PAIRS)
    small = [summarize_pair(p, side) for p in pairs if p['op'] in (FIRST,SECOND) for side in ('dt','pr')]
    assert len(small) == 40
    assert all(r['plain_mmad'] == 0 and r['issued_cube_ops'] == r['logical_cube_ops'] for r in small)
    job = next(job for job in read(FREEZE)['jobs'] if job['id']=='moe')
    kernels = {}
    for path in (ROOT/'runs/model-v7').glob('moe*/result.json'):
        for k in (read(path).get('measurement') or {}).get('kernels', []):
            kernels[k['db_sha256']] = k
    big = {}
    for op in (FIRST,SECOND):
        by_chip = {}
        for chip_id, record in job['predictions'][op].items():
            chip = read(ROOT/'configs/chips'/(chip_id+'.json'))
            w = record['work']
            cube = w['cube_ops'] / (chip['cube_tops']['8'] * 1e6)
            cold_hbm = (w['logical_read_bytes']+w['logical_write_bytes'])/(chip['hbm_bandwidth_gbs']*1000)
            by_chip[chip_id] = dict(public_us=record['us'], status=record['status'], cube_floor_us=cube,
                        cold_single_transfer_hbm_us=cold_hbm, ratio_to_cube_floor=record['us']/cube,
                        logical_cube_ops=w['cube_ops'], logical_read_bytes=w['logical_read_bytes'],
                        logical_write_bytes=w['logical_write_bytes'], cube_peak_fp8_tops=chip['cube_tops']['8'],
                        hbm_peak_gbs=chip['hbm_bandwidth_gbs'])
        big[op] = dict(ref_work_units=work(op,job['shape'],job['attrs']), chips=by_chip,
                       reference_prefix=audit_templates(op,job,kernels))
    evidence = dict(source_pairs=str(PAIRS.relative_to(ROOT)), pairs_sha256=digest(PAIRS),
                    freeze=str(FREEZE.relative_to(ROOT)), freeze_sha256=digest(FREEZE), template_sha256=digest(MODELS),
                    shape=job['shape'],attrs=job['attrs'], geometry=job['geometry'], large=big,
                    closed_ca=small, measurement_role='archived development replay; no new CA')
    (OUT/'mx-audit.json').write_text(json.dumps(evidence,indent=2)+'\n')


if __name__ == '__main__':
    main()
