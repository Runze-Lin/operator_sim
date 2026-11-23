"""Collect or reprocess a frozen CANN 9.2.1 QuantMatmulV5 ND CA case."""

import argparse
import json
import os
import re
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from calibration.__main__ import CASE_FIELDS, digest, encoded, ingest, read, require, unseal
from calibration.msopprof import CORE, sha

OBJECT = {
    'a4w4': ('quant_batch_matmul_v3',
             'QuantBatchMatmulV3_ND_ND_mxfp4e2m1_mxfp4e2m1_float8_e8m0_bf16_high_performance'),
    'a8w4': ('quant_batch_matmul_v4',
             'QuantBatchMatmulV4_ND_ND_float8_e4m3fn_float4_e2m1_out_bf16_high_performance'),
}
MEMORY_CONFIG = {
    'Ascend950DT_9582': 'hbm3_8000M.ini',
    'Ascend950PR_957b': 'lpddr5_6400M_bailu.ini',
}


def write_json(path, data):
    if path.exists():
        require(read(path) == data, f'Existing result differs: {path}')
        return
    with path.open('x') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write('\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def resolve_case(frozen, case_id):
    unseal(frozen, 'freeze')
    case = next((c for c in frozen['cases'] if c['id'] == case_id), None)
    require(case and case['op'] == 'matmul' and case['api'] == 'aclnnQuantMatmulV5'
            and case['prefix'] is None, 'Expected isolated dense aclnnQuantMatmulV5')
    m, n, k = case['shape']
    require(case['outputs'] == {'Y': m*n} and k % 64 == 0, 'Unsupported native probe shape/output')
    chip = case['chip']
    require(chip in MEMORY_CONFIG and frozen['chips'][chip]['frequency_ghz'] == 1.65,
            'Expected 950DT or 950PR CA at 1.65 GHz')
    if case['dtype'] == 'mxfp4' and case['attrs'] == {}:
        return case, 'a4w4'
    require(case['dtype'] == 'fp8_e4m3fn' and case['attrs'] == {'weight_dtype': 'mxfp4'},
            'Unsupported dtype/attributes for this native driver')
    return case, 'a8w4'


def capture(case, route, freeze_sha, raw, probe, launcher, minutes):
    probe = probe.resolve(strict=True)
    cmd = ['msopprof', 'simulator', '--soc-version=' + case['chip'], '--dump=false',
           '--timeout=' + str(minutes), '--output=' + str(raw / 'profile'),
           str(probe), route, *map(str, case['shape'])]
    if launcher:
        cmd = ['bash', str(launcher.resolve(strict=True)), *cmd]
    raw.mkdir(parents=True, exist_ok=False)
    (raw / 'freeze.sha256').write_text(freeze_sha + '\n')
    start = now()
    begin = monotonic()
    reason = 'launcher failed'
    status = None
    try:
        with (raw / 'run.log').open('x') as log:
            task = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=raw,
                                    start_new_session=True)
            try:
                status = task.wait(timeout=minutes*60)
                reason = None if status == 0 else 'nonzero exit'
            except subprocess.TimeoutExpired:
                reason = 'host timeout'
                os.killpg(task.pid, signal.SIGTERM)
                try:
                    status = task.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(task.pid, signal.SIGKILL)
                    status = task.wait()
    finally:
        write_json(raw/'cost.json', dict(started_utc=start, finished_utc=now(), command=cmd,
                                        probe_sha256=sha(probe), wall_seconds=monotonic()-begin,
                                        exit_code=status, stop_reason=reason))


def core_windows(simulator, roster):
    # The merged Chrome trace omits inter-pipe events, including trailing
    # WAIT_INTRA_BLOCK on the mixed kernel's AIVs. The physical shards retain
    # those retirements, use a common clock, and reproduce the official rows.
    shards = {p.parent.name:p for p in simulator.glob('core*.*core*/trace.json')}
    require(set(shards) == set(roster), 'Physical-core shards differ from launched roster')
    cores, hashes = {}, {}
    for name, path in sorted(shards.items()):
        source = path.read_bytes()
        events = [e for e in json.loads(source)['traceEvents'] if e['ph'] == 'X']
        require(events and all(e['ts'] >= 0 and e['dur'] >= 0 for e in events),
                f'Invalid device duration on {name}')
        lo = min(e['ts'] for e in events)
        hi = max(e['ts'] + e['dur'] for e in events)
        ends = [e['ts'] for e in events if e['name'] == 'END']
        require(len(ends) == 1 and abs(hi-lo-roster[name]) < .011,
                f'Physical END or vendor per-core duration mismatch on {name}')
        cores[name] = dict(first_tick=round(lo*1650), end_tick=round(ends[0]*1650),
                           retire_tick=round(hi*1650), end_count=1)
        hashes[name] = digest(source)
    return cores, hashes


def convert(case, route, raw, opp):
    chip = case['chip']
    m, n, k = case['shape']
    cost = read(raw/'cost.json')
    require(cost['exit_code'] == 0 and cost['stop_reason'] is None, 'CA acquisition did not finish')
    log = (raw/'run.log').read_text()
    require(re.findall(r'^runtime_soc=(\S+)', log, re.M) == [chip] and MEMORY_CONFIG[chip] in log,
            'SKU or external memory model mismatch')
    tag = (f'case=matmul api=aclnnQuantMatmulV5 route={route} M={m} N={n} K={k} '
           f'layout=ND_NDT scales=E8M0 group=32 out=BF16 workspace=')
    cases = re.findall(r'^case=[^\r\n]+', log, re.M)
    require(len(cases) == 1 and re.fullmatch(re.escape(tag) + r'\d+', cases[0]),
            'Shape, layout or precision differs from native probe')
    require(re.findall(r'^correctness[^\r\n]*', log, re.M) == [f'correctness count={m*n} bad=0']
            and 'SMOKE_PASS' in log and 'SMOKE_FAIL' not in log and 'All task success.' in log
            and 'Profiling data parse finished.' in log, 'Incomplete correctness or profiling')
    kernels = re.findall(r'Start profiling on kernel: (\S+)', log)
    folder, stem = OBJECT[route]
    require(len(kernels) == 1 and re.fullmatch(re.escape(stem) + r'_\d+(?:_mix_ai[cv])?', kernels[0]),
            f'Unexpected runtime kernels {kernels}')
    profiles = list((raw/'profile').glob('OPPROF_*'))
    require(len(profiles) == 1, 'Expected exactly one profile')
    base = profiles[0]
    actual = base/'dump/aicore_binary.o'
    reference = opp/folder/(stem+'.o')
    require(reference.resolve() != actual.resolve(), 'Supply independent OPP library')
    obj_sha = sha(reference)
    require(obj_sha == sha(actual), 'Actual CA image differs from installed OPP library')
    rows = CORE.findall(log)
    roster = {name: float(us) for name, us in rows}
    require(roster and len(roster) == len(rows), 'Missing or duplicate per-core profiling rows')
    cores, hashes = core_windows(base/'simulator', roster)
    return dict(schema=1, kind='ca-receipt', case={field:case[field] for field in CASE_FIELDS},
                provider='CANN 9.2.1 ops-nn', timing='all-core-instruction', complete=True,
                started_utc=cost['started_utc'], finished_utc=cost['finished_utc'],
                run_log_sha256=sha(raw/'run.log'), correctness={'Y': {'checked':m*n,'bad':0}},
                stages=[dict(kernel=kernels[0], kernel_binary_sha256=obj_sha,
                             trace_sha256=digest(encoded(hashes)), physical_shard_sha256=hashes,
                             ticks_per_us=1650, launched_cores=sorted(roster), cores=cores)],
                freeze_sha256=(raw/'freeze.sha256').read_text().strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('freeze','raw','opp'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--launcher', type=Path, help='Optional shell script that activates CANN then execs its arguments')
    parser.add_argument('--case', required=True)
    parser.add_argument('--minutes', type=int, default=20)
    parser.add_argument('--convert-only', action='store_true', help='Reprocess an existing completed acquisition')
    args = parser.parse_args()
    try:
        frozen = read(args.freeze)
        case, route = resolve_case(frozen, args.case)
        raw = args.raw.resolve()
        if not args.convert_only:
            require(args.probe and args.minutes >= 1, 'Provide --probe and a positive timeout')
            capture(case, route, frozen['sha256'], raw, args.probe, args.launcher, args.minutes)
        receipt = convert(case, route, raw, args.opp)
        measurement = ingest(frozen, receipt)
        write_json(raw/'receipt.json', receipt)
        write_json(raw/'measurement.json', measurement)
        print(f"{args.case} {measurement['ca_us']:.6f} us")
    except (OSError, ValueError, TypeError, KeyError) as err:
        parser.exit(1, f'{err}\n')


if __name__ == '__main__':
    main()
