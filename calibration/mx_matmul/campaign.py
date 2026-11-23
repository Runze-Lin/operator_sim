"""Run a sealed dense MX validation set with bounded CA concurrency and restart support."""

import argparse
import fcntl
import json
import re
import shutil
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from time import monotonic

from calibration.__main__ import ingest, read, require
from calibration.mx_matmul.collect import OBJECT, capture, convert, resolve_case, write_json
from calibration.mx_matmul.native_phases import summarize


def enough_space(root, disk_gib, memory_gib):
    if shutil.disk_usage(root).free < disk_gib * 2**30:
        return 'disk reserve'
    line = next(row for row in Path('/proc/meminfo').read_text().splitlines() if row.startswith('MemAvailable:'))
    if int(line.split()[1]) * 1024 < memory_gib * 2**30:
        return 'memory reserve'
    return None


def run_case(frozen, case, route, folder, probe, opp, launcher, minutes):
    raw = folder/case['id']
    # Resuming or taking a single case from the queue on another worker must
    # never start a second simulation or inspect a partial receipt.
    with (folder/('.' + case['id'] + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return collect_case(frozen, case, route, raw, probe, opp, launcher, minutes)


def collect_case(frozen, case, route, raw, probe, opp, launcher, minutes):
    start = monotonic()
    if not raw.exists():
        capture(case, route, frozen['sha256'], raw, probe, launcher, minutes)
    require((raw/'freeze.sha256').read_text().strip() == frozen['sha256'], 'Different freeze at ' + str(raw))
    if (raw/'receipt.json').exists():
        receipt = read(raw/'receipt.json')
    else:
        receipt = convert(case, route, raw, opp)
    measurement = ingest(frozen, receipt)
    require(measurement['case_id'] == case['id'] and measurement['data_role'] == 'prospective',
            'Not bound to a prospective acquisition')
    write_json(raw/'receipt.json', receipt)
    write_json(raw/'measurement.json', measurement)
    if (raw/'phases.json').exists():
        phases = read(raw/'phases.json')
    else:
        phases = summarize(raw)
        write_json(raw/'phases.json', phases)
    require(abs(phases['full_window_us']-measurement['ca_us']) < .002,
            'Independent physical-shard phase extraction produced a different window')
    cost = read(raw/'cost.json')
    return dict(case_id=case['id'], ca_us=measurement['ca_us'], aics=phases['active_mmad_aics'],
                producer_aivs=phases['active_vector_instruction_aivs'], ca_wall_s=round(cost['wall_seconds'],3),
                campaign_task_s=round(monotonic()-start,3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('freeze', 'raw-root', 'probe', 'opp'):
        parser.add_argument('--'+option, type=Path, required=True)
    parser.add_argument('--launcher', type=Path)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--minutes', type=int, default=20)
    parser.add_argument('--reserve-disk-gib', type=int, default=48)
    parser.add_argument('--reserve-memory-gib', type=int, default=32)
    parser.add_argument('--case', dest='selected', action='append', help='Limit processing to named frozen case(s)')
    args = parser.parse_args()
    require(args.jobs > 0 and args.minutes > 0 and args.reserve_disk_gib >= 0 and args.reserve_memory_gib >= 0,
            'Invalid concurrency/timeout/resources')
    frozen = read(args.freeze)
    planned = [resolve_case(frozen, c['id']) for c in frozen['cases']]
    if args.selected:
        planned = [(c,r) for c,r in planned if c['id'] in args.selected]
        require({c['id'] for c,r in planned} == set(args.selected), 'Unknown case selected')
    for c,r in planned:
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', c['id']), 'Invalid case identifier')
        op, name = OBJECT[r]
        require((args.opp/op/(name+'.o')).is_file(), 'Missing installed object ' + name)
    root = args.raw_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    probe = args.probe.resolve(strict=True)
    launcher = args.launcher.resolve(strict=True) if args.launcher else None
    planned.sort(key=lambda cr: (cr[0]['shape'][0]*cr[0]['shape'][1]*cr[0]['shape'][2], cr[0]['id']))
    running, failures = {}, []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        while planned or running:
            while planned and len(running) < args.jobs:
                reason = enough_space(root, args.reserve_disk_gib, args.reserve_memory_gib)
                if reason:
                    if not running:
                        raise RuntimeError(f'{reason} reached with {len(planned)} cases pending; resume after capacity recovers')
                    break
                case, route = planned.pop(0)
                print(json.dumps({'event':'start_or_resume', 'case':case['id']}), flush=True)
                job = executor.submit(run_case, frozen, case, route, root, probe, args.opp.resolve(), launcher, args.minutes)
                running[job] = case['id']
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for job in finished:
                ident = running.pop(job)
                try:
                    result = job.result()
                    print(json.dumps(dict(event='qualified', **result)), flush=True)
                except Exception as error:
                    failures.append(ident)
                    print(json.dumps(dict(event='incomplete', case=ident, message=str(error))), flush=True)
    if failures:
        sys.exit('Incomplete cases: '+', '.join(failures))


if __name__ == '__main__':
    main()
