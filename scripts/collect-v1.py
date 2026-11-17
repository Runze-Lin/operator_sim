#!/usr/bin/env python3
"""Bounded, serial, resumable CA collection with content-addressed cache."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from operator_model.ca import extract, sha256


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def tree_size(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['calibration', 'holdout', 'repeat'], required=True)
    parser.add_argument('--case')
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--case-timeout', type=int)
    parser.add_argument('--profiler-timeout', type=int)
    parser.add_argument('--version', choices=['0.1','0.2','0.3'], default='0.1')
    args = parser.parse_args()
    generation = 'v' + args.version.split('.')[-1]
    spec_path = ROOT / ('configs/sampling-' + generation + '.json')
    out = ROOT / 'results' / ('v' + args.version)
    implementation = ROOT / 'operator_model' / ('model.py' if args.version == '0.1' else 'model_' + generation + '.py')
    spec = json.loads(spec_path.read_text())
    if args.case and not any(c['id']==args.case and c['split']==args.split for c in spec['cases']):
        raise SystemExit('Requested case is not in this split')
    run_root = ROOT / 'runs' / ('model-' + generation)
    run_root.mkdir(parents=True, exist_ok=True)
    ledger_path = run_root / 'ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    prefix = ROOT / 'toolchains/cann/9.2.1/cann-9.2.1'
    identity = {
        'soc_version': spec['soc_version'], 'cann_version': spec['cann_version'],
        'host_binary_sha256': sha256(ROOT / 'build/smoke/three_ops'),
        'package_manifest_sha256': sha256(ROOT / 'docs/cann-packages.json'),
        'platform_sha256': sha256(prefix / 'aarch64-linux/data/platform_config/Ascend950DT_9582.ini'),
        'sim_config_sha256': sha256(prefix / 'tools/simulator/dav_3510/lib/config.json'),
        'capture': {'dump': True, 'launch_count': 10},
        'input_pattern': 'deterministic-bf16-v1',
    }
    if args.split == 'holdout':
        freeze = json.loads((out / 'frozen-predictions.json').read_text())
        if freeze['model_sha256'] != sha256(out / 'model.json'):
            raise SystemExit('Model changed after prediction freeze')
        if freeze['manifest_sha256'] != sha256(spec_path):
            raise SystemExit('Sampling manifest changed after prediction freeze')
        if freeze['implementation_sha256'] != sha256(implementation):
            raise SystemExit('Predictor implementation changed after prediction freeze')
        if 'shared_implementation_sha256' in freeze and freeze['shared_implementation_sha256'] != sha256(ROOT/'operator_model/model.py'):
            raise SystemExit('Shared predictor implementation changed after prediction freeze')
        for path,expected in freeze.get('artifact_sha256',{}).items():
            if sha256(ROOT/path)!=expected:raise SystemExit('Frozen artifact changed: '+path)
    budget = dict(spec['budget'])
    if args.attempt < 1 or (args.attempt > 1 and not args.case):
        raise SystemExit('Retries require an explicit --case and positive --attempt')
    if args.case_timeout:
        if not 1 <= args.case_timeout <= 600: raise SystemExit('Timeout must be within 600 seconds')
        budget['case_timeout_seconds'] = args.case_timeout
    if args.profiler_timeout:
        if not 1 <= args.profiler_timeout <= 9: raise SystemExit('Profiler timeout must be within 9 minutes')
        budget['profiler_timeout_minutes'] = args.profiler_timeout
    for case in spec['cases']:
        if case['split'] != args.split or (args.case and args.case != case['id']):
            continue
        case_timeout = args.case_timeout or case.get('timeout_seconds',budget['case_timeout_seconds'])
        profiler_timeout = args.profiler_timeout or case.get('profiler_timeout_minutes',budget['profiler_timeout_minutes'])
        if not 1 <= case_timeout <= 600 or not 1 <= profiler_timeout <= 9:
            raise SystemExit('Per-case timeout outside bounded slow-case policy')
        key_input = {'identity': identity, 'op': case['op'], 'shape': case['shape'], 'attrs': spec['attributes'][case['op']]}
        key = hashlib.sha256(json.dumps(key_input, sort_keys=True).encode()).hexdigest()[:20]
        folder = run_root / (case['id'] + '-' + key + ('' if args.attempt == 1 else '-attempt' + str(args.attempt)))
        result_path = folder / 'result.json'
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous['status'] == 'ok':
                # Validate artifacts, do not silently trust a stale cache entry.
                report = Path(previous['measurement']['report_path'])
                if sha256(report / 'simulator/trace.json') != previous['measurement']['trace_sha256']:
                    raise SystemExit('Cache trace hash mismatch')
                print('CACHE', case['id'], flush=True)
                continue
            print('PREVIOUS_FAILURE', case['id'], 'not retried automatically', flush=True)
            continue
        used_wall = sum(x['wall_seconds'] for x in ledger)
        used_cpu = sum(x['cpu_seconds'] for x in ledger)
        if (used_wall + case_timeout > budget['wall_seconds'] or
                used_cpu >= budget['cpu_seconds'] or tree_size(run_root) >= budget['raw_bytes'] or
                len(ledger) >= budget['max_attempts']):
            print('BUDGET_STOP', flush=True)
            break
        if folder.exists():
            raise SystemExit(f'Incomplete prior attempt at {folder}; inspect before resuming')
        folder.mkdir()
        save(folder / 'input.json', key_input)
        cmd = ['bash', str(ROOT / 'scripts/cann-run.sh'), 'msopprof', 'simulator',
               '--soc-version=' + spec['soc_version'], '--dump=true', '--launch-count=10',
               '--timeout=' + str(profiler_timeout), '--output=' + str(folder / 'profile'),
               str(ROOT / 'build/smoke/three_ops'), case['op'], *map(str, case['shape'])]
        start = time.perf_counter()
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        print('START', case['id'], case['shape'], flush=True)
        timed_out = False
        with (folder / 'run.log').open('w') as log:
            process = subprocess.Popen(cmd, cwd=folder, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=case_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = 124
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        result = {
            'case': case, 'cache_key': key, 'identity': identity, 'command': cmd, 'attempt':args.attempt,
            'finished_utc': datetime.now(timezone.utc).isoformat(),
            'wall_seconds': time.perf_counter() - start,
            'cpu_seconds': after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
            'returncode': code, 'timed_out': timed_out, 'status': 'failed',
        }
        if code == 0:
            try:
                result['measurement'] = extract(folder / 'profile', folder / 'run.log')
                result['status'] = 'ok'
            except (ValueError, OSError, KeyError) as error:
                result['error'] = str(error)
        result['raw_bytes'] = tree_size(folder)
        save(result_path, result)
        ledger.append({k: result[k] for k in ('cache_key', 'wall_seconds', 'cpu_seconds', 'raw_bytes', 'status')})
        save(ledger_path, ledger)
        print('DONE', case['id'], result['status'], round(result['wall_seconds'], 2), 'seconds', flush=True)
    # Publish compact records, never raw traces or installation assets.
    attempts = [json.loads(p.read_text()) for p in run_root.glob('*/result.json')]
    selected = {}
    for record in sorted(attempts, key=lambda r:r['finished_utc']):
        selected[record['case']['id']] = record
    records = list(selected.values())
    import_path = out / 'imported-calibration.json'
    if import_path.exists():
        records += json.loads(import_path.read_text())['records']
    save(out / 'dataset.json', {'manifest_sha256': sha256(spec_path), 'records': records, 'attempts':[{k:r.get(k) for k in ('case','attempt','status','returncode','error','wall_seconds','cpu_seconds','raw_bytes','finished_utc')} for r in attempts]})


if __name__ == '__main__':
    main()
