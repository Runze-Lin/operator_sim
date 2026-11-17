#!/usr/bin/env python3
"""Summarize the three CA smoke runs without treating summed latency as utilization."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(name):
    log_path = ROOT / 'runs/smoke-950dt' / (name + '.log')
    log = log_path.read_text()
    reports = sorted((log_path.parent / name).glob('OPPROF_*'))
    if len(reports) != 1:
        raise RuntimeError(f'{name}: expected one report, found {len(reports)}')
    report = reports[0]
    trace_path = report / 'simulator/trace.json'
    trace = json.loads(trace_path.read_text())
    events = [e for e in trace['traceEvents'] if e['ph'] == 'X']
    pipes = collections.Counter(e['tid'] for e in events)
    cores = []
    for path in sorted(report.rglob('*_instr_exe.csv')):
        with path.open() as stream:
            # Vendor CSV's trailing detail JSON is not RFC4180 quoted. Only
            # consume the six well-formed leading fields; use trace for detail.
            rows = list(csv.DictReader(stream))
        counts = collections.Counter()
        opcodes = collections.Counter()
        for row in rows:
            counts[row['pipe']] += int(row['call_count'])
            opcodes[row['instr']] += int(row['call_count'])
        core = path.parent.name
        raw = report / 'dump' / (core + '.instr_log.dump')
        timing = re.search(r'^' + re.escape(core) + r'\s+([\d.]+)\s+([\d.]+)\s*$', log, re.M)
        cores.append({
            'core': core,
            'csv_rows': len(rows),
            'csv_call_count_by_pipe': dict(sorted(counts.items())),
            'csv_call_count_by_opcode': dict(sorted(opcodes.items())),
            'raw_instruction_records': sum('(PC:' in line for line in raw.open()),
            'profiler_duration_us': float(timing[1]) if timing else None,
            'profiler_running_us': float(timing[2]) if timing else None,
        })
    has_vec = any(pipes[p] for p in ('VECTOR', 'RVECEX'))
    checks = {
        'correctness': 'SMOKE_PASS' in log and 'bad=0' in log,
        'target': 'runtime_soc=Ascend950DT_9582' in log,
        'profiler_finished': 'Profiling data parse finished.' in log and 'All task success.' in log,
        'instruction_records': bool(cores) and all(c['raw_instruction_records'] > 0 for c in cores),
        'timeline': bool(events),
        'expected_compute_pipes': has_vec if name == 'quant' else bool(pipes['CUBE']) and (name != 'fa' or has_vec),
    }
    return {
        'name': name, 'checks': checks,
        'report': str(report.relative_to(ROOT)),
        'kernels': re.findall(r'Start profiling on kernel: (\S+)', log),
        'correctness': re.findall(r'^correctness .+$', log, re.M),
        'cores': cores, 'timeline_events_by_pipe': dict(sorted(pipes.items())),
        'zero_duration_events_by_pipe': dict(sorted(collections.Counter(e['tid'] for e in events if e['dur'] == 0).items())),
        'trace_sha256': digest(trace_path),
        'kernel_object_sha256': digest(report / 'dump/aicore_binary.o'),
        'warnings': sorted(set(re.findall(r'^.*(?:\[WARN\]|\[error\]).*$', log, re.M))),
    }


if __name__ == '__main__':
    result = {
        'date': '2026-11-17', 'soc_version': 'Ascend950DT_9582', 'cann_version': '9.2.1',
        'scope': 'One synthetic input per operator; no sweep or calibration',
        'cases': [summarize(name) for name in ('quant', 'matmul', 'fa')],
        'interpretation': [
            'CSV rows aggregate repeated PCs: use call_count, not row count, for dynamic counts.',
            'CSV call counts, raw records and timeline events have different filtering/representation; do not assume equality.',
            'Pipeline counts include synchronization instructions: FA CUBE has 21 calls but only 2 MMAD calls.',
            'Metadata tracks are excluded from timeline counts; only phase X events count.',
            'Durations are rounded per-core profiler values, not end-to-end operator latency.',
            'Overlapping instruction latency is not additive; no utilization estimated by this script.',
            'Some vector events have zero duration. Source-level debug mapping is absent in vendor binaries.',
            'Host wall time was not separately instrumented for these initial runs.',
        ],
    }
    destination = ROOT / 'docs/smoke-950dt-results.json'
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    for case in result['cases']:
        print(case['name'], case['checks'])
    if not all(all(case['checks'].values()) for case in result['cases']):
        raise SystemExit('One or more smoke checks failed')
