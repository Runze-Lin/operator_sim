"""Strict CA extraction. Raw data never enters the online predictor."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def union_length(intervals):
    total, end = 0.0, float('-inf')
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError('Negative duration')
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def extract(directory, log_path):
    directory, log_path = Path(directory), Path(log_path)
    log = log_path.read_text(errors='replace')
    required = ['SMOKE_PASS', 'runtime_soc=Ascend950DT_9582', 'All task success.', 'Profiling data parse finished.']
    if not all(s in log for s in required) or 'SMOKE_FAIL' in log:
        raise ValueError('Incomplete/incorrect CA run: inspect run.log')
    reports = list(directory.glob('OPPROF_*'))
    if len(reports) != 1:
        raise ValueError(f'Expected one kernel report, found {len(reports)}')
    report = reports[0]
    kernels = re.findall(r'Start profiling on kernel: (\S+)', log)
    if len(kernels) != 1:
        raise ValueError('v0.1 requires exactly one captured kernel; multi-kernel APIs need composition')
    trace_path = report / 'simulator/trace.json'
    trace = json.loads(trace_path.read_text())
    events = [e for e in trace['traceEvents'] if e['ph'] == 'X']
    if not events:
        raise ValueError('Empty instruction timeline')
    if any(not math.isfinite(e[k]) for e in events for k in ('ts','dur')):
        raise ValueError('Non-finite trace timestamp or duration')
    # Chrome trace ts/dur are microseconds. displayTimeUnit is presentation only.
    # Also validate against this tool's per-core duration_time(us), below.
    begin = min(e['ts'] for e in events)
    end = max(e['ts'] + e['dur'] for e in events)
    by_core = defaultdict(list)
    for e in events:
        if e['dur'] < 0:
            raise ValueError('Negative trace duration')
        by_core[e['pid']].append(e)
    cores = []
    for name, core_events in sorted(by_core.items()):
        start = min(e['ts'] for e in core_events)
        finish = max(e['ts'] + e['dur'] for e in core_events)
        span = finish - start
        timing = re.search(r'^' + re.escape(name) + r'\s+([\d.]+)\s+([\d.]+)\s*$', log, re.M)
        if not timing or abs(span - float(timing[1])) > max(.03, span * .02):
            raise ValueError(f'{name}: timeline units/scope do not match profiler duration')
        intervals = defaultdict(list)
        counts = Counter()
        matrix_shapes = Counter()
        cube_flops = 0
        for e in core_events:
            counts[e['tid']] += 1
            detail = json.loads(e.get('args', {}).get('detail', '{}'))
            if not any(detail.get(k, 0) for k in ('is_sync', 'is_wait', 'is_duration_excluded')):
                intervals[e['tid']].append((e['ts'], e['ts'] + e['dur']))
            if e['name'] == 'MMAD' and all(k in detail for k in ('m', 'n', 'k')):
                cube_flops += 2 * detail['m'] * detail['n'] * detail['k']
                matrix_shapes[(detail['m'],detail['n'],detail['k'])] += 1
        path = report / 'simulator' / name / (name + '_instr_exe.csv')
        opcodes, pipe_counts = Counter(), Counter()
        with path.open() as stream:
            # The trailing detail field has non-standard quoting; only use
            # leading well-formed columns. The trace has valid JSON detail.
            for row in csv.DictReader(stream):
                opcodes[row['instr']] += int(row['call_count'])
                pipe_counts[row['pipe']] += int(row['call_count'])
        vector = [pair for pipe, pairs in intervals.items() if pipe.startswith('RVEC') or pipe == 'VECTOR' for pair in pairs]
        busy = {p: union_length(v) for p, v in sorted(intervals.items())}
        busy['VECTOR_FAMILY'] = union_length(vector)
        cores.append({
            'name': name, 'kind': 'AIC' if 'cubecore' in name else 'AIV',
            'span_us': span, 'start_us': start - begin,
            'profiler_duration_us': float(timing[1]),
            'busy_union_us_by_pipe': busy,
            'busy_fraction_by_pipe': {p: t / (end - begin) for p, t in busy.items()},
            'dynamic_opcodes': dict(opcodes), 'dynamic_calls_by_pipe': dict(pipe_counts),
            'timeline_event_counts': dict(counts), 'executed_cube_flops': cube_flops,
            'matrix_instruction_shapes': [{'m':m,'n':n,'k':k,'count':count} for (m,n,k),count in sorted(matrix_shapes.items())],
        })
    return {
        'reference_us': end - begin, 'reference_scope': 'cross-core instruction makespan',
        'kernel': kernels[0], 'cores': cores,
        'active_aic': sum(c['kind'] == 'AIC' for c in cores),
        'active_aiv': sum(c['kind'] == 'AIV' for c in cores),
        'correctness': re.findall(r'^correctness .+$', log, re.M),
        'trace_sha256': sha256(trace_path),
        'binary_sha256': sha256(report / 'dump/aicore_binary.o'),
        'report_path': str(report),
        'warnings': sorted(set(re.findall(r'^.*(?:\[WARN\]|\[error\]).*$', log, re.M))),
        'zero_duration_events': sum(e['dur'] == 0 for e in events),
        'cache_state': 'fresh process; shipped CA config has L2 cache_enable=0',
    }
