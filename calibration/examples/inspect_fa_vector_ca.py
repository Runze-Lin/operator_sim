"""Read one small v2.1 FA CA trace and independently check timing and RVEC counts."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


def inspect(folder, chip, dtype):
    log = (folder / 'run.log').read_text()
    memory_file = 'hbm3_8000M.ini' if chip == 'Ascend950DT_9582' else 'lpddr5_6400M_bailu.ini'
    accepted = [memory_file, 'All task success.', 'Profiling data parse finished.', 'SMOKE_PASS']
    gold = re.findall(r'^correctness .*$', log, re.M)
    kernels = re.findall(r'Start profiling on kernel: (\S+)', log)
    if (not all(tag in log for tag in accepted) or 'SMOKE_FAIL' in log
            or re.findall(r'^runtime_soc=(\S+)', log, re.M) != [chip]
            or len(gold) != 1 or 'bad=0' not in gold[0] or 'elements=576' not in gold[0]
            or len(kernels) != 1 or not kernels[0].startswith('FlashAttentionScore_')):
        raise ValueError('Incomplete simulator/golden output or unexpected dispatch')
    roster = {name: float(duration) for name, duration in re.findall(
        r'^(core\d+\.(?:cube|vec)core\d+)\s+([\d.]+)\s+[\d.]+\s*$', log, re.M)}
    trace, = folder.glob('profile/OPPROF_*/simulator/trace.json')
    payload = trace.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    cores = {}
    ends = Counter()
    mmads = Counter()
    positive = defaultdict(Counter)
    zero = defaultdict(Counter)
    for event in json.loads(payload)['traceEvents']:
        if event.get('ph') != 'X':
            continue
        core, name = event['pid'], event['name']
        t = event['ts']
        duration = event['dur']
        if core not in roster or not math.isfinite(t+duration) or duration < 0:
            raise ValueError('Unknown core or invalid timestamp')
        begin, stop = cores.get(core, (t, t))
        cores[core] = min(begin, t), max(stop, t+duration)
        if name == 'END':
            ends[core] += 1
        if name == 'MMAD':
            args = json.loads(event['args']['detail'])
            fields = ('matrix_a_dtype', 'matrix_b_dtype', 'matrix_c_dtype')
            mmads[tuple(args.get(k, 'unreported') for k in fields)] += 1
        if event['tid'].startswith('RVEC'):
            (positive if duration else zero)[core][name] += 1
    expected = ('bf16', 'bf16', 'f32') if dtype == 'bf16' else ('f16', 'f16', 'f32')
    expected_count = 66 if dtype == 'bf16' else 6
    if (sum(mmads.values()) != expected_count or not mmads[expected]
            or set(mmads) - {expected, ('unreported',)*3}):
        raise ValueError('Unexpected MMAD count or operand precision')
    if set(cores) != set(roster) or ends != Counter({name: 1 for name in roster}):
        raise ValueError('Missing END or active core')
    if any(abs((stop-begin)-roster[c]) > max(.03, .02*roster[c]) for c,(begin,stop) in cores.items()):
        raise ValueError('Per-core window disagrees with simulator')
    return dict(kernel=kernels[0], trace_sha256=sha, correctness=gold[0],
                ca_us=max(stop for _,stop in cores.values())-min(start for start,_ in cores.values()),
                mmads_by_precision={','.join(k):v for k,v in mmads.items()},
                rvec_positive_duration_opcode_counts_by_aiv=positive,
                rvec_zero_duration_opcode_records_by_aiv=zero)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw', type=Path)
    parser.add_argument('chip', choices=['Ascend950DT_9582', 'Ascend950PR_957b'])
    parser.add_argument('dtype', choices=['bf16', 'fp16'])
    args = parser.parse_args()
    print(json.dumps(inspect(args.raw, args.chip, args.dtype), indent=2))
