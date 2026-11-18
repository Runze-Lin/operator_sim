"""Qualify exact-SKU paired captures using the common cross-core trace window."""
import json
import math
from collections import defaultdict
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model.ca import sha256

OUT = ROOT/'results/v0.13/sku'
holdout = sys.argv[1:] == ['--holdout']
if sys.argv[1:] and not holdout:
    raise SystemExit('Usage: report.py [--holdout]')
budget = json.loads((OUT/('holdout-budget.json' if holdout else 'budget.json')).read_text())


def summarize(directory, soc):
    log = (directory/'run.log').read_text(errors='replace')
    if 'SMOKE_FAIL' in log:
        raise ValueError('Kernel validation failure')
    if re.findall(r'^runtime_soc=(\S+)', log, re.M) != [soc]:
        raise ValueError('Runtime SKU mismatch')
    for marker in ('All task success.', 'Profiling data parse finished.', 'SMOKE_PASS'):
        if marker not in log:
            raise ValueError('Missing ' + marker)
    correct = re.findall(r'^correctness (.+)', log, re.M)
    if not correct or not all(re.search(r'\bbad=0\b', row) for row in correct):
        raise ValueError('Numerical validation failed')
    kernel, = re.findall(r'Start profiling on kernel: (\S+)',log)
    trace, = directory.glob('profile/OPPROF_*/simulator/trace.json')
    per_core = defaultdict(list)
    for e in json.loads(trace.read_text())['traceEvents']:
        if e['ph'] == 'X':
            t,d = e['ts'],e['dur']
            if not(math.isfinite(t) and math.isfinite(d) and d >= 0):
                raise ValueError('Malformed timing')
            per_core[e['pid']].append((t,t+d))
    spans = []
    for core, intervals in per_core.items():
        a = min(p[0] for p in intervals)
        b = max(p[1] for p in intervals)
        vendor, = re.findall(r'^'+re.escape(core)+r'\s+([.\d]+)\s+[.\d]+\s*$',log,re.M)
        if abs((b-a)-float(vendor)) > max(.03, .02*(b-a)):
            raise ValueError('Profiler/core timing disagreement')
        spans.append((a,b))
    ca_cfg, = re.findall(r'Using config from SO dir: (\S+)', log)
    return dict(soc=soc, kernel=kernel, correctness=correct, events=sum(map(len,per_core.values())),
                active_aiv=sum('veccore' in core for core in per_core),
                active_aic=sum('cubecore' in core for core in per_core),
                ca_us=max(p[1] for p in spans)-min(p[0] for p in spans),
                trace_sha256=sha256(trace), kernel_binary_sha256=sha256(trace.parents[1]/'dump/aicore_binary.o'),
                ca_config_sha256=sha256(ca_cfg), trace=str(trace.relative_to(ROOT)))


pairs = []
for case in budget['cases']:
    dt_record = ROOT/case['dt_result']
    dt = summarize(dt_record.parent, 'Ascend950DT_9582')
    archived = json.loads(dt_record.read_text())['measurement']
    assert dt['trace_sha256'] == archived['trace_sha256'] and abs(dt['ca_us']-archived['reference_us']) < 1e-8
    pr_dir = ROOT/('runs/sku13-holdout' if holdout else 'runs/sku13-profile')/case['op']
    pr = summarize(pr_dir, budget['sku'])
    cost = json.loads((pr_dir/'cost.json').read_text())
    assert cost['exit_code'] == 0
    cost['raw_bytes'] = sum(p.stat().st_size for p in pr_dir.rglob('*') if p.is_file())
    infra = ROOT/'runs/sku13'/case['op']
    failed = None
    if not holdout and (infra/'cost.json').exists():
        log = (infra/'run.log').read_text()
        failed = dict(**json.loads((infra/'cost.json').read_text()),
                      runtime_soc=re.findall(r'runtime_soc=(\S+)',log), timing_eligible=False,
                      socket_error='connect server failed' in log)
    pairs.append(dict(op=case['op'], shape=case['shape'], dt=dt, pr=pr, pr_cost=cost,
                      pr_to_dt_ratio=pr['ca_us']/dt['ca_us'], rejected_infrastructure_attempt=failed))
if holdout:
    freeze = json.loads((OUT/'holdout-freeze.json').read_text())
    for pair, predictions in zip(pairs, freeze['predictions']):
        assert pair['shape'] == predictions['shape'] and pair['op'] == predictions['op']
        pair.update(frozen=predictions, pr_error_percent=100*(predictions['pr_us']/pair['pr']['ca_us']-1))
(OUT/('holdout.json' if holdout else 'pairs.json')).write_text(json.dumps(pairs,indent=2)+'\n')

if not holdout:
    memory_models = []
    for pair in pairs:
        entry = dict(op=pair['op'])
        for key in ('dt', 'pr'):
            directory = (ROOT/pair[key]['trace']).parents[3]
            log = (directory/'run.log').read_text()
            entry[key] = [line for line in log.splitlines() if 'Loading device model file' in line
                          or 'Channels,' in line or line.startswith('soc version:')]
        memory_models.append(entry)
    (OUT/'memory-models.json').write_text(json.dumps(memory_models,indent=2)+'\n')
