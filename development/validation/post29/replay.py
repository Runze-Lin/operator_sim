"""Replay real dispatch order with zero profiler attributes passed to OperatorStream."""
from collections import defaultdict
from hashlib import sha256
from itertools import groupby
from math import isclose, prod
from pathlib import Path
from statistics import median
import json
import re
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model import OperatorStream, estimate

OUT = ROOT / 'results/v0.29/post'
OP = 'mhc_post'
DT, PR, FUTURE = 'Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'
source_hashes = {}
rows = {}
physical = {}
failed = []
timings = []


def read(file):
    file = Path(file)
    data = (ROOT / file).read_bytes()
    source_hashes[str(file.relative_to(ROOT) if file.is_absolute() else file)] = sha256(data).hexdigest()
    return json.loads(data)


class Dispatch:
    def __init__(self, sku):
        self.sku = sku
        self.old = OperatorStream(sku)
        self.new = OperatorStream(sku, track_post_history=True)
        self.future = OperatorStream(FUTURE, track_post_history=True)

    def forget(self):
        for stream in (self.old, self.new, self.future):
            stream.forget_history()

    def call(self, shape, op=OP, attrs=None):
        old = self.old.advance(op, shape, attrs)
        t = perf_counter_ns()
        new = self.new.advance(op, shape, attrs)
        timings.append((perf_counter_ns()-t)/1e6)
        future = self.future.advance(op, shape, attrs)
        if op != OP:
            return
        diag = new.projection['timing_diagnostics']
        assert old.work == new.work and old.active_aiv == new.active_aiv
        return dict(chip=self.sku, shape=shape, old_us=old.latency_us, ordered_us=new.latency_us,
                    stateless_us=estimate(OP, shape, chip=self.sku).latency_us,
                    extrapolated_970_us=future.latency_us, target_plan=diag['target_plan'],
                    model=diag['model'], scenario=diag.get('ordered_scenario', diag['context']))


def add(case, time_us, kernel, trace, digest, role, metric, registry=None):
    digest = digest or sha256((ROOT / trace).read_bytes()).hexdigest()
    existing = rows.get(digest)
    if existing:
        assert existing['chip'] == case['chip'] and existing['shape'] == case['shape']
        assert isclose(existing['ca_us'], time_us, abs_tol=0.004), (existing, time_us)
        assert isclose(existing['ordered_us'], case['ordered_us'], abs_tol=1e-8)
        existing['roles'].append(role)
    else:
        rows[digest] = dict(**case, ca_us=time_us, kernel=kernel, trace=trace,
                            sha256=digest, roles=[role], metric=metric)
    if registry is not None:
        rows[digest]['registration'] = registry


def mixed_archive():
    src = read('results/v0.22/post-stream/replay.json')
    for chain in src['chains']:
        s = Dispatch(chain['chip'])
        for r in chain['stages']:
            case = s.call(r['shape'], r['op'], r['attrs'])
        assert r['op'] == OP and case['scenario'] == 'unknown' and case['old_us'] == case['ordered_us']
        add(case, r['ca_us'], r['kernel'], 'results/v0.22/post-stream/replay.json', r['trace_sha256'],
            'mixed Pre/Sinkhorn/Post', 'published full window')

    for sku in (DT, PR):
        j = read('results/post-candidate/paired-' + sku + '.json')
        s = Dispatch(sku)
        for r in j['rows']:
            v = r['stage']; case = s.call(j['shape'])
            add(case, v['ca_us'], v['kernel'], v['trace'], v['trace_sha256'],
                'archived paired cold/hot process', 'published full window')

    # Twenty isolated v0.21 launches (including the requalified v0.22 cases)
    # are additional to the older phase archive. Prove the cold prefix from
    # the host's entire launched-kernel roster.
    for r in read('results/v0.22/evaluation.json')['rows']:
        if r['op'] != OP or r['cohort'] != 'primary':
            continue
        receipt = read(r['source'])
        assert receipt['qualified']
        v = r['receipt']
        trace = v['trace'][0] if isinstance(v['trace'],list) else v['trace']
        digest = v['trace_sha256'][0] if isinstance(v['trace_sha256'],list) else v['trace_sha256']
        raw = ROOT / trace.split('/profile/')[0]
        log = (raw/'run.log').read_text()
        t,_,d = r['shape']
        assert re.findall(r'Start profiling on kernel: (\S+)',log) == [v['kernel']]
        assert f'case=mhc_post api=aclnnMhcPost T={t} N=4 D={d} dtype=BF16' in log
        case = Dispatch(r['chip']).call(r['shape'])
        add(case, r['ca_us'], v['kernel'], trace, digest, 'v0.21 isolated first launch', 'published full window')

    archive = read('results/post-candidate/phase-evidence.json')
    for (_, sku), group in groupby(sorted(archive, key=lambda r:(r['process'], r['chip'])),
                                  key=lambda r:(r['process'], r['chip'])):
        stream = Dispatch(sku); prior = -1
        def ordinal(row):
            match = re.search(r'/(\d+)_entry_kernel_\d+\.db$', row['trace'])
            return int(match[1]) if match else 0
        for r in sorted(group, key=ordinal):
            i = ordinal(r)
            if i != prior + 1 or r['state'] == 'after_pre_sinkhorn':
                stream.forget()
            elif i == 0 and r['prior_aiv'] != 0:
                stream.forget()
            case = stream.call(r['shape'])
            prior = i
            if case['scenario'] != 'unknown':
                assert r['active'] == case['target_plan']['active_aiv']
            add(case, r['us'], r['kernel'], r['trace'], r['sha256'], 'historical phase evidence',
                r['backend'])


def registered():
    registered = read('results/v0.27/post/registration.json')
    originals = {(r['chip'], r['group'], r['i']):r for r in registered['rows']}
    ordinary = read('results/v0.27/post/score.json')
    ordinary_traces = {r['sha256']:r for r in ordinary['measured']}
    repeat = read('results/v0.26/post/score.json')
    repeat_traces = {r['trace_sha256']:r for r in repeat['independent'] if r['eligible_repeat']}
    runs = {}
    for version in (24,25,26,28):
        for file in sorted((ROOT/f'results/v0.{version}/post/receipts').glob('*.json')):
            j = read(file)
            if not j.get('rows'):
                failed.append(dict(receipt=file.name, version=version, reason=j.get('error', j.get('scope','no complete window'))))
                continue
            sku = j['chip']; stream = Dispatch(sku)
            raw = ROOT / j.get('raw', 'runs/post' + str(version) + '/' + file.stem)
            log = (raw/'run.log').read_text()
            blocks = dict(re.findall(r'^BATCH_BEGIN (\d+)\s*$\n(.*?)^BATCH_END \1\s*$', log, re.M|re.S))
            for r in sorted(j['rows'], key=lambda a:a['i']):
                i = r['i']; shape = r['shape']; block = blocks[str(i)]
                assert f'case=mhc_post api=aclnnMhcPost T={shape[0]} N=4 D={shape[2]} dtype=BF16' in block
                assert re.findall(r'^correctness name=output count=(\d+) bad=(\d+)\b', block, re.M) == [(str(prod(shape)), '0')]
                name, = re.findall(r'Start profiling on kernel: (\S+)', block)
                case = stream.call(shape); plan = case['target_plan']
                assert name.endswith('_6' if plan['route'] == 'post_token' else '_5')
                key = (version, sku, j.get('group', file.stem), i)
                runs[key] = case
                if not r['timing_qualified']:
                    failed.append(dict(receipt=file.name, version=version, index=i,
                                       shape=shape, reason=r.get('reason', 'published timing gate rejected')))
                    continue
                stage = r['stage']; digest = stage['trace_sha256']
                assert stage['kernel'] == name and stage['active_aiv'] == plan['active_aiv']
                role = f'v0.{version} dispatch replay'
                if digest in ordinary_traces:
                    target = originals[(sku, j['group'], i)]
                    assert isclose(target['us'],case['ordered_us'],abs_tol=1e-8)
                    assert isclose(target['base'],case['old_us'],abs_tol=1e-8)
                    role = 'v0.27 pre-CA ordinary token'
                elif digest in repeat_traces:
                    assert isclose(repeat_traces[digest]['policy_us'],case['ordered_us'],abs_tol=1e-8)
                    role = 'v0.25/26 pre-CA adjacent token repeat'
                add(case, stage['ca_us'], name, stage['trace'], digest, role, 'official inclusive full window', dict(version=version, group=key[2], index=i))
    for version in (24,25,27):
        j = read(f'results/v0.{version}/post/physical-summary.json')
        for r in j['rows']:
            if not r['qualified']:
                continue
            ver = int(r['version'].split('.')[1]) if 'version' in r else version
            sku = r.get('sku',r.get('chip')); i = r.get('i', r.get('ordinal'))
            case = runs[(ver, sku, r['group'], i)]
            key = r['trace']
            physical[key] = dict(**case, trace=key, ca_us=r['ca_us'], effective_aiv=r['active'],
                                 metric='first entry to last physical retirement, synthetic labels excluded')
    assert all(x in rows for x in ordinary_traces.keys() | repeat_traces.keys())


def stats(rs):
    result = dict(windows=len(rs), shapes=len({tuple(r['shape']) for r in rs}))
    for k in ('ordered_us','old_us','stateless_us'):
        errors = [abs(r[k] / r['ca_us'] - 1) for r in rs]
        result[k] = dict(wape=100*sum(abs(r[k]-r['ca_us']) for r in rs)/sum(r['ca_us'] for r in rs),
                         maximum=100*max(errors))
    return result


def main():
    mixed_archive(); registered()
    src = read('results/v0.27/history-bounds.json')
    channel = read('results/v0.28/post/channel-score.json')
    groups = defaultdict(list)
    for r in rows.values():
        labels = ['all distinct archived windows'] + r['roles']
        if r['model'] == 'post_token_ordered':
            labels += ['selected ordinary token', 'selected '+r['scenario']]
        else:
            labels.append('not selected')
            assert r['ordered_us'] == r['old_us']
        if r['target_plan']['route'] == 'post_channel':
            labels.append('channel kernel')
        for key in labels:
            groups[r['chip'], key].append(r)
    for r in physical.values():
        groups[r['chip'], 'real-instruction-only window (separate metric)'].append(r)
    report = dict(scope='v0.29 integration replay. Inputs are shape/attrs, SKU, and host dispatch order. No CA observation supplies H. No v0.29 independent data.',
                  summary={s:{k:stats(v) for (sku,k),v in groups.items() if sku==s}for s in (DT,PR)},
                  timings_ms=dict(calls=len(timings),median=median(timings),p95=sorted(timings)[int(.95*(len(timings)-1))],maximum=max(timings)),
                  paired_shape_only_lower_bounds={s:src['cohorts']['mHC Post/'+s] for s in (DT, PR)},
                  pending_channel_gate=channel['qualifies_to_enable_on_ordered_channel_repeat'],
                  channel_repeat_counts={s:channel['metrics'][s]['all_repeat']['distinct_shapes'] for s in (DT,PR)},
                  source_sha256=source_hashes, rejected=failed, rows=list(rows.values()),
                  physical_rows=list(physical.values()))
    (OUT/'score.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
