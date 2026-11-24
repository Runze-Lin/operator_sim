"""Rebuild the Q32 single-wave receipt from the complete package and CA trace."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import median
import zipfile

from calibration.__main__ import require
from calibration.candidate_indexer.fused_qsli_report import WHEEL_KEY
from calibration.msopprof import CORE, sha
from calibration.mx_matmul.collect import MEMORY_CONFIG, core_windows, trace_events

SKU = 'Ascend950DT_9582'
FREQUENCY = 1650


def summarize(raw, wheel):
    freeze = json.loads(Path(__file__).with_name('qsli-1mi-wave-freeze.json').read_text())
    log = raw / (SKU + '.log')
    text = log.read_text()
    require(MEMORY_CONFIG[SKU] in text and 'All task success.' in text and 'Traceback' not in text,
            'Profiler / SKU qualification failed')
    fixture = json.loads(re.search(r'^FIXTURE (\{.*\})$', text, re.M)[1])
    golden = re.search(r'^WAVE_GOLDEN_PASS (\d+) ([0-9a-f]{64})$', text, re.M)
    require(golden and golden[1] == '16384' and golden[2] == fixture['golden_sha256'],
            'Expected the full 32-row independent CPU golden')
    metadata = fixture['metadata']
    require(set(metadata) == set(map(str, range(32))), 'Expected 32 LI rows and no FD')
    for i in range(32):
        last = [1, 0, 0] if i == 31 else [0, i+1, 0]
        require(metadata[str(i)][:7] == [1, 0, i, 0, *last], 'Unexpected row ownership')
    (profile,) = (raw / 'profile').glob('OPPROF_*')
    (kernel,) = re.findall(r'Start profiling on kernel: (\S+)', text)
    require('kernel_QuantSparseLightningIndexer' in kernel, 'Unexpected device operator')
    provider_event=json.loads((raw/'provider.json').read_text())
    require(provider_event['counts'] == {'installed_hit': 1} and provider_event['events'][0]['key'] == WHEEL_KEY,
            'Native installed AOT key mismatch')
    entry = f'ops/_native/operators/qsli/variants/{WHEEL_KEY}/provider.so'
    with zipfile.ZipFile(wheel) as archive:
        host = archive.read(entry)
        sidecar = json.loads(archive.read(entry+'.meta.json'))
    require(hashlib.sha256(host).hexdigest() == sidecar['binary_sha256'], 'Bad provider checksum')
    elf = (profile / 'dump/aicore_binary.o').read_bytes()
    require(host.find(elf) >= 0, 'Device dump must be present byte-for-byte in the provider')
    roster = CORE.findall(text)
    require(len(roster) == 96 and len(dict(roster)) == len(roster), 'Incomplete SKU profile')
    sim = profile / 'simulator'
    cores, hashes = core_windows(sim, {c: float(t) for c,t in roster})
    origin = min(c['first_tick'] for c in cores.values())
    end = max(c['retire_tick'] for c in cores.values())
    rows, transitions, issue_gaps = [], Counter(), []
    for i in range(32):
        shard = sim / f'core{i}.cubecore0' / 'trace.json'
        events = []
        for e in trace_events(shard):
            if e.get('ph') == 'X' and e.get('name') == 'MMAD_MX' and e['dur'] > 0:
                detail = json.loads(e['args']['detail'])
                require(tuple(detail[n] for n in ('m','n','k')) == (32,512,128) and
                        tuple(detail[n] for n in ('matrix_a_dtype','matrix_b_dtype','matrix_c_dtype')) == ('fp4_e2m1','fp4_e2m1','f32'),
                        'Unexpected precision or dot product size')
                issue = round(e['ts']*FREQUENCY)-origin
                retire = round((e['ts']+e['dur'])*FREQUENCY)-origin
                require(retire > issue, 'Invalid MMAD interval')
                events.append((issue,retire))
        events.sort()
        require(len(events) == 32, f'Expected 32 matrix tiles for row {i}')
        issue_gaps.extend(b[0]-a[0] for a,b in zip(events,events[1:]))
        # Count cores with a live observed MMAD in each half-open interval.
        # Adjacent or overlapping issues on one core occupy that core once.
        start, stop = events[0]
        for left, right in events[1:]:
            if left <= stop:
                stop = max(stop,right)
            else:
                transitions[start] += 1
                transitions[stop] -= 1
                start, stop = left, right
        transitions[start] += 1
        transitions[stop] -= 1
        rows.append(dict(metadata_query=i, cube=shard.parent.name, tiles=len(events),
                         first_issue_cycles=events[0][0], last_retire_cycles=events[-1][1]))
    live = peak = 0
    for tick, change in sorted(transitions.items()):
        live += change
        peak = max(live,peak)
    us = (end-origin)/FREQUENCY
    predicted = freeze['predictions'][SKU]['us']
    interval_freeze = json.loads(Path(__file__).with_name('qsli-interval-wave-prediction.json').read_text())
    require(interval_freeze['shape'] == freeze['shape'], 'Interval prediction did not cover this input')
    interval_predicted = interval_freeze['resource_price_us']
    return dict(sku=SKU, shape=freeze['shape'], attrs=freeze['attrs'], role='frozen Q32 wave; separate from Q1 samples',
                kernel=kernel, native_key=WHEEL_KEY, golden_positions=int(golden[1]), fixture=fixture,
                device_sha256=hashlib.sha256(elf).hexdigest(), provider_sha256=sidecar['binary_sha256'],
                elf_inside_provider_at=host.find(elf), device_us=us, prediction_us=predicted,
                signed_error_pct=100*(predicted/us-1), interval_prediction_us=interval_predicted,
                interval_prediction_signed_error_pct=100*(interval_predicted/us-1), elapsed_cycles=end-origin,
                launched_aic=32, launched_aiv=64, issuing_aic_cores=32, observed_row_tiles=1024,
                peak_simultaneous_mmad_cores=peak, median_mmad_issue_gap_cycles=median(issue_gaps),
                first_mmad_issue_cycles=min(r['first_issue_cycles'] for r in rows),
                last_mmad_retire_cycles=max(r['last_retire_cycles'] for r in rows),
                last_finishing_core=max(cores, key=lambda c:cores[c]['retire_tick']),
                trace_set_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
                wall_s=float((raw/(SKU+'.wall')).read_text().strip()), log_sha256=sha(log), rows=rows,
                host_profile_annotation_warning='HOST_RANGE_POP_ERROR_AFTER_ENQUEUE' in text)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw', type=Path)
    p.add_argument('--wheel', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    args.out.write_text(json.dumps(summarize(args.raw,args.wheel),indent=2)+'\n')
