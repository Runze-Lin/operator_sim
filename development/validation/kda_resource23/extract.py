"""Extract retired resources and VF subqueue counts from qualified development CA."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/kda-resource23'
SEED = ROOT / 'results/chunk-recovery/fla2691/v2/verified.json'
FULL = ROOT / 'results/kda-v23/ascend950dt-t64-h1.json'
FAMILIES = ('chunk_kda_fwd_prepare', 'chunk_fwd_h', 'chunk_kda_fwd_finalize')


def read(path):
    return json.loads(path.read_bytes())


def extract(path):
    receipt = read(path)
    mechanism_only = not receipt['qualified']
    if mechanism_only:
        assert receipt['exact_stage_elfs'] and receipt['full_windows'] and receipt['actual_sku_verified'] and not receipt['sync_errors']
    stages = {}
    for stage in receipt['stages']:
        if stage['family'] not in FAMILIES:
            continue
        path_trace = ROOT/stage['aggregate']
        raw = path_trace.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == stage['aggregate_sha256']
        first = stage['first_tick']
        events = json.loads(raw)['traceEvents']
        dma, vf, mm, rv = [], [], [], []
        for e in events:
            if e['ph'] != 'X':
                continue
            desc = e['args'].get('detail', '')
            if not desc.startswith('{'):
                continue
            d = json.loads(desc)
            if d.get('is_sync') or d.get('is_wait') or d.get('is_duration_excluded'):
                continue
            start = round(e['ts'] * 1650) - first
            end = round((e['ts'] + e['dur']) * 1650) - first
            r = {'begin':start, 'end':end, 'core':e['pid']}
            if e['name'] == 'VF':
                vf.append(dict(r, counts=Counter()))
            elif e['name'] == 'MMAD':
                mm.append(dict(r, mnk=[d[k] for k in ('m','n','k')]))
            elif all(k in d for k in ('src_mem','dst_mem','processed_bytes')):
                dma.append(dict(r, route=d['src_mem']+'->'+d['dst_mem'], size=d['processed_bytes']))
            elif e['tid'].startswith('RVEC'):
                rv.append((r, e['tid']))
        vf.sort(key=lambda v:v['begin'])
        unmatched = Counter()
        for r, lane in rv:
            # Some per-lane retirements cross the VF launch endpoint. Assign by issue.
            earlier = [v for v in vf if v['core']==r['core'] and v['begin']<=r['begin']]
            if earlier:
                earlier[-1]['counts'][lane] += 1
            else:
                unmatched[lane] += 1
        assert not unmatched, unmatched
        stages[stage['family']] = dict(family=stage['family'], kernel=stage['kernel'],
            text_sha256=stage['device_text_sha256'], aggregate=stage['aggregate'], sha256=stage['aggregate_sha256'],
            ca_window_ticks=[first,stage['last_tick']], actual_aic=stage['active_aic'], actual_aiv=stage['active_aiv'],
            mmads=sorted(mm,key=lambda r:r['begin']), vfs=vf, transfers=sorted(dma,key=lambda r:r['begin']))
    assert list(stages)==list(FAMILIES)
    return dict(receipt=str(path.relative_to(ROOT)), qualified=not mechanism_only,
                use="instruction counters only" if mechanism_only else "qualified development primitives", stages=stages)


if __name__ == '__main__':
    data={str(n):extract(p) for n,p in ((2,SEED),(64,FULL))}
    (OUT/'dt-primitive-services.json').write_text(json.dumps(data,indent=2)+'\n')
