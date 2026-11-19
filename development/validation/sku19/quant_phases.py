"""Extract first payload/VF/write clocks, independent of kernel wall time."""
import json
from pathlib import Path
from hashlib import sha256
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'development/validation/sku17'))
from campaign_collect import events


def observations(path):
    groups = {}
    for e in events(path):
        if e.get('ph') != 'X':
            continue
        record = groups.setdefault(e['pid'], {'read': [], 'write': [], 'simd': None})
        ts = e['ts']
        record['start'] = min(record.get('start', ts), ts)
        if e['name'] == 'MOV_SRC_TO_DST_ALIGNv2':
            d = json.loads(e['args']['detail'])
            dest = 'read' if d['instr_type'] == 'MTE2' else 'write'
            record[dest].append([ts, ts+e['dur'], d['processed_bytes']])
        if e['tid'].startswith('RVEC'):
            record['simd'] = min(record['simd'] or ts, ts)
    origin = min(c['start'] for c in groups.values())
    for c in groups.values():
        for queue in ['read', 'write']:
            c[queue].sort()
            for e in c[queue]:
                e[:2] = [round((t-origin)*1650) for t in e[:2]]
        c['start'] = round((c['start']-origin)*1650)
        c['simd'] = round((c['simd']-origin)*1650)
    return groups


if __name__ == '__main__':
    data = []
    for ver in ('v0.18','v0.17'):
        rows = json.loads((ROOT/'results'/ver/'sku/pairs.json').read_text())
        for r in rows:
            if r['op'] != 'quant':
                continue
            for side in ['dt','pr']:
                trace = r[side]['trace'][0]
                if not trace.endswith('.json'):
                    continue
                digest = sha256()
                with (ROOT/trace).open('rb') as source:
                    for block in iter(lambda: source.read(2**20), b''):
                        digest.update(block)
                assert digest.hexdigest() == r[side]['trace_sha256'][0], trace
                cores = observations(ROOT/trace)
                data.append(dict(ver=ver, shape=r['shape'], side=side, trace=trace,
                                 sha256=r[side]['trace_sha256'][0], cores=cores))
                print(ver,r['shape'],side,len(cores),flush=True)
    (ROOT/'results/v0.19/quant-dma-samples.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
