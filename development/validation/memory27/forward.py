"""Register or score the read-service test on the already queued 156 MiB CA calls."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from read_windows import examine
from retention import OUT, ROOT, read

CACHE = ROOT / 'results/v0.25/cache'
REG = OUT / 'read-freeze.json'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def register():
    original = read(CACHE/'freeze.json')
    cases = []
    for j in original['jobs']:
        slug = j['id']
        if (CACHE/(slug+'-result.json')).exists() or (CACHE/(slug+'-prospective-traffic.json')).exists():
            raise ValueError('outcome already returned: '+slug)
        cases.append(dict(id=slug, shape=j['shape'], op=j['op'], sku=j['sku'],
                          read_only_exceeds_nominal=('950PR' in j['sku'])))
    files = [CACHE/'freeze.json', Path(__file__), Path(__file__).with_name('read_windows.py'),
             Path(__file__).with_name('retention.py')]
    result = dict(utc=datetime.now(timezone.utc).isoformat(), cases=cases,
                  diagnostic='Use conservative descriptor windows (floor(issue)-1, ceil(retire)+1 cycle); '
                  'test if peak read-only interval rate exceeds target nominal GB/s by >1%. '
                  'Prices, total-time forecasts, collection order and budgets remain those in the original contract.',
                  source_sha256={str(f.relative_to(ROOT)):digest(f) for f in files})
    with REG.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')


def score():
    registration = read(REG)
    for name, fingerprint in registration['source_sha256'].items():
        if digest(ROOT/name) != fingerprint:
            raise ValueError('source changed since freeze: '+name)
    outcomes = []
    for case in registration['cases']:
        traffic = CACHE / (case['id']+'-prospective-traffic.json')
        result = CACHE / (case['id']+'-result.json')
        if not result.exists():
            continue
        if not read(result)['qualified']:
            outcomes.append(dict(id=case['id'], qualified=False))
            continue
        item = examine(traffic)
        if (item['shape'], item['chip'], item['op']) != (case['shape'], case['sku'], case['op']):
            raise ValueError('receipt mismatch')
        above = item['peak_rate_interval']['rate_GBps'] > 1.01 * item['nominal_GBps']
        outcomes.append(dict(id=case['id'], qualified=True, matched=(above==case['read_only_exceeds_nominal']), result=item))
    (OUT/'read-holdout.json').write_text(json.dumps(dict(registration=digest(REG), rows=outcomes), indent=2)+'\n')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('action', choices=('register','score'))
    {'register':register, 'score':score}[args.parse_args().action]()
