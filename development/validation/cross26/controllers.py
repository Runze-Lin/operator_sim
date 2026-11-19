"""Preserve native memory counters from the existing Pro queue without repeating CA."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PRO = ROOT / 'results/v0.23/pro-prefill'
OUT = ROOT / 'results/v0.26/memory'
SNAP = ROOT / 'runs/memory26-controller-snapshots'
sys.path.insert(0, str(ROOT / 'development/validation/large20'))
from dram import counters


def observe(job):
    case = job['id']
    folder = PRO / 'raw' / case
    native = folder / 'log'
    kept = SNAP / case / 'log'
    for item in native.glob('*.log'):
        if not any(term in item.name.lower() for term in ('dmc', 'ddr', 'state', 'cache', 'memory', 'stat')):
            continue
        kept.mkdir(parents=True, exist_ok=True)
        try:
            os.link(item, kept / item.name)
        except (FileExistsError, FileNotFoundError):
            pass
    receipt = PRO / (case + '.json')
    if not receipt.exists():
        return False
    dest = OUT / (case + '-controller.json')
    if dest.exists():
        return True
    measured = json.loads(receipt.read_text())
    names = sorted(p.name for p in kept.glob('*.log'))
    totals = counters(kept.parent) if case.endswith('-dt') and measured.get('qualified') else None
    payload = dict(shape=job['shape'], sku=job['sku'], qualified_full_CA=measured.get('qualified'),
                   observed_files=names, native_controller_totals=totals,
                   origin='Native process memory-counter periods. A missing complete channel set is unknown.',
                   receipt=str(receipt.relative_to(ROOT)))
    tmp = dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2)+'\n')
    tmp.replace(dest)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    jobs = [j for j in json.loads((PRO/'freeze.json').read_text())['jobs'] if '256ki' in j['id']]
    while True:
        ended = [observe(j) for j in jobs]
        if all(ended) or not args.watch:
            break
        time.sleep(15)


if __name__ == '__main__':
    main()
