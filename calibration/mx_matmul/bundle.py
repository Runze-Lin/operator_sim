"""Make a portable, validated two-version reference from a finished MX campaign."""
import argparse
import gzip
from pathlib import Path

from calibration.__main__ import encoded, read, require
from calibration.mx_matmul.compare import compare, version_pair


def bundle(geometries, budget, old_timing, old_plans, new_timing, new_plans, raw):
    archive = {'geometries': read(geometries), 'budget': read(budget), 'observations': []}
    for timing, schedule in ((old_timing, old_plans), (new_timing, new_plans)):
        freeze = read(timing)
        version = 'v' + '.'.join(freeze['model_version'].split('.')[:2])
        archive[version] = {'freeze': freeze, 'schedules': read(schedule)}
    require(version_pair(archive)[-1] == version, 'Supply the older timing freeze first')

    for case in archive[version]['freeze']['cases']:
        ident = case['id']
        require(Path(ident).name == ident and ident not in ('', '.', '..'), 'Invalid case directory')
        folder = raw / ident
        cost = read(folder / 'cost.json')
        receipt = read(folder / 'receipt.json')
        require(cost['exit_code'] == 0 and cost['stop_reason'] is None and cost['wall_seconds'] > 0
                and receipt['case']['id'] == ident and receipt['started_utc'] == cost['started_utc']
                and receipt['finished_utc'] == cost['finished_utc'], f'Acquisition cost mismatch: {ident}')
        archive['observations'].append({'receipt': receipt, 'ca_wall_s': cost['wall_seconds'],
                                        'phases': read(folder / 'phases.json')})
    compare(archive)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('geometries', 'budget', 'old-timing', 'old-plans', 'new-timing', 'new-plans', 'raw', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    archive = bundle(args.geometries, args.budget, args.old_timing, args.old_plans,
                     args.new_timing, args.new_plans, args.raw)
    payload = gzip.compress(encoded(archive), mtime=0)
    with args.output.open('xb') as out:
        out.write(payload)


if __name__ == '__main__':
    main()
