"""Check an installed wheel from an isolated interpreter against a release freeze."""
import argparse
import json
import math
from pathlib import Path
import statistics
from time import perf_counter_ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path, help='directory containing rms/ and post/ freezes')
    args = parser.parse_args()
    start = perf_counter_ns()
    import operator_model
    from operator_model import estimate

    import_ms = (perf_counter_ns() - start) / 1e6
    if 'site-packages' not in Path(operator_model.__file__).parts:
        raise ValueError('Expected an installed wheel outside the checkout')
    if args.release.name != 'v' + operator_model.__version__.rsplit('.', 1)[0]:
        raise ValueError('The release directory does not match the installed version')
    results = {}
    for folder in ('rms', 'post'):
        cases = json.loads((args.release / folder / 'freeze.json').read_text())['cases']
        times = []
        for case in cases:
            attrs = case['attrs']
            for inputs, expected in ((attrs, case['prediction']),
                                     ({**attrs, 'resident_aiv': case['prior_aiv']}, case['sequence_prediction'])):
                for _ in range(25):
                    start = perf_counter_ns()
                    actual = estimate(case['op'], case['shape'], inputs)
                    times.append((perf_counter_ns() - start) / 1e6)
                    if (actual.status != 'estimated' or actual.active_aiv != expected['predicted_active_aiv']
                            or not math.isclose(actual.latency_us, expected['predicted_us'], rel_tol=1e-9)):
                        raise ValueError(f"Installed model disagrees with {case['id']}")
        if max(times) >= 100:
            raise ValueError(f'Query exceeded 100 ms for {folder}')
        results[folder] = dict(calls=len(times), first_ms=times[0], median_ms=statistics.median(times),
                               max_ms=max(times))
    print(json.dumps(dict(version=operator_model.__version__, import_ms=import_ms,
                          method='isolated installed wheel; 25 in-process calls for each shape with/without '
                                 'history; first includes lazy loading; device latency is checked against the freeze',
                          checks=results), indent=2))


if __name__ == '__main__':
    main()
