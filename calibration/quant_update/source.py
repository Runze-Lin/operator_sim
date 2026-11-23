"""Attach actual exported system inputs to the frozen continuous-quant geometries."""
import argparse
import csv
import gzip
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from calibration.__main__ import read, require, unseal
from calibration.msopprof import sha
from calibration.system_quant_shapes import captured_shapes
from operator_model import estimate
from operator_model.operators.quant.mxfp4 import contiguous_input


def nodes(export, frozen, catalog):
    cases = unseal(frozen, 'freeze')['cases']
    expected = {tuple(case['shape']) for case in cases}
    examples = {}
    with gzip.open(export, 'rt', encoding='utf-8-sig', newline='') as infile:
        for node in csv.DictReader(infile):
            if (node['raw_name'] != 'npu::npu_dynamic_mx_quant' or node['rank'] != '0'
                    or node['history_tokens'] != '131072'):
                continue
            tensor, = json.loads(node['inputs'])
            shape = tuple(tensor['logical_shape'])
            if shape not in expected or shape in examples or not contiguous_input(shape, tensor['stride']):
                continue
            example = {key: node[key] for key in ('case', 'parallelism', 'history_tokens', 'batch', 'stage',
                                                  'rank', 'node_id', 'module_path')}
            example.update(shape=list(shape), dtype=tensor['dtype'], input_stride=tensor['stride'],
                           storage_offset=tensor['storage_offset'], source='exported_system_simulation',
                           contiguous_call_count=catalog[shape][True])
            examples[shape] = example
            if expected == examples.keys():
                break
    require(expected == examples.keys(), 'A frozen geometry is absent from the actual contiguous exports')
    return dict(export_name=export.name, export_sha256=sha(export), examples=list(examples.values()))


def benchmark(catalog):
    shapes = sorted(shape for shape, counts in catalog.items() if counts[True])
    chips = ('Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT')
    for chip in chips:
        estimate('quant_mxfp4', [9, 128], chip=chip)
    samples = []
    for _ in range(3):
        for chip in chips:
            for shape in shapes:
                start = perf_counter_ns()
                estimate('quant_mxfp4', shape, chip=chip)
                samples.append(((perf_counter_ns()-start)/1e6, shape, chip))
    samples.sort()
    worst, shape, chip = samples[-1]
    return dict(num_shapes=len(shapes), passes=3, samples=len(samples),
                median_ms=median(s[0] for s in samples), p99_ms=samples[99*len(samples)//100][0],
                max_ms=worst, worst_shape=shape, worst_chip=chip,
                method='3 warmed public estimate calls per observed continuous geometry and SKU, no input result cache; single process')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('export', 'freeze', 'output'):
        cli.add_argument('--'+name, required=True, type=Path)
    cli.add_argument('--benchmark', type=Path, help='Optionally measure and write public API query costs')
    args = cli.parse_args()
    catalog = captured_shapes(args.export)
    args.output.write_text(json.dumps(nodes(args.export, read(args.freeze), catalog), indent=2) + '\n')
    if args.benchmark:
        args.benchmark.write_text(json.dumps(benchmark(catalog), indent=2) + '\n')
