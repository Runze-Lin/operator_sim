"""Compare full rank-0 inference tasks against their archived per-node Roofline ledgers."""
import argparse
import csv
import gzip
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
from time import perf_counter


def near(a, b):
    return isclose(a, b, rel_tol=1e-9, abs_tol=1e-7)


def cases(matrix):
    for model in ('v4-pro', 'v41-flash', 'kimi-k31'):
        for layout, label in (('TP8/EP8', ''), ('TP8/EP32', '-tp8-ep32'), ('DEP64', '-dp64-ep64')):
            for batch in (1, 16, 64):
                suffix = f'-b{batch}' if label or batch != 1 else ''
                yield matrix / (model + label + suffix), model, layout, batch


def capture(job, *, ledger=False):
    # Import in the worker so the summary and merge step need only the standard library.
    from sim_platform.api.inference import run_inference

    case, model, layout, batch, mode, scratch = job
    output = scratch / mode / case.name
    chain = ['roofline'] if mode == 'roofline' else ['operator_sim', 'roofline']
    before = perf_counter()
    result = run_inference(case / 'task.toml', overrides={
        'output.directory': str(output), 'exports.enabled': False, 'prediction.backends': chain,
    })
    wall = perf_counter() - before
    old_dir = case / 'results/l1'
    archive = json.loads((old_dir / 'inference.json').read_text())
    old = archive['statistics'][0]
    [stat] = result.statistics
    [memory] = result.static_memory
    if stat.local_batch_size != batch or not near(memory.peak_bytes / 1e9, old['memory']['peak_gb']):
        raise ValueError(f'Batch or peak memory changed for {case.name}')
    if stat.accepted_length != 7 or len(result.evaluations) != 2:
        raise ValueError(f'Unexpected decode schedule for {case.name}')
    assigned = defaultdict(lambda: [0, 0., 0.])
    status = Counter()
    digest = sha256()
    old_total = new_total = 0.
    paired_nodes = []
    for index, ((stage, _, repeats), evaluation) in enumerate(
        zip(stat.stage_durations_us, result.evaluations, strict=True)
    ):
        if repeats != 1 or stage != ('target', 'draft')[index]:
            raise ValueError(f'Unexpected stage for {case.name}')
        with (old_dir / f'{index:03d}_verify_{stage}.csv').open(newline='') as source:
            baseline = {row['id']: row for row in csv.DictReader(source) if row['record_type'] == 'operator'}
        graph = evaluation.graph
        costs = dict(evaluation.execution_costs)
        predictions = dict(evaluation.predictions)
        if len(baseline) != len(graph.nodes):
            raise ValueError(f'Node count changed for {case.name}/{stage}')
        # Native node IDs, all tensor occurrences, attributes, resources and edges.
        for item in (*graph.nodes, *graph.dependencies):
            digest.update(repr(item).encode())
            digest.update(b'\n')
        for node in graph.nodes:
            row = baseline[node.ref.node_id]
            if node.raw_name != row['raw_name'] or graph.graph_id != row['graph_id']:
                raise ValueError(f'Node identity changed at {node.ref}')
            value = costs[node.ref].duration_us
            previous = float(row['duration_us'])
            prediction = predictions.get(node.ref)
            state = prediction.status if prediction else 'skipped_' + node.category
            if prediction and state == 'success':
                state = prediction.backend_name
            status[state] += 1
            if ledger:
                paired_nodes.append((stage, node.ref.node_id, node.raw_name, state, value, previous))
            if state == 'operator_sim':
                counts = assigned[node.raw_name]
                counts[0] += 1
                counts[1] += previous
                counts[2] += value
            elif state != (row['backend_name'] if row['prediction_status'] == 'success' else row['prediction_status']) or not near(previous, value):
                raise ValueError(f'Unmatched fallback {case.name}/{node.ref} {state}: {value} vs {previous}')
            old_total += previous
            new_total += value
    # The selected decode tasks have a serial DES timeline. Fail on overlap before
    # interpreting sums of node deltas as contributions to a round's latency gap.
    if not near(old_total, old['duration_us']) or not near(new_total, stat.duration_us):
        raise ValueError(f'Serial attribution invalid for {case.name}')
    summary = {
        'case': case.name, 'model': model, 'layout': layout, 'batch': batch, 'peak_gb': memory.peak_bytes / 1e9,
        'accepted_length': stat.accepted_length, 'duration_us': stat.duration_us,
        'target_us': stat.stage_durations_us[0][1], 'draft_us': stat.stage_durations_us[1][1],
        'tpot_us': stat.tpot_us, 'archive_us': old['duration_us'], 'status': dict(status),
        'nodes': sum(status.values()), 'mapped': dict(assigned), 'wall_seconds': wall,
        'graph_sha256': digest.hexdigest(), 'task_sha256': sha256((case / 'task.toml').read_bytes()).hexdigest(),
    }
    (output / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    if ledger:
        with gzip.open(output / 'nodes.csv.gz', 'wt', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(('stage', 'id', 'raw', 'source', 'us', 'roofline_us'))
            writer.writerows(paired_nodes)
    return case.name, stat.duration_us


def compare(matrix, scratch, dest):
    rows = []
    per_op = []
    for case, model, layout, batch in cases(matrix):
        base, fine = [json.loads((scratch / mode / case.name / 'comparison.json').read_text())
                      for mode in ('roofline', 'hybrid')]
        for key in ('graph_sha256', 'task_sha256', 'nodes', 'peak_gb', 'accepted_length'):
            if base[key] != fine[key]:
                raise ValueError(f'Different {key} for {case.name}')
        if not near(base['archive_us'], base['duration_us']):
            raise ValueError(f'Roofline differs from archive for {case.name}')
        matches = fine['mapped']
        hit = sum(values[0] for values in matches.values())
        old_hit = sum(values[1] for values in matches.values())
        new_hit = sum(values[2] for values in matches.values())
        if not near(fine['duration_us'] - base['duration_us'], new_hit - old_hit):
            raise ValueError(f'Latency gap not accounted for in {case.name}')
        row = dict(case=case.name, model=model, layout=layout, local_batch=batch, peak_gb=fine['peak_gb'],
                   graph_sha256=fine['graph_sha256'], task_sha256=fine['task_sha256'], nodes=fine['nodes'],
                   hit_nodes=hit, fallback_priced_nodes=fine['status'].get('roofline', 0),
                   skipped_zero=fine['status'].get('skipped_zero', 0), miss=fine['status'].get('miss', 0),
                   placeholder=fine['status'].get('placeholder', 0), fail=fine['status'].get('fail', 0),
                   base_hit_us=old_hit, hybrid_hit_us=new_hit, fallback_us=base['duration_us'] - old_hit)
        for mode, summary in (('base', base), ('hybrid', fine)):
            for key in ('duration_us', 'target_us', 'draft_us', 'tpot_us', 'wall_seconds'):
                row[f'{mode}_{key}'] = summary[key]
        rows.append(row)
        for op, (n, old, new) in sorted(matches.items()):
            per_op.append(dict(case=case.name, raw_name=op, nodes=n, base_us=old, hybrid_us=new))
    dest.mkdir(parents=True, exist_ok=True)
    for filename, items in [('system-decode-compare.csv', rows), ('system-decode-operators.csv', per_op)]:
        with (dest / filename).open('w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=items[0].keys(), lineterminator='\n')
            writer.writeheader()
            writer.writerows(items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('matrix', type=Path, help='decode-950dt-128k directory with original task.toml and L1 outputs')
    parser.add_argument('scratch', type=Path, help='separate working directory for fresh runs')
    parser.add_argument('mode', choices=['roofline', 'hybrid', 'compare'])
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('reference'))
    args = parser.parse_args()
    matrix, scratch = args.matrix.resolve(), args.scratch.resolve()
    if scratch == matrix or matrix in scratch.parents or scratch in matrix.parents:
        parser.error('scratch and original matrix must use separate directories')
    if args.mode == 'compare':
        compare(matrix, scratch, args.output)
    else:
        jobs = [(*item, args.mode, scratch) for item in cases(matrix)]
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for case, duration in pool.map(capture, jobs):
                print(f'{case}  {duration:.3f} us', flush=True)


if __name__ == '__main__':
    main()
