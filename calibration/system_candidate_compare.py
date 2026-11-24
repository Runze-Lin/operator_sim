"""Rank actual fallback nodes and price the same 27 captured tasks with three backends."""
import argparse
import csv
import gzip
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import zip_longest
from pathlib import Path

from calibration.system_compare import capture, cases, near


CANDIDATE = 'inference::candidate_filtered_indexer'
REFERENCE = Path(__file__).parent / 'reference'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_table(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def summary(scratch, run, name):
    return json.loads((scratch / run / name / 'comparison.json').read_text())


def ledger(scratch, run, name):
    with gzip.open(scratch / run / name / 'nodes.csv.gz', 'rt', newline='') as stream:
        yield from csv.DictReader(stream)


def baseline(old, stored):
    for key in ('graph_sha256', 'task_sha256'):
        require(old[key] == stored[key], f"v1.7 {key} changed for {old['case']}")
    for key in ('duration_us', 'target_us', 'draft_us'):
        require(near(old[key], float(stored['hybrid_' + key])), f"v1.7 {key} changed for {old['case']}")
    require(near(old['archive_us'], float(stored['base_duration_us'])), 'Roofline time changed')
    require(old['status']['operator_sim'] == int(stored['hit_nodes']), 'v1.7 hit count changed')


def check_fresh_roofline(scratch, name, stored):
    pure = summary(scratch, 'roofline', name)
    require(not pure['status'].get('operator_sim'), 'Unexpected operator-sim hit in pure Roofline')
    for key in ('graph_sha256', 'task_sha256'):
        require(pure[key] == stored[key], f'Pure Roofline {key} changed for {name}')
    for key in ('duration_us', 'target_us', 'draft_us'):
        require(near(pure[key], float(stored['base_' + key])), f'Pure Roofline {key} changed for {name}')
    require(pure['nodes'] == int(stored['nodes']) and near(pure['peak_gb'],float(stored['peak_gb'])),
            f'Pure Roofline graph size or memory changed for {name}')
    return pure['wall_seconds']


def rank_fallback(matrix, scratch, stored, output):
    counts, costs = Counter(), Counter()
    for directory, *_ in cases(matrix):
        old = summary(scratch, 'v17', directory.name)
        base = stored[directory.name]
        baseline(old, base)
        rows = [r for r in ledger(scratch, 'v17', directory.name) if r['source'] == 'roofline']
        require(len(rows) == int(base['fallback_priced_nodes']), 'Fallback count changed')
        require(near(sum(float(r['us']) for r in rows), float(base['fallback_us'])), 'Fallback time changed')
        for row in rows:
            counts[row['raw']] += 1
            costs[row['raw']] += float(row['us'])
    total = sum(costs.values())
    roof_total = sum(float(s['base_duration_us']) for s in stored.values())
    rows = [dict(rank=i, raw_name=raw, nodes=counts[raw], fallback_us=us,
                 pct_of_fallback=100*us/total, pct_of_all_roofline=100*us/roof_total)
            for i, (raw, us) in enumerate(costs.most_common(), 1)]
    write_table(output / 'system-v17-fallback.csv', rows)


def candidate_first_sensitivity(directory, scratch):
    """Reprice only the candidate consumers in a verified serial DES round."""
    from operator_model import estimate
    from operator_model.sim_backend import OperatorSimBackend
    from sim_platform.hardware.loading import load_hardware
    import tomllib

    with (directory / 'task.toml').open('rb') as stream:
        hardware_path = tomllib.load(stream)['platform']['hardware']
    hardware = load_hardware(directory / hardware_path).to_model_hardware()
    chip = OperatorSimBackend()._resolve_chip(hardware.name, hardware.chip)
    billed = {(r['stage'], r['id']): float(r['us']) for r in ledger(scratch, 'candidate', directory.name)
              if r['raw'] == CANDIDATE}
    costs = []
    for stage in ('target', 'draft'):
        with gzip.open(directory / 'native' / f'000_rank0_{stage}.json.gz', 'rt') as stream:
            graph = json.load(stream)
        for node in graph['nodes']:
            if node['name'] != CANDIDATE:
                continue
            q, keys, weights, positions, blocks = [graph['tensors'][idx]['shape'] for idx in node['inputs']]
            output = graph['tensors'][node['outputs'][0]]['shape']
            rows, heads, dim = q
            batch, length, key_dim = keys
            require(rows % batch == 0 and key_dim == dim and weights == [rows, heads]
                    and positions == [rows] and output[0] == blocks[0] == rows, 'Bad candidate geometry')
            shape = (batch, rows // batch, heads, dim, length, blocks[-1], output[-1])
            attrs = {'implementation': 'li_pipeline_projection', 'block_size': 8}
            full = estimate('candidate_filtered_indexer', shape, attrs, chip=chip, dtype='bf16')
            compact = estimate('candidate_filtered_indexer', shape, dict(attrs, score_policy='candidate_first'),
                               chip=chip, dtype='bf16')
            require(full.status == compact.status == 'extrapolated', 'Candidate evidence class changed')
            require(full.latency_us and compact.latency_us and
                    near(full.latency_us, billed.pop((stage, str(node['id'])))), 'Scenario differs from DES route')
            costs.append(compact.latency_us)
    require(len(costs) == 6 and not billed, 'Unmatched candidate consumer')
    return sum(costs)


def compare_case(scratch, name, base):
    old = summary(scratch, 'v17', name)
    new = summary(scratch, 'candidate', name)
    baseline(old, base)
    for k in ('graph_sha256', 'task_sha256', 'nodes', 'peak_gb', 'batch', 'accepted_length'):
        require(old[k] == new[k], f'Changed {k}: {name}')
    for k in ('miss', 'fail', 'skipped_zero', 'placeholder'):
        require(old['status'].get(k, 0) == new['status'].get(k, 0), f'Changed status {k}: {name}')
    candidate = 0
    old_stage_us, new_stage_us, stage_counts = Counter(), Counter(), Counter()
    for a, b in zip_longest(ledger(scratch, 'v17', name), ledger(scratch, 'candidate', name)):
        require(a and b, f'Node count changed: {name}')
        for key in ('stage', 'id', 'raw', 'roofline_us'):
            require(a[key] == b[key], f'Changed {key}: {name}')
        if a['raw'] == CANDIDATE:
            require(a['source'] == 'roofline' and b['source'] == 'operator_sim',
                    f"Candidate not priced: {name}/{b['stage']}/{b['id']}")
            candidate += 1
            stage_counts[a['stage']] += 1
            old_stage_us[a['stage']] += float(a['us'])
            new_stage_us[b['stage']] += float(b['us'])
        else:
            require(a['source'] == b['source'] and near(float(a['us']), float(b['us'])),
                    f"Unexpected node change: {name}/{a['stage']}/{a['id']} {a['raw']}")
    counts = {'target': 4, 'draft': 2} if old['model'] == 'v41-flash' else {}
    require(stage_counts == counts, 'Unexpected target/draft candidate count')
    for stage in ('target', 'draft'):
        expected = old[stage + '_us'] - old_stage_us[stage] + new_stage_us[stage]
        require(near(expected, new[stage + '_us']), f'Unexplained {stage} DES change: {name}')
    prev_us, next_us = sum(old_stage_us.values()), sum(new_stage_us.values())
    net = new['duration_us'] - old['duration_us']
    require(near(old['duration_us'] - prev_us + next_us, new['duration_us']), 'Unexplained DES gap')
    require(new['status']['operator_sim'] == old['status']['operator_sim'] + candidate, 'Unexpected coverage')
    roof_us = float(base['base_duration_us'])
    before = float(base['base_hit_us'])
    return dict(case=name, layout=base['layout'], local_batch=int(base['local_batch']), nodes=old['nodes'],
                peak_gb=new['peak_gb'], graph_sha256=old['graph_sha256'], task_sha256=old['task_sha256'],
                roofline_us=roof_us, hybrid17_us=old['duration_us'], hybrid_new_us=new['duration_us'],
                hybrid_new_target_us=new['target_us'], hybrid_new_draft_us=new['draft_us'],
                tpot_us=new['tpot_us'], candidate_nodes=candidate, candidate_old_us=prev_us,
                candidate_new_us=next_us, candidate_new_target_us=new_stage_us['target'],
                candidate_new_draft_us=new_stage_us['draft'], other_node_change_us=net - (next_us - prev_us),
                old_hit_nodes=int(base['hit_nodes']),
                new_hit_nodes=new['status']['operator_sim'], old_hit_budget_us=before,
                new_hit_budget_us=before+prev_us, fallback_nodes=new['status']['roofline'],
                fallback_us=float(base['fallback_us'])-prev_us, old_wall_s=old['wall_seconds'],
                new_wall_s=new['wall_seconds'])


def comparison_section(rows):
    """Keep the report's coverage and two matched time comparisons in sync with the ledger."""
    def totals(selected):
        budget = sum(row['roofline_us'] for row in selected)
        return [100 * sum(row[field] for row in selected) / budget
                for field in ('old_hit_budget_us', 'new_hit_budget_us')]

    flash = [row for row in rows if row['case'].startswith('v41-flash')]
    old, new = totals(rows)
    old_flash, new_flash = totals(flash)
    candidate_us = sum(row['candidate_new_us'] for row in flash)
    old_us = sum(row['candidate_old_us'] for row in flash)
    added = sum(row['candidate_nodes'] for row in rows)
    priced = sum(row['new_hit_nodes'] + row['fallback_nodes'] for row in rows)
    total_hit = sum(row['new_hit_nodes'] for row in rows)
    numerators_ms = [sum(row[field] for row in rows)/1000 for field in ('old_hit_budget_us','new_hit_budget_us')]
    denominator_ms = sum(row['roofline_us'] for row in rows)/1000
    families = [('V4 Pro', 'v4-pro'), ('V4.1 Flash', 'v41-flash'), ('Kimi K3.1 Pro', 'kimi-k31')]
    lines = [
        '## 三方同图结果',
        '',
        f'按原 Roofline 预算加权，27 组整体覆盖率从 **{old:.2f}% 升至 {new:.2f}%**，'
        f'其中 Flash 九组从 **{old_flash:.2f}% 升至 {new_flash:.2f}%**。'
        f'在同一个 {denominator_ms:,.3f} ms 分母上，接管的预算由 {numerators_ms[0]:,.3f} ms 增至 {numerators_ms[1]:,.3f} ms。'
        f'新增 {added} 个外推节点，新后端共接管 {total_hit:,}/{priced:,} 个成功定价节点。'
        f'新增的 {new-old:.2f} 个百分点全部来自候选 Indexer 的融合排程外推。新接管的节点中，与该融合实现和系统形状匹配的完整 CA 时间窗为 **0/{added}**。'
        '950DT/PR 上已有的小规模分解路径实验及具体形状见[算例](examples/candidate-filtered-indexer.md)。',
        '', '每个模型各有九个配置。下表时间为这些独立配置各执行一轮的总和。', '',
        '| 模型 | 原 Roofline ms | v1.7 混合 ms | 新混合 ms | 按原耗时加权的接管比例 v1.7 → 新版 |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for label, prefix in families + [('全部', '')]:
        group = [r for r in rows if r['case'].startswith(prefix)]
        times = [sum(r[key] for r in group)/1000 for key in ('roofline_us','hybrid17_us','hybrid_new_us')]
        c0, c1 = totals(group)
        lines.append(f'| {label} | {times[0]:,.3f} | {times[1]:,.3f} | {times[2]:,.3f} | {c0:.2f}% → {c1:.2f}% |')
    lines.extend(['',
        f'这 54 次调用在九个不同配置上的旧预算合计 **{old_us/1e3:,.3f} ms**，'
        f'新版合计 **{candidate_us/1e3:,.3f} ms**。单独运行任一配置请读下表的一行。'
        '下表均为单轮毫秒，差值百分比以相应旧列为分母。不同布局捕获的单卡 Indexer 几何相同，'
        '同一 B 的候选 Indexer 价格在三种布局中一致，其余算子使得整轮差值百分比各异。',
        '',
        '| Flash 布局 | B | 原 Roofline ms | v1.7 混合 ms | 新混合 ms | 对 v1.7 | 对 Roofline | 六次 Indexer 旧 → 新 ms |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    for row in flash:
        roof, previous, current = [row[k]/1000 for k in ('roofline_us', 'hybrid17_us', 'hybrid_new_us')]
        lines.append(f"| {row['layout']} | {row['local_batch']} | {roof:.3f} | {previous:.3f} "
                     f"| {current:.3f} | {100*(current/previous-1):+.2f}% | {100*(current/roof-1):+.2f}% "
                     f"| {row['candidate_old_us']/1000:.3f} → {row['candidate_new_us']/1000:.3f} |")
    lines.extend(['',
        f"以 TP8/EP8、B=1 而言，四次 target 的新版单次预测均为 {flash[0]['candidate_new_target_us']/4:.3f} µs，"
        f"两次 draft 均为 {flash[0]['candidate_new_draft_us']/2:.3f} µs。"
        '旧回退把点积全部记在 Vector。新版假设 BF16 点积在 Cube 上执行，由 Vector 归约、筛选并产生排序索引，'
        '两者读取同一张完整 key 表。这一排程变化解释了方向。是否能实现预测差距还需要匹配部署实现的整窗计时。',
        '',
        'Pro 与 Kimi 共 18 个对照任务没有这条候选节点，逐节点比较和两阶段 DES 均复现 v1.7。'
        '27 组静态峰值逐组相同，节点定价失败数为零。各次执行仍保留原有缺价和占位。'
        '[全部 27 组时延、覆盖分子及分母](../calibration/reference/system-candidate-comparison.csv)',
        '',
        '### 先缩小 key 集合的敏感性',
        '',
        '另用 54 次调用各自的本卡输入计算 `candidate_first`。假定候选为不同的块，最新的可见块固定入选，'
        '尾块按实际可见 key 数裁剪。该排程需承担候选 gather 和索引重映射。对每个节点先用同一芯片配置复算全表价格并与正式 DES 核对，'
        '再用候选优先时间替换。前述 27 个系统图已验证每个阶段的价格可串行相加，因此下表通过替换六个节点计算单轮敏感性，'
        '其余节点、依赖与通信价格固定。它描述另一种部署假设。以上主表才是重新捕获且执行 DES 的参考顺序。',
        '',
        '| Flash 布局 | B | 全表打分单轮 ms | 候选优先单轮 ms | 六次候选优先 Indexer 合计 ms |',
        '| --- | ---: | ---: | ---: | ---: |',
    ])
    for r in flash:
        lines.append(f"| {r['layout']} | {r['local_batch']} | {r['hybrid_new_us']/1000:.3f} | "
                     f"{r['candidate_first_round_scenario_us']/1000:.3f} | {r['candidate_first_only_us']/1000:.3f} |")
    return '\n'.join(lines)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('matrix', type=Path)
    cli.add_argument('scratch', type=Path)
    cli.add_argument('mode', choices=('v17', 'candidate', 'rank', 'compare'))
    cli.add_argument('--output', type=Path, default=REFERENCE)
    cli.add_argument('--jobs', type=int, default=4)
    cli.add_argument('--fresh-roofline', action='store_true',
                     help='also verify a fresh pure Roofline run from calibration.system_compare in scratch/roofline')
    args = cli.parse_args()
    matrix, scratch = args.matrix.resolve(), args.scratch.resolve()
    if matrix == scratch or matrix in scratch.parents or scratch in matrix.parents:
        cli.error('scratch must be separate from the original captures')
    if args.mode in ('rank', 'compare'):
        args.output.mkdir(parents=True, exist_ok=True)
        with (REFERENCE / 'system-decode-compare.csv').open() as source:
            stored = {r['case']: r for r in csv.DictReader(source)}
        if args.mode == 'rank':
            rank_fallback(matrix, scratch, stored, args.output)
        else:
            rows = []
            for case, *_ in cases(matrix):
                row = compare_case(scratch, case.name, stored[case.name])
                if args.fresh_roofline:
                    row['fresh_roofline_wall_s'] = check_fresh_roofline(scratch, case.name, stored[case.name])
                sensitivity = candidate_first_sensitivity(case, scratch) if row['candidate_nodes'] else 0.
                row['candidate_first_only_us'] = sensitivity
                row['candidate_first_round_scenario_us'] = row['hybrid_new_us'] - row['candidate_new_us'] + sensitivity
                rows.append(row)
            write_table(args.output / 'system-candidate-comparison.csv', rows)
            print(comparison_section(rows))
    else:
        if args.mode == 'v17':
            import operator_model
            require(operator_model.__version__ == '1.7.0', 'v17 requires the frozen v1.7.0 model')
        jobs = [(*item, args.mode, scratch) for item in cases(matrix)]
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for name, duration in pool.map(partial(capture, ledger=True), jobs):
                print(f'{name} {duration:.3f} us', flush=True)


if __name__ == '__main__':
    main()
