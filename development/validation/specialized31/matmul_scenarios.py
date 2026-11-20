"""Join the released public Matmul predictions to qualified workload-specific windows."""
from collections import defaultdict
from math import isclose
from pathlib import Path
from statistics import median


def write_scenarios(archive, query, root, out, chips, scenarios, older_workloads):
    measured = defaultdict(list)
    for row in archive:
        if row['op'] == 'matmul' and row['attrs'] == {}:
            measured[tuple(row['shape']), row['chip']].append(row)
    dt, pr, future = chips
    tables = {'decode': [], 'prefill': []}
    counts = defaultdict(lambda: [0, 0, 0, 0])
    flash = defaultdict(dict)
    indexed = {tuple(item['shape']): dict(item, scenarios=list(item['scenarios'])) for item in scenarios}
    for older in older_workloads:
        item = indexed.setdefault(tuple(older['shape']), dict(shape=older['shape'], scenarios=[], targets={}))
        scenario = {k: older[k] for k in ('phase', 'model', 'node', 'tokens_per_request')}
        scenario['batch'] = older['requests']
        if scenario not in item['scenarios']:
            item['scenarios'].append(scenario)
    def order(item):
        first = item['scenarios'][0]
        return first['model'], first['node'], item['shape']
    for item in sorted(indexed.values(), key=order):
        shape = item['shape']
        scenarios_for_shape = item['scenarios']
        first = scenarios_for_shape[0]
        phase = first['phase']
        assert all(s['phase'] == phase and s['model'] == first['model'] and s['node'] == first['node']
                   and s['batch'] * s['tokens_per_request'] == shape[0] for s in scenarios_for_shape)
        cells = []
        seen = []
        for chip in (dt, pr):
            pred = query('matmul', shape, {}, chip).latency_us
            rows = measured[tuple(shape), chip]
            ca = median(r['ca_us'] for r in rows) if rows else None
            seen.append(bool(rows))
            if rows:
                assert all(isclose(r['current_us'], pred, abs_tol=1e-8) for r in rows)
                old_reference = item['targets'].get(chip, {}).get('ca')
                source = rows[0]['source']
                if len(rows) != 1 or not isinstance(source, str) or not (root / source).is_file():
                    source = 'results/v0.31/replay.json'
                if old_reference and isclose(old_reference['ca_us'], ca, abs_tol=1e-8):
                    source = old_reference['source']
                assert (root / source).is_file()
                cells.append(f'{pred:.3f} / [{ca:.3f}](../../../{source})')
                if phase == 'prefill' and first['node'] == 'indexer.wk':
                    flash[chip][shape[0]] = ca
            else:
                cells.append(f'{pred:.3f} / —')
        tally = counts[phase]
        for i, v in enumerate((True, *seen, all(seen))):
            tally[i] += int(v)
        label = '、'.join(f"{s['batch']}×{s['tokens_per_request']:,}" for s in scenarios_for_shape)
        pred_future = query('matmul', shape, {}, future).latency_us
        tables[phase].append(f"| {first['model']} · {first['node']} | {label} | {shape} | "
                             f"{cells[0]} | {cells[1]} | {pred_future:.3f} |")
    lines = ['# 三模型 Matmul 的真实宽度、batch 和长轴覆盖', '',
             '按单卡一次连续 BF16 ND 投影计时。输入矩阵为 A[M,K] 和 B[K,N]，输出为 C[M,N]。'
             '下列预测均来自 v0.31 公开 estimate()。CA 为完整计时窗，同几何有多个合格窗时取中位数。'
             '独立算子调用与服务侧的融合、补齐、量化或 chunk 配置可能不同。', '',
             '| 数据集 | 不同 [M,N,K] | DT 已测 | PR 已测 | 同形两端均已测 |',
             '|---|---:|---:|---:|---:|']
    for phase in tables:
        lines.append(f"| {phase} | " + ' | '.join(map(str, counts[phase])) + ' |')
    lines += ['', 'decode 的 M 等于本卡这一步实际提交的行数。B1 校验 8 个草稿加 1 个主模型位置，'
              '与 B9 普通 decode 都给 M=9，共用同一条设备证据。B9、11、16 同时校验 Q9 时对应 81、99、144 行。'
              'prefill 按每请求完整交付计算；有 chunk 时把 Q 替换为各请求本步提交的 token 数再求和。'
              '970DT 全为架构外推。表中 — 表示归档中缺完整 CA 设备时间。', '']
    header = ['| 模型 · 投影 | B×Q | [M,N,K] | 950DT 预测 / CA µs | 950PR 预测 / CA µs | 970DT 外推 µs |',
              '|---|---|---|---:|---:|---:|']
    for phase, rows in tables.items():
        lines.extend(['## ' + phase, '', *header, *rows, ''])
    lines.extend(['## 大 M 的斜率', '',
                  '同一 Flash indexer.wk 权重的两个相邻已测 M 之间，用 Δ设备时间 / Δ行数计算新增一行的平均成本。'
                  '下表的倍数来自端点实测。输出乘法量随 M 线性增长，启动、并行轮次和读写越过缓存容量会改变局部斜率。', '',
                  '| 芯片 | 相邻已测 M | 行数倍数 | 实测时间倍数 | 增量 ns/行 |',
                  '|---|---|---:|---:|---:|'])
    for chip in (dt, pr):
        ordered = sorted(flash[chip].items())
        for (m1, t1), (m2, t2) in zip(ordered, ordered[1:]):
            lines.append(f'| {chip[6:11]} | {m1:,} → {m2:,} | {m2/m1:.2f} | {t2/t1:.2f} | '
                         f'{1000*(t2-t1)/(m2-m1):.4f} |')
    lines.extend(['', '[模型 config、TP 及投影推导](../../v0.29/matmul/workloads.md) · '
                  '[同形对比与覆盖角色](report.md) · [B11 的流水和指令](device/example.md) · '
                  '[长 prefill 的容量模型与正式回执](../../v0.30/matmul/comparison.md)', ''])
    (Path(out) / 'matmul/scenarios.md').write_text('\n'.join(lines))
