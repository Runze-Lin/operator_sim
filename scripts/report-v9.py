"""Publish the current model, keeping old calibration data separate from fresh CA validation."""
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from operator_model.predictor import predict, predict_v8

OUT = ROOT / 'results/v0.9'
GROUPS = {'addmm': ('addmm',), 'mhc': ('mhc_pre', 'mhc'),
          'vector': ('rmsnorm', 'mhc_sinkhorn', 'mhc_post')}
CHANGED = {op for group in GROUPS.values() for op in group}


def read(path):
    return json.loads(path.read_text())


def metrics(rows, field='predicted_us'):
    if not rows:
        return None, None
    errors = [abs(r[field] - r['reference_us']) for r in rows]
    total = sum(r['reference_us'] for r in rows)
    return sum(errors) / total, max(e / r['reference_us'] for e, r in zip(errors, rows))


def percent(value):
    return 'N/A' if value is None else f'{value:.2%}'


def evaluate():
    old = read(ROOT / 'results/v0.8/evaluation.json')
    prior_shapes = {(r['op'], tuple(r['shape'])) for r in old['rows']}
    rows = [r for r in old['rows'] if r['op'] not in CHANGED]
    for group, ops in GROUPS.items():
        for r in read(OUT / group / 'rows.json'):
            assert r['op'] in ops
            if r['role'] == 'validation' and (r['op'], tuple(r['shape'])) in prior_shapes:
                raise ValueError(f"Reused development shape in validation: {r['op']} {r['shape']}")
            rows.append(r)
    for row in rows:
        op, shape, attrs = row['op'], row['shape'], row.get('attrs', {})
        duration = []
        for _ in range(5):
            start = time.perf_counter_ns()
            value = predict(op, shape, attrs)['predicted_us']
            duration.append((time.perf_counter_ns() - start) / 1e6)
        if not math.isclose(value, row['predicted_us'], rel_tol=1e-8):
            raise ValueError(f'Frozen prediction changed for {op} {shape}: {value}, {row["predicted_us"]}')
        row['query_max_ms'] = max(duration)
        assert row['query_max_ms'] < 1000 and row['reference_us'] > 0
        if op in CHANGED:
            previous = predict_v8(op, shape, attrs)['predicted_us']
            if row.get('previous_us') is not None and not math.isclose(row['previous_us'], previous, rel_tol=1e-8):
                raise ValueError(f'Previous prediction disagrees for {op} {shape}')
            row['previous_us'] = previous
    summaries = []
    for prior in old['operator_summary']:
        op = prior['op']
        selected = [r for r in rows if r['op'] == op]
        val = [r for r in selected if r['role'] == 'validation']
        dev = [r for r in selected if r['role'] == 'development']
        assert len(val) + len(dev) == len(selected)
        assert len({tuple(r['shape']) for r in val}) == len(val)
        assert len(val) >= 10 or op == 'kda_chunk'
        wape, worst = metrics(val)
        stat = dict(op=op, name=prior['name'], method=prior['method'], independent=len(val),
                    development=len(dev), wape=wape, max_ape=worst,
                    query_max_ms=max((r['query_max_ms'] for r in selected), default=None))
        if op in CHANGED:
            stat['same_set_v08_wape'], stat['same_set_v08_max_ape'] = metrics(val, 'previous_us')
        summaries.append(stat)
    return rows, summaries


def plot(summaries):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'build/mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax, matched) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1.3, 1]})
    qualified = [r for r in summaries if r['wape'] is not None]
    positions = range(len(qualified))
    ax.barh(positions, [100*r['wape'] for r in qualified], color='#377f77')
    for i, row in enumerate(qualified):
        ax.text(100*row['wape'], i, f"  {100*row['wape']:.1f}", va='center', fontsize=8)
    ax.set_yticks(list(positions), [r['op'] for r in qualified])
    ax.set_xlim(0, 1.2*max(100*r['wape'] for r in qualified))
    ax.set_title('Current model')
    changed = [r for r in summaries if r['op'] in CHANGED]
    for i, row in enumerate(changed):
        pair = [100*row['wape'], 100*row['same_set_v08_wape']]
        matched.plot(pair, [i, i], color='#b2bdc2', lw=2)
        matched.scatter(pair[0], i, color='#377f77', label='v0.9' if i == 0 else None, zorder=3)
        matched.scatter(pair[1], i, color='#a5482a', marker='D', label='v0.8' if i == 0 else None, zorder=3)
    matched.set_yticks(range(len(changed)), [r['op'] for r in changed])
    matched.set_title('Same fresh shapes')
    matched.legend(loc='best', fontsize=8)
    for axis in (ax, matched):
        axis.invert_yaxis()
        axis.set_xlabel('Independent WAPE (%)')
        axis.grid(axis='x', alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT/'validation.png', dpi=140)
    current = ROOT/'results/current'
    (current/'all-operators.png').write_bytes((OUT/'validation.png').read_bytes())
    svg = current/'all-operators.svg'
    fig.savefig(svg, metadata={'Date': None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def describe(row):
    attrs = row.get('attrs', {})
    settings = ''
    if 'num_iters' in attrs:
        settings += f", I={attrs['num_iters']}"
    if 'expert_counts' in attrs:
        settings += f", 专家行数={attrs['expert_counts']}"
    return f"`{row['shape']}`{settings}"


def report(rows, summaries):
    apis = {r['op']:r for r in read(ROOT/'results/v0.7/evaluation.json')['operator_summary']}
    lines = ['# 当前整套算子误差分析','',
             '| 算子 | 方法 | 开发 / 独立 shape | 独立 WAPE / 最大 | 最慢查询 ms |',
             '|---|---|---:|---:|---:|']
    for s in summaries:
        duration = 'N/A' if s['query_max_ms'] is None else f"{s['query_max_ms']:.3f}"
        lines.append(f"| {s['name']} | {s['method']} | {s['development']} / {s['independent']} | "
                     f"{percent(s['wape'])} / {percent(s['max_ape'])} | {duration} |")
    lines.extend(['','WAPE 为独立样本绝对耗时偏差之和除以其 CA 耗时之和。最大误差取最差单例的绝对百分比。六条改动路径的 v0.8 样本全部按开发回放，独立列只使用冻结后新测的 shape。其余路径复用原冻结模型及其独立样本。KDA chunk 的 10 个结构检查另计，完整 CA 耗时仍为 0 个。',
                  '', '图中菱形表示在本轮同一批新 shape 上回放旧模型的结果。柱形表示当前默认模型。',
                  '', '![逐算子模型误差](../results/v0.9/validation.png)', '',
                  '## 测量口径','',
                  '参考为 CANN 9.2.1 安装包在 Ascend950DT_9582 CA 上的跨核指令窗口。串行链相加各实际 kernel 的净窗口；主机 launch、两次调用之间的空闲和卡间通信由系统层另计。批采逐次同步并检查数值，后续调用可能保留热状态。拆分 mHC 的 Pre 与整链共享采集，MoE 两组件与整链同理。每个 shape 对默认 Python 入口调用五次，结果不缓存，表中取所有调用的最大墙钟；首次使用参数时的读取计入。', '',
                  '## 逐算子、逐 shape 的误差',''])
    for s in summaries:
        op = s['op']
        api = apis[op]
        lines.extend([f"### {s['name']}",'', f"安装包接口为 {api['library']} 的 `{api['api']}`。",''])
        selected = [r for r in rows if r['op'] == op]
        val = [r for r in selected if r['role'] == 'validation']
        if not val:
            lines.extend(['普通整块和双头 safe gate 路径都在主 kernel 触发重复 set_flag。辅助 kernel 单独完成不足以给出整条调用时间。10 个冻结 shape 的逻辑工作量、依赖和稀疏 CPU 前向已核对。[结构验证](../results/v0.8/cube/chunk-structure-validation.json)',''])
            continue
        worst = max(val,key=lambda r:abs(r['predicted_us']/r['reference_us']-1))
        lines.extend([f"最大偏差出现在 {describe(worst)}，预测 {worst['predicted_us']:.3f} µs，CA 为 {worst['reference_us']:.3f} µs。",'',
                      '| shape 与属性 | 用途 | 预测 µs | CA µs | 有符号误差 | 实际 kernel | 活跃 AIC / AIV |',
                      '|---|---|---:|---:|---:|---|---|'])
        for r in sorted(selected, key=lambda item: item['role'] != 'validation'):
            role = '独立' if r['role']=='validation' else '开发'
            lines.append(f"| {describe(r)} | {role} | {r['predicted_us']:.3f} | {r['reference_us']:.3f} | "
                         f"{r['predicted_us']/r['reference_us']-1:+.1%} | {r['kernel']} | {r['active_aic']} / {r['active_aiv']} |")
        lines.append('')
    lines.extend(['## 误差来源与成本','', (OUT/'diagnosis.md').read_text().strip().replace('](vector/', '](../results/v0.9/vector/'), '',
                  '所有可计时样例的预测、正确性和数据来源保存在[逐例 JSON](../results/v0.9/evaluation.json)。模型版本变化、同集对照及离线成本见[迭代记录](../results/v0.9/changes.md)。',''])
    text='\n'.join(lines)
    (ROOT/'docs/error-analysis.md').write_text(text)
    (OUT/'error-analysis.md').write_text(text.replace('../results/v0.9/','./').replace('../results/v0.8/','../v0.8/'))


if __name__ == '__main__':
    rows, summary = evaluate()
    plot(summary)
    report(rows, summary)
    costs = {group: read(OUT/group/'costs.json') for group in GROUPS}
    payload = json.dumps(dict(version='v0.9', operator_summary=summary, rows=rows, collection_cost=costs),
                         ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    (OUT/'evaluation.json').write_text(payload)
    (ROOT/'results/current/evaluation.json').write_text(payload)
