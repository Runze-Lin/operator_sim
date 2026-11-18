"""Publish the current CA validation report. Run from a source checkout."""
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from operator_model.backend import estimate
from operator_model.predictor import predict_v10

OUT = ROOT / 'results/v0.11'
PREVIOUS = ROOT / 'results/v0.10'
CAMPAIGNS = {'mhc_post': 'post', 'rmsnorm': 'rms'}


def load(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def errors(rows, field='predicted_us'):
    if not rows:
        return None, None
    deviation = [abs(row[field] - row['reference_us']) for row in rows]
    return (sum(deviation) / sum(row['reference_us'] for row in rows),
            max(d / row['reference_us'] for d, row in zip(deviation, rows)))


def pct(value):
    return f'{value:.2%}' if value is not None else 'N/A'


def split_groups(rows):
    small = [r for r in rows if r['shape'][0] <= 4 and r['shape'][-1] >= 2048]
    other = [r for r in rows if r['shape'][0] > 4 or r['shape'][-1] < 2048]
    return [('all', rows), ('small_wide', small), ('ordinary', other)]


def collect():
    previous = load(PREVIOUS / 'evaluation.json')
    fresh = []
    for op, folder in CAMPAIGNS.items():
        campaign_rows = load(OUT / folder / 'rows.json')
        prior_shapes = {tuple(r['shape']) for r in previous['rows'] if r['op'] == op}
        replay_shapes = {tuple(r['shape']) for r in campaign_rows if r['role'] == 'development'}
        if replay_shapes != prior_shapes or any(r['op'] != op for r in campaign_rows):
            raise ValueError(f'Incomplete development replay for {op}')
        frozen = {c['id']: c for c in load(OUT / folder / 'freeze.json')['cases']}
        validation = [r for r in campaign_rows if r['role'] == 'validation']
        if {r['id'] for r in validation} != set(frozen):
            raise ValueError(f'Missing or unexpected holdout for {op}')
        for row in validation:
            case = frozen[row['id']]
            if (row['shape'] != case['shape'] or row['attrs'] != case['attrs']
                    or row['predicted_us'] != case['prediction']['predicted_us']
                    or row['previous_us'] != case['previous_us']):
                raise ValueError(f"Changed shape, attributes or frozen prediction for {row['id']}")
        fresh.extend(campaign_rows)
    old_shapes = {(r['op'], tuple(r['shape'])) for r in previous['rows']}
    rows = [r for r in previous['rows'] if r['op'] not in CAMPAIGNS]
    for row in fresh:
        if row['role'] == 'validation' and (row['op'], tuple(row['shape'])) in old_shapes:
            raise ValueError(f"Reused validation shape {row['shape']}")
        rows.append(row)
    for r in rows:
        times = []
        for _ in range(5):
            start = perf_counter_ns()
            pred = estimate(r['op'], r['shape'], r.get('attrs', {}))
            times.append((perf_counter_ns() - start) / 1e6)
        if pred.latency_us is None or not math.isclose(pred.latency_us, r['predicted_us'], rel_tol=1e-8):
            raise ValueError(f"Installed interface disagrees with frozen prediction for {r['op']} {r['shape']}")
        r['query_max_ms'] = max(times)
        if r['query_max_ms'] >= 100 or not math.isfinite(r['reference_us']) or r['reference_us'] <= 0:
            raise ValueError('Invalid CA time or query exceeded 100 ms')
        if r['op'] in CAMPAIGNS:
            prior = predict_v10(r['op'], r['shape'], r.get('attrs', {}))['predicted_us']
            if r.get('previous_us') is not None and not math.isclose(prior, r['previous_us'], rel_tol=1e-8):
                raise ValueError('Incorrect same-shape v0.10 prediction')
            r['previous_us'] = prior
    summaries = []
    for operator in previous['operator_summary']:
        op = operator['op']
        sample = [r for r in rows if r['op'] == op]
        val = [r for r in sample if r['role'] == 'validation']
        dev = [r for r in sample if r['role'] == 'development']
        if (len(val) != len({tuple(r['shape']) for r in val}) or len(sample) != len(val) + len(dev)
                or (len(val) < 10 and op != 'kda_chunk')):
            raise ValueError(f'Invalid evaluation split for {op}')
        wape, worst = errors(val)
        summary = dict(op=op, name=operator['name'], method=operator['method'],
                       independent=len(val), development=len(dev), wape=wape, max_ape=worst,
                       query_max_ms=max((r['query_max_ms'] for r in sample), default=None))
        if op in CAMPAIGNS:
            if len(val) != 10:
                raise ValueError(f'Expected 10 new shapes for {op}')
            summary['groups'] = {}
            for group, group_rows in split_groups(val):
                if not group_rows:
                    raise ValueError(f'Empty {group} for {op}')
                prev, prev_max = errors(group_rows, 'previous_us')
                now, now_max = errors(group_rows)
                summary['groups'][group] = dict(count=len(group_rows), wape=now, max_ape=now_max,
                                                v010_wape=prev, v010_max_ape=prev_max)
        summaries.append(summary)
    return rows, summaries


def figure(summary):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'build/mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    items = [s for s in summary if s['wape'] is not None]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    current = [s['wape'] * 100 for s in items]
    ax.barh(range(len(items)), current, color='#39776b')
    for index, value in enumerate(current):
        ax.text(value, index, f'  {value:.1f}%', va='center', fontsize=8)
    ax.set_yticks(range(len(items)), [s['op'] for s in items])
    ax.set_xlim(0, max(current)*1.19)
    ax.invert_yaxis()
    ax.set_xlabel('WAPE across independent CA shapes (%)')
    ax.grid(axis='x', alpha=.18)
    fig.tight_layout()
    fig.savefig(OUT / 'validation.png', dpi=144)
    current_dir = ROOT / 'results/current'
    (current_dir/'all-operators.png').write_bytes((OUT/'validation.png').read_bytes())
    svg = current_dir/'all-operators.svg'
    fig.savefig(svg, metadata={'Date': None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def shape_label(row):
    a = row.get('attrs', {})
    label = f"`{row['shape']}`"
    if 'num_iters' in a:
        label += f"，I={a['num_iters']}"
    if 'expert_counts' in a:
        label += f"，专家行数={a['expert_counts']}"
    return label


def subgroup_table(summaries):
    lines = ['## 改动算子的同集对照', '',
             '默认调用只传 shape。少 token 宽通道组在冻结时定义为 T≤4 且末维 D≥2048，普通组为其余输入。两个版本使用相同库调用的 CA 窗口。', '',
             '| 算子 | 独立分组 | shape 数 | v0.10 WAPE → 当前 | v0.10 最大误差 → 当前 |',
             '|---|---|---:|---:|---:|']
    labels = {'all': '全部', 'small_wide': '少 token 宽通道', 'ordinary': '普通'}
    for s in summaries:
        for group, g in s.get('groups', {}).items():
            lines.append(f"| {s['name']} | {labels[group]} | {g['count']} | "
                         f"{pct(g['v010_wape'])} → {pct(g['wape'])} | {pct(g['v010_max_ape'])} → {pct(g['max_ape'])} |")
    return lines


def render(rows, summaries):
    apis = {s['op']: s for s in load(ROOT/'results/v0.7/evaluation.json')['operator_summary']}
    lines = ['# 算子耗时预测与 CA 校准误差', '',
             '| 算子 | 方法 | 开发 / 独立 shape | 独立 WAPE / 最大误差 | 最慢查询 ms | CA 计时状态 |',
             '|---|---|---:|---:|---:|---|']
    for s in summaries:
        duration = f"{s['query_max_ms']:.3f}" if s['query_max_ms'] is not None else 'N/A'
        lines.append(f"| {s['name']} | {s['method']} | {s['development']} / {s['independent']} | "
                     f"{pct(s['wape'])} / {pct(s['max_ape'])} | {duration} | "
                     f"{'完整计时' if s['independent'] else '仅结构核对'} |")
    lines += ['', '每行对应后端的一条默认预测路径。WAPE 是绝对时延偏差之和除以 CA 参考时延之和；较慢的样例权重较大。最大误差是单例最大的绝对相对误差。表内误差只使用独立 shape。KDA chunk 有 10 个结构检查，尚无完整 CA 耗时。',
              '', '![各路径的独立 CA 误差](../results/v0.11/validation.png)', '',
              '## 怎样把这些误差用于系统仿真', '',
              '参考平台为 CANN 9.2.1 的 Ascend950DT_9582 CA。单 kernel 取最早指令至最晚指令的跨核净窗口，多 kernel 串行链求各窗口之和。框架需另计 host launch、通信、kernel 间间隙和跨算子重叠。多个组件共享一次链采集，整链误差可能包含相反方向的阶段偏差。A7/A8 尚无对应实测。',
              '', '模型输入使用每卡实际 shape、布局、dtype、迭代数或专家 token 分布。对批量顺序执行的 trace，缓存与控制状态可能随请求顺序变化。查询列为一个 Python 进程逐个 shape 调用系统后端五次的最大墙钟，按次重算且包含各路径首次读参数；返回的设备时延以 µs 计。',
              '', *subgroup_table(summaries), '']
    analysis = (OUT/'analysis.md').read_text().strip()
    for folder in (*CAMPAIGNS.values(), 'scaling'):
        analysis = analysis.replace(f']({folder}/', f'](../results/{OUT.name}/{folder}/')
    lines += [analysis, '', '## 逐算子与逐 shape 的证据', '']
    for s in summaries:
        op = s['op']
        info = apis[op]
        sample = [r for r in rows if r['op'] == op]
        lines += [f"### {s['name']}", '', f"安装包调用 {info['library']} 的 `{info['api']}`。", '']
        if not sample:
            lines += ['普通整块和双头 safe gate 路径在主 kernel 均触发重复 set_flag，时延为 N/A。[CPU 与分块结构核对](../results/v0.8/cube/chunk-structure-validation.json)', '']
            continue
        lines += ['| shape 与属性 | 用途 | 预测 µs | CA µs | 有符号误差 | 实际 kernel | 活跃 AIC / AIV |',
                  '|---|---|---:|---:|---:|---|---|']
        for r in sorted(sample, key=lambda x: x['role'] != 'validation'):
            role = '独立' if r['role'] == 'validation' else '开发'
            lines.append(f"| {shape_label(r)} | {role} | {r['predicted_us']:.3f} | {r['reference_us']:.3f} | "
                         f"{r['predicted_us']/r['reference_us']-1:+.1%} | {r['kernel']} | {r['active_aic']} / {r['active_aiv']} |")
        lines.append('')
    lines += ['[机器可读校验数据与 CA 采集成本](../results/v0.11/evaluation.json) · [本次冻结与同集比较](../results/v0.11/changes.md)', '']
    text = '\n'.join(lines)
    (ROOT/'docs/error-analysis.md').write_text(text)
    (OUT/'error-analysis.md').write_text(text.replace('../results/v0.11/', './').replace('../results/v0.8/', '../v0.8/'))


if __name__ == '__main__':
    observations, operators = collect()
    figure(operators)
    render(observations, operators)
    data = dict(version=OUT.name, rows=observations, operator_summary=operators,
                collection_cost={folder: load(OUT/folder/'costs.json') for folder in CAMPAIGNS.values()})
    dump(OUT/'evaluation.json', data)
    (ROOT/'results/current/evaluation.json').write_bytes((OUT/'evaluation.json').read_bytes())
