"""Publish the PR independent comparison and retain the complete DT calibration report."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model import estimate
from development.validation.report_v13 import relocate
from metrics import comparison, plot

OUT = ROOT/'results/v0.14'
OLD = ROOT/'results/v0.13'
OPS = ('quant','matmul','fa')
NAMES = {'quant':'DynamicMXQuant','matmul':'Matmul','fa':'FA'}
API = {'quant':'ops-nn / aclnnDynamicMxQuant','matmul':'ops-nn / aclnnMatmul',
       'fa':'ops-transformer / aclnnFlashAttentionScoreV2'}
PR = 'Ascend950PR_957b'


def load(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def prior_dev():
    dev = []
    for raw in load(OLD/'sku/pairs.json') + load(OLD/'sku/holdout.json'):
        pr = estimate(raw['op'],raw['shape'],chip=PR)
        dt = estimate(raw['op'],raw['shape'])
        dev.append(dict(op=raw['op'],shape=raw['shape'],pr_ca_us=raw['pr']['ca_us'],
                        dt_ca_us=raw['dt']['ca_us'],pr_us=pr.latency_us,dt_us=dt.latency_us))
    return dev


def pct_pair(metrics):
    return f"{metrics['wape']:.2%} / {metrics['max_ape']:.2%}"


def scale_analysis(rows):
    lines = ['### 工作量范围与尾块','',
             '分别按 quant 输入元素数、Matmul 有用 Cube 运算数和 FA 的 QK+PV 有用 Cube 运算数排序。'
             '每种算子在自己的十例中取较小五例和较大五例，显示数据规模对本组误差的影响。','',
             '| 算子 | 工作量范围 | 小五例 WAPE，DT复用 → PR | 大五例 WAPE，DT复用 → PR | 活跃核数完全命中 |',
             '|---|---:|---:|---:|---:|']
    for op in OPS:
        def magnitude(row):
            if op == 'quant':
                return math.prod(row['shape'])
            if op == 'matmul':
                return 2 * math.prod(row['shape'])
            return 4 * math.prod(row['shape'])
        group = sorted((r for r in rows if r['op']==op),key=lambda r:(magnitude(r),r['id']))
        split = [comparison(group[:5]),comparison(group[5:])]
        hits = sum((r['pr_aic'] or 0,r['pr_aiv'] or 0)==(r['actual_aic'],r['actual_aiv']) for r in group)
        values = [f"{v['dt_us']['wape']:.1%} → {v['pr_us']['wape']:.1%}" for v in split]
        lines.append(f"| {NAMES[op]} | {magnitude(group[0]):,}…{magnitude(group[-1]):,} | "
                     f"{values[0]} | {values[1]} | {hits}/{len(group)} |")
    return lines


def report(rows, summaries, stats, dev, wheel, cost):
    def sizes(op):
        return [r for r in rows if r['op']==op]
    note = ['# 算子耗时预测与 CA 校准误差','',
            '| 算子 | DT 开发 / 独立 | DT 独立 WAPE / 最大误差 | PR 开发 / 独立 | PR 独立 WAPE / 最大误差 | 最慢查询 ms（DT / PR） | 校准状态 |',
            '|---|---:|---:|---:|---:|---:|---|']
    for s in summaries:
        op = s['op']
        ndev = sum(r['op']==op for r in dev)
        qt = [ch['per_op_max_ms'].get(op) for ch in wheel['checks'].values()]
        wall = ' / '.join(f'{t:.3f}' for t in qt) if all(t is not None for t in qt) else 'N/A'
        dt = pct_pair(s) if s['independent'] else 'N/A'
        pr = pct_pair(stats[op]['pr_us']) if op in stats else 'N/A'
        qualification = 'DT、PR 完整 CA' if op in stats else 'DT 完整，PR 外推' if s['independent'] else '仅结构'
        note.append(f"| {s['name']} | {s['development']} / {s['independent']} | {dt} | "
                    f"{ndev} / {len(sizes(op))} | {pr} | {wall} | {qualification} |")
    note += ['', 'WAPE 为全部绝对误差之和除以同组 CA 总耗时，最大误差取最差单例的绝对相对误差。'
            'PR 的 5 个旧样例均列为开发，独立列只包括冻结之后在 PR 首次采集的 shape。DT 沿用已冻结的独立集。'
            '查询列为已安装 wheel 在同一进程内逐条调用的墙钟峰值，单位 ms。','',
            relocate((OUT/'chips.md').read_text().strip(),OUT,ROOT/'docs'),'',
            '## 冻结 PR 预测与直接复用 DT 的同集比较','',
            '“复用 DT”是在相同 shape、dtype、属性上运行 DT 预测器，然后将预测时间直接用作 PR 时间。'
            '两种预测共用一个 PR CA 分母。另列 v0.13 原跨芯片预测，用于区分本轮改动的效果。','',
            '| 算子 | 独立 shape | v0.14 PR WAPE / 最大误差 | 直接复用 DT WAPE / 最大误差 | v0.13 PR WAPE / 最大误差 | 比直接复用更准 / 持平 / 更差 |',
            '|---|---:|---:|---:|---:|---:|']
    for op in OPS:
        s=stats[op]
        note.append(f"| {NAMES[op]} | {s['n']} | {pct_pair(s['pr_us'])} | {pct_pair(s['dt_us'])} | "
                    f"{pct_pair(s['prior_pr_us'])} | {s['wins']} / {s['ties']} / {s['losses']} |")
    note += ['', '![三个算子的独立 shape 逐点有符号误差](../results/v0.14/sku-comparison.png)','',
             '### 分核命中后，偏差出在哪里','',
             '对具备两端 CA 的点，PR 的总误差可以精确拆成两项。第一项是 DT 预测本身的偏差。'
             '第二项是芯片切换时的增量误差，即（PR 预测−DT 预测）减去（PR CA−DT CA）。'
             '下面给出逐点拆分后取算术平均的有符号值；异号误差可相消。','',
             '| 算子 | 双端 CA 点数 | DT 基础偏差 µs | 芯片增量偏差 µs | PR 总偏差 µs |',
             '|---|---:|---:|---:|---:|']
    for op in OPS:
        s=stats[op]
        if s.get('paired_count'):
            a,b=s['reference_error_mean_us'],s['transfer_error_mean_us']
            note.append(f"| {NAMES[op]} | {s['paired_count']} | {a:+.3f} | {b:+.3f} | {a+b:+.3f} |")
    note += ['', *scale_analysis(rows), '', relocate((OUT/'interpretation.md').read_text().strip(),OUT,ROOT/'docs'),'', '### 逐 shape 验证','',
             '| 实际库入口 / 每卡 shape | DT CA µs | PR CA µs | 复用 DT µs | PR 预测 µs | PR 有符号误差 | 预计 / 实际活跃核（C,V） | 实际 PR kernel |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        dt='N/A' if r['dt_ca_us'] is None else f"{r['dt_ca_us']:.3f}"
        note.append(f"| {API[r['op']]} `{r['shape']}` | {dt} | {r['pr_ca_us']:.3f} | {r['dt_us']:.3f} | "
                    f"{r['pr_us']:.3f} | {r['pr_us']/r['pr_ca_us']-1:+.2%} | "
                    f"({r['pr_aic']},{r['pr_aiv']}) / ({r['actual_aic']},{r['actual_aiv']}) | `{r['kernel']}` |")
    instructions = load(OUT/'sku/instructions.json')
    matched = sum(r['max_mmad_on_one_core']==r['predicted_max_mmad'] for r in instructions if r['op']=='matmul')
    note += ['', f'按队列复核完整 PR 指令时间线。Matmul 的最多单核 MMAD 次数与分块预测在 {matched}/10 例一致。'
             'Quant 的 RVEC 分成发起、加载、存储和执行等子队列，合计是时间线事件数；FA 的 MMAD 另按核计数。'
             '这些计数可核对分支和循环，直接乘单条指令延迟会重复计入重叠。'
             '[逐 shape 队列与 MMAD 计数](../results/v0.14/sku/instructions.json)', '',
             'Quant 检查每个量化值和 scale 字节；Matmul 检查全部 BF16 输出；FA 检查全部注意力主输出，辅助统计未验。'
             '每例还检查运行时 SKU、PR 的 LPDDR5 子模型、实际启动 kernel、跨核首末指令，以及各核时间窗与 profiler 的差值。'
             '逐例二进制和 trace 的 SHA-256、golden、有效核和采集成本见[原始验收记录](../results/v0.14/sku/holdout.json)。']
    note += ['', '### 开发回放','',
             '| PR 旧样例 | PR CA µs | 直接复用 DT µs | v0.14 PR µs | 有符号误差 |','|---|---:|---:|---:|---:|']
    for r in dev:
        note.append(f"| {r['op']} {r['shape']} | {r['pr_ca_us']:.3f} | {r['dt_us']:.3f} | "
                    f"{r['pr_us']:.3f} | {r['pr_us']/r['pr_ca_us']-1:+.2%} |")
    note += ['', cost, '', '[预测及误差 JSON](../results/v0.14/evaluation.json) · [安装与墙钟验证](../results/v0.14/wheel-check.json) · [变更摘要](../results/v0.14/changes.md)','']
    old = (OLD/'error-analysis.md').read_text()
    note.append(relocate(old[old.index('## 950DT 已校准路线的同集对照'):], OLD,ROOT/'docs'))
    return '\n'.join(note)


def main():
    rows = load(OUT/'sku/holdout.json')
    dev = prior_dev()
    olds = {(r['op'],tuple(r['shape'])+(1,1) if r['op']=='fa' and len(r['shape'])==3 else tuple(r['shape'])) for r in dev}
    for r in rows:
        r['kernel'] = r['pr']['kernel']
        r['actual_aic'],r['actual_aiv'] = r['active_aic'],r['active_aiv']
        normalized = tuple(r['shape']) + ((1,1) if r['op']=='fa' and len(r['shape'])==3 else ())
        if (r['op'],normalized) in olds:
            raise ValueError('Previously used PR shape')
        olds.add((r['op'],normalized))
        actual = estimate(r['op'],r['shape'],chip=PR)
        dt = estimate(r['op'],r['shape'])
        for projected, frozen in [(actual.latency_us,r['pr_us']),(dt.latency_us,r['dt_us'])]:
            if not math.isclose(projected,frozen,rel_tol=1e-10):
                raise ValueError('Model no longer agrees with frozen prediction')
        diagnostics = actual.projection['timing_diagnostics']
        r['pr_model_route'] = diagnostics.get('method') or diagnostics.get('mode') or 'phase_projection'
    stats = {}
    for op in OPS:
        group = [r for r in rows if r['op']==op]
        if len(group)!=10:
            raise ValueError(f'Expected ten qualified {op} shapes, got {len(group)}')
        stats[op] = comparison(group)
    wheel = load(OUT/'wheel-check.json')
    current = load(ROOT/'results/v0.12/evaluation.json')
    summary = current['operator_summary']
    for s in summary:
        s['query_max_ms'] = wheel['checks']['Ascend950DT_9582']['per_op_max_ms'].get(s['op'])
    cost = relocate((OUT/'cost.md').read_text().strip(),OUT,ROOT/'docs')
    evaluation = dict(version='v0.14', reference_dataset='results/v0.12/evaluation.json',
                      dt_summary=summary, pr_summary=stats, development=dev,
                      holdout='sku/holdout.json', attempts='sku/attempts.json',
                      freeze='sku/freeze.json', query_wall=wheel,
                      rows=[{k:r[k] for k in ('id','op','shape','dt_ca_us','pr_ca_us','pr_us','dt_us',
                          'prior_pr_us','pr_model_route','actual_aic','actual_aiv','pr_aic','pr_aiv','kernel')} for r in rows])
    write(OUT/'evaluation.json',evaluation)
    plot(rows,OUT/'sku-comparison.png')
    rendered = report(rows,summary,stats,dev,wheel,cost)
    (ROOT/'docs/error-analysis.md').write_text(rendered)
    (OUT/'error-analysis.md').write_text(relocate(rendered,ROOT/'docs',OUT))
    # Preserve the established schema and scope of each DT record.
    current['version']='v0.14'
    current['previous_version']='v0.13'
    current['reference_calibration_release']='v0.12'
    for r, ref_row in zip(current['rows'],load(OLD/'evaluation.json')['rows'],strict=True):
        if (r['op'],r['shape']) != (ref_row['op'],ref_row['shape']):
            raise ValueError('DT archive reference order differs')
        r['previous_us'] = r['predicted_us']
        r['query_max_ms'] = wheel['checks']['Ascend950DT_9582']['archived_row_ms'][ref_row['id']]
    for s in summary:
        for group in s.get('groups',{}).values():
            group.update(previous_wape=group['wape'], previous_max_ape=group['max_ape'])
    old_pairs = load(OLD/'sku/pairs.json') + load(OLD/'sku/holdout.json')
    for r, prediction in zip(old_pairs, dev):
        pr = estimate(r['op'], r['shape'], chip=PR)
        r.update(role='development',target_us=prediction['pr_us'],dt_estimate_us=prediction['dt_us'],
                 target_aic=pr.active_aic,target_aiv=pr.active_aiv)
    for r in rows:
        r.update(target_us=r['pr_us'], dt_estimate_us=r['dt_us'], target_aic=r['pr_aic'],
                 target_aiv=r['pr_aiv'], pr_to_dt_ratio=r['pr_ca_us']/r['dt_ca_us'] if r['dt_ca_us'] else None)
    current['paired_sku']=old_pairs+rows
    current['pr_summary']=stats
    current['query_wall']={chip: measurements['per_op'] for chip, measurements in wheel['checks'].items()}
    attempts = load(OUT/'sku/attempts.json')
    current['collection_cost']={'reference_archive':'results/v0.12/evaluation.json',
        'new_pr_success_seconds':sum(a['seconds'] for a in attempts if a['qualified']),
        'new_pr_rejected_seconds':sum(a['seconds'] for a in attempts if not a['qualified']),
        'pr_attempts':'results/v0.14/sku/attempts.json'}
    write(ROOT/'results/current/evaluation.json',current)
    print(json.dumps(stats,indent=2))


if __name__ == '__main__':
    main()
