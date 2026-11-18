"""Publish the portable backend's two-SKU evidence from archived, qualified CA data."""
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from operator_model import estimate

OUT = ROOT/'results/v0.13'
OLD = ROOT/'results/v0.12'
DOC = ROOT/'docs/error-analysis.md'
PR = 'Ascend950PR_957b'


def load(path):
    return json.loads(path.read_text())


def group_error(rows):
    if not rows:
        return 'N/A'
    errors = [abs(r['target_us']-r['pr']['ca_us']) for r in rows]
    return (f"{sum(errors)/sum(r['pr']['ca_us'] for r in rows):.2%} / "
            f"{max(e/r['pr']['ca_us'] for e,r in zip(errors,rows)):.2%}")


def figure(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,3.7), layout='constrained')
    for offset, label, color, vals in [
        (-.26, '950DT CA', '#7f96a9', [r['dt']['ca_us'] for r in rows]),
        (0, '950PR CA', '#086774', [r['pr']['ca_us'] for r in rows]),
        (.26, '950PR prediction', '#cf7334', [r['target_us'] for r in rows])]:
        ax.bar([i+offset for i in range(len(rows))], vals, .24, label=label, color=color)
    ax.set_xticks(range(len(rows)),[r['op']+'\n'+'×'.join(map(str,r['shape']))+'\n'+r['role'] for r in rows])
    ax.set_ylabel('Device time (µs)')
    ax.legend(frameon=False, ncols=3)
    ax.spines[['right','top']].set_visible(False)
    fig.savefig(OUT/'sku-comparison.png', dpi=155)
    plt.close(fig)


def relocate(text, old_dir, new_dir):
    def link(match):
        address = match[1]
        if address.startswith(('https:', 'http:', '#')):
            return match[0]
        return ']('+os.path.relpath(old_dir/address, new_dir)+')'
    return re.sub(r'\]\(([^() ]+)\)', link, text)


def main():
    archive = load(OUT/'evaluation.json')
    rows = []
    for file, role in [('pairs.json','development'), ('holdout.json','holdout')]:
        for row in load(OUT/'sku'/file):
            r = estimate(row['op'],row['shape'],chip=PR)
            if role == 'holdout' and not math.isclose(r.latency_us,row['frozen']['pr_us'],rel_tol=1e-10):
                raise ValueError('Disagrees with frozen holdout prediction')
            rows.append(dict(**row,role=role,target_us=r.latency_us,dt_estimate_us=estimate(row['op'],row['shape']).latency_us,target_aic=r.active_aic,target_aiv=r.active_aiv))
    figure(rows)
    note = ['# 算子耗时预测与 CA 校准误差', '',
            '| 算子 | DT 开发 / 独立 | DT 独立 WAPE / 最大误差 | PR 开发 / 独立 | PR 独立 WAPE / 最大误差 | 校准状态 |',
            '|---|---:|---:|---:|---:|---|']
    for s in archive['dt_summary']:
        dev = [r for r in rows if r['op']==s['op'] and r['role']=='development']
        val = [r for r in rows if r['op']==s['op'] and r['role']=='holdout']
        dt = f"{s['wape']:.2%} / {s['max_ape']:.2%}" if s['wape'] is not None else 'N/A'
        status = ('DT 完整，PR 配对'+('及冻结验证' if val else '开发') if dev else
                  'DT 完整，PR 外推' if s['independent'] else '仅结构')
        note.append(f"| {s['name']} | {s['development']} / {s['independent']} | {dt} | "
                    f"{len(dev)} / {len(val)} | {group_error(val)} | {status} |")
    note.extend(['','WAPE = 绝对误差之和 / CA 参考时间之和。最大误差取单例的绝对相对误差。DT 使用已有冻结独立集，更新接口后逐点回放。PR 独立指预测冻结后首次在该 SKU 采集的 shape；同一 shape 可以有更早的 DT 记录。每个 PR 独立组仅一例，WAPE 与最大误差相等。960 / 970 / 980 均无实测。','',
        '## 950PR 与 950DT 的配对检验','',
        '两端均为 CANN 9.2.1，输入合同、数值校验和计时口径一致。逐例核对运行时 SKU、安装包实际 kernel、黄金输出、二进制与 trace 指纹、各核窗口和跨核设备时间。下列三个开发例用于辨认两种芯片分核差异，独立例在调整分块后冻结。', '',
        '| 算子与本卡 shape | PR 角色 | DT CA µs | PR CA µs | 原样复用 DT 预测 µs | PR 预测 µs | 有符号误差 | PR 预测 / 实测活跃核（C,V） |',
        '|---|---|---:|---:|---:|---:|---:|---|'])
    for r in rows:
        ca = r['pr']['ca_us']
        note.append(f"| {r['op']} `{r['shape']}` | {'独立' if r['role']=='holdout' else '开发'} | {r['dt']['ca_us']:.3f} | "
                    f"{ca:.3f} | {r['dt_estimate_us']:.3f} | {r['target_us']:.3f} | {r['target_us']/ca-1:+.2%} | "
                    f"({r['target_aic']},{r['target_aiv']}) / ({r['pr']['active_aic']},{r['pr']['active_aiv']}) |")
    good_cost = sum(r['pr_cost']['seconds'] for r in rows)
    rejected = sum((r['rejected_infrastructure_attempt'] or {}).get('seconds',0) for r in rows)
    note += ['', '![同 shape 设备时间，按开发和独立标注](../results/v0.13/sku-comparison.png)', '',
             '在 PR 的开发点，quant [64,1024] 的活跃 AIV 从 DT 的 64 变成 44，Matmul [128,256,512] 的活跃 AIC 从 32 变成 16，FA 维持 1 AIC / 2 AIV。这使按目标核数重做分块成为 quant 与 Matmul 的校准项。两款平台配置了不同片外存储子模型。同一小算子只有合成的设备时延，无法单独识别持续外存带宽、缓存命中率或实际 HBM 字节。', '',
             '冻结的 quant [512,256] 在两端的预测都受同一个 5.430 µs 控制下限约束，核数虽命中，shape 在 PR 上的额外时延变化未被捕获。Matmul [129,256,256] 在两端均为 20 AIC，PR 阶段缩放使预测从 3.947 µs 增至 3.987 µs，CA 却从 DT 3.664 µs 降至 PR 3.609 µs。该点已有 DT 基线偏差，迁移倍率额外放大偏差；需要阶段级证据才能将剩余误差分到读、算、控制。两个独立点都没有显示相对直接复用 DT 耗时预测的精度优势。', '',
             f'新增 {len(rows)} 个合格 PR CA 样例，成功进程墙钟累计 {good_cost:.1f} s。另有三次通信失败、运行时 SKU 不符的尝试，额外累计 {rejected:.1f} s，未计入精度。DT 复用既有 trace。[实际 ACLNN、外存子模型与核数](../results/v0.13/sku/README.md) · [开发配对摘要](../results/v0.13/sku/pairs.json) · [冻结合同](../results/v0.13/sku/holdout-freeze.json) · [独立配对原始摘要](../results/v0.13/sku/holdout.json)', '',
             '## 在线预测的开销','',
             '| 先载入芯片配置 | DT 最慢 ms | PR 最慢 ms |','|---|---:|---:|']
    timings = archive['query_wall']
    for s in archive['dt_summary']:
        op = s['op']
        if s['independent']:
            note.append(f"| {s['name']} | {timings['Ascend950DT_9582'][op]['max_ms']:.3f} | {timings[PR][op]['max_ms']:.3f} |")
    note += ['', f"同一 Python 进程对 {len(archive['rows'])} 条 DT 既有计时输入，在两种配置上各调用三次，均含工作量统计；目标为单次 100 ms。[逐条预测、逻辑字节与计时](../results/v0.13/evaluation.json)",'']
    old = (OLD/'error-analysis.md').read_text()
    old = old[old.index('## 改动算子的同集对照'):].replace('## 改动算子的同集对照','## 950DT 已校准路线的同集对照')
    note += [relocate(old, OLD, DOC.parent)]
    DOC.write_text('\n'.join(note))
    (OUT/'error-analysis.md').write_text(relocate(DOC.read_text(), DOC.parent, OUT))
    current = load(OLD/'evaluation.json')
    current.update(version='v0.13', previous_version='v0.12', reference_calibration_release='v0.12',
                   operator_summary=archive['dt_summary'], paired_sku=rows,
                   query_wall=archive['query_wall'])
    for r, replay in zip(current['rows'],archive['rows'],strict=True):
        r['previous_us'] = r['predicted_us']
        r['query_max_ms'] = replay['dt_query_max_ms']
    for summary in current['operator_summary']:
        for group in summary.get('groups',{}).values():
            group['previous_wape'] = group['wape']
            group['previous_max_ape'] = group['max_ape']
    current['collection_cost'] = {'reference_archive': 'results/v0.12/evaluation.json',
                                  'new_pr_success_seconds':good_cost,'new_pr_rejected_seconds':rejected}
    (ROOT/'results/current/evaluation.json').write_text(json.dumps(current,indent=2,ensure_ascii=False)+'\n')
    metrics = {r['op']: group_error([r]) for r in rows if r['role']=='holdout'}
    print(json.dumps(metrics))
    print('CA seconds',good_cost,rejected)


if __name__ == '__main__':
    main()
