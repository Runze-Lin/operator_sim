"""Summarize the two-SKU development replays and frozen holdout separately."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model import estimate

out = ROOT/'results/v0.13/sku'
development = json.loads((out/'pairs.json').read_text())
validation = json.loads((out/'holdout.json').read_text())
report = ['# 950PR 开发回放与独立误差', '',
          '预测先按 28 AIC、56 AIV 重算支持的分核策略，再按目标芯片资源给各阶段估时。'
          '下表的核数均来自设备指令记录或模型预计的有效分核。误差为（预测/CA−1）×100%。', '']
for label, data in [('开发回放', development), ('冻结独立验证', validation)]:
    report.extend(['## '+label, '', '| shape | DT CA / 模型（µs） | PR CA / 模型（µs） | PR 误差 | PR 活跃核，实测 / 预计 |',
                   '|---|---:|---:|---:|---|'])
    for row in data:
        op, shape = row['op'],row['shape']
        is_cube = op=='matmul'
        def cores(aic, aiv):
            return f'{aic} AIC + {aiv} AIV' if op=='fa' else f'{aic} AIC' if is_cube else f'{aiv} AIV'
        if 'frozen' in row:
            frozen = row['frozen']
            dt, pr = frozen['dt_us'], frozen['pr_us']
            active = cores(frozen['pr_aic'],frozen['pr_aiv'])
        else:
            dt = estimate(op,shape).latency_us
            target = estimate(op,shape,chip=row['pr']['soc'])
            pr = target.latency_us
            active = cores(target.active_aic,target.active_aiv)
        b,c = row['dt']['ca_us'],row['pr']['ca_us']
        report.append(f'| {op} {shape} | {b:.3f} / {dt:.3f} | {c:.3f} / {pr:.3f} | {100*(pr/c-1):+.2f}% | '
                      +cores(row['pr']['active_aic'],row['pr']['active_aiv'])+' / '+active+' |')
    report.append('')
seconds = sum(x['pr_cost']['seconds'] for x in validation)
mib = sum(x['pr_cost']['raw_bytes'] for x in validation)/2**20
report += [f'独立验证两次新发射合计 {seconds:.1f} 进程秒，落盘 {mib:.1f} MiB。'
           '冻结了模型代码、在线参数、输入清单和预测。每条路线仅 1 个新 PR 点，单点绝对相对误差也等于该路线当前的独立 WAPE。'
           '[冻结清单](holdout-freeze.json) · [逐例实测](holdout.json) · [开发对照与两端外存子模型](README.md)', '']
(out/'comparison.md').write_text('\n'.join(report))
