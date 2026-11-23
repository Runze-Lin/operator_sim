"""Build the per-contract CA window tables from the published reference and public estimates."""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
from math import isclose
from pathlib import Path
import re

import operator_model
from calibration.__main__ import FUTURE, HERE, encoded, metrics, predict, read

DT = 'Ascend950DT_9582'
PR = 'Ascend950PR_957b'
ROLES = {'replay': '开发回放', 'independent': '冻结独立'}
SOURCE = '../../calibration/reference/windows.jsonl'
NA = 'N/A'


def page_name(contract):
    return re.sub('[^a-z0-9_]+', '-', contract.lower()).strip('-') + '.md'


def number(value, digits=4, signed=False):
    if value is None:
        return NA
    precision = '.3g' if 0 < abs(value) < 10**-digits else f'.{digits}f'
    return format(value, ('+' if signed else '') + precision)


def percent(fraction):
    return NA if fraction is None else number(100*fraction, 3) + '%'


def table(headings, body):
    def line(cells):
        return '| ' + ' | '.join(str(cell).replace('|', r'\|').replace('\n', ' ') for cell in cells) + ' |'
    return '\n'.join([line(headings), line(['---']*len(headings)), *map(line, body)])


@lru_cache(maxsize=None)
def reprice(case_bytes, sku):
    chip = operator_model.load_chip(sku)
    result = predict(operator_model, json.loads(case_bytes), chip)
    return result.latency_us, result.work['dtype'], result.rates['roofline']


def load_windows(path):
    groups = defaultdict(list)
    with path.open(encoding='utf-8') as archive:
        for index, line in enumerate(archive, 1):
            row = json.loads(line)
            case = encoded({k: row.get(k) for k in ('op', 'shape', 'attrs', 'dtype', 'prefix')})
            for sku, field, destination in [(row['chip'], 'predicted_us', 'roofline'),
                                             (FUTURE, 'extrapolated_970dt_us', 'future_roofline')]:
                current, dtype, roofline = reprice(case, sku)
                if not isclose(current, row[field], abs_tol=1e-7, rel_tol=1e-10):
                    raise ValueError(f'Archived prediction differs on line {index} for {sku}: {row[field]} vs {current}')
                row[destination] = roofline
                row['input_dtype'] = dtype
            row['line'] = index
            groups[row['contract']].append(row)
    return groups


def utilization(rows):
    eligible = [row for row in rows if row['roofline']['utilization'] is not None]
    ideal = sum(row['roofline']['ideal_us'] for row in eligible)
    modeled = sum(row['predicted_us'] for row in eligible)
    return len(eligible), ideal / modeled if modeled else None


def index_table(groups, role):
    body = []
    combined = []
    for rows in groups.values():
        combined.extend(rows)
    for name, rows in [*groups.items(), (None, combined)]:
        cells = [f'[{name}]({page_name(name)})' if name else '**全部合同窗**']
        for chip in (DT, PR):
            selected = [r for r in rows if r['chip'] == chip and r['data_role'] == role]
            count, value = utilization(selected)
            cells += [f'{count}/{len(selected)}', percent(value)]
        body.append(cells)
    return table(['合同', 'DT 可比/完整窗', 'DT 等效利用率', 'PR 可比/完整窗', 'PR 等效利用率'], body)


def render_index(groups):
    count = sum(len(rows) for rows in groups.values())
    sections = ['# 按合同查阅 CA 完整窗', f'共 {len(groups)} 个输入与计时合同、{count:,} 条记录。选合同可查看每次调用的本卡 shape、输入精度和前序、模型与 CA 设备耗时、查询耗时及 970DT 外推。同一 shape 的不同运行各占一行。',
                '可比窗指能够同时取得等效 Roofline 理想时间和同一输入完整模型时间的窗口。合同汇总按 `Σ理想时间 / Σ模型时间` 计算，分子和分母取完全相同的可比窗。单次占比可超过 100%，例如来自 L2 的数据也可能计入逻辑搬运量。这里的分母是设备预测，CA 用于校验预测误差。970DT 列只有外推。',
                '[总误差与计时口径](../error-analysis.md) · [建模方法](../modeling-strategy.md) · [逐窗原始数据](../../calibration/reference/windows.jsonl)']
    for role, title in ROLES.items():
        sections += [f'## {title}', index_table(groups, role)]
    sections += ['## 只有 CA 实测的输入', '[FP8 cache 解包、非连续 FP4 量化、残差加和广播乘的双端整窗](measured-only.md)。这些输入还没有可独立评价的价格。', '## 其它精度与实现',
                '表内合同以参考数据的当前登记为准。表外精度与实现（如 A8W4 专家 GMM 或低精度 FA）应按其各自的完整窗确定 CA 误差与等效利用率，缺少匹配证据时为 N/A。[精度与实现覆盖](../precision-coverage.md)']
    sections += ['## 复算与更新',
                 '在安装了本仓库包的环境，从仓库根目录执行 `python -m calibration.report_pages --check` 可重算全部页面并检查归档耗时。更新参考记录后，执行 `python -m calibration.report_pages` 重新生成。[生成工具](../../calibration/report_pages.py)']
    return '\n\n'.join(sections) + '\n'


def dictionary(values):
    return {value: i for i, value in enumerate(dict.fromkeys(values), 1)}


def code(value):
    return '`' + json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '`'


def link_id(prefix, number, anchor=False):
    name = f'{prefix}{number}'
    return f'<a id="{name.lower()}"></a>{name}' if anchor else f'[{name}](#{name.lower()})'


def implementation(row):
    return (row['api'], row['provider'], tuple(row['kernel']))


def render_contract(contract, rows):
    sections = [f'# {contract} · 逐窗误差', '[全部合同](README.md) · [总误差分析](../error-analysis.md) · [原始逐窗记录](../../calibration/reference/windows.jsonl)']
    if not rows:
        sections += ['950DT、950PR 均有 0 个合格完整窗。模型对照、CA 误差和等效利用率为 N/A，970DT 无对应校准窗。']
        evidence = {
            'attn_res_fwd · resident inference prefill': '此为 UB 驻留的推理 prefill，物化输出的样本列在另一个合同。[AttnResFwd 算例与证据边界](../examples/attn_res_fwd.md)',
            'candidate_filtered_indexer': '系统捕获了候选块索引及其形状。算例分别核算全表打分与候选优先的工作量。[候选 Indexer 算例](../examples/candidate-filtered-indexer.md)',
            'cache_load_mxfp4': '显式 AntiMxQuant 有[独立双端校准](cache_load_mxfp4_antimx.md)。系统捕获了 packed E2M1 值、按 32 元素分组并补齐的 E8M0 scale 和 BF16 物化输出。[缓存读取算例](../examples/cache_load.md)',
            'cache_load_fp8': '默认调用没有派发身份。显式 AscendAntiQuant 有原生实测，[FP8 的双端设备时间](measured-only.md)。系统捕获了 E4M3FN 值、逐行 FP32 scale 和 BF16 物化输出。[缓存读取算例](../examples/cache_load.md)',
            'routed_grouped_matmul_mxfp4': '显式 GroupedMatmulV5 有[双端校准](routed_grouped_matmul_mxfp4_v5.md)。系统捕获了路由专家的 A4W4 分组乘法，工作量取决于专家行分布。稠密 Matmul 的 CA 窗属于另一合同。[路由专家 GMM 算例](../examples/routed-grouped-matmul.md)',
        }
        if contract in evidence:
            sections.append(evidence[contract])
        return '\n\n'.join(sections) + '\n'
    sections.append('模型和 CA 均为设备微秒（µs）。有符号误差为 `100 × (模型 / CA − 1)`。等效 Roofline 利用率 U 为同一输入的理想时间除以模型设备时间。表内同时列出理想时间的控制项；合同汇总只对可比窗按 `Σ理想时间 / Σ模型时间` 聚合。查询列为归档 CPU 墙钟毫秒（ms），顺序为 950 / 970。970DT 的时间与 U 都是外推。')

    attributes = dictionary(encoded(r['attrs']) for r in rows if r['attrs'])
    histories = dictionary(encoded(r['prefix']) for r in rows if r.get('prefix'))
    kernels = dictionary(implementation(r) for r in rows)
    reasons = dictionary(r[key]['reason'] for r in rows for key in ('roofline','future_roofline')
                         if r[key]['utilization'] is None and r[key].get('reason'))

    def roof_cells(info):
        why = info.get('reason')
        bound = str(info['bound']) if info['utilization'] is not None else (f'原因 {reasons[why]}' if why else NA)
        return [number(info['ideal_us'], 6), bound, percent(info['utilization'])]

    summary = []
    for chip in (DT, PR):
        for role, label in ROLES.items():
            selected = [r for r in rows if r['chip'] == chip and r['data_role'] == role]
            error = metrics(selected)
            count, value = utilization(selected)
            summary.append([chip, label, error['shapes'], len(selected), number(error['wape_pct'],2),
                            number(error['max_pct'],2), count, percent(value)])
    sections += ['## 完整窗与误差', table(['芯片','数据角色','不同 shape','完整窗','WAPE %','最大绝对误差 %','U 可比窗','汇总 U'],summary),
                 '下表的行号链接到 JSONL 原记录，其中包含全长 trace 指纹和已归档的收据字段。A、H、K 分别链接到本页的属性、前序和实现。前序列的“未传”表示本次调用未提供流历史，“空流”表示已知从空的 OperatorStream 开始；显式 resident_aiv 归在属性内。']
    for chip in (DT, PR):
        for role, label in ROLES.items():
            selected = [r for r in rows if r['chip'] == chip and r['data_role'] == role]
            if not selected:
                continue
            body = []
            for row in selected:
                attrs = row['attrs']
                attr = ('未传' if attrs is None else '{}') if not attrs else link_id('A', attributes[encoded(attrs)])
                prefix = row.get('prefix')
                history = ('未传' if prefix is None else '空流') if not prefix else link_id('H', histories[encoded(prefix)])
                ref = f"[{row['line']}]({SOURCE}#L{row['line']})"
                lower, bound, fraction = roof_cells(row['roofline'])
                future = row['future_roofline']
                fu = percent(future['utilization'])
                if future['utilization'] is None and future.get('reason'):
                    fu += f"（原因 {reasons[future['reason']]}）"
                body.append([ref,'×'.join(map(str,row['shape'])),row['input_dtype'],f'{attr} / {history}',link_id('K', kernels[implementation(row)]),
                             number(row['predicted_us']),number(row['ca_us']),number(row['signed_error_pct'],2,True),
                             f'{lower} / {bound}',fraction,number(row['extrapolated_970dt_us']),fu,
                             f"{number(row['query_ms'],3)} / {number(row['query_970_ms'],3)}"])
            sections += [f'## {chip} · {label}',table(['原始行','本卡 shape','主输入 dtype','属性 / 前序','实现','模型 µs','CA µs','误差 %','理想 µs / 控制项','U','970 外推 µs','970 外推 U','查询 ms（950 / 970）'],body)]
    sections += ['## 实际调用的库实现', table(['编号','ACLNN 接口','实现提供方','设备 kernel，按执行顺序'],
                                  ((link_id('K', i, anchor=True),api,provider,' → '.join(f'`{name}`' for name in stages))
                                   for (api,provider,stages),i in kernels.items()))]
    if attributes:
        sections += ['## 调用属性',table(['编号','原始 attrs'],((link_id('A', i, anchor=True),code(json.loads(key))) for key,i in attributes.items()))]
    if histories:
        sections += ['## 可见的调用前序',table(['编号','按提交顺序的完整 prefix'],((link_id('H', i, anchor=True),code(json.loads(key))) for key,i in histories.items()))]
    if reasons:
        sections += ['## Roofline 缺项说明',table(['编号','N/A 原因'],((i,why) for why,i in reasons.items()))]
    return '\n\n'.join(sections) + '\n'


def report(output, check=False):
    folder = HERE / 'reference'
    contracts = read(folder / 'summary.json')
    rows = load_windows(folder / 'windows.jsonl')
    unknown = set(rows) - set(contracts)
    if unknown:
        raise ValueError(f'Unlisted contracts: {sorted(unknown)}')
    groups = {name: rows[name] for name in contracts}
    pages = {page_name(name): render_contract(name, items) for name, items in groups.items()}
    if len(pages) != len(groups) or 'README.md' in pages:
        raise ValueError('Ambiguous contract filenames')
    from calibration.diagnostics import render as measured_only
    pages['measured-only.md'] = measured_only()
    pages['README.md'] = render_index(groups)
    if check:
        for filename, content in pages.items():
            if not (output / filename).is_file() or (output / filename).read_text(encoding='utf-8') != content:
                raise ValueError(f'Missing or stale report: {filename}')
    else:
        output.mkdir(exist_ok=True, parents=True)
        for filename, content in pages.items():
            (output / filename).write_text(content, encoding='utf-8')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, default=HERE.parent/'docs'/'errors')
    cli.add_argument('--check', action='store_true', help='Verify generated Markdown without writing files.')
    args = cli.parse_args()
    report(args.output, args.check)
