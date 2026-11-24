"""Build the per-contract CA window tables from the published reference and public estimates."""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
from math import isclose
from pathlib import Path
import re

import operator_model
from operator_model.roofline import cold_io_window
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
    if fraction is None:
        return NA
    value = 100 * fraction
    return (format(value, '.3g') if 0 < abs(value) < .01 else number(value, 3)) + '%'


def table(headings, body):
    def line(cells):
        return '| ' + ' | '.join(str(cell).replace('|', r'\|').replace('\n', ' ') for cell in cells) + ' |'
    return '\n'.join([line(headings), line(['---']*len(headings)), *map(line, body)])


@lru_cache(maxsize=None)
def reprice(case_bytes, sku):
    chip = operator_model.load_chip(sku)
    result = predict(operator_model, json.loads(case_bytes), chip)
    rates = result.rates['logical_roofline']
    logical = dict(ideal_us=rates['reference_us'], utilization=rates['time_factor'],
                   hbm_us=rates['logical_io_us'], cube_us=rates['cube_us'], vector_us=rates['vector_us'],
                   bound=rates['bound'], reason=rates['reason'])
    # Use the exact deployment policy after replaying the original stream history;
    # a stateless estimate here would silently change the calibrated schedule.
    cold = None
    if sku != FUTURE and logical['hbm_us'] is not None:
        cold = cold_io_window(result.latency_us, result.work, chip)
    return result.latency_us, result.work['dtype'], logical, result.rates['physical_roofline'], cold


def load_windows(path):
    groups = defaultdict(list)
    with path.open(encoding='utf-8') as archive:
        for index, line in enumerate(archive, 1):
            row = json.loads(line)
            case = encoded({k: row.get(k) for k in ('op', 'shape', 'attrs', 'dtype', 'prefix')})
            for sku, field, destination in [(row['chip'], 'predicted_us', 'roofline'),
                                             (FUTURE, 'extrapolated_970dt_us', 'future_roofline')]:
                current, dtype, roofline, physical, cold = reprice(case, sku)
                if not isclose(current, row[field], abs_tol=1e-7, rel_tol=1e-10):
                    raise ValueError(f'Archived prediction differs on line {index} for {sku}: {row[field]} vs {current}')
                row[destination] = roofline
                if sku == row['chip']:
                    row['physical_roofline'] = physical
                    row['cold_us'] = None if cold is None else cold['duration_us']
                    row['cold_u'] = (roofline['ideal_us'] / cold['duration_us']
                                     if cold is not None and roofline['ideal_us'] is not None else None)
                row['input_dtype'] = dtype
            row['line'] = index
            groups[row['contract']].append(row)
    return groups


def metric_range(values):
    values = [v for v in values if v is not None]
    return f'{percent(min(values))}-{percent(max(values))}' if values else NA


def window_ranges(rows):
    return [metric_range(r['roofline']['utilization'] for r in rows),
            metric_range(r['cold_u'] for r in rows)]


def summary_ranges(markdown, groups):
    """Refresh both logical ranges without changing the CA validation scores."""
    ranges = {}
    for name, rows in groups.items():
        per_chip = []
        for chip in (DT, PR):
            cohort = [r for r in rows if r['chip'] == chip]
            factors = [r['roofline']['utilization'] for r in cohort if r['roofline']['utilization'] is not None]
            factor_range = f'{min(factors):.4g}–{max(factors):.4g}×' if factors else NA
            per_chip.append((factor_range, metric_range(r['cold_u'] for r in cohort)))
        ranges[page_name(name)] = [' / '.join(pair) for pair in zip(*per_chip)]

    lines = markdown.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.startswith('| 算子与输入合同（逐窗） |'))
    headings = lines[start].split('|')
    columns = [headings.index(' 原 CA 窗逻辑时间系数（倍，DT / PR） '),
               headings.index(' 冷读写逻辑 U 范围（DT / PR） ')]
    seen = set()
    for i in range(start + 2, len(lines)):
        if not lines[i].startswith('|'):
            break
        cells = lines[i].split('|')
        link = re.fullmatch(r'\s*\[[^\]]+\]\(errors/([^/)]+\.md)\)\s*', cells[1])
        if not link:
            raise ValueError(f'Missing contract link in error summary: {cells[1]}')
        name = link[1]
        if name == 'measured-only.md':
            values = [f'{NA} / {NA}'] * len(columns)
        else:
            values = ranges[name]
            seen.add(name)
        for column, value in zip(columns, values):
            cells[column] = f' {value} '
        lines[i] = '|'.join(cells)
    if seen != ranges.keys():
        raise ValueError(f'Contracts missing from error summary: {sorted(ranges.keys() - seen)}')
    return ''.join(lines)


def policy_audit(groups):
    body = []
    for sku, label in [(DT, '950DT'), (PR, '950PR')]:
        cohort = [r for rows in groups.values() for r in rows if r['chip'] == sku]
        eligible = [r for r in cohort if r['cold_u'] is not None]
        original_over = [r for r in eligible if r['roofline']['utilization'] > 1]
        def exceed(engine):
            return sum(r['roofline'][engine] is not None and r['roofline'][engine] > r['predicted_us']
                       for r in cohort)
        body.append([label, f'{len(eligible):,} / {len(cohort):,}',
                     *window_ranges(eligible), len(original_over),
                     sum(r['roofline']['ideal_us'] > r['ca_us'] for r in original_over),
                     f"{exceed('cube_us')} / {exceed('vector_us')} / {exceed('hbm_us')}",
                     sum(r['cold_u'] > 1 for r in eligible),
                     sum(r['roofline']['cube_us'] is None for r in cohort)])
    return table(['芯片', '逻辑 U 可比 / 全部', '原窗 U 范围', '冷读写 U 范围',
                  '原窗 U >100%', '其中换 CA 分母仍 >100%', '各项越线 Cube / Vector / HBM',
                  '冷读写 U >100%', '缺 Cube 峰值'], body)


def refresh_policy_audit(markdown, groups):
    start, end = '<!-- roofline-policy-audit:start -->', '<!-- roofline-policy-audit:end -->'
    before, rest = markdown.split(start)
    _, after = rest.split(end)
    return f'{before}{start}\n\n{policy_audit(groups)}\n\n{end}{after}'


def index_table(groups, role):
    body = []
    combined = []
    for rows in groups.values():
        combined.extend(rows)
    for name, rows in [*groups.items(), (None, combined)]:
        cells = [f'[{name}]({page_name(name)})' if name else '**全部合同窗**']
        for chip in (DT, PR):
            selected = [r for r in rows if r['chip'] == chip and r['data_role'] == role]
            count = sum(r['roofline']['utilization'] is not None for r in selected)
            cells += [f'{count}/{len(selected)}', *window_ranges(selected)]
        body.append(cells)
    return table(['合同', 'DT 可比/完整窗', 'DT 原窗 U 范围', 'DT 冷读写 U 范围', 'PR 可比/完整窗', 'PR 原窗 U 范围', 'PR 冷读写 U 范围'], body)


def render_index(groups):
    count = sum(len(rows) for rows in groups.values())
    known = [(row['physical_roofline']['known_lower_bound_us'], row) for rows in groups.values()
             for row in rows if row['physical_roofline']['known_lower_bound_us'] is not None]
    below_model = sum(row['physical_roofline']['below_known_bound'] for _, row in known)
    below_ca = sum(row['ca_us'] < floor * (1 - 1e-12) for floor, row in known)
    sections = ['# 按合同查阅 CA 完整窗', f'共 {len(groups)} 个输入与计时合同、{count:,} 条记录。选合同可查看每次调用的本卡 shape、输入精度和前序、模型与 CA 设备耗时、查询耗时及 970DT 外推。同一 shape 的不同运行各占一行。',
                '可比窗指能够同时取得逻辑 Roofline 基准时间和同一输入完整模型时间的窗口。下表分芯片与角色取全部可比窗的最小值和最大值。两个 U 使用同一个逻辑基准；原窗的分母是默认排程时间，冷读写的分母是所有逻辑读写都经过 HBM 时的场景预测。CA 误差只对原窗计分；970DT 只有外推。',
                '[总误差与计时口径](../error-analysis.md) · [建模方法](../modeling-strategy.md) · [逐窗原始数据](../../calibration/reference/windows.jsonl)']
    sections.append(f'已知物理下界覆盖 {len(known):,}/{count:,} 个窗口，其中模型低于下界 {below_model} 窗、CA 低于下界 {below_ca} 窗。当前可核实的项为稠密 Cube 乘加。完整物理利用率还需要同窗 HBM 计数及适用的 Vector 指令吞吐。')
    for role, title in ROLES.items():
        sections += [f'## {title}', index_table(groups, role)]
    sections += ['## 只有 CA 实测的输入', '[FP8 cache 解包、非连续 FP4 量化、残差加和广播乘的双端整窗](measured-only.md)。这些输入还没有可独立评价的价格。', '## 其它精度与实现',
                '表内合同以参考数据的当前登记为准。表外精度与实现（如 A8W4 专家 GMM 或低精度 FA）应按其各自的完整窗确定 CA 误差与逻辑时间系数，缺少匹配证据时为 N/A。[精度与实现覆盖](../precision-coverage.md)']
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
        sections += [('BF16 全表融合参考排程在两端均缺完整 CA。A5 的 MXFP4 原生消费层另有固定 metadata 的完整设备窗；包含 AICPU、准备输入和后处理的部署端到端时间仍为外推。'
                      if contract == 'candidate_filtered_indexer' else
                      '950DT、950PR 均有 0 个合格完整窗。模型对照、完整窗误差和物理利用率为 N/A，970DT 无对应校准窗。')]
        evidence = {
            'attn_res_fwd · resident inference prefill': '此为 UB 驻留的推理 prefill，物化输出的样本列在另一个合同。[AttnResFwd 算例与证据边界](../examples/attn_res_fwd.md)',
            'candidate_filtered_indexer': 'BF16 图按 Hugging Face 参考的全表打分、候选掩码和 top-k 定价。A5 发布的 MXFP4 QSLI 先收集候选再用 Cube 打分。A5 的证据见[融合实现与组件对照](flash-a5-qsli.md)。两种显式分解有双端 CA，逐个 kernel 可查。[分解实测与核身份](../../calibration/candidate_indexer/README.md) · [候选 Indexer 算例](../examples/candidate-filtered-indexer.md)',
            'cache_load_mxfp4': '显式 AntiMxQuant 有[独立双端校准](cache_load_mxfp4_antimx.md)。系统捕获了 packed E2M1 值、按 32 元素分组并补齐的 E8M0 scale 和 BF16 物化输出。[缓存读取算例](../examples/cache_load.md)',
            'cache_load_fp8': '默认调用没有派发身份。显式 AscendAntiQuant 有原生实测，[FP8 的双端设备时间](measured-only.md)。系统捕获了 E4M3FN 值、逐行 FP32 scale 和 BF16 物化输出。[缓存读取算例](../examples/cache_load.md)',
            'routed_grouped_matmul_mxfp4': '显式 GroupedMatmulV5 有[双端校准](routed_grouped_matmul_mxfp4_v5.md)。系统捕获了路由专家的 A4W4 分组乘法，工作量取决于专家行分布。稠密 Matmul 的 CA 窗属于另一合同。[路由专家 GMM 算例](../examples/routed-grouped-matmul.md)',
        }
        if contract in evidence:
            sections.append(evidence[contract])
        return '\n\n'.join(sections) + '\n'
    sections.append('模型和 CA 均为设备微秒（µs）。有符号误差为 `100 × (模型 / CA − 1)`。原窗逻辑 U 的分子是 `max(Cube、Vector、逻辑 HBM 的峰值时间)`，分母是默认排程设备耗时。冷读写列保留同一排程与分子，把分母换成 `max(原窗 µs, (逻辑读字节 + 逻辑写字节) / HBM 带宽)`，假设冷读且输出写回并充分重叠。冷读写没有对应的 CA 真值，只用于场景比较。物理利用率需要同窗 HBM 控制器流量，本页所有记录均为 N/A。两个逻辑 U 都未截断。查询列为归档 CPU 墙钟毫秒（ms），顺序为 950 / 970。970DT 两列均为原窗策略的外推。')

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
            count = sum(r['roofline']['utilization'] is not None for r in selected)
            summary.append([chip, label, error['shapes'], len(selected), number(error['wape_pct'],2),
                            number(error['max_pct'],2), count, *window_ranges(selected)])
    sections += ['## 完整窗与误差', table(['芯片','数据角色','不同 shape','完整窗','WAPE %','最大绝对误差 %','U 可比窗','原窗 U 范围','冷读写 U 范围'],summary),
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
                floor = row.get('physical_roofline', {}).get('known_lower_bound_us')
                body.append([ref,'×'.join(map(str,row['shape'])),row['input_dtype'],f'{attr} / {history}',link_id('K', kernels[implementation(row)]),
                             number(row['predicted_us']),number(row['ca_us']),number(row['signed_error_pct'],2,True),
                             f'{lower} / {bound}',fraction,number(row['cold_us']),percent(row['cold_u']),number(floor),number(row['extrapolated_970dt_us']),fu,
                             f"{number(row['query_ms'],3)} / {number(row['query_970_ms'],3)}"])
            sections += [f'## {chip} · {label}',table(['原始行','本卡 shape','主输入 dtype','属性 / 前序','实现','原窗模型 µs','CA µs','原窗误差 %','逻辑基准 µs / 控制项','原窗逻辑 U','冷读写模型 µs','冷读写逻辑 U','已知物理下界 µs','970 原窗外推 µs','970 原窗逻辑 U','查询 ms（950 / 970）'],body)]
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
    files = {output / filename: content for filename, content in pages.items()}
    docs = HERE.parent / 'docs'
    if output.resolve() == (docs / 'errors').resolve():
        summary = docs / 'error-analysis.md'
        overview = summary_ranges(summary.read_text(encoding='utf-8'), groups)
        files[summary] = refresh_policy_audit(overview, groups)
    if check:
        for path, content in files.items():
            if not path.is_file() or path.read_text(encoding='utf-8') != content:
                raise ValueError(f'Missing or stale report: {path}')
    else:
        output.mkdir(exist_ok=True, parents=True)
        for path, content in files.items():
            path.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, default=HERE.parent/'docs'/'errors')
    cli.add_argument('--check', action='store_true', help='Verify generated Markdown without writing files.')
    args = cli.parse_args()
    report(args.output, args.check)
