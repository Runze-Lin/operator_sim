"""List every packed-quant shape in a rank-0 decode export, checking layout before linking CA."""
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from operator_model import estimate
from operator_model.operators.quant.mxfp4 import contiguous_input
from calibration.__main__ import FUTURE, HERE, require
from calibration.report_pages import DT, PR, table


def captured_shapes(path):
    groups = defaultdict(Counter)
    with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as src:
        for node in csv.DictReader(src):
            if node['raw_name'] != 'npu::npu_dynamic_mx_quant' or int(node['history_tokens']) != 131072:
                continue
            x, = json.loads(node['inputs'])
            y, scale = json.loads(node['outputs'])
            spec = json.loads(node['attributes'])['semantic']
            shape = x['logical_shape']
            location = f"{node['case']}/{node['stage']}#{node['node_id']}"
            require(int(node['rank']) == 0 and x['dtype'] == 'torch.bfloat16' and spec['round_mode'] == 'round'
                    and spec['group_size'] == 32 and spec['padding'] == [0, 0]
                    and spec['logical_dtype'] == 'fp4_e2m1' and spec['scale_dtype'] == 'float8_e8m0fnu'
                    and shape == x['physical_shape'] == y['logical_shape'] == spec['logical_shape']
                    and y['dtype'] == 'torch.float4_e2m1fn_x2' and scale['dtype'] == 'torch.float8_e8m0fnu'
                    and y['physical_shape'] == shape[:-1] + [shape[-1]//2]
                    and scale['physical_shape'] == shape[:-1] + [(shape[-1]+63)//64, 2],
                    f'Unexpected quant encoding at {location}')
            dense = contiguous_input(shape, x['stride'])
            groups[tuple(shape)][dense] += 1
    require(groups, 'No packed block32 quant calls in the selection')
    return groups


def render(groups):
    measured = defaultdict(list)
    with (HERE/'reference'/'windows.jsonl').open() as src:
        for line in src:
            row = json.loads(line)
            if row['contract'] == 'quant_mxfp4':
                measured[tuple(row['shape']), row['chip']].append(row['ca_us'])
    dense = sum(g[True] for g in groups.values())
    views = sum(g[False] for g in groups.values())
    matched = sum(g[True] for shape, g in groups.items() if (shape, DT) in measured and (shape, PR) in measured)
    body = []
    for shape, counter in sorted(groups.items()):
        prices = []
        for chip in (DT, PR, FUTURE):
            if counter[True]:
                result = estimate('quant_mxfp4', shape, dtype='bf16', chip=chip)
                require(result.latency_us is not None, f'Unexpected unsupported contiguous geometry {shape}/{chip}')
                prices.append(f'{result.latency_us:.3f}')
            else:
                prices.append('N/A')
        reference = []
        for chip in (DT, PR):
            values = measured.get((shape,chip), []) if counter[True] else []
            reference.append(', '.join(f'{us:.3f}' for us in values) or 'N/A')
        body.append(['×'.join(map(str,shape)), counter[True], counter[False], *prices, *reference])
    return '\n\n'.join(['# 128Ki decode 系统中的全部 packed FP4 量化形状',
        f'27 个常规 run 的 rank 0 共调用 {dense+views:,} 次，覆盖 {len(groups)} 个不同的原始 shape。按输入的元素 stride 核对，其中 {dense:,} 次连续，{views:,} 次读取非连续视图。连续输入中有 {matched:,} 次与双端 CA 的维度和布局相符；计数按系统调用次数加权，同一 CA 值只代表该几何的一次实测。系统没有记录 scale_alg，映射时假设 OCP（0）。',
        '时间单位为设备 µs。表中的预测和 CA 只适用于同一行的连续调用。非连续视图的时间均为 N/A。未取得完整 CA 的连续 shape 给出资源外推；970DT 全部为外推。相同算子在系统账本上的时间来自系统后端，未用作真值。单维长度为 1 时该轴 stride 可任取，其他轴必须满足行主序连续性。',
        table(['本卡 BF16 输入 shape','连续调用','视图调用','950DT 预测','950PR 预测','970DT 外推','950DT CA','950PR CA'], body),
        '执行 `python -m calibration.system_quant_shapes "$EXPORT"` 可从导出的 `modeling-inputs.csv.gz` 复算本页。另一个 history=131198 的边界样例单独保留。[算子覆盖盘点](system-workload-coverage.md) · [逐 CA 窗与 Roofline 利用率](errors/quant_mxfp4.md) · [模型和校准算例](examples/quant-mxfp4.md)']) + '\n'


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('export', type=Path, help='decode-950dt-128k export directory')
    cli.add_argument('--output', type=Path, help='Output file (default: stdout)')
    args = cli.parse_args()
    report = render(captured_shapes(args.export/'modeling-inputs.csv.gz'))
    if args.output:
        args.output.write_text(report, encoding='utf-8')
    else:
        print(report, end='')
