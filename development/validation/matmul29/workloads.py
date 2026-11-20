"""Join the application, decode and branch-boundary Matmul registrations by geometry."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.matmul28.overview import receipt
OUT = ROOT / 'results/v0.29/matmul'
PREVIOUS = ROOT / 'results/v0.28/matmul'
DT, PR = 'Ascend950DT_9582', 'Ascend950PR_957b'


def read(path):
    return json.loads(path.read_bytes())


def render():
    cases = {}
    for item in read(PREVIOUS / 'overview.json')['rows']:
        shape = tuple(item['shape'])
        endpoints = {c:dict(item['targets'][c], **receipt(item,tag)) for c,tag in ((DT,'dt'),(PR,'pr'))}
        cases[shape] = dict(name=f"{item['model']} · {item['node']}",
                            scenarios=[(item['phase'], item['requests'], item['tokens_per_request'])],
                            endpoints=endpoints, future=item['extrapolated_970dt_us'])

    decode = read(OUT / 'decode/freeze.json')
    prices = read(OUT / 'decode/score.json')['unique_rows']
    for call in decode['scenarios']:
        m = call['m']
        pair = prices[str(m)]
        shape = tuple(pair[DT]['shape'])
        case = cases.setdefault(shape, dict(name='Kimi-K3 · mla.q_b_proj', scenarios=[],
                                     endpoints=pair.copy(), future=pair[DT]['extrapolated_970_us']))
        dispatch = ('decode', call['batch'], call['query'])
        if dispatch not in case['scenarios']:
            case['scenarios'].append(dispatch)
    recovery = OUT / 'decode/retry/receipt.json'
    if recovery.exists():
        record = read(recovery)
        if record['qualified']:
            endpoint = cases[tuple(record['shape'])]['endpoints'][DT]
            # Do not overwrite a previously timed original attempt.
            if not endpoint.get('ca_us'):
                endpoint.update(ca_us=record['ca_us'], status='qualified on recovery')
    register = read(OUT / 'timing-freeze.json')
    observed = read(OUT / 'score.json')
    for item in register['rows']:
        if item['group'] != 'boundary':
            continue
        shape = tuple(item['shape'])
        pair = {}
        for chip in (DT, PR):
            timed = next((r for r in observed['rows'] if r['shape']==list(shape) and r['chip']==chip
                          and r['role']=='frozen timing'), None)
            pending = next((r for r in observed['pending'] if r['shape']==list(shape) and r['chip']==chip), {})
            pair[chip] = dict(us=item['old'][chip], ca_us=timed['ca_us'] if timed else None,
                              status='qualified' if timed else pending.get('status','pending'))
        cases[shape] = dict(name='Flash · indexer.weights_proj', scenarios=[('切 K 边界',None,None)],
                            endpoints=pair,future=item['old']['Ascend970DT'])
    paired = sum(all(case['endpoints'][chip].get('ca_us') for chip in (DT,PR)) for case in cases.values())
    lines = ['# 三个模型的单卡 Matmul 几何与实测', '',
             f"共 {len(cases)} 个不同的 [M,N,K]，其中 {paired} 个已有 DT、PR 同形完整 CA。每个几何只计一次。"
             '输入 X[M,K]、权重 W[K,N] 均为连续 BF16 ND。表中预测统一用公开路由，'
             '各条件候选的同集比较见分表。两个 SKU 的 CA 只有在数值、库实现和全核结束均通过时才填入。', '',
             '| 模型 · 投影 | 场景 B×Q | [M,N,K] | DT 公开 / CA (µs) | PR 公开 / CA (µs) | 970DT 公开外推 (µs) |',
             '|---|---|---|---:|---:|---:|']
    for shape, case in cases.items():
        cells = []
        for chip in (DT,PR):
            endpoint = case['endpoints'][chip]
            forecast = endpoint['us'] if 'us' in endpoint else endpoint['old_us']
            ca = endpoint.get('ca_us')
            result = (f'{ca:.3f}' if ca else '未获合格窗' if endpoint['status']=='unqualified' else '待返回')
            if endpoint['status']=='qualified on recovery':
                result += '（补测）'
            cells.append(f'{forecast:.3f} / {result}')
        labels = '、'.join(f'{phase} {b}×{q}' if b else phase for phase,b,q in case['scenarios'])
        lines.append(f"| {case['name']} | {labels} | {list(shape)} | {' | '.join(cells)} | {case['future']:.3f} |")
    lines += ['', 'Q 是本次派发中每个请求提交的行数。普通 decode 取 1，'
              'MTP 推 8 个 token 时主模型校验取 9。整批 B8×128Ki 与 B8×256Ki 的 M 分别为 '
              '1,048,576 和 2,097,152；使用 chunk 时从该次实际提交行数定价。表中时间只覆盖独立库调用。',
              '', '[基础 28 形与原始回执](../../v0.28/matmul/README.md) · [Kimi 同宽 decode 与补测](decode/host-placement.md) · '
              '[Kimi 逐批误差](decode/report.md) · [切 K 边界](report.md) · [大形权重常驻模型与 CA](full-load/report.md)', '']
    output = OUT/'workloads.md'
    tmp = output.with_suffix('.md.tmp')
    tmp.write_text('\n'.join(lines))
    tmp.replace(output)


if __name__ == '__main__':
    render()
