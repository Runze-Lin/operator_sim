"""Read qualified Kimi decode traces on the vendor-wide clock, with no new CA."""
from collections import defaultdict
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model import estimate

OUT = ROOT / 'results/v0.31/matmul/kimi'
DECODE = ROOT / 'results/v0.29/matmul/decode'


def inspect(path, shape, chip):
    receipt = json.loads(path.read_bytes())
    if not receipt.get('qualified'):
        return None
    stage, = receipt['stages']
    pred = estimate('matmul', shape, chip=chip)
    plan = pred.projection['target_plan']
    stages = pred.projection['stages_us']
    if plan['waves'] != 1 or stages.get('shared_link_floor') or stages.get('additional_output_waves'):
        return None
    source = (ROOT / stage['trace']).read_bytes()
    assert sha256(source).hexdigest() == stage['trace_sha256']
    events = [e for e in json.loads(source)['traceEvents'] if e['ph'] == 'X']
    core = max(stage['windows'], key=lambda name: stage['windows'][name][1])
    reads = [e['ts'] for e in events if e['pid'] == core and e['name'] == 'MOV_OUT_TO_L1_MULTI_ND2NZ']
    mm = sorted((e for e in events if e['name'] == 'MMAD' and e['pid'] == core), key=lambda e:e['ts'])
    meta = [json.loads(e['args']['detail']) for e in mm]
    if not mm or not reads or sum(p['k'] for p in meta) != shape[2]:
        return None
    depth = 0
    pages = []
    for e, p in zip(mm, meta):
        if depth % plan['read_page_k'] == 0:
            pages.append(e['ts'])
        depth += p['k']
    first = min(e['ts'] for e in events)
    last = max(e['ts'] + e['dur'] for e in events)
    knots = [first, min(reads), pages[0], pages[-1], max(e['ts'] + e['dur'] for e in mm), last]
    observed = [b-a for a,b in zip(knots, knots[1:])]
    modeled = [stages['launch'], stages['read_fill']+stages['l0_lead'], stages['page_intervals'],
               stages['last_mmad'], stages['write_tail']]
    assert min(observed) >= 0 and isclose(sum(observed), receipt['ca_us'], abs_tol=.002)
    assert isclose(sum(modeled), pred.latency_us, abs_tol=1e-8)
    return dict(shape=shape,chip=chip,receipt=str(path.relative_to(ROOT)),trace_sha256=stage['trace_sha256'],
                last_core=core,physical_kernel=stage['kernel'],active_aic=stage['active_aic'],
                issued_mmad=stage['mmad'],observed_mmad_k=[p['k'] for p in meta],page_k=plan['read_page_k'],
                page_intervals_us=[b-a for a,b in zip(pages,pages[1:])],
                model_us=modeled,ca_us=observed,model_total_us=pred.latency_us,device_us=receipt['ca_us'])


def render():
    rows = []
    archived = defaultdict(list)
    for j in json.loads((DECODE/'freeze.json').read_bytes())['targets']:
        shape = j['shape']
        for tag,chip in [('dt','Ascend950DT_9582'),('pr','Ascend950PR_957b')]:
            stem = '-'.join(map(str,shape))+'-'+tag
            paths = [DECODE/(stem+'.json'), ROOT/'results/v0.28/matmul'/(stem+'-drain.json')]
            if j['m'] == 9 and tag == 'dt':
                paths.insert(0,DECODE/'retry/receipt.json')
            for path in paths:
                if path.is_file() and (row := inspect(path,shape,chip)):
                    rows.append(row)
                    archived[chip].append(j['m'])
                    break
    lines = ['# Kimi MLA 查询投影的页发起间隔', '',
             '输入为 TP8 的连续 BF16 Matmul，N=2304，K=1536。每端沿用已合格的整个设备窗，'
             '取最后完成的 AIC，按实际 MMAD 的 K 深度累计到每页 256。每页首条 MMAD 的发起时差包含依赖和等待。'
             '表中每格为模型 / CA，单位 µs。首行启动从全片首指令量到关键核第一笔操作数读。', '',
             '| M | 芯片 | 启动 | 读至首条 MMAD | 第一至最后一页首发 | 末页计算退休 | 尾写 | 整窗 |',
             '|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        pairs = [f'{p:.3f} / {a:.3f}' for p,a in zip(r['model_us'],r['ca_us'])]
        lines.append(f"| {r['shape'][0]} | {r['chip'][6:11]} | {' | '.join(pairs)} | "
                     f"{r['model_total_us']:.3f} / [{r['device_us']:.3f}](../../../../{r['receipt']}) |")
    lines += ['', 'DT B16 的 18 个 AIC 共发 216 条 MMAD，与公开模型一致。关键核六页的五个首发间隔'
              '实测为 0.473、0.675、0.442、0.827、0.542 µs，均低于预测的 0.842 µs。'
              '这五个间隔合计贡献约 1.25 µs 高估，占全窗 1.79 µs 偏差的约 70%。'
              'DT 的 M=99 使用相同 K，关键核首末 MMAD 之间的预算已接近实测。'
              '给所有同 K 的 shape 使用相同折扣会破坏宽 M 的时间。当前证据能定位误差区间，不能单独量出 HBM 服务率。', '',
              'M=9 的 DT 两次尝试及 PR 一次均用完墙钟预算，未产生逐指令终窗和数值结果。'
              'DT 两次从派发日志可见 18 个 AIC，三份原始运行都加载与完整 M=11 对照相同的已安装 ELF。'
              'DT 重试中模拟器报 retire_sc 的断言在超时终止之后出现。采集的进度计数不能作为该形的设备耗时。'
              '[原始 DT](../../../v0.29/matmul/decode/9-2304-1536-dt.json) · '
              '[DT 重试](../../../v0.29/matmul/decode/retry/receipt.json) · '
              '[原始 PR](../../../v0.29/matmul/decode/9-2304-1536-pr.json)', '',
              '[逐页与总窗数据](pages.json) · [三模型完整场景](../scenarios.md)', '']
    assert 16 in archived['Ascend950DT_9582']
    (OUT/'pages.md').write_text('\n'.join(lines))
    (OUT/'pages.json').write_text(json.dumps(dict(rows=rows),ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    render()
