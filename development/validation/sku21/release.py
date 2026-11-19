"""Refresh the current calibration report from completed receipts without launching CA."""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import median
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model import __version__
from development.validation.sku21.acceptance import CHANGED, CHIPS, NAMES, OUT, read, report, score, write
from development.validation.sku19.campaign import digest


def snapshot():
    files = set((ROOT/'operator_model').rglob('*.py'))
    for line in (ROOT/'MANIFEST.in').read_text().splitlines():
        if line.startswith(('include results/','include configs/')):
            files.add(ROOT/line.split(maxsplit=1)[1])
    files.update(ROOT/name for name in ('pyproject.toml','setup.py','MANIFEST.in'))
    manifests = sorted(OUT.glob('*/freeze.json'))
    write(OUT/'model.json',dict(version=__version__,
        sources={str(p.relative_to(ROOT)):digest(p) for p in sorted(files)},
        freezes={str(p.relative_to(ROOT)):digest(p) for p in manifests},
        wheel_sha256=read(OUT/'wheel-check.json')['wheel_sha256'],
        qualification='Each campaign seals predictions before measurement. This release index fingerprints the assembled backend and those immutable registrations.'))


def pairs(out):
    by_key = defaultdict(dict)
    for r in out['rows']:
        if r['cohort']=='primary' and 'old_us' in r:
            key = (r['op'],tuple(r['shape']),json.dumps(r['attrs'],sort_keys=True))
            assert r['chip'] not in by_key[key], key
            by_key[key][r['chip']] = r
    values = []
    for (op,shape,_), endpoints in sorted(by_key.items()):
        if len(endpoints)!=2:
            continue
        dt,pr = (endpoints[CHIPS[side]] for side in ('dt','pr'))
        values.append(dict(op=op,shape=shape,attrs=dt['attrs'],dt_ca=dt['ca_us'],pr_ca=pr['ca_us'],
            dt_us=dt['us'],pr_us=pr['us'],dt_old=dt['old_us'],pr_old=pr['old_us'],
            speed_model=dt['us']/pr['us'],speed_ca=dt['ca_us']/pr['ca_us'],speed_old=dt['old_us']/pr['old_us'],
            direct_dt_as_pr_us=dt['us'],future_us=dt['future_us']))
    write(OUT/'pairs.json',dict(role='post-freeze unseen pairs; DT/PR is DT time divided by PR time. Direct reuse forecasts PR with the current DT time and predicts 1x.',rows=values))
    lines=['# 同一新 shape 的 DT、PR 时间和加速比','',
        '时间单位为 µs，加速比取 DT 耗时除以 PR 耗时。表中每一行的两端均已通过数值与设备窗口核对。把该行的 DT 预测原样移作 PR 预测，对应加速比恒为 1。970DT 仅有外推。','',
        '| 算子、shape | DT 旧 / 当前 / CA | PR 旧 / 当前 / CA | 加速比 旧 / 当前 / CA | 970DT 外推 |',
        '|---|---:|---:|---:|---:|']
    for r in values:
        lines.append(f"| {NAMES[r['op']]} {r['shape']} | {r['dt_old']:.3f} / {r['dt_us']:.3f} / {r['dt_ca']:.3f} | {r['pr_old']:.3f} / {r['pr_us']:.3f} / {r['pr_ca']:.3f} | {r['speed_old']:.3f} / {r['speed_model']:.3f} / {r['speed_ca']:.3f} | {r['future_us']:.3f} |")
    lines+=['','[单端先完成的 shape 和历史样本](shape-errors.md) · [可复算的成对数值](pairs.json)']
    (OUT/'pairs.md').write_text('\n'.join(lines)+'\n')
    return values


def changes(out,paired):
    measured = [r for r in out['rows'] if r['cohort']=='primary' and r['op'] in CHANGED]
    stats = read(OUT/'wheel-check.json')['overall']
    lines = ['# v0.21 方法与同集变化','',
        'Addmm 按目标 AIC 选择分波，并单独计算列偏置读入 L1、送入偏置缓冲的服务。Cast 与 Sigmoid 共用全片缓存、读写端口和发起间隔模型。mHC Sinkhorn 把同一向量发射可容纳的两行合并，PreSinkhorn 按目标核数分 K，AttnResFwd 按活动通道和缓冲访问计算尾段服务。RMSNorm 与 mHC Post 根据前序核驻留定价，系统可按本 rank 的派发次序维护状态。KDA chunk 增加变长序列、gate 和状态的工作量统计。','',
        f'本版八条有新时延预测的路线，各登记 10 个新 shape，双端共 160 个窗口。当前合格 {len(measured)} 个，形成 {len(paired)} 对。下表只在同一组已经完成的窗口上比较 v0.20 和 v0.21。每端满十个不同 shape 后再判断独立组是否达标。','',
        '| 算子 | 芯片 | 已实测 / 登记 | 旧 WAPE / 最大 | 当前 WAPE / 最大 |','|---|---|---:|---:|---:|']
    for op in NAMES:
        if op not in CHANGED:
            continue
        for side in ('dt','pr'):
            group = [r for r in measured if r['op']==op and r['chip']==CHIPS[side]]
            old,new = score(group,'old_us'),score(group)
            def cell(m):
                return f"{m['wape']:.2%} / {m['max_ape']:.2%}" if m else 'N/A'
            lines.append(f'| {NAMES[op]} | {side.upper()} | {len(group)} / 10 | {cell(old)} | {cell(new)} |')
    costs = []
    for root in ('vector21','fusion21','dispatch21','rms21','fusion21-retry','dispatch21-retry'):
        for p in sorted((ROOT/'runs'/root).glob('*/cost.json')):
            item = read(p)
            seconds = item.get('seconds',item.get('wall_s'))
            if seconds is not None:
                costs.append(dict(source=str(p.relative_to(ROOT)),seconds=seconds,stop=item.get('stop_reason')))
    write(OUT/'collection-cost.json',dict(scope='available terminal host cost records in campaign run folders, includes failures; concurrent wall times do not add to elapsed time',rows=costs))
    lines+=['',f"隔离安装的公开 Python 入口共 {stats['n']} 种输入，中位 {stats['median_ms']:.3f} ms，P95 {stats['p95_ms']:.3f} ms，最慢 {stats['max_ms']:.3f} ms。"]
    if costs:
        times=[r['seconds'] for r in costs]
        lines+=[f'目前收齐的 {len(times)} 份宿主成本回执（含失败）里，CA 墙钟中位 {median(times):.1f} 秒，最长 {max(times):.1f} 秒。多队列并行，按单任务计时。']
    lines+=['','### 大输入的采集范围','',
        'B8×32k 的全宽 Cast 输入为 [262144,5120]，读 2.5 GiB、写 5 GiB，预测在 64 个 DT AIV 上每核最多 993 轮。原 v0.20 预测 2014.328 µs 在启动前冻结；v0.21 的 2514.820 µs 是运行中的后续预测。记录全核 wrapper 从开始到完成的设备窗口，另要求 1,342,177,280 个输出逐元素校验。原 4 小时预算按同一进程增加 2.5 小时，原始数据仍限 8 GiB。']
    terminal=OUT/'large/full-hidden-terminal.json'
    if terminal.exists():
        native=read(terminal)
        lines.append(f"终态墙钟 {native['cost']['seconds']/3600:.2f} 小时。"+(f"完整 core wrapper 窗口 {native['core_envelope_us']:.3f} µs。" if native['qualified_core_envelope'] else '本次未取得完整且正确的全核窗口，CA 时间为 N/A。'))
    else:
        lines.append('全宽 Cast 仍在进行，CA 时间待测。')
    lines+=['同一低开销队列登记 Kimi 的动态量化 [262144,7168]，独立于已经取到全指令窗口的窄输出 Cast、Sigmoid。[全宽 Cast 续测依据](large/full-hidden-continuation.json) · [Kimi 量化采集合同](../v0.20/full-quant/freeze.json)','',
            '另以 Addmm [16,64,256] 检查单 CPU worker 能否替代默认采集。DT 的数值、二进制、核数和 MMAD 计数一致，但全指令窗口由 3.588 µs 变成 3.448 µs，超过事前设定的一个设备时钟容差。PR 单 worker 在 420 秒上限前未完成。本次独立统计继续使用原来的采集口径。[DT 对照](dispatch/host-workers-dt.json) · [PR 对照](dispatch/host-workers-pr.json)']
    lines+=['','无历史输入的 mHC Post 在历史批次仍超误差门槛。KDA chunk 缺少通过校验的主 kernel 设备窗口，所有芯片耗时填 N/A。完整大 hidden 的 B8×32k 和稀疏注意力分支按实际采集状态披露；长序列小宽度的投影、Cast、Sigmoid 有单独的完整 CA。970DT 全部为架构情景外推。','',
        '[建模方法](../../docs/modeling-strategy.md) · [总体误差与超限样本](../../docs/error-analysis.md) · [按 tile 或阶段上下对照](../../docs/pipeline-comparisons.md) · [双端耗时与加速比](pairs.md) · [预测开销](wheel-check.json) · [CA 成本](collection-cost.json) · [模型指纹](model.json)']
    (OUT/'changes.md').write_text('\n'.join(lines)+'\n')


def inputs():
    files = [OUT/'fusion/evaluation.json',OUT/'vector/evaluation.json',OUT/'vector/holdout.json',
             ROOT/'results/v0.20/rms/records.json',OUT/'wheel-check.json',OUT/'dispatch/stream-replay.json']
    for folder in ('dispatch','rms'):
        files.extend(sorted((OUT/folder/'receipts').glob('*.json')))
    files.extend(sorted((OUT/'dispatch/retries').glob('*.json')))
    files.extend(sorted((OUT/'large').glob('*terminal.json')))
    return sha256(b''.join(f.read_bytes() for f in files)).hexdigest()


def refresh():
    result=report()
    paired=pairs(result)
    changes(result,paired)
    snapshot()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--follow',action='store_true')
    follow=parser.parse_args().follow
    previous=None
    while (ROOT/'operator_model/__init__.py').read_text().find("0.21.0")>=0:
        current=inputs()
        if previous!=current:
            refresh()
            previous=current
        if not follow:
            break
        time.sleep(60)


if __name__=='__main__':
    main()
