"""Compare Vector operators on identical rank-local shapes with existing whole-device CA."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if not sys.flags.isolated:
    sys.path.insert(0,str(ROOT))
from operator_model import estimate, __version__

OUT = ROOT/'results/v0.28'
SOURCES = {'quant':'results/v0.25/coverage/audit.json','rmsnorm':'results/v0.22/evaluation.json'}


def collect(op, path):
    groups = defaultdict(list)
    for row in json.loads((ROOT/path).read_bytes())['rows']:
        if row['op'] != op or row.get('ca_us') is None:
            continue
        if op == 'rmsnorm' and row['attrs']:
            continue
        receipt = row['receipt'] if op == 'rmsnorm' else row
        group = (row['chip'],tuple(row['shape']))
        observation = dict(ca_us=row['ca_us'], trace=receipt['trace_sha256'], kernel=receipt['kernel'])
        if observation not in groups[group]:
            groups[group].append(observation)
    return groups


def main():
    if __version__ != '0.27.0':
        raise RuntimeError('Run with python -I from the archived v0.27 wheel environment')
    quant = collect('quant',SOURCES['quant'])
    rms = collect('rmsnorm',SOURCES['rmsnorm'])
    shapes = sorted(quant.keys() & rms.keys())
    results=[]
    for chip, shape in shapes:
        for op, raw in [('quant',quant),('rmsnorm',rms)]:
            result=estimate(op,shape,chip=chip)
            cases=raw[chip,shape]
            low,high=min(v['ca_us'] for v in cases),max(v['ca_us'] for v in cases)
            results.append(dict(op=op,chip=chip,shape=shape,latency_us=result.latency_us,ca_range_us=[low,high],
                                ape_range_pct=sorted([100*abs(result.latency_us/low-1),100*abs(result.latency_us/high-1)]),
                                active_aiv=result.active_aiv,observations=cases))
    checksums={f:sha256((ROOT/f).read_bytes()).hexdigest() for f in SOURCES.values()}
    (OUT/'vector-matched.json').write_text(json.dumps(dict(role='Historical comparison of matching shape and target; invocation order, output dtype and work differ. Ranges retain all distinct archived observations.',sha256=checksums,rows=results),indent=2)+'\n')
    lines=['# 同样的 shape，RMSNorm 与 MXQuant 的偏差','',
           'BF16 输入，调用方只给 shape 和芯片。两套历史归档有四种两端相同的几何。下表列 v0.27 公开入口的原预测和该几何归档 CA 的范围，设备时间单位均为 µs。两种库算子的输出、指令依赖和前序协议各自成立，表内对比只控制了芯片和输入几何。','',
           '| 每卡 [token, 通道] | 芯片 | DynamicMXQuant 预测 / CA | RMSNorm 预测 / CA | MXQuant / RMS 活动 AIV |',
           '|---|---|---:|---:|---:|']
    def times(row):
        lo,hi=row['ca_range_us']
        observed=f'{lo:.3f}' if round(lo,3)==round(hi,3) else f'{lo:.3f}–{hi:.3f}'
        return f"{row['latency_us']:.3f} / {observed}"
    for q,r in zip(results[::2],results[1::2]):
        lines.append(f"| {list(q['shape'])} | {q['chip']} | {times(q)} | {times(r)} | {q['active_aiv']} / {r['active_aiv']} |")
    lines.extend(['','当行数只有 1、宽为 128/256 时，RMSNorm 的无前序点在两端都明显偏低。到 [36,5120]，950PR 的 RMSNorm 比 Quant 更接近 CA。算子名称和 Vector 峰值都不足以解释这个差异。','',
                  'RMSNorm 归约一整行才能获得 rstd；归一化输出和小体积统计量有两个完成边界。MXQuant 按 32 个元素的微缩放组计算并写量化值与 scale，组之间可以流水推进。二者都可能受 MTE、首次和局部写竞争影响，但参与关键路径的依赖不同。对于大 Kimi Quant，超过 L2 后主要需要解释共享读写和控制器的空数据拍。[Quant 大形](../v0.27/quant-channel-service.md) · [RMSNorm 逐核读与前序](../v0.27/rms/report.md) · [算子身份、源文件与全部观测](vector-matched.json)'])
    (OUT/'vector-matched.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
