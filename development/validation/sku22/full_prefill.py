"""Show the separate, low-volume whole-hidden calibration jobs as their terminal receipts arrive."""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from operator_model import estimate
from development.validation.sku19.campaign import CHIPS, read, write


def publish(archive):
    cases=[('Flash','cast_bf16_fp32',[262144,5120],'results/v0.21/large/full-hidden-terminal.json','runs/prefill20-native/cast_262144_5120/run.log'),
           ('Kimi','quant',[262144,7168],'results/v0.20/full-quant/native.json','runs/kimi-full-quant20/run.log')]
    rows=[]
    for model,op,s,src,log in cases:
        rec=read(archive/src) if (archive/src).exists() else None
        if rec is None:state='已启动，等待数值及全核窗口' if (archive/log).exists() else '排在前一项之后'
        elif rec['qualified_core_envelope']:state='完整数值与全核 wrapper 包络'
        else:state='已结束，未取得合格包络'
        forecasts={c:estimate(op,s,chip=c) for c in CHIPS.values()}
        rows.append(dict(model=model,op=op,shape=s,state=state,source=src,measured_envelope_us=rec.get('core_envelope_us') if rec and rec['qualified_core_envelope'] else None,
                         measured_instruction_us=None,terminal=rec is not None,active={c:r.active_aiv for c,r in forecasts.items()},
                         predicted_us={c:r.latency_us for c,r in forecasts.items()},cost=rec.get('cost') if rec else None,
                         logical_read_bytes=forecasts[CHIPS['dt']].work['logical_read_bytes'],logical_write_bytes=forecasts[CHIPS['dt']].work['logical_write_bytes']))
    out=ROOT/'results/v0.22'
    if (out/'full-prefill.json').exists() and (out/'full-prefill.md').exists() and read(out/'full-prefill.json')==rows:
        return all(r['terminal'] for r in rows)
    write(out/'full-prefill.json',rows)
    doc=['# B8 × 32k 整批 · 真实 hidden 的低输出量 CA','', '每条请求 32,768 个 token，单卡合并 8 条，共 262,144 行。这里使用 Flash 的 hidden=5120 和 Kimi 的 hidden=7168。低输出量采集保留全部结果的数值核验及每颗 AIV 的运行边界；这种采集不导出逐指令时间线。表中的模型预测针对指令窗口，原生 CA 一列是额外的 wrapper 全核包络，单独统计。两项测试覆盖指定独立库调用，实际部署是否物化 Cast 或 MX 量化还取决于量化方法和融合选择。','', '| 模型、算子、每卡 shape | DT 预测 µs | DT 原生包络 µs | PR 预测 µs | 970DT 外推 µs | 状态 |','|---|---:|---:|---:|---:|---|']
    for r in rows:
        pred=r['predicted_us'];obs=r['measured_envelope_us'];v=f'{obs:.3f}' if obs is not None else 'N/A'
        doc.append(f"| {r['model']} {r['op']} {r['shape']} | {pred[CHIPS['dt']]:.3f} | {v} | {pred[CHIPS['pr']]:.3f} | {pred[CHIPS['future']]:.3f} | {r['state']} |")
    doc+=['','在每卡 B8×256k 下已有完整逐指令对照的窄张量，Flash 为 indexer 投影输出 `[2097152,32]`，Kimi 为 beta `[2097152,12]`。上述 full-hidden 合同使用全通道激活，两者在同一层可以对应不同位置。[Flash 投影和 hidden 区别](../../docs/workloads/deepseek-flash.md) · [逐算子模型与实测图](../../docs/pipeline-comparisons.md) · [运行状态及逐芯片字节数](full-prefill.json)']
    (out/'full-prefill.md').write_text('\n'.join(doc)+'\n');return all(r['terminal'] for r in rows)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive',type=Path,default=ROOT);parser.add_argument('--follow',action='store_true');args=parser.parse_args()
    while not publish(args.archive.resolve()) and args.follow:time.sleep(60)
