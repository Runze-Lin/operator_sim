"""Calculate scaling of one real weight, with measured and projected time kept distinct."""
import json
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/v0.32/matmul'
CHIPS = ('Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT')
LABELS = ('950DT','950PR','970DT 外推')
CONTEXT = (32768,65536,131072,262144)


def measure(installation):
    sys.path.insert(0, str(Path(installation).resolve()))
    import operator_model
    if not sys.flags.isolated or operator_model.__version__ != '0.32.0':
        raise RuntimeError('Use -I and the frozen v0.32 installation')
    score = json.loads((OUT/'score.json').read_bytes())['rows']
    data, table = [], []
    for context in CONTEXT:
        shape = [8*context,128,512]
        for chip, label in zip(CHIPS,LABELS):
            result = operator_model.estimate('matmul', shape, dtype='bf16', chip=chip)
            plan = result.projection['target_plan']
            diagnostic = result.projection['timing_diagnostics']
            measured = [r for r in score if r['shape']==shape and r['chip']==chip]
            ca = median(r['ca_us'] for r in measured) if measured else None
            work = diagnostic['internal_work']
            d = dict(batch=8, prompt_tokens=context, chip=chip, shape=shape, waves=plan['waves'],
                     mmads=plan['mmads_total'], cube_ops=result.work['cube_ops'],
                     useful_input_bytes=result.work['logical_read_bytes'], useful_output_bytes=result.work['logical_write_bytes'],
                     modeled_l2_output_bytes=work['possible_dirty_output_bytes'], modeled_us=result.latency_us,
                     stages_us=diagnostic['stages_us'], wave_segments=diagnostic['service_segments'], ca_us=ca,
                     measured_rows_per_us=shape[0]/ca if ca else None, sources=[r['source'] for r in measured])
            data.append(d)
            actual = '—' if not measured else f"[{ca:.3f}](../../../{measured[0]['source']})"
            table.append(f"| {context//1024}Ki | {shape[0]:,} | {label} | {d['waves']} | "
                         f"{d['modeled_l2_output_bytes']/2**20:.1f} | {result.latency_us:.3f} | {actual} | "
                         +(f"{shape[0]/ca:,.0f}" if ca else '—')+" |")
    opts = json.loads((OUT/'freeze.json').read_bytes())
    report = ['# 八条等长请求，共用同一组投影权重','',
              '取 DeepSeek-V4.1-Flash 的 BF16 indexer.wk，K=512、每卡 N=128。每批 B=8，把每条请求当前全部 prompt 一次交给此 kernel。'
              '有效乘加工作量为 2MNK ops，输入 A 每行 1024 B、输出 C 每行 256 B，权重只有 128 KiB。每增加一行，算术和张量字节固定增加。'
              '设备时间还取决于每波的核数、读写是否同时占用带宽，以及有多少 C 可以留在 L2。服务配置启用分块 prefill 时，换成本次真正提交的行数。','',
              '| 每请求 token | 本卡 M | 芯片 | 预测波数 | 模型可留 L2 的 C（MiB） | 预测 µs | 完整 CA µs | 实测行/µs |',
              '|---:|---:|---|---:|---:|---:|---:|---:|',*table,'','## 大形会不会趋近线性','']
    for chip,label in zip(CHIPS[:2],LABELS):
        rows=[d for d in data if d['chip']==chip and d['ca_us'] is not None]
        for small,big in zip(rows,rows[1:]):
            delta_m=big['shape'][0]-small['shape'][0]
            slope=1000*(big['ca_us']-small['ca_us'])/delta_m
            ratio=big['ca_us']/small['ca_us']
            forecast=1000*(big['modeled_us']-small['modeled_us'])/delta_m
            report.append(f"- {label}，{small['prompt_tokens']//1024}Ki → {big['prompt_tokens']//1024}Ki。"
                          f"行数 ×{big['shape'][0]/small['shape'][0]:g}，CA 时长 ×{ratio:.2f}，"
                          f"这一区间每新增一行平均增加 {slope:.3f} ns，模型为 {forecast:.3f} ns。")
    report+=['','有限差分包含两次测量之间的容量与尾波变化。模型在同一分支、L2 脏输出已饱和、重复满波占主导时，'
             '每增加一整波的资源工作近似固定，因而预测呈分段线性。两点能给区间斜率，是否进入固定斜率需要更多同权重长形的完整窗。'
             '950DT、950PR 的已测行保留原回执，970DT 全部来自处理器配置外推。',
             '',f"64Ki/256Ki 两端预测已冻结，模型登记时间为 {opts['utc']}。"
             '[资源公式与波级算例](method.md) · [全部分支与新形回执](report.md) · [机器可读的分段、三端数值及来源](scaling.json)','']
    (OUT/'scaling.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (OUT/'scaling.md').write_text('\n'.join(report))


if __name__=='__main__':
    measure(sys.argv[1])
