"""Refresh the v0.20 release snapshot using only finished, local CA receipts."""
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku20.report import render, summary, display, CHIPS, V, LABEL, read, write


def release():
    rows=render()
    wheel=read(V/'wheel-smoke.json');q=wheel['overall']
    frozen=[V/'rms/freeze.json',V/'large/cast-v20-freeze.json',V/'newop/holdout-freeze.json']
    files=[p for base in ('operator_model','configs/chips') for p in (ROOT/base).rglob('*')
           if p.suffix in ('.py','.json')]
    files.extend(ROOT/line.removeprefix('include ') for line in (ROOT/'MANIFEST.in').read_text().splitlines() if line.startswith('include '))
    write(V/'model.json',dict(version='0.20.0', prior_commit='c057e1efc2148567c4e2e3cefa6281dc62ea58d1',
        installed_wheel_sha256=wheel['installed_wheel_sha256'],
        source_and_profiles={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(set(files+frozen))},online_dependencies='Python >=3.10, packaged JSON; no CANN'))
    bench=f"{q['n']:,} 个不同的单次查询。中位 {q['median_ms']:.3f} ms，P95 {q['p95_ms']:.3f} ms，最慢 {q['max_ms']:.3f} ms"
    readme=ROOT/'README.md'
    text=re.sub('对三种芯片做了 [^。]+。中位 [^；]+；导入另计 [0-9.]+ ms',
        '对三种芯片做了 '+bench+f"；导入另计 {wheel['import_ms']:.2f} ms",readme.read_text())
    readme.write_text(text)
    strategy=ROOT/'docs/modeling-strategy.md'
    text=re.sub(r'当前 v0.20 在独立安装的 wheel 上测了 [^\n]*',
        '当前 v0.20 在独立安装的 wheel 上对三种芯片做了 '+bench+'。包括 8×256k 的整批输入。'
        '[逐算子调用开销](../results/v0.20/wheel-smoke.json)',strategy.read_text())
    strategy.write_text(text)
    def errs(m):return 'N/A' if m['n']==0 else f"{m['wape']:.2%} / {m['max_ape']:.2%}"
    lines=['# v0.20 · 整批 prefill、末核时间轴与写回边界','',
        '按三份模型配置及固定版本 vLLM Ascend 导出 B8 下的 32k、128k、256k，每个 kernel 的 head_dim、hidden、TP/DP/EP、KV 与调度截面随模型计算。54 份形状清单附层号和条件分支，27 次外层 decoder 符号执行核对调用次数。候选融合节点单独标记；线上预测无需权重或 CANN。[真实几何](../../docs/prefill-shapes.md) · [固定依赖和范围](workloads/source-lock.json)','',
        'RMSNorm 与 BF16→FP32 Cast 现分别计算每核 MTE、共享 L2 及在设备窗口内必须离芯片的字节。输入预留和目标容量限制可留输出。新增 Kimi KDA 的独立 FP32 beta Sigmoid 及其工作量、分核、UB 四缓冲、exp/div 和 DMA 定价。参考模型的单请求 shape 上限现在与整批 quant／BF16 matmul 的资源外推分开。[全部公式与算例](../../docs/modeling-strategy.md)','',
        '复用了 MatMulV3 两波各两页的真实 trace。各计时路线现在有同轴上下图，分别标出模型关键路径或可识别的阶段时长、全核窗口与真实末核指令。KDA chunk 缺少合格的 CA 窗口。[Matmul](../../docs/examples/matmul.md) · [Quant 五轮 UB](../../docs/examples/quant.md) · [Kimi 32k 整批](../../docs/examples/sigmoid_fp32.md)','',
        '| 新冻结独立集 | 已完成 DT / PR / 成对（登记 shape） | DT WAPE / 最大 | PR WAPE / 最大 | DT/PR 倍速 WAPE / 最大 |',
        '|---|---|---|---|---|']
    for op,title in LABEL.items():
        cohort=[r for r in rows if r['op']==op and r['role']=='独立'];d=summary(cohort)
        lines.append(f"| {title} | {d['dt']['n']} / {d['pr']['n']} / {d['speed']['n']}（{len(cohort)}） | {errs(d['dt'])} | {errs(d['pr'])} | {errs(d['speed'])} |")
    cost=[r.get(s+'_observation',{}).get('seconds') for r in rows for s in ('dt','pr')]
    wall=[x for x in cost if x]
    lines+=['','[同集 v0.19 对照、逐 shape、970DT 外推和全部路线摘要](../../docs/error-analysis.md) · [冻结模型与输入指纹](model.json) · [数值、时间和数据角色](evaluation.json)','',
        f"独立安装的 wheel 用三种芯片执行 {bench}。以上 {len(wall)} 个已完成且有时间的正式窗口，CA 单次宿主墙钟 {min(wall)/60:.1f}–{max(wall)/60:.1f} 分钟；各进程累计 {sum(wall)/3600:.2f} 小时，并发时间重叠。"
        '[预测墙钟](wheel-smoke.json) · [RMS 每端](rms/costs.json) · [Cast 原始成本及低记录量实验](large/report.md) · [Sigmoid](newop/measurements.md)','',
        '32k Flash Cast 的 DT 原生通道记录显示 16 MiB 输入产生约 16.04 MiB 片外读，写命令为零。对同形 PR 的 Cast，旧误差 +162.9%，本次开发重放 −7.3%。RMSNorm [2048,5120] 的 PR 开发重放从 +77.64% 到 −6.46%。对 Sigmoid 的写回敏感性只用于诊断；正式表保留采集前冻结的模型。整批 [262144,5120] 另走低记录量 64 核原生 wrapper 协议，wrapper 启动等待独立于规范指令窗口，单列记录。未完、失去资格和没有 970 CA 的情形均不计入 WAPE。','']
    (V/'changes.md').write_text('\n'.join(lines))

if __name__=='__main__':release()
