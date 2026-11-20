"""Publish the conditional Post audit and distinguish absent device windows."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.28/post'
def load(n):return json.loads((OUT/n).read_text())
def summary(g,key):
    return 'N/A' if not g else f"{g[key]['wape']:.2f}% / {g[key]['maximum']:.2f}%"


def main():
    replay=load('history-replay.json'); score=load('channel-score.json') if (OUT/'channel-score.json').exists() else None
    lines=['# 内置 mHC Post 的前序条件与误差','',
           '相同输入在首次、扩核、隔形复用和紧邻完全重复时有不同的控制与搬运成本。调用方现在可以分别查询这些情景、直接指定成功前序，或输入自己的状态分布。仅有 shape 时不会替调用方构造概率。','',
           '| 芯片 / 数据角色 | 窗 / 不同形 | 条件模型 WAPE / 最大 | v0.27 同流 WAPE / 最大 | 仅 shape WAPE / 最大 |',
           '|---|---:|---:|---:|---:|']
    for sku,group in replay['by_sku'].items():
        for key,label in [('v0.24 mixed history replay','混合前序开发回放'),('v0.27 original forward ordinary token','原冻结 token 换形/首次'),('published forward exact token repeat','原冻结 token 紧邻重复')]:
            r=group[key]
            lines.append(f"| {sku} / {label} | {r['windows']} / {r['shapes']} | {summary(r,'selected_us')} | {summary(r,'old_us')} | {summary(r,'stateless_us')} |")
        if score:
            r=score['metrics'][sku]['all_repeat']
            lines.append(f"| {sku} / 通道紧邻重复（原冻结及新增） | {r['n']} / {r['distinct_shapes']} | {summary(r,'candidate_us')} | {summary(r,'public_known_us')} | {summary(r,'public_shape_only_us')} |")
    lines.append(r"""
## 分核分支与状态

已安装 ops-transformer 的 BF16 `aclnnMhcPost` 使用四路输入及 residual 矩阵。每个输出为 $Y_{tid}=h^{out}_{td}h^{post}_{ti}+\sum_{j=1}^4 X_{tjd}H_{tji}$。一个 token 写回四路 BF16 输出。这里的向量工作是逐通道的流混合，与 RMSNorm 跨通道的平方和归约不同。vLLM 扩展的 `HcPost` 有独立实现合同。

令 $T,D,C$ 分别为 token 数、通道宽度与本片 AIV 数。$2T<C$ 时走通道分核入口 `_5`。单个 token 的片宽超过 256 且分片翻倍后仍不超过核数时将通道对半切。得到 $J$ 片，各片至多 $d=\lceil D/J\rceil$ 列。其余形状走 token 分核入口 `_6`，每个核至多 $\lceil T/C\rceil$ 行。芯片变化时重新选择入口。例如 `[29,4,320]` 在 64-AIV DT 走 `_5`，在 56-AIV PR 走 `_6`。

对 `_5` 紧邻重复，每片两笔有效输入为 $b=32\lceil 2d/32\rceil$ 与 $4b$ 字节，对应发起时刻 $0,g$。就绪时刻以周期表示为 $L=\max(l_0+bk/1024,g+l_1+4bk/1024)$。$k$ 为目标 MTE2 每 KiB 服务周期，缺失时继承 950 的有效端口价格。$g$ 为后一道读的发起间隔，$l_0,l_1$ 是扣除字节服务后的固定响应。直接复用 v0.24 CA 前冻结的两个开发窗口的价格，分别为 63.00、177.43 和 166.21 周期。加 26.00 周期的交接后进入向量段，按每片的 64 通道组数及目标每 AIV 的 FP32 吞吐缩放。四次输出写可逐笔在 CA 核对，现在的相同条件模型为写回提供聚合时间。整芯片 HBM 与 L2 字节服务下界独立检查。

对 `_6`，每行的四类读为 $64,16,2D,8D$ 字节，另有一次 $8D$ 写，宽度超 UB 容量时拆片。首次发起、首至次行发起及后续行稳态分别定价。上一条成功内置 Post 是通道入口还是 token 入口，决定普通换形的第一次发起与入口控制成本。后续的读、写发起间隔受向量工作量与搬运服务约束。价格缺少前一入口的特定观测时使用先前的状态价和可共享的稳态间隔。本次没有重新拟合总时间。[参数及原始 SHA](parameters.json)

$H$ 为同一设备同一有序流上调用方已知的同家族 AIV 高水位，$A$ 为本次预测的活动核数。$H=0$ 视作首次，$0<H<A$ 为扩核，$H\ge A$ 可选普通复用。紧邻同形还要求上一条已成功派发相同的 BF16、ND、四流、含 residual 的内置算子。只知 $H$ 无法判断精确重复。另一个内核执行之后不应把更早的相同 shape 传为紧邻前序。

## 算例

- **通道分核。** DT `[1,4,192]` 首次 CA 为 4.528 µs，紧邻重复为 1.742 µs。冻结端口模型给紧邻重复 1.889 µs，普通复用价为 2.280 µs。两个输入 384 B 和 1536 B 分别在 0.770 与 0.808 µs 发起，至 0.875 与 0.937 µs 完成。四笔 384 B 写分别对应四路输出，首末发起时刻为 1.205 和 1.631 µs，最后一笔于 1.736 µs 完成。首次/重复的实际控制阶段为 1.892/0.770 µs，读阶段为 0.979/0.183 µs，计算到写阶段为 1.447/0.678 µs。[逐笔地址与实测](../../v0.24/post/small-figure-data.json)
- **token 分核。** `[127,4,448] → [157,4,1088]` 在 DT 对应 `reuse_after_token`，预测 3.490 µs，CA 3.207 µs。同一顺序在 PR 对应 `expand_after_token`，预测 4.488 µs，CA 4.527 µs。[登记的预测和完整窗](../../v0.27/post/report.md)
- **尚未拆开的误差。** `[2,4,448]` 从一个 AIV 扩到四个，DT/PR 各自最后完成者均为新物理核，仅加入偏移就有 1.458/1.534 µs。`[1,4,192]` 隔形回访可以仍然接近热态。另一个 `[3,4,7168]` 回访则比同形紧邻热态明显慢。普通区间模型未进一步预判所有物理分区何时可用。[前序核名单及时间](../../v0.24/post/topology.md)

## 系统调用

```python
from operator_model.post_history import estimate_post_history

history = estimate_post_history([157, 4, 1088], chip='Ascend950DT_9582', dtype='bf16')
known_us = history.for_predecessor(64, [127, 4, 448]).latency_us
lower_us, upper_us = history.range_us
for state, case in history.scenarios.items():
    print(state, case.latency_us, case.stages_us)
# prior 是调用方对当前合同观测或显式假设的概率分布。
# average_us = history.expected_us(prior)
```

`expected_us` 只接受当前入口可行的状态、有限非负概率与总和 1，缺项概率视为零。970DT 或修改过的同名芯片有 `target_extrapolated=True` 标记。仅凭 shape 即使事后任选常数，同一组真实调用的 WAPE 下界仍为 DT 27.64%、PR 28.81%。这里返回的最小和最大值是模拟的状态范围。混合开发组的真实 CA 只有 DT 10/21、PR 12/22 落入该范围，不能用它来保证预测误差。[逐窗核对](history-replay.json) · [原不可辨识下界](../../v0.27/history-bounds.json)

## 三对独立小形

每形每 SKU 用一个新进程，在同一设备流真实执行首次和连续重复各一次。核对完整数值 golden、安装库 ELF、`_5` 入口、实际活动核、每核物理 END 和导出 END_LABEL、官方 duration、总写回字节。每次离线限 600 秒和 1 GiB。使用结果前冻结了两端三形及全部参数，禁止失格后补形。合并原合格的独立通道重复窗口后，预定两端各至少十个不同形，WAPE≤25%、最大≤50%，且同集 WAPE 优于旧普通复用才准入。[冻结合同](registration.json)
""")
    if score:
        lines+=['| SKU | 新 shape | 状态 | 预测 µs | CA µs | 相对误差 |','|---|---|---|---:|---:|---:|']
        for r in score['rows']:
            if r['cohort'].startswith('v0.28'):
                lines.append(f"| {r['chip']} | {r['shape']} | {r['role']} | {r['candidate_us']:.3f} | {r['ca_us']:.3f} | {100*(r['candidate_us']/r['ca_us']-1):+.2f}% |")
        receipts = sorted((OUT/'receipts').glob('*.json'))
        lines += ['', '| SKU 与形状 | 离线墙钟秒 | 峰值 MiB | 结果 |',
                  '|---|---:|---:|---|']
        for path in receipts:
            receipt = json.loads(path.read_text())
            cost = receipt['cost']
            status = cost.get('stop_reason') or ('完整采集' if receipt.get('rows') else receipt.get('reason', '待验证'))
            lines.append(f"| {path.stem} | {cost['seconds']:.1f} | {cost['peak_bytes']/2**20:.1f} | {status} |")
        lines.append('')
        if score['complete']:
            status = '达到预登记的通道重复准入门槛。' if score['qualifies_to_enable_on_ordered_channel_repeat'] else '预登记的通道重复准入门槛未达到。'
            lines.append(status)
        else:
            lines.append(f'六项预算已有 {len(receipts)} 项结束，须等已登记的逐形结果齐全后再作准入判断。')
        lines.append('失格或仍在运行的调用无正式设备时间。现有完整窗和成本见[同集评分](channel-score.json)和[逐形回执](receipts/)。')
        diagnostic_path=OUT/'partial-evidence.json'
        found=load(diagnostic_path.name)['jobs'] if diagnostic_path.exists() else {}
        if any(x.stem not in found and not json.loads(x.read_text()).get('rows') for x in receipts):
            python=ROOT/'toolchains/envs/cann-9.2.1/bin/python'
            subprocess.run([str(python) if python.exists() else sys.executable,
                            str(Path(__file__).with_name('partials.py'))],check=True)
            found=load(diagnostic_path.name)['jobs']
        for name,scope in found.items():
            for observation in scope['observed']:
                stage=observation['stage']
                lines.append(f"离线超时的 {name} 中，第 {observation['ordinal']+1} 次调用可以单独核对。"
                             f"全 {observation['full_output_elements']} 个元素正确，{stage['active_aiv']} 个活动 AIV 的写回共 "
                             f"{stage['writes_bytes']} B，设备窗 {stage['ca_us']:.3f} µs，冻结预测 "
                             f"{observation['frozen_prediction_us']:.3f} µs。整对未完成，未计入同集 WAPE。")
        if found:
            lines.append('各超时任务的真实启动次数、官方 duration 表计数、数值与逐核时钟见[部分证据](partial-evidence.json)。')
    cross=load('cross-chip.json')
    lines+=['', '## 同流 DT/PR 耗时比', '',
            '定义比值为 DT 设备微秒数除以 PR 设备微秒数，大于 1 表示 PR 较快。只有双端实测且完整输入序列逐字节相同、调用序号相同的窗口配对。方向只在实测差超过 5% 时统计。', '',
            '| 数据角色 | 双端配对 | 条件模型比值 WAPE | 旧同流比值 WAPE | 条件模型判对方向 |',
            '|---|---:|---:|---:|---:|']
    for label,title in [('v0.24 mixed history replay','混合开发回放'),
                        ('v0.27 original forward ordinary token','原冻结 token 换形/首次'),
                        ('published forward exact token repeat','原冻结 token 重复')]:
        g=cross['metrics'][label]; selected=g['models']['selected_us']; old=g['models']['old_us']
        lines.append(f"| {title} | {g['paired_windows']} | {selected['wape_pct']:.2f}% | {old['wape_pct']:.2f}% | {selected['direction_correct_at_5pct']}/{g['measured_ratios_outside_5pct']} |")
    lines+=['', '例如 [16,4,768] 紧邻重复时，CA 为 DT 2.125 µs、PR 2.005 µs。条件模型重算目标分片后给 DT/PR 比值 0.926，实测为 1.060。DT 分到 64 个 AIV，每片 192 列。PR 为 32 个 AIV，每片 384 列。模型随片宽给 PR 多分了 0.157 µs 的主体时间，实际 PR 主体只多 0.012 µs。控制到首操作数的模型两者同为 0.898 µs，CA 分别为 1.055、0.935 µs。两项差异解释了此处快慢倒置，单窗无法进一步分离首条控制指令、分区接纳和端口争用各占多少。四个单端窗保留于误差原表，只在比值统计中排除。[逐对双端数据与完整批次指纹](cross-chip.json)', '']
    ms=max(v['max_ms'] for v in replay['warm_python_30_queries'])
    first=max(v['first_query_after_module_import_ms'] for v in replay['fresh_python_queries'])
    lines+=['',f'在独立 Python 进程中加载 API 后，分别首次查询小形、普通形及大形，三次最大 {first:.3f} ms。',f'已在两个 950 SKU 与 970DT 上各查四个小形至大形，每形连续 30 次。预热 Python 进程的单次最长为 {ms:.3f} ms。在线查询不需要 CA。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
