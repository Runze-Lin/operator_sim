# Kimi 融合 KDA V1 的工作分配与同步资格

| 950 路线 | 独立完整时延样本 | 950DT WAPE / 最大 | 950PR WAPE / 最大 | 已核对的设备结果 |
|---|---:|---:|---:|---|
| 默认 `aclnnChunkKdaFwd`，单个 FULL 主核 | 0 | N/A | N/A | 单序列 DT、PR 全输出正确，存在重复 SET |
| 同一接口，四个物理主核 | 0 | N/A | N/A | DT 双序列的四组输出及七次全核窗口完整，首末两个主阶段存在重复 SET。PR 无该例实测 |

同样是 B8 的 32k prompt，TP8 时每请求每步只交 2,048 token 会选 V1，一次交完则选 V2。对 V1 四段路线，本轮可以从每个 tile 推导出 Finalize 的两项矩阵乘法、分核工作量与计算下界。在原始小例上，这些结果逐一对应到实际 GM 地址和矩阵指令。可校准的整核时间仍为空。本轮新增 CA 调用数为零。[逐指令证据](partial-stages.json) · [历史完整调用与黄金值](../../v0.27/kda/host-control/report.md)

## 系统调用

```python
from operator_model import plan_kimi_kda_prefill

p = plan_kimi_kda_prefill([2048] * 8, attention_tp=8, chip="Ascend950PR_957b")
w = p["v1_work"]["finalize"]
print(w["work_items"], w["logical_cube_ops"])       # 3072, 9663676416
print(round(w["cube_useful_work_floor_us"], 3))     # 25.632
print(p["v1_work"]["timing_qualified"])              # False
print(p["main_kernels_us"], p["full_dispatch_us"])  # None, None
```

`v1_work` 还提供不可校准的具体原因、每个已分配 AIC 的逻辑运算数、BF16 峰值和目标芯片的外推标记。FULL 的独立 Finalize 工作字段为空，V2 及空调度的 `v1_work` 为空。当目标没有 BF16 峰值，仅计算量可用。[公共调用实现](../../../operator_model/kimi_kda.py) · [工作及价格计算](../../../operator_model/kda_v1_work.py)

## 分块、分核与计算下界

限定 Kimi 官方 stable 分派和 FLA A5 26.9.1，BF16、64 行一块、$K=V=128$，传规范的累计长度和块索引。每卡实际头数为 $H$，每条请求本轮的有效长度为 $s_i$。各条独立取整得到 $C=\sum_i\lceil s_i/64\rceil$ 个块和 $W=CH$ 个块头任务。$W\ge4096$ 切换到 V2。V1 的 $C=1$ 为 FULL，$C>1$ 在 950 上依序发出 Gate/Prepare、PostWU、FwdH、Finalize。显式累计长度排除了 dense 特化。此几何取 tiling key 2，单改 token 或头数不改变这个 key。[发行版 host 分派](https://github.com/flashserve/flash-linear-attention-npu/blob/5cc699069c7f91a35aabd583032c8720a0b626fa/fla/ops/ascendc/kda/chunk_kda_fwd/op_host/op_api/aclnn_chunk_kda_fwd.cpp#L762-L767) · [所用软件及源码位置](source-map.json)

Finalize 中每个有效 $t$ 行的块头需要 $Q_gH$ 和 $A_{qk}V_{new}$ 两个乘法，形状分别是 $[t,K][K,V]$ 和 $[t,t][t,V]$。每次乘加计两个操作，每任务有用工作量为

$$F(t)=2tKV+2t^2V.$$

按块优先、块内 head 递增给任务编号 $j$。编号为 $a$ 的 AIC 处理 $j=a,a+A,\ldots$，$A$ 是芯片 AIC 核数。因此整段总工作量是 $\sum_j F(t_j)$，最长核的工作是 $F_\mathrm{max}=\max_a\sum_{j\bmod A=a}F(t_j)$。若整芯片 BF16 峰值是 $P_{16}$ TFLOP/s、各 AIC 对称，则这部分有用工作在理想计算条件下至少需要 $10^{-6}F_\mathrm{max} A/P_{16}$ 微秒。填充、指令发起、传输、同步、其他阶段均在下界以外。970DT 同时采用同一软件映射的假设。[各芯片配置、每核分工、完整计算](system-contracts.json)

| B8、TP8，每请求本次提交 | 块头任务 | 调用 | Finalize 逻辑 Cube ops | DT / PR / 970DT 计算下界 µs |
|---:|---:|---|---:|---:|
| 512 | 768 | V1 四段 | 2,415,919,104 | 5.592 / 6.524 / 1.493 |
| 2,048 | 3,072 | V1 四段 | 9,663,676,416 | 22.370 / 25.632 / 5.545 |
| 2,688 | 4,032 | V1 四段 | 12,683,575,296 | 29.360 / 33.554 / 7.251 |
| 2,689 | 4,128 | V2 | — | 另走 V2 |
| 32k / 128k / 256k | 49,152 / 196,608 / 393,216 | V2 | — | 另走 V2 |

## 每个实际块怎样对应到物理核

从既有 950DT 的 26.9.1 原始记录复核两条各一行、单头、累计长 [0,1,2] 的显式 V1。O、Aqk、Akk、最终 VK 全部过原始黄金值。四段各启动 32 AIC / 64 AIV，发出有效矩阵指令的核数如下。

| 核内阶段 | 异常整调用中的观测窗 µs | 做 MMAD 的 AIC | MMAD 次数 | 指令对应 Cube ops | 重复 SET 回调 |
|---|---:|---:|---:|---:|---:|
| Gate/Prepare | 37.439 | 2 | 16 | 1,376,256 | 114 |
| PostWU | 5.364 | 0 | 0 | 0 | 0 |
| FwdH | 50.190 | 2 | 2 | 65,536 | 0 |
| Finalize | 11.785 | 2 | 4 | 66,048 | 18 |

两个任务对应 core0、core1。各核的 `MMAD(1,128,128)` 与 `MMAD(1,128,1)` 恰好产生 $F(1)=33,024$ ops。按同一指令 PC 对应读描述符，core1 相较 core0 的源地址逐一偏移 256 B（Qg 的下一 token）、32,768 B（下一块头的 BF16 128×128 状态）、128 B（Aqk 的下一行跨 64 个 BF16 槽）、256 B（vNew 的下一 token）。两笔中间 FP32 写的目的地址也各差 512 B，正是一行 128 个 FP32。两核 M→MTE1 事件 0、1、2 均在同号首次 WAIT 前第二次 SET。core0 事件 0 在 tick 221,766 和 223,653 置旗，第二条 ELF 偏移 `0x16935c` 和真实回调同址，首条 MMAD 在 230,224 才发起。中间两个阶段本次零回调，但缺少其独立所有输出的黄金值。这四个局部观察窗都不进入正常时延校准。[各物理 owner 的指令与地址](partial-stages.json)

## 哪些几何仍有冲突

对每个有工作量的规范 varlen 四段输入，至少有一个 AIC 处理非空块。$1\le t<64$ 时使用普通 `ComputeOutputCube`，$t=64$ 调用专门的 `ComputeOutputCubeStagedArch35`。在发行版源码中两个分支均在本地构造 `Resource<KdaArchTag>`。外层 Finalize 分派已建立 `TPipe pipe`，内层 Resource 的 pipe 成员在构造函数体中执行 `Destroy()` 之前已经构造。因此让长度都整除 64 仍会进入两层所有权的源码路径。尾块的实际二次 SET 已经在两工作的物理核上证实。未实测满块的故障 PC 或回调数量不作推断。单块 FULL 所有合法属性也尚未穷尽。[已发布内核的分支和位置](source-map.json) · [独立真实编译器的 TPipe 生命周期微对照](../../v0.25/kda-v1/compiler-controls.json)

2026-11-20 再次读取官方 GitHub Release，所有三个已发布 A5/aarch64 wheel 的长度和 SHA256 均与本机文件字节相符。26.10.0 正式与 feature 设备 `.text` 清单相同，26.10.0 本身已有数值通过但同步失败的 DT 实测。系统 CANN 8.5.0 的内建 kernel 目录只列出 ascend910b，项目 CANN 9.2.1 内建库与 FLA 插件分别标记身份。本次可见发布范围中未发现修正上述冲突的替代库。[实物与发布清单](release-inventory.json) · [原始 API 快照](official-releases.json.gz) · [官方发布页](https://github.com/flashserve/flash-linear-attention-npu/releases)

在源码目录可运行 `python3 development/validation/kda29/audit.py audit results/v0.29/kda/official-releases.json.gz`，复核原始 CA 及工作账本。另有独立的按工作区段计算的 CPU 对照，100 个随机不等长样本与逐任务枚举一致。[审计程序](../../../development/validation/kda29/audit.py) · [随机样本与边界](../../../development/validation/kda29/test_audit.py)

在两个 950 配置与 970DT 上，用两条一行、B8×2,048、B8×2,688 以及 1,024 个单 token 请求，以公共入口各测 80 次，共 960 次。此次同进程单次最长 3.632 ms。全部 V1 行仍保持主阶段时间表为空、主核耗时和整调用耗时为 None。[公共接口数值、性能与源文件指纹](public-api.json)
