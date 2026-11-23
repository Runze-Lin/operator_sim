# 拆分 mHC · 三个 kernel 的设备窗

输入同为 [4,4,1024]，Pre 将 4 个展平为 4096 的 token 投影到 24 列，Sinkhorn 使用 20 轮，Post 将外部层输出和四路残差重新混合。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [4, 4, 1024]
attrs = {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20}
result = estimate('mhc', shape, attrs, chip='Ascend950PR_957b')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

按 Pre→Sinkhorn→Post 依次安排三个库调用。模拟给外部层设置恒等连接，以隔离这三段。PR 返回的三个阶段分别对应三个设备窗。组合的逻辑流量累计中间张量的写与下一段读。

公开工作量统计为 Cube **786,432**、Vector **230,984** 个有用标量等效操作。逻辑读 **467,628 B**、写 **42,000 B**。片外实际流量取决于重用与缓存。

各子模型沿用各自的参数与已安装 kernel 证据，本例合计 9.247+5.578+3.334≈18.160 µs。若部署采用融合 PreSinkhorn 加 Post，应使用该融合调用的实际阶段和排程。

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `pre` | 9.247 |
| `sink` | 5.578 |
| `post` | 3.334 |
| **整窗** | **18.160** |

## 本例各阶段怎样定价

**工作单位。** [T,N,D]=[4,4,1024] 按 Pre → Sink → Post 依次下发。对每个独立设备窗，将 950DT 的开发阶段回放拟合成四组资源预算。阶段 q 的目标 µs 为 a_q f时钟 + b_q W_q f_q；a 是固定 µs，b 是每个工作单位的增量 µs，W 由该实现的分块推出。DT/PR 同频，f时钟=1，本例没有新一波。向量增量按每核 FP32 峰值换算 fv=(27/64)/(23/56)=1.027174。若拟合中计算增量取零，对应的独占时间不可辨识；逻辑工作和实际计算指令仍在。

| 子窗 | W=(计算块,向量单位,流量 KiB,控制轮) 的形成 |
|---|---|
| Pre | 输入扁平 K=4×1024=4096，输出 O=N²+2N=24，MK 网格分到的本地 K=256。W=(⌈4/16⌉×⌈256/32⌉,⌈4/64⌉×⌈1024/256⌉,(4KO+2TK)/1024,⌈4/16⌉)=(8,4,416,1)。 |
| Sink | 4×4 的 FP32 矩阵，20 次归一化，W=(0,⌈4/64⌉×20×4²,4TN²/1024,1)=(0,320,0.25,1)。 |
| Post | 切通道直至每片 256，j=4，⌈Tj/64⌉=1 波。W=(0,1×⌈D/(64j)⌉×N(N+1),4TND/1024,1)=(0,80,64,1)。 |

**给搬运重定价。** 因局部 MTE 带宽未知，局部倍率取 1；片外倍率为 4000/1400。令 B 为该子调用接口上的逻辑读写字节，u 为 DT 分段预算之和，压力 p=min(1,B/(1000×4000×u))，最终 f搬运=1+p×(4000/1400−1)。这是将参考成本分配给共享链路的近似，不等于实测 HBM 流量。下表的 W流量只作参考分段拟合的工作单位；B 则计入实际接口的所有张量。

| 子窗 | B=逻辑读+写（B） | DT 的 u（µs） | p | f搬运 |
|---|---:|---:|---:|---:|
| Pre（含公开中间结果） | 426092+8976=435068 | 9.225955 | 0.01178924 | 1.02189430 |
| Sink | 4TN²+4TN²=512 | 5.578448 | 0.00002295 | 1.00004261 |
| Post | (2TD(N+1)+4TN(N+1))+2TND=74048 | 3.322115 | 0.00557235 | 1.01034866 |

Pre 的读分成 2TND 的 BF16 输入和 4(O·ND+3+O) 的 FP32 权重、三个缩放与偏置。写包含 2TD 的隐藏输出、4TN 的门控、4TN² 的矩阵与 4T(1+O+N) 的中间结果，总和为 8976 B。

**每段代入。** 950DT 各行以 f搬运=fv=1 先相加即可得到上表的 u。950PR 的价格如下，常数与结果均为 µs。

| 子窗 | Compute / Vector | Traffic / Control | 合计 |
|---|---|---|---:|
| Pre | 0；0.998017+0×4×fv | 2.314089+0.002311555×416×1.02189430；3.699636+1.252606×1 | 9.247009 |
| Sink | 0；3.928485+0×320×fv | 0.549632+0.2049623×0.25×1.00004261；1.049091 | 5.578450 |
| Post | 0；0.995960+0.005010101×80×fv | 0.649866+0.002192646×64×1.01034866；1.135152 | 3.334459 |

组合整窗为 **18.15992 µs**。这条拆分实现的三个独立设备窗可相加；中间张量的写后读已分别进入相邻调用的逻辑字节。独立 Sinkhorn、内置独立 Post 与 vLLM 的 Post 各有自己的实现合同。[拆分调度与工作量](../../src/operator_model/operators/mhc/pre_dispatch.py) · [参考阶段系数](../../src/operator_model/data/parameters/mhc_partition.json) · [子窗资源比例](../../src/operator_model/scheduling/projection.py) · [逻辑张量统计](../../src/operator_model/workload.py)

## 如何从 CA 得到这些参数

**边界。** 接口 dtype、字节和三个设备调用的顺序给工作量。用实现的 16 行和 256 列对齐推导单位；频率、吞吐、带宽取目标配置。DT/PR 的带宽压力 p 是逻辑字节加权假设；校准的 a、b 对应 DT 可见阶段，不能当作物理端口的峰值。

**选择真实样本列。** 只取历史 950DT 串行三 kernel 的 mhc 调用，按序号 0/1/2 配 Pre/Sink/Post。对 SQLite 的 ExecutedInstructions，读取非排除记录的 (ExecInstrTickStart,ExecInstrTickEnd,queue_type)，在每类核内选最晚完成者，再取整体最长窗；全核最早指令是零点。每一 cycles 同时活跃两引擎时按 CUBE、SIMD、搬运（MTE/FIX）、控制/等待优先唯一归属。四列合计必须等于该 kernel 的实际设备窗。

**从数据库到一条拟合标签。** 950DT [4,4,1024] 的 Pre，源 DB SHA-256 前缀 35ee3398c468，最后完成核 CoreTypeId=2、ExecInstrCoreId=0。全窗 14404 cycles；重新扫描得到独占 Cube 0、Vector 1847、搬运 6240、其他控制与空闲 6317。除以 1650 即 (0,1.11939,3.78182,3.82848) µs，整窗 8.72970 µs；本行对应工作 (8,4,416,1)。它是一个目标点。Pre-MK 开发组的 Cube 标签全部为 0，得 a=b=0。Vector 的非负斜率落在零边界，发布的固定 0.99801653 µs 是该组 11 个 Vector 观测的平均值。[读 SQLite 并提取重叠集](https://github.com/Runze-Lin/operator_sim/blob/2059edce197af60ddee6cfbc7c78d40c715ed76f/scripts/fusion-v8-freeze.py) · [训练样本与存档位置](https://github.com/Runze-Lin/operator_sim/blob/2059edce197af60ddee6cfbc7c78d40c715ed76f/results/v0.8/fusion/rows.json) · [训练 DB 指纹清单](https://github.com/Runze-Lin/operator_sim/blob/2059edce197af60ddee6cfbc7c78d40c715ed76f/results/v0.9/mhc/freeze.json)

**全组复算。** Pre-MK 11 条、Pre-ND 2 条、Sink 13 条、短 token Post 12 条、整行 Post 1 条，分别用每资源独占 µs 作 y、其分块单位作 x。若 x 仅一值，设 a=median(y)、b=0。否则比较可行的非负普通最小二乘直线和仅固定/仅单位项两条非负边界，取平方残差和最小者。用本机归档数据库执行这一步，五支的每个固定项和斜率均与发布文件相同。Git 保留 shape、哈希、整窗、摘要和拟合过程；重建逐 cycles 互斥标签仍需对应原始 DB，不能从整窗单独还原。[每个子窗拟合目标及规则](https://github.com/Runze-Lin/operator_sim/blob/2059edce197af60ddee6cfbc7c78d40c715ed76f/scripts/mhc-v9-freeze.py) · [非负直线的实现](https://github.com/Runze-Lin/operator_sim/blob/2059edce197af60ddee6cfbc7c78d40c715ed76f/operator_model/model.py) · [完整冻结系数](https://github.com/Runze-Lin/operator_sim/blob/2059edce197af60ddee6cfbc7c78d40c715ed76f/results/v0.9/mhc/model.json)

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 18.127 | 18.220 | -0.51% | — / — |
| Ascend950PR_957b | 18.160 | 18.525 | -1.97% | 16 / 16 |
| Ascend970DT（外推） | 16.608 | N/A | N/A | 16 / 16 |

核数字段的 “—” 表示该参考路径未返回这一项。

整链时间可能很接近 CA，即便个别阶段正负误差抵消。观察端到端系统时还要加入 attention/MLP 以及调用间隙。

示例收据的调用为 `aclnnMhcPre + aclnnMhcSinkhorn + aclnnMhcPost`，provider 为 CANN 9.2.1 ops-transformer，实际记录的 kernel 为 `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic`、`MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0`、`MhcPost_59130ccee6717ff2c0f15b485e731162_5`。

复核时，在逐窗记录中筛选 `contract=mhc`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/mhc/pre_dispatch.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
