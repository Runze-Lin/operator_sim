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

对每阶段 q 取目标成本 Tq=aq×fscalar + bq×Wq×fq。a 是在 950DT 拟合的固定阶段预算（µs），b 是每一工作单位增加的 µs，f 是目标的相对资源价。把 950DT 的 CA 分段拟合到 compute、vector、traffic、control 后，零斜率只说明在这些样本上没有分辨出独立增量。

本例三个 kernel 有顺序依赖。表中 W 依次为计算块、Vector 工作单位、KiB 和控制轮次。

| 子窗 | 工作如何形成 | 目标 PR 时长 |
|---|---|---:|
| Pre | MK 核内 K=256，W=(8,4,416,1)，分段代入见 [Pre](mhc_pre.md)。 | 9.24701 µs |
| Sinkhorn | 一波 4×4 矩阵做 20 次归一化，W=(0,20×16,4×4×16/1024,1)=(0,320,0.25,1)。 | 5.57845 µs |
| Post | 小 token 切 4 个通道片，W=(0,⌈1024/(64×4)⌉×20,4×4×4×1024/1024,1)=(0,80,64,1)。 | 3.33446 µs |

这里的 Sink 和 Post 都使用拆分实现的独立预算。以 Post 为例，固定项为 0.99596+0.64987+1.13515 µs，Vector 增量为 0.0050101×80×1.02717，traffic 增量为 0.00219265×64×1.01035 µs。三窗相加得到 **18.15992 µs**。切换到独立 Sinkhorn 或 vLLM Post API 会改变实现和校准合同。

[M/K 分块与各阶段单位](../../src/operator_model/operators/mhc/pre_dispatch.py) · [目标端定价](../../src/operator_model/scheduling/projection.py) · [分支系数与单位](../../src/operator_model/data/parameters/mhc_partition.json)

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
