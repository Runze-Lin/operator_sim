# vLLM HcPost · 三个 token 与两个通道块

[B,S,N,D]=[1,3,4,5120]。输出 61,440 个 BF16 值，每个值需要 5 次乘和 4 次加，共 552,960 ops。这一扩展的输入系数为 FP32。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [1, 3, 4, 5120]
attrs = {'implementation': 'vllm_hc_post'}
result = estimate('mhc_post', shape, attrs, chip='Ascend950PR_957b')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

3 个 AIV 各一行。每行 D 轴切成 4096+1024 两个块。调度器分派系数读、主残差读、向量混合及输出写，关键核的两块受缓冲槽和数据就绪时刻约束。

公开工作量统计为 Cube **0**、Vector **552,960** 个有用标量等效操作。逻辑读 **153,840 B**、写 **122,880 B**。片外实际流量取决于重用与缓存。

可识别的参数包括首次系数读发起的 3183 tick、初次 VF 准入的 18 tick 以及两次系数发起之间的 47 tick。它们来自参考核指令边界。目标分核改变发起离散与共享流量，最小 UB 预算 213184 B 限定该排程。 [参数及单位](../../src/operator_model/data/parameters/vllm_hc_post.json)

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `critical_core` | 5.136 |
| `exposed_shared_limit` | 0.000 |
| **整窗** | **5.136** |

## 本例各阶段怎样定价

[1,3,4,5120] 在 950PR 请求 3 个有效 AIV。每个核一行，两个宽度分片为 4096、1024。每片需要系数、x 和 residual 都就绪后发起 VF，即 VF_start=max(x_end,res_end,coeff_end)+input_ready，还须等待上一片的 cast 槽。下一次写等待 max(上一写退休+槽周转, 本片cast完成+写发起间隔)。每个 DMA 的完成时间为发起时刻+固定响应+字节数×每字节价格，其中单核 DMA 价来自同一路径的 950DT 描述符窗口。

| 关键核 rank=2 上的事件 | 起始零点以来的周期 |
|---|---:|
| 系数首发、首 VF、首片写发起 | 3216.31 / 4878.31 / 7147.45 |
| 1024 尾片写发起 | 8043.35 |
| 输出写退休 | 8043.35+306.34115+8192 B×0.01398214 周期/B = 8464.24 |
| 核退休 | 8464.24+10 = 8474.24 |

尾片的 8192 B 来自 4 路×1024 个 BF16 值。三核排程中最迟的核给出 8474.24 周期；共享服务下界约 181.31 周期，已被隐藏。整窗 **8474.24/1650=5.13590 µs**。公共表只导出关键核和额外共享预算；上述 VF 至写回的区间同时包含重叠工作和等待。

[逐片调度与寄存器处理](../../src/operator_model/operators/mhc/vllm_post.py) · [描述符、VF 和槽响应价格](../../src/operator_model/data/parameters/vllm_hc_post.json)

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 5.084 | 5.122 | -0.73% | 0 / 3 |
| Ascend950PR_957b | 5.136 | 5.024 | +2.22% | 0 / 3 |
| Ascend970DT（外推） | 4.658 | N/A | N/A | 0 / 3 |

相同名称中的 Post 不表示相同实现。只有显式选择 vllm_hc_post 才调用这条模型。小形有限核的测量能约束启动，对更多行共享缓存和队列竞争的解释仍有限。

示例收据的调用为 `aclnnHcPost`，provider 为 vLLM Ascend custom OPP，实际记录的 kernel 为 `HcPost_d2fe6614722b27afa6f8d964c215447e_1`。

## 相同实现的 1024-token 调用

对于 [1,1024,4,5120]，每个 AIV 有多次 token 工作。950PR 预测 52.850 µs，完整 CA 为 44.868 µs，高估 17.79%。950DT 预测 45.169 µs，已有 trace 只给出 ≥38.186 µs 的下界，完整误差未知。970DT 的 24.696 µs 是外推。[大形对照数据](../../calibration/reference/large-shapes.json)

复核时，在逐窗记录中筛选 `contract=vllm_hc_post`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/mhc/vllm_post.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
