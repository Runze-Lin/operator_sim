# RMSNorm · 每卡 8192 个 Flash token

Flash 每卡按 4 个请求、每请求 2048 个已调度 token 计算。输入元素 8192×5120=41,943,040。共享 γ 有 5120 个 BF16。Y 有 83,886,080 B，另输出 8192×4=32,768 B 的 rstd。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [8192, 5120]
attrs = {}
result = estimate('rmsnorm', shape, attrs, chip='Ascend950PR_957b')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

DT 每个 AIV 分到 128 行，按 5 行/块得到 25 个满块和 3 行尾块。PR 最忙核 147 行，分 29 个满块和 2 行尾块。每块读、归约、乘 γ、写 rstd 和 Y，写描述符可以与邻块向量运算重叠。

公开工作量统计为 Cube **0**、Vector **167,788,544** 个有用标量等效操作。逻辑读 **83,896,320 B**、写 **83,918,848 B**。片外实际流量取决于重用与缓存。

分块时先将 D 补齐到 16，并按整行双缓冲、γ 与归约临时区核算 UB。写回还区分按 128 列计价的向量工作和按 KiB 计价的 Y 描述符。有效竞争窗口 128 B 是 950 上相邻 FP32 统计量写者的经验粒度。多个块连续时核对实际采用的共享读写约束。 [参数及单位](../../src/operator_model/data/parameters/rms_writeback.json)

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `arrival` | 0.563 |
| `SCALAR` | 1.626 |
| `other` | 0.004 |
| `MTE2` | 1.288 |
| `SIMD` | 41.573 |
| `MTE3` | 0.377 |
| `shared_io` | 16.191 |
| **整窗** | **61.622** |

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 39.379 | 44.868 | -12.23% | 0 / 64 |
| Ascend950PR_957b | 61.622 | 70.485 | -12.57% | 0 / 56 |
| Ascend970DT（外推） | 21.582 | N/A | N/A | 0 / 119 |

两端都低估大形全窗。图中末核的读发起间隔与一次读描述符占用时长可分别测量，实际发起会因 UB 槽位和共享写路径延后。上栏整窗只说明模型预算，对未辨识的逐 tile 时刻保持空缺。

示例收据的调用为 `aclnnRmsNorm`，provider 为 CANN 9.2.1 ops-nn，实际记录的 kernel 为 `RmsNorm_normal_all_bf16_high_performance_5000`。

![两款芯片末核的真实读和写](assets/rmsnorm.png)

上方只标出可识别的模型整窗。单条 CA 横杠为在途描述符，下方数据支持逐条读出发起与退休。 [图上使用的数值](assets/rmsnorm.json)

复核时，在逐窗记录中筛选 `contract=rmsnorm · msopprof`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/rmsnorm/pipeline.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
