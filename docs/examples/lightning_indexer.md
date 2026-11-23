# LightningIndexer · 一个 query 扫描 8192 个 key

Q[1,1,8,128]、K[1,8192,1,128]、权重[1,1,8] 均为 BF16，key 占 2 MiB。每个 key 有八个点积，Cube 工作为 2×8192×8×128=16,777,216 ops。输出 top512 写 2048 B。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [1, 1, 8192, 8, 128]
attrs = {'dtype': 'bf16',
 'layout': 'BSND',
 'return_values': False,
 'sparse_count': 512,
 'sparse_mode': 3}
result = estimate('lightning_indexer', shape, attrs, chip='Ascend950DT_9582')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

一个四 query 分组只装入一行。8192/128=64 页，单 AIC 共发 64 条 MMAD。排序 AIV 执行 256 条正时长直方图指令。DT 库启动 32 个 AIC 和 64 个 AIV，真正做这两种有效工作的各为一个。

公开工作量统计为 Cube **16,777,216**、Vector **188,416** 个有用标量等效操作。逻辑读 **2,099,216 B**、写 **2,048 B**。片外实际流量取决于重用与缓存。

参考首次 MMAD 为 4587 tick，页到页的读发起间隔为 416 tick，score 到 histogram 接力为 1160 tick，三者分别由首次、稳态和交接事件得到。目标若有已知 MTE2 带宽，用每页 32768 B 的实际端口服务与补齐乘加服务重新比较。 [参数及单位](../../src/operator_model/data/parameters/indexer.json)

下表取 950DT 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `first_mmad` | 2.780 |
| `page_intervals` | 15.884 |
| `last_mmad` | 0.079 |
| `score_to_histogram` | 0.703 |
| `histogram_and_output` | 2.425 |
| `additional_shared_service` | 0.000 |
| **整窗** | **21.871** |

## 本例各阶段怎样定价

本例为 DT、单 query、8192 个可见 key、8 个头、depth=128。8 行头对齐到 M=16，8192/128=64 个键页。一个完整键页为 128×128×2=32768 B。整个 CV 窗分为下面五项，单位为 DT 周期。

| 阶段 | 公式与本例 |
|---|---|
| 首 MMAD | 参考关键核的第一次 MMAD 在 4587 周期。 |
| 相邻页 | (64−1)×max(单页读发起间隔416, 2×16×128×128/8181.818)=26208。计算项使用每核 BF16 ops/周期。 |
| 尾页退休 | 66.88053 周期响应 + 2×16×128×128/8181.818 = 130.96。 |
| 结果可排序 | Score 到首个 histogram 的等待为 1160。 |
| histogram 与输出 | 每 256 keys 一组，共 32 组、8 次 histogram/组。一个扫描区间需 540+48.75×32=2100 周期，输出/退休需 635.857+39.5357×32≈1901 周期。 |

合计 **(4587+26208+130.96+1160+4001)/1650=21.87088 µs**；HBM 下界约 0.525 µs 已被流水覆盖。页间隔包含高速本地读取与发起节奏，端口明确时以目标服务率重算；首发、尾部和 histogram 响应来自 950DT 分段 CA。超过 8192 key 时需要在同一行重复扫描，目前按行间交接作代理。

[键页与 histogram 计数](../../src/operator_model/operators/attention/indexer.py) · [页和扫描边界价格](../../src/operator_model/data/parameters/indexer.json)

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 21.871 | 22.709 | -3.69% | 1 / 1 |
| Ascend950PR_957b | 21.926 | N/A | N/A | 1 / 1 |
| Ascend970DT（外推） | 20.046 | N/A | N/A | 1 / 1 |

DT 原始数据库有 120,521 条记录，96 个参与核各有唯一物理 END，设备跨度 (43,410−5940)/1650=22.709 µs，完整 512 个索引集合与 CPU 一致。Q1 的 DT 样本属于开发回放。其他 Q 或候选数的 PR 不替代此形状。

示例收据的调用为 `aclnnLightningIndexer`，provider 为 CANN 9.2.1 ops-transformer，实际记录的 kernel 为 `entry_kernel_31b1b`。

复核时，在逐窗记录中筛选 `contract=lightning_indexer`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/attention/indexer.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
