# Chunk KDA 算例

输入轴顺序为 **B,T,Hq,Hv,K,V**。本例比较同一输入契约在 950DT、950PR 与假设的 970DT 上的设备指令窗口。

```python
from operator_model import estimate

shape = [1, 64, 2, 2, 128, 128]
attrs = {'dtype': 'bf16', 'beta_dtype': 'fp32', 'use_gate_in_kernel': True, 'output_final_state': True, 'chunk_size': 64, 'safe_gate': True, 'lower_bound': -0.03125, 'export_gk': True}
for chip in ('Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'):
    r = estimate('kda_chunk', shape, attrs, chip=chip, dtype='bf16', timing_mode='auto')
    print(r.latency_us, r.status, r.active_aic, r.active_aiv, r.work)
```

## 输入到工作量

该路线调用 CANN 9.2.1 的 ops-transformer 算子族。实际调用接口为 `aclnnChunkKdaFwd`。
工作量接口给出的 Cube 操作为 20,971,520，Vector 计数为 32,768。逻辑读/写分别为 None / 262144 B。此处 None 表示缺少计数；片外实际流量须另行观测，逻辑字节不含中间层重复搬运。

## 分块、资源和关键路径

块长 64，每序列 1 块，B×Hv=1×2=2 条独立状态链。每条链的 FP32 状态为 K×V，总状态 131,072 B。实块长 L=64 时，每头保留下三角 QK 点积 L(L+1)/2、严格下三角 KK 点积 L(L−1)/2，各点积 2K 次操作。

| 阶段 | 依赖 | 资源 | 本例工作（标量等效 ops） |
|---|---|---|---:|
| qk | 输入 | cube_bf16 | 1,064,960 |
| kk | qk | cube_bf16 | 1,032,192 |
| wu | kk | cube_bf16 | 4,194,304 |
| state_update | wu | cube_bf16 | 8,388,608 |
| state_decay | state_update | vector_alu | 32,768 |
| finalize | state_decay | cube_bf16 | 6,291,456 |

其中 WU 和 finalize 是稠密等效工作。状态跨块串行；三角求逆等未取得执行级映射。依赖图可以输出，设备时间、实际分核、稳态发起间隔和计时误差均为 N/A。

```mermaid
flowchart LR
  QK --> KK --> WU --> StateUpdate --> Decay --> Finalize
```

[归档结构与验证证据](../../results/v0.8/cube/chunk-structure-validation.json)

## 设备时间及误差

| 平台 | 模型 (µs) | CA (µs) | 有符号误差 | 预测 AIC / AIV | CA AIC / AIV |
|---|---:|---:|---:|---|---|
| 950DT | N/A | N/A | N/A | N/A / N/A | N/A |
| 950PR | N/A | N/A | N/A | N/A / N/A | N/A |
| 970DT 外推 | N/A | N/A | N/A | N/A / N/A | N/A |

## CA 流水与模型差距

本次 PR 尝试：ValueError('process/budget: CA synchronization/instruction failure')。实际宿主开启 safe gate、lower_bound=−0.03125 并导出 GK；冻结清单遗漏了这些属性，默认门控不同。上面的结构演算采用宿主实际属性。950DT 的同类同步失败已归档。两端没有通过正确性与完整窗口检查的本例 trace；无法画出合格流水图或计算误差。

[总建模方法](../modeling-strategy.md) · [全部 shape 的误差](../error-analysis.md) · [970DT 配置与来源](../../configs/chips/Ascend970DT.json)
