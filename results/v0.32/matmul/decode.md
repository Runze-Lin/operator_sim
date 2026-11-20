# 三个模型的 decode：B1、9、11、16 与主模型 Q9

M 是**这个 rank 交给一次 Matmul 的行数**。普通 decode 每个请求一行 Q=1；MTP 一次提出八个草稿时主模型校验 Q=9。B=1 开 MTP 和 B=9 普通 decode 得到同一个 M=9，共用设备证据；B=9 的九个请求也各开 MTP，得到 M=81。B=11 的两种情况是 M=11/99，B=16 是 M=16/144。公式仅指本卡无 padding、无额外请求时的一次投影，非注意力 batch 轴推断。

以下形状依据三模型配置与权重推导的真实 TP8 逻辑尺寸；实测合同为安装包 aclnnMatmul、连续无偏 BF16 ND。图部署若重新量化、转置或融合，按新合同核验。每个已有完整设备窗按 trace 指纹去重；同一形状多样本取中位数。每个模型列一个主要投影和一个替代投影，**同一模型内两个真实投影并非重复样本**。
当前 42 种唯一几何，DT 20、PR 19 种已有合格时间，18 种双端配对。空白项只表示这份合同尚无合格设备时间。


| 模型 · 投影 | B×Q → M | 每卡 [M,N,K] | 950DT 预测 / CA µs | 950PR 预测 / CA µs | 970DT 外推 µs |
|---|---|---|---:|---:|---:|
| DeepSeek-V4.1-Flash · indexer.wk | 1×1 → 1 | [1, 128, 512] | 4.230 / [4.161](../../../results/v0.30/matmul-dispatch/device/1-dt-retired.json) | 4.230 / [4.030](../../../results/v0.30/matmul-dispatch/device/1-pr-retired.json) | 3.877 |
| DeepSeek-V4.1-Flash · indexer.wk | 1×9、9×1 → 9 | [9, 128, 512] | 4.234 / [4.325](../../../results/v0.30/matmul-dispatch/device/9-dt-retired.json) | 4.234 / [4.127](../../../results/v0.30/matmul-dispatch/device/9-pr-retired.json) | 3.881 |
| DeepSeek-V4.1-Flash · indexer.wk | 11×1 → 11 | [11, 128, 512] | 4.235 / [4.310](../../../results/v0.30/matmul-dispatch/device/11-dt-retired.json) | 4.235 / [4.137](../../../results/v0.30/matmul-dispatch/device/11-pr-retired.json) | 3.882 |
| DeepSeek-V4.1-Flash · indexer.wk | 16×1 → 16 | [16, 128, 512] | 4.237 / [4.371](../../../results/v0.30/matmul-dispatch/device/16-dt-retired.json) | 4.237 / [4.186](../../../results/v0.30/matmul-dispatch/device/16-pr-retired.json) | 3.884 |
| DeepSeek-V4.1-Flash · indexer.wk | 9×9 → 81 | [81, 128, 512] | 4.380 / [4.402](../../../results/v0.30/matmul-dispatch/device/81-dt-retired.json) | 4.380 / [4.290](../../../results/v0.30/matmul-dispatch/device/81-pr-retired.json) | 4.015 |
| DeepSeek-V4.1-Flash · indexer.wk | 11×9 → 99 | [99, 128, 512] | 4.392 / [4.568](../../../results/v0.30/matmul-dispatch/device/99-dt-retired.json) | 4.392 / [4.332](../../../results/v0.30/matmul-dispatch/device/99-pr-retired.json) | 4.026 |
| DeepSeek-V4.1-Flash · indexer.wk | 16×9 → 144 | [144, 128, 512] | 4.412 / [4.479](../../../results/v0.30/matmul-dispatch/device/144-dt-retired.json) | 4.413 / [4.530](../../../results/v0.30/matmul-dispatch/device/144-pr-retired.json) | 4.045 |
| DeepSeek-V4.1-Flash · dsa.wq_a | 1×1 → 1 | [1, 1280, 5120] | 11.168 / — 无合格窗 | 12.488 / — 无合格窗 | 10.996 |
| DeepSeek-V4.1-Flash · dsa.wq_a | 1×9、9×1 → 9 | [9, 1280, 5120] | 11.288 / [10.215](../../../results/v0.28/matmul/9-1280-5120-dt-drain.json) | 12.631 / [13.642](../../../results/v0.28/matmul/9-1280-5120-pr-drain.json) | 10.999 |
| DeepSeek-V4.1-Flash · dsa.wq_a | 11×1 → 11 | [11, 1280, 5120] | 11.318 / [10.153](../../../results/v0.31/matmul/device/11-1280-5120-dt.json) | 12.667 / [13.579](../../../results/v0.31/matmul/device/11-1280-5120-pr.json) | 11.000 |
| DeepSeek-V4.1-Flash · dsa.wq_a | 16×1 → 16 | [16, 1280, 5120] | 11.394 / — 无合格窗 | 12.756 / — 无合格窗 | 11.002 |
| DeepSeek-V4.1-Flash · dsa.wq_a | 9×9 → 81 | [81, 1280, 5120] | 12.653 / — 已排队 | 13.916 / — 已排队 | 11.133 |
| DeepSeek-V4.1-Flash · dsa.wq_a | 11×9 → 99 | [99, 1280, 5120] | 12.982 / — 无合格窗 | 14.237 / — 无合格窗 | 11.171 |
| DeepSeek-V4.1-Flash · dsa.wq_a | 16×9 → 144 | [144, 1280, 5120] | 13.777 / — 无合格窗 | 15.040 / — 无合格窗 | 11.188 |
| DeepSeek-V4-Pro · indexer.weights_proj | 1×1 → 1 | [1, 64, 7168] | 5.695 / [5.585](../../../results/v0.28/matmul/1-64-7168-dt-drain.json) | 5.573 / [5.467](../../../results/v0.28/matmul/1-64-7168-pr-drain.json) | 13.865 |
| DeepSeek-V4-Pro · indexer.weights_proj | 1×9、9×1 → 9 | [9, 64, 7168] | 5.731 / [6.079](../../../results/v0.28/matmul/9-64-7168-dt-drain.json) | 5.658 / [5.367](../../../results/v0.28/matmul/9-64-7168-pr-drain.json) | 13.869 |
| DeepSeek-V4-Pro · indexer.weights_proj | 11×1 → 11 | [11, 64, 7168] | 5.741 / [5.688](../../../results/v0.28/matmul/11-64-7168-dt-drain.json) | 5.679 / [5.833](../../../results/v0.28/matmul/11-64-7168-pr-drain.json) | 13.870 |
| DeepSeek-V4-Pro · indexer.weights_proj | 16×1 → 16 | [16, 64, 7168] | 5.763 / [6.108](../../../results/v0.28/matmul/16-64-7168-dt-drain.json) | 5.732 / [5.619](../../../results/v0.28/matmul/16-64-7168-pr-drain.json) | 13.872 |
| DeepSeek-V4-Pro · indexer.weights_proj | 9×9 → 81 | [81, 64, 7168] | 6.249 / [— 未测 · 主机满载未预约](../../../results/v0.32/matmul/pro-b9-mtp/host-check.json) | 6.215 / [— 未测 · 主机满载未预约](../../../results/v0.32/matmul/pro-b9-mtp/host-check.json) | 14.003 |
| DeepSeek-V4-Pro · indexer.weights_proj | 11×9 → 99 | [99, 64, 7168] | 6.475 / [7.056](../../../results/v0.31/matmul/device/99-64-7168-dt.json) | 6.461 / [6.888](../../../results/v0.31/matmul/device/99-64-7168-pr.json) | 14.015 |
| DeepSeek-V4-Pro · indexer.weights_proj | 16×9 → 144 | [144, 64, 7168] | 7.080 / [7.378](../../../results/v0.28/matmul/144-64-7168-dt-drain.json) | 7.113 / [7.332](../../../results/v0.28/matmul/144-64-7168-pr-drain.json) | 14.033 |
| DeepSeek-V4-Pro · dsa.wq_a | 1×1 → 1 | [1, 1536, 7168] | 14.611 / — 无合格窗 | 18.820 / — 无合格窗 | 14.098 |
| DeepSeek-V4-Pro · dsa.wq_a | 1×9、9×1 → 9 | [9, 1536, 7168] | 14.731 / — 无合格窗 | 18.981 / — 无合格窗 | 14.101 |
| DeepSeek-V4-Pro · dsa.wq_a | 11×1 → 11 | [11, 1536, 7168] | 14.761 / — 无合格窗 | 19.022 / — 无合格窗 | 14.102 |
| DeepSeek-V4-Pro · dsa.wq_a | 16×1 → 16 | [16, 1536, 7168] | 14.835 / — 已排队 | 19.123 / — 已排队 | 14.105 |
| DeepSeek-V4-Pro · dsa.wq_a | 9×9 → 81 | [81, 1536, 7168] | 16.271 / — 无合格窗 | 20.437 / — 无合格窗 | 14.252 |
| DeepSeek-V4-Pro · dsa.wq_a | 11×9 → 99 | [99, 1536, 7168] | 16.637 / — 无合格窗 | 20.801 / — 无合格窗 | 14.273 |
| DeepSeek-V4-Pro · dsa.wq_a | 16×9 → 144 | [144, 1536, 7168] | 17.505 / — 无合格窗 | 21.710 / — 无合格窗 | 14.290 |
| Kimi-K3 · mla.q_b_proj | 1×1 → 1 | [1, 2304, 1536] | 8.618 / [6.680](../../../results/v0.28/matmul/1-2304-1536-dt-drain.json) | 8.621 / [7.494](../../../results/v0.28/matmul/1-2304-1536-pr-drain.json) | 5.633 |
| Kimi-K3 · mla.q_b_proj | 1×9、9×1 → 9 | [9, 2304, 1536] | 8.626 / [— 采集未合格](../../../results/v0.29/matmul/decode/9-2304-1536-dt.json) | 8.647 / [— 采集未合格](../../../results/v0.29/matmul/decode/9-2304-1536-pr.json) | 5.637 |
| Kimi-K3 · mla.q_b_proj | 11×1 → 11 | [11, 2304, 1536] | 8.628 / [6.939](../../../results/v0.28/matmul/11-2304-1536-dt-drain.json) | 8.653 / [7.608](../../../results/v0.28/matmul/11-2304-1536-pr-drain.json) | 5.638 |
| Kimi-K3 · mla.q_b_proj | 16×1 → 16 | [16, 2304, 1536] | 8.632 / [6.840](../../../results/v0.29/matmul/decode/16-2304-1536-dt.json) | 8.669 / [— 采集未合格](../../../results/v0.29/matmul/decode/16-2304-1536-pr.json) | 5.640 |
| Kimi-K3 · mla.q_b_proj | 9×9 → 81 | [81, 2304, 1536] | 9.073 / [— 采集未合格](../../../results/v0.29/matmul/decode/81-2304-1536-dt.json) | 9.253 / [9.008](../../../results/v0.29/matmul/decode/81-2304-1536-pr.json) | 5.788 |
| Kimi-K3 · mla.q_b_proj | 11×9 → 99 | [99, 2304, 1536] | 9.168 / [9.407](../../../results/v0.28/matmul/99-2304-1536-dt-drain.json) | 9.388 / [9.847](../../../results/v0.28/matmul/99-2304-1536-pr-drain.json) | 5.866 |
| Kimi-K3 · mla.q_b_proj | 16×9 → 144 | [144, 2304, 1536] | 11.877 / [10.964](../../../results/v0.29/matmul/decode/144-2304-1536-dt.json) | 11.895 / [— 采集未合格](../../../results/v0.29/matmul/decode/144-2304-1536-pr.json) | 5.866 |
| Kimi-K3 · mla.kv_b_proj | 1×1 → 1 | [1, 3072, 512] | 5.271 / — 无合格窗 | 5.834 / — 无合格窗 | 4.110 |
| Kimi-K3 · mla.kv_b_proj | 1×9、9×1 → 9 | [9, 3072, 512] | 5.280 / — 已排队 | 5.851 / — 已排队 | 4.113 |
| Kimi-K3 · mla.kv_b_proj | 11×1 → 11 | [11, 3072, 512] | 5.283 / — 无合格窗 | 5.855 / — 无合格窗 | 4.114 |
| Kimi-K3 · mla.kv_b_proj | 16×1 → 16 | [16, 3072, 512] | 5.289 / — 无合格窗 | 5.865 / — 无合格窗 | 4.116 |
| Kimi-K3 · mla.kv_b_proj | 9×9 → 81 | [81, 3072, 512] | 5.741 / — 无合格窗 | 6.376 / — 无合格窗 | 4.313 |
| Kimi-K3 · mla.kv_b_proj | 11×9 → 99 | [99, 3072, 512] | 5.839 / — 无合格窗 | 6.490 / — 无合格窗 | 4.391 |
| Kimi-K3 · mla.kv_b_proj | 16×9 → 144 | [144, 3072, 512] | 6.887 / — 无合格窗 | 6.809 / — 无合格窗 | 4.733 |

Pro 的同一 indexer.weights_proj [81,64,7168]（B9×Q9）目前只有 host workspace 和本模型预报。已有设备登记和全部相关原始调用均未发现相同采集。三次 2 秒宿主采样只见 2、1、3 个具有至少 80% 空闲的物理核；256 核主机上的旧 Kimi 槽已有锁主和等待者。因此此端点没有另排长队，表里也不填 CA。[资源核对](pro-b9-mtp/host-check.json)。

本轮只新增三个从未采集也未预约的几何：Flash dsa.wq_a 的 [81,1280,5120]、Pro dsa.wq_a 的 [16,1536,7168]、Kimi mla.kv_b_proj 的 [9,3072,512]。三形共六个目标端点，已在采前登记唯一 ELF、数值、分核和 MMAD 预测；在现有 Kimi 大形双端释放同一把锁后逐一执行。前序等待不计为已采或独立准确性。

[新增六端冻结](decode/freeze.json) · [实际排队和 CA 状态](decode/state.json) · [逐几何机器数据](decode.json) · [prefill 双端实测和冻结盲测](report.md)
