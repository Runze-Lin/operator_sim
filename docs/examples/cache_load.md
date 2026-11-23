# CacheLoad · 将压缩缓存物化为 BF16

系统图的 `inference::cache_load` 读取当前有效的缓存行和对应 scale，生成一份 BF16 张量。工作量取决于**本次读出的逻辑形状**，包括本卡 batch、有效行、头或其它前缀维度以及尾轴宽度。已分配缓存的最大长度可能更大。物化意味着后续算子的正常输入读与本次算子的 BF16 输出写是两个接口事件，实际是否经过 HBM 取决于部署实现。

## 一次 Kimi 长上下文调用

```python
from operator_model import estimate

result = estimate(
    "cache_load", [64, 131081, 2, 16, 64],
    dtype="mxfp4", chip="Ascend950DT_9582",
    attrs={"storage_shape": [64, 131081, 2, 16, 32],
           "scale_shape": [64, 131081, 2, 16, 1, 2]},
)
print(result.status, result.latency_us)  # unavailable, None
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
# 9127432192 34362097664
```

此形状取自系统框架 `0.1.0-preview.1` 的 950DT、Kimi、DP64/EP64、batch 64 的 rank-0 目标图。历史长度 131072，本次验证 9 个位置，因此有效行轴为 131081。同一目标图的按 shape 计数表有 10 次这样的调用。一次调用共有 R=64×131081×2×16=268453888 行，每行逻辑宽度 D=64，输出 E=R×D=17181048832 个元素。

| 一次调用的工作 | 计算 | 结果 |
|---|---|---:|
| 读 FP4 值 | 每两个值打包进一个 E2M1x2 字节，R×⌈D/2⌉ | 8,590,524,416 B |
| 读 E8M0 scale | 每 32 个值共用一因子，两个尺度成对存放，R×2×⌈D/64⌉ | 536,907,776 B |
| BF16 物化写 | E×2 B | 34,362,097,664 B |
| 缩放 | 每个有效输出应用一次其所属组的 scale | 17,181,048,832 次 |

每个元素满足 `Y[r,j] = BF16(decode(E2M1[r,j]) × scale[r,j//32])`。E8M0 表示 2 的整数次幂，真正的内核可以用指数调整。模型将数学上的缩放记入 `vec_ops_by_class["scale_application"]`，实际执行精度记作 `unspecified`，没有把它解释成具体条数的浮点乘指令。`input_value_decodes`、`bf16_result_values` 单列有效值数，可与字节数一同用于实现设计。

**奇数宽度和 scale 补齐。** 输入逻辑 [3,65] 时每行有 33 个 packed 字节，最后一个半字节为空。每行使用 3 个真实 scale，物理 scale 张量 [3,2,2] 每行有 4 槽。假设完整读取声明的有效行张量，值和 scale 共 99+12=111 B，输出 390 B。实现可能跳过无用槽；公开的 `physical_hbm_bytes` 为未知。

## FP8 路径

V4-Pro 目标图还出现一张 [1,1024,512] 的 E4M3FN 缓存，物理形状与 BF16 输出形状相同，FP32 scale 形状为 [1,1024]，每行一个。调用 `estimate("cache_load", [1,1024,512], dtype="fp8_e4m3fn")`，值占 524288 B，scale 占 4096 B，输出占 1048576 B；每次调用共缩放 524288 个值。FP8 缓存这里采用逐行 FP32；带 block32 E8M0 的 MXFP8 是另一份布局。FP8 图上的 `group_size=32` 只是同一生产器携带的元数据，FP8 路径的 scale 数始终按行推导。

## 系统图如何接入

1. 确认源节点为 `inference::cache_load`，出现两个输入、一个物化 BF16 输出。源代码在 BF16 缓存上直接返回原值。注意力或索引器直接消费低精度缓存时，应按对应融合实现计价。
2. 将 BF16 **输出**的本卡维度传给 `shape`。FP4 的值张量形状只在尾轴除以二。用源图的 `semantic.storage_precision` 及两输入 dtype 确认路径，将实际物理输入形状放入可选 `storage_shape`、`scale_shape` 可检查不一致。源适配器仅在 `storage_precision="fp4"` 时进入 packed 验证，其余路径验证 FP8 字节、逐行 FP32 scale 和 BF16 输出。API 当前支持 block32 FP4，源 builder 允许的其它分组需要单独定义合同。
3. 对发布快照的 `modeling-inputs.csv.gz` 逐行核验，3,435 次图节点中 2,415 次为 E2M1x2/E8M0，1,020 次为 E4M3FN/FP32，输出都为 BF16；这是该批工作负载的实录。原始快照 SHA-256 为 `07e72a917a5553d1f152a53c54fb689633118f9113a90333c86607eaa1807102`。上述 Kimi 和 V4-Pro 两份目标图的按 shape 汇总表指纹分别以 `d0b413beb90b` 和 `bc100cfcb938` 开头。[固定版缓存布局](https://github.com/ZhanluModelSim/inference-models-simulator/blob/e09a6f866bb55c84455a905bf91421aa4b6291f2/src/inference_models/common/cache.py) · [有效行切片和源节点](https://github.com/ZhanluModelSim/inference-models-simulator/blob/e09a6f866bb55c84455a905bf91421aa4b6291f2/src/inference_models/layers/layout.py) · [实际语义适配器](https://github.com/ZhanluModelSim/operator-performance-models/blob/c266554fa186d82201f4acdd034dc7b228a07d3a/src/operator_performance_models/adaptors/vec/cache_io.py)

## 校准和耗时边界

950 的 Ascend C `AntiQuantize` 接口列出了按组 E2M1/E8M0 → BF16、按 token E4M3FN/FP32 → BF16 的数据类型组合，对连续方向字节对齐有约束。系统捕获只提供软件层的物化语义；没有确认这批节点在任一 SKU 上实际调用哪个库 kernel、如何处置尾组、使用多少 AIV，或读/转换/写怎样流水。因此默认输出保留精确的张量工作量，950DT/PR 的 CA、误差与 970DT 的时间均为 N/A。单看 Kimi 本例，若 43,489,529,856 B 逻辑流量全走 4 TB/s HBM，且能持续达到峰值，仅搬运即需 10872.38 µs。这个数字随缓存命中、写回策略和并发改变，不能作为设备预测或 CA 实测。[950 接口精度与对齐](https://gitcode.com/cann/asc-devkit/blob/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/SIMD-API/adv_api/quantization/AntiQuantize.md) · [工作量实现](../../src/operator_model/operators/vector/cache_load.py)
