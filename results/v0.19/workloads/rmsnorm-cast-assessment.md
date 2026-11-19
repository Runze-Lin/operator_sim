# DeepSeek 的归一化双输出在 950 上如何落地

解码块在 MoE 前调用 `rms_norm_cast(x)`。这个 Python 方法名涵盖多条路径。模型假跑可以确定调用次数和张量合同，要判断本机能否测 fused kernel 还需核查平台能力和注册。

| 固定 vLLM Ascend v0.30.0 路线 | 输入输出与 950 可达性 | CA 资格 |
|---|---|---|
| DeepSeek-V4.1-Flash | `DeviceOperator.rms_norm_cast` 先检查 `RMS_NORM_CAST` capability，再检查自定义包和符号。A5 profile 中缺少该 capability，方法返回 None，归一化结果另做 `.float()`。每层符号假跑会看到一次这个方法。 | 可分别用 CANN 9.2.1 的 ops-nn `aclnnRmsNorm` 和 ops-math `aclnnCast` 验证组件。具体框架图是否又融合，仍需实际加载整网捕获。 |
| DeepSeek-V4-Pro | 自定义算子开关打开时，模型直接调 `torch.ops._C_ascend.npu_rms_norm_cast`。适配器请求 `aclnnRmsNormCast`，该提交的 OpDef 只注册 `ascend910b` 和 `ascend910_93`。开关关闭时使用相同的“归一化后升为 FP32”路径。 | 本机 950 OPP 缺少无残差的同签名 fused kernel；未验证 Pro 在打开开关的配置下能跑通，不把 fallback 当作自动降级。 |
| Kimi-K3 | 本轮逐层符号路径未发现同名双输出方法。 | 不将 DeepSeek 的次数迁到 Kimi。 |

目标 custom kernel 接受 BF16 或 FP16 二维视图和同 dtype gamma，输出与 x 同形的低精度 y 及 FP32 精确归一化值。fallback 则把已舍入低精度结果转成 FP32，两种路径可能产生不同的 FP32 路由输入。安装包虽有 `aclnnAddRmsNormCast`，其 ABI 还接受残差，并返回更多张量；给它补零会额外搬运和计算。该 API 的 CA 窗口不能替代模型请求的双输出 kernel 的测量。

对于 BF16 [T,D]，自定义 fused 路径的逻辑输入约 2TD+2D B，输出 6TD B。独立 Cast 组件每元素从 BF16 提升到 FP32，输入 2TD B，输出 4TD B，执行 TD 次类型转换。fallback 的第一个算子还有本身的输入/输出开销，候选时间为“普通 RMSNorm +独立 Cast”两次设备调用之和。排队、kernel 间热状态及编译器融合影响整网调用时间。970 的所有时间仍为外推。

**源码与 ABI 核验。** 当前 tag 对应提交 `c09d3e20b9c45c80b34ae3781440d90904a44023`。以下文件核对了精确分支和输入/输出，OpDef 和适配器对应 Git blob `395d9d4b8caccc359ab4640a269fe20fa150129d`、`728de5b9f0e7bd287cdea350572ab4fabe1edd43`。本地已装 CANN 的 include、ascend950 kernel 配置与实际加载 kernel 用于核对计时资格。[Flash 模型](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/models/deepseek_v41/model.py#L844) · [Pro 模型](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/models/deepseek_v4/model.py#L734) · [设备分支](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/device/device_op.py#L64) · [A5 能力](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/device/hardware_profile.py#L333) · [Torch 适配器](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/csrc/moe/rms_norm_cast/rms_norm_cast_torch_adpt.h) · [custom OpDef](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/csrc/moe/rms_norm_cast/op_host/rms_norm_cast_def.cpp)

## 实际假跑 shape 的分解示例

编排器可分别调用如下两行代码。若图中两个 kernel 在同一 stream 串行执行，组件时间相加是忽略启动间隙的估计。这里 CPU 侧指定的芯片与每卡输入固定。

```python
from operator_model import estimate

shape = [72, 7168]  # 本卡 batch=8，每个请求校验 Q=9
rms = estimate('rmsnorm', shape, chip='Ascend950PR_957b', timing_mode='resource')
cast = estimate('cast_bf16_fp32', shape, chip='Ascend950PR_957b', timing_mode='resource')
component_sum_us = rms.latency_us + cast.latency_us
```

假跑下每个 decoder 层对应的 post-attention 双输出组件耗时如下。按“普通归一化 + Cast”回退策略计价。每格为 RMSNorm + Cast 设备时间，单位 µs。所有数字由模型预测，970DT 及超出全量 CA 范围的大 shape 均为外推。

| 模型 | 场景 | 本卡 [T,D] | 950DT | 950PR | 970DT 外推 |
|---|---|---|---:|---:|---:|
| DeepSeek-V4-Pro | unchunked_prefill | 262144×7168 | 1881.11 + 2819.63 | 5371.15 + 8054.13 | 770.88 + 783.91 |
| DeepSeek-V4-Pro | chunked_prefill | 2048×7168 | 17.10 + 23.08 | 43.64 + 63.98 | 10.91 + 7.87 |
| DeepSeek-V4-Pro | verify_8 | 72×7168 | 5.29 + 3.20 | 5.09 + 3.27 | 6.15 + 3.18 |
| DeepSeek-V4.1-Flash | unchunked_prefill | 262144×5120 | 1344.24 + 2014.33 | 3837.23 + 5753.25 | 552.18 + 560.21 |
| DeepSeek-V4.1-Flash | chunked_prefill | 2048×5120 | 13.19 + 16.79 | 31.65 + 46.00 | 9.06 + 6.66 |
| DeepSeek-V4.1-Flash | verify_8 | 72×5120 | 5.10 + 3.02 | 4.89 + 3.01 | 6.06 + 3.09 |

独立 Cast 的校准、参数与 CA 复核见[Cast 算例](../../../docs/examples/cast_bf16_fp32.md)。
