# DeepSeek 的 vLLM HcPost

DeepSeek V4.1 Flash 与 V4 Pro 的 attention 和 FFN 输出进入 `torch.ops._C_ascend.npu_hc_post`，由 vLLM Ascend 绑定调用 `aclnnHcPost`。这里使用锁定提交的 arch35 分核、核函数和生成的 ACLNN 库，通过私有 OPP 载入 CANN 9.2.1；BF16 激活搭配 FP32 post/comb。另一个可单独调用的库算子 `aclnnMhcPost` 有自己的分核和计时模型。[锁定的调用绑定](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/csrc/torch_binding.cpp#L1434-L1444) · [arch35 host tiling](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/csrc/moe/hc_post/op_host/hc_post_tiling_arch35.h) · [CANN MhcPost 算例](mhc_post.md)

## 从三个 token 算到张量工作量

```python
from operator_model import estimate
r = estimate("mhc_post", [1, 3, 4, 5120], {"implementation": "vllm_hc_post"},
             chip="Ascend950DT_9582", dtype="bf16")
print(r.active_aiv, r.work["vec_ops"], r.work["logical_read_bytes"], r.work["logical_write_bytes"])
# 3 552960 153840 122880
print(r.status, r.latency_us)  # unavailable None
```

1. 输入 x 为 [1,3,5120]，残差为 [1,3,4,5120]，系数 post 为 [1,3,4]、comb 为 [1,3,4,4]。记 T=B×S 为提交的 token 数，H=4 为残差路数，D 为通道宽度。对每个 token 和输出路 j，逐通道计算 `out[j] = x × post[j] + Σᵢ residual[i] × comb[i,j]`。每个输出通道共 5 次乘和 4 次加，有效 FP32 算术量为 THD(2H+1)。例中 61,440 个 BF16 输出，共 552,960 个 FP32 算术操作，另有类型转换。
2. 逻辑读字节为 `2TD(1+H)+4T(H+H²)`，写字节为 `2THD`。2 和 4 为 BF16 与 FP32 字节数。post 和 comb 的系数在一个 token 的所有 D 分块间复用。三个 token 合计读 153,840 B、写 122,880 B。核内实际搬运还包括元数据和片上寄存器传输。
3. 每个 AIV 领取完整 token，使用 `A=min(T,AIV数)` 个核；每核 `⌊T/A⌋` 或 `⌈T/A⌉` 行。每行先读两份系数，再按源代码固定的 4096 通道拆 D。5120 拆为 4096+1024，分别搬运 x、四路残差和输出。3 个 AIV 共 6 组 D 分块、6 笔系数读、12 笔激活读和 6 笔数据写，0 次 Cube 指令。源码的尾块写回用有效元素数，BF16 转换指令按分配的整块区域执行，不能仅用有效输出字节推断整个 VF 时长。[BF16 核函数](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/csrc/moe/hc_post/op_kernel/hc_post_bfloat16.h)
4. DT 与 PR 的公开设备窗口分别为 **5.122 和 5.024 µs**。两边都实际运行 3 个 AIV，61,440 个输出逐元素与 FP64 参考舍入的 BF16 位值相等。设备代码段 SHA256 一致，逐核核验物理 END、结束标记、官方时长和汇总 trace 中的同步事件。当前接口支持工作量与目标芯片分核预测，阶段定价和耗时误差待独立校准，970DT 暂无耗时数值。[DT 原始计时回执](../../results/v0.23/hc-post/Ascend950DT_9582-3-5120.json) · [PR 原始计时回执](../../results/v0.23/hc-post/Ascend950PR_957b-3-5120.json)

Flash 每卡 B8×32k 整批会给绑定传 [1,262144,4,5120]，首维 1 来自模型把本卡 token 展平后补维。DT 按 64 个 AIV 每核 4096 行，每行两块；PR 最忙核 4682 行。Pro 同样有 4 路残差，但 D=7168，每行拆成 4096+3072。970DT 在相同实现假设下按 120 个 AIV 最多分 2185 行/核。这三条大 shape 的分核是源码推导。[可复算实现](../../operator_model/hc_post_layout.py) · [全部校准输入与源码哈希](../../results/v0.23/hc-post/development.json)
