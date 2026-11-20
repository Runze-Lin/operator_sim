# RMSNorm 的逐核归约与双路写回

输入是单卡 BF16 的 `x[T,D]`、共享 BF16 权重 `γ[D]`、epsilon（本次 CA 为 1e-6）、芯片 JSON，以及可选的前一次成功派发信息。安装包调用 `aclnnRmsNorm`，实测 kernel 为 `RmsNorm_normal_all_bf16_high_performance_5000`。每行产生 BF16 的 `y[T,D]` 和独立 FP32 的 `rstd[T]`。每行先求 `rstd = 1/sqrt(sum(x²)/D + eps)`，再算 `y = x · rstd · γ`。完整的 x 归约、γ 的读入、两个结果的就绪与提交顺序，决定了本算子的关键路径。MXQuant 的量化分组、scale 数据类型与写入协议有各自的实现。

## 使用

v0.28 的常规调用 `estimate("rmsnorm", [T,D], chip=..., dtype="bf16")` 在无前序且每核一、两块时使用本节三情景中心。已知 H 的公共估时沿用之前的分核与扇区调度。以下专项接口暴露逐核读、各 VF 发起与写回的完整可解释路径，支持显式区分相同合同立即复跑。

```python
from operator_model.rmsnorm_pipeline import rmsnorm_pipeline

entry = rmsnorm_pipeline([4, 784], chip="Ascend950PR_957b", dtype="bf16",
                         resident_aiv=1, diagnostics=True)
repeat = rmsnorm_pipeline([4, 784], chip="Ascend950PR_957b",
                          resident_aiv=4, immediate_repeat=True)
unknown = rmsnorm_pipeline([4, 784], chip="Ascend950PR_957b")
# entry["us"], entry["stages"], entry["bounds_cycles"]
# unknown["us"], unknown["lower_us"], unknown["upper_us"], unknown["scenarios"]
```

`us` 为预计设备耗时。条件调用的 `stages` 记录各 AIV 各 UB 块的发起、返回和逻辑起始行，单位为目标时钟周期。区分参与 AIV 数 `A` 与前序 AIV 数 `H`。H=0 表示进程首次，H>0 表示知道前序覆盖范围。形状、dtype、内存布局、设备、流和参数相同且确知上一调用成功时才传 `immediate_repeat=True`。复跑实验还核对了对应输入和输出 GM 地址。实际系统中仅知道 H≥A 时应使用普通已覆盖前序。

## 如何分核、分块与构造依赖

1. 每核最多处理 `m=ceil(T / Cᵥ)` 行，实际使用 `A=ceil(T/m)` 个 AIV。末核只取剩余行。按目标 UB 容量、宽度和既有 key 5000 策略得每块上限 `R`，核内分成 `r=min(R, 剩余行数)` 的块。形状在合法的 BF16 整行路径内，D 为 16 的倍数且不超过 8192。特殊短行转置分支另有已有后端限制。
2. 读取 γ 一次并搬入本核的 x。对第 j 块令 `Bⱼ=2D·Σᵢrᵢⱼ` 字节，求 `Lⱼ=Bⱼ f/BW_HBM` 周期，f 以 GHz，带宽用十进制 GB/s。共享链路与每核 MTE2 各自施加下界。γ 的发起取决于 lane 状态和同时启动的 AIV 数。1 个 AIV 前序可能使另一物理分区首次启动；DT/PR 均按两个平衡分区，单核的 AIV 数来自配置的 `ceil(AIV/AIC)`。一个分区内部区分原先已活动的 lane、同核 sibling、整核新增。不同状态的发起响应由已知前序的旧 CA 描述符校准。[分区与扇区证据](../../v0.22/rms-lane-parameters.json)
3. 开始归约需要两条读响应都就绪。若 `b` 为归约 VF 发起时刻，`b ≥ max(Eγ,Ex)+4` 周期。下一个块可在前一块归一化描述符结束前约 50 周期发起归约，但仍须满足本块的读依赖。x 首个发起晚于 γ 约 96 周期。每条读的式子为 `max(k·Pᵣ, aₛ + bₛ·k·Pᵣ/Pᵣ₀ + cₛ·Lⱼ)`，k 为本核 KiB，Pᵣ 为目标局部端口周期/KiB，s 为 lane 状态；γ 没有全体 x 的流量项。立即复跑的额外入口流量按缓存驻留情景估算，同时保留 L2 容量不足后的 HBM 下界。响应可以相互重叠。
4. A5 默认 256 B Vector 寄存器。一个寄存器可装 64 个 FP32，成对寄存器每组处理 128 个值。令 `g=r·ceil(D/128)`、`d=r·ceil(log₂(ceil(D/64)))`。前者表征 VF 主循环，后者是归约树深度项。以 `V` 表示目标相对参考的每周期 FP32 吞吐倒数，以 `s`、`n` 表示 rstd VF 和归一化 VF 的发起。所有时间先用目标周期计算：

   ```text
   s         = b + 17.28 + V·(2.82r + 4.47g + βd)
   n         = s + 113.46 + 0.117r
   E_rstd_VF = s + 209.40 + 0.201r
   E_norm    = n + 142.86 + V·(1.57r + 6.225g + β'd)
   ```

   两个 `β` 在本安装包样本上的非负阶段拟合接近零，独立树项尚不可辨识。代码保留了组数和树深工作量。实测 VF 描述符有重叠，上式用下一阶段的真实发起间隔。图中用斜线表示归约发起至下一 VF 发起；CA 还可以画出归约的完整响应。固定发起成本保留周期单位，仅变动工作随实际向量吞吐调整。
5. rstd 写发起为 `Iᵣ=E_rstd_VF+4`，y 写发起为 `Iᵧ=E_norm+5`。y 每块写 `2Dr` 字节，rstd 写 `4r` 字节。对一个 128 B 扇区中的写者按真实可发起时间排序；第 i 个写的预计完成时刻为 `Fᵢ=max(Iᵢ+αₛ,Fᵢ₋₁+157)`。αₛ 表示同状态独占扇区的响应，157 周期来自旧 CA 对同扇区相邻写者的间隔。一个描述符跨多个扇区时取最后提交者，各扇区可并行；整块 y 走自己的写端口响应。尾部开销为 9 周期。端口和共享带宽下界与两路完成取最大值，再除以 `1000f` 得到全设备微秒数。[逐阶段参数、样本数](prices.json)
6. 未知前序时计算进程首次、只覆盖一个 lane 的前序、立即复跑三个可观察情景。短、长资源时间分别记作 l、u，输出 `2lu/(l+u)`。它使两端的相对误差幅度相等。它只覆盖这三个模型情景，不能保证覆盖任一真实前序或 CA 波动。使用已知流状态可以直接选一个条件 DAG；调用方若只能提供 shape/dtype，理想点估计也有可算的下界。[不可辨识性与实测误差](report.md)
7. 上述逐块归约路径目前支持每核至多两块。超过两块时复用已有多轮 RMS 流水，按首次、相邻两轮的增量与尾轮计算，保留端口/带宽约束，避免逐 token 展开。这个工作量较大的后备路径没有本轮新的大张量 CA 资格。[32k、128k、256k × B8 的耗时与查询开销](query-cost.json)

## 参数从哪里来

- 设备工作点由芯片 JSON 给出。DT 为 64 AIV、32 AIC、1.65 GHz、HBM 4000 GB/s，PR 为 56/28、1.65 GHz、1400 GB/s。970DT 用户情景为 120/120、1.8 GHz、14400 GB/s，L2 带宽 30000 GB/s。970 的频率、逐核 FP32、使用核数、HBM 与 L2 都参与定价。无新芯片实测的分区策略、寄存器分组、扇区宽度、前端唤醒和 VF 发起周期继承参考假设。几何和写者提交间隔、尾部排空周期可通过 `RmsGeometry` 替换，阶段价格可经 `prices` 替换。若芯片的 ISA、库算法或分区结构变化，需换实现策略；只改峰值不足以核验新芯片。
- 每个状态的 `gamma_*` 截距/斜率为 γ 描述符发起周期/新增 AIV，`gamma_spread_*` 表示同一状态发起跨度。PR 首次有效载荷移位来自此前独立的首次派发校准。`gamma_response_*`、`x_response_*`、`y_service_*` 按实际描述符的 KiB、聚合竞争周期和状态估计，均受配置端口限制。`second_x_*` 表示两次 x 读发起间隔。在扩核的旧读序列中缺少双块控制，暂继承已覆盖态两次 x 发起间隔，本轮 H=1→双块给出独立检查。`rstd_solo_*` 取每窗最快 rstd 写响应再跨窗拟合。
- 训练只读取之前两个版本的完整逐核轨迹，按真实 ELF PC、访问地址和字节数定位。stage-samples 对每窗、lane 状态、块大小只计一次组统计。阶段拟合为非负 Huber 鲁棒最小二乘；覆盖次数很少的占用趋势使用中位截距。输入里没有整算子耗时标签，芯片名也不选取总耗时系数。物理核启动跨度、扇区退休和 VF 依赖都可在不同芯片上单独复查。[开发窗口索引](milestones.json) · [冻结预测及数据角色](freeze.json)
- 安装包 ELF、golden、指令 PC 与完整窗是本轮实际证据。`ops-nn` 的公开 Ascend 950 实现和 CANNBot 的 `RmsNormRegBasePerf` 说明用于核对数据流；公开源码与二进制的一一版本对应关系尚未证明。[Ascend C RMSNorm 950 知识卡](../../../third_party/cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/norm/rms_norm_950.md)
