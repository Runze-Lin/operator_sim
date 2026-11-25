# 原生 Vector 算子：指令类别、分核与阶段替换

`estimate` 的 `attrs={"vector_model":"instruction_classes"}` 会换掉所选实现的 Vector 服务估计。本文的已有实现默认仍用原法，可用 `calibrated` 显式选中；这个选择器与限制证据资格的 `timing_mode="calibrated"` 不同。后者不会把试验中的指令法标为已校准。950DT、950PR 的 CA 窗口用于**比较预测**；没有拿整窗误差倒推“每条指令几周期”。970DT 只能外推。`projection.instruction_samples` 保存每类第一块的计数、代表性依赖链与周期下界；这是分析模型，不能当成反汇编或硬件指令计数器。

先沿 host 分核确定忙核实际处理的行和列，每行的末寄存器独立取整。例如 BF16 输入升为 FP32 计算时，256 B 寄存器容纳 64 个 FP32 元素：对 `r` 行、每行 `w` 个元素，寄存器工作组数 `g = r × ceil(w / 64)`。按类别计发起次数 `n[k]` 后，按 `n[k] × II[k]` 得该类的端口需求。`II[k]` 是同类指令能相隔多少核心周期再次发起，不是结果延迟。再分别计算总发起需求 `sum(n) / issue_width`、UB 端口需求 `(load_service + store_service) / ub_ports`、联合算术端口需求 `sum(alu_service) / alu_lanes`，以及一条最长候选生产链的串行延迟 `sum(latency[k])`。取上述下界及各类独占端口下界的最大值作为这块的估计。周期除以 `1000 × GHz` 得设备微秒。新实现、通路共享或调度不理想时，该下界式可能乐观。

默认 `issue_width=2`、`ub_ports=2`、`alu_lanes=2`；EXP、DIV、RSQRT 的默认 II 为 4，其余为 1。依赖延迟以旧 FA 模型的寄存器通路为先验（load/store 9、算术/转换 7、reduce 16、exp 18、divide 17 周期）；RSQRT、重排分别暂借 17、7。三组均为待校准先验，并非华为公布的各指令速度。可以在芯片配置的 `details` 覆盖 `vf_<类别>_ii_cycles`、`vf_<类别>_latency_cycles` 和端口数。请单独调整新芯片的 VF 性能参数；整芯片 Cube 峰值翻倍不会让这些 Vector 指令自动变快。DMA、核数、频率与各算子的 tiling 另按目标芯片计算。

| 实现 | 忙核计算步骤；如何排回整窗 |
|---|---|
| DynamicMxQuant BF16→MX8 / BF16→MX4 | 每 tile 有 `b=r×2×ceil(w/64)` 个含填充的 scale 槽，数据与 scale 循环分别 `ceil(b/8)`、`ceil(b/128)` 次。估入读寄存器、求分组最大值、指数处理、重排、转换和 store；MX4 再打包。用现有双缓冲流水重算，包括共享小 scale 写队列、尾块服务、HBM 下界。小 MX8 的原法默认可能走 DT 模板，此处走可拆分的资源图，因此两种整窗之差不能全归因于 VF。 |
| AntiMxQuant MX4→BF16 | 保留 row×column 分区，以本 tile 交错 scale/data 的真实循环计数估解包、BF16 转换、乘 scale 和 store。重新选取最晚 cohort，重算读/计算/写重叠、首尾和共享带宽。FP8 仅有工作量或测时、没有可调度实现的路径继续 N/A。 |
| RMSNorm BF16 | 分为 square/reduce、单行倒方根、归一化/乘 gamma/写 y 三个 VF，前两步需要行内归约。1–2 个 UB tile 时逐核重排 gamma/x 的 ready、rstd/y 两条退役路径、共享 rstd 扇区，保留原逻辑的阶段固定成本。超过两个 UB tile 用多块资源模型替换忙核 SIMD 并重新检查共享带宽下界，跨 tile 相依只能近似。前驱未知时按可达场景取相对误差的极小极大中心 `2×L×H/(L+H)`。流中只知道前驱下界时，指令法采样最小前缀和完全覆盖，不声称这是所有中间前缀的严格包络。 |
| 内置 mHC Post BF16 / vLLM HcPost | `n=4` 条输入支路。每 `g` 组做 `(n+1)g` 次输入读取、`n(n+1)g` 次混合算术。保留按通道/按 token 的分核，按已知历史重排行内传输或多片双槽；未知历史分别重估首次/扩核/复用/重复。vLLM 的另一实现保留完整 16384 个 FP32 槽的输出转换以及写回，其两次 VF 单独计价。 |
| mHC Sinkhorn FP32 | N4 行列归一化 helper 每轮依次执行；编译器布局以每矩阵 32 B 对齐的 UB 块为粒度，256 B 寄存器一趟可容 8 矩阵。分别计转置/重排、加和、除法和存储。保留 launch、softmax 入场、扇出/写回和 UB 容量约束；若模型遗漏 helper 之间的固定同步，这类小矩阵的循环时间会明显低估。 |
| mHC Pre、Pre+Sinkhorn、split mHC | Pre 以 K 分片转换/归约、输出混合两段计数。split Pre 的非 VF 分项是原 DT 分支先验，按频率迁移，未识别的流量/核间等待仍会失真；三 kernel 组合显式串联 pre、sink、post。Pre+Sinkhorn 保留已计价的 Cube 碎片、读端口和总带宽下界，仅给两段 Vector 工作重新计价。替代团队布局（≥128 token 或非 1:2 AIC:AIV）属于另一个 K-split 假设，不能视同验证。 |
| AttnResFwd BF16 | 分别建立权重转换、prefix add、`valid+1` 行的打分/归约/exp、加权混合和输出归一化。驻留推理保留读 gamma、单槽输出和下一行 overlap；带 materialize 的路径保留原阶段 DMA 和控制截距。已无时间资格的合同不生成总时长。 |
| MoE gating top-k FP32 | `ceil(E/64)` 个专家组，逐行先 sigmoid，加 bias，假设做 K 轮最大值归约与去除，最后归一化。工作计数包含选择比较与重排，不只数 FLOP，原调度的控制/读/写原样保留。这是候选排序算法；若安装的实现用了排序网络，误差随 E/K 变化，不可把此假定解释为已还原该库源码。 |

表中 `r,w` 是**单个忙核 tile** 的有效行数和宽度；`E,K` 分别是路由专家数和选出数；文中“保留”表示复用原模型的同项假设/阶段，并非新收集了该阶段硬件观测。逐合同同集结果与 CPU 估计耗时见[方法对比](vector-method-comparison.md)。

**可复核小例子：** 对 `estimate('quant_mxfp4',[9,128],{'vector_model':'instruction_classes'})`，该 DT tile 为 `r=1,w=128`，所以 `b=4`，两种循环均为 1 次。候选向量体计 `predicate=2, load=3, arithmetic=7, reduce=1, permute=5, convert=6, store=2`，共 26 条。两路总发起下界是 13 周期；读→归约→转换→打包→写回链是 `9+16+7+7+9=48` 周期，比发起端口下界长。因此该 tile 在 950DT 采用 `134.5+48=182.5` 周期的 Vector 阶段：134.5 是原 FP4 tile 交接常数，48 是显式指令类别价格。整算子还要过 DMA 和多核的 scale 队列，不能把 182.5 周期当整窗。可通过 `result.projection['timing_diagnostics']['first_tile']` 对照这部分。

**证据边界。** 分核、块尺寸和包数来自各算子现有代码中的 950 策略；逐类数、端口并行度及关键链属于可替换的实现假设。CANNBot 的 `common/glossaries/ascendc_simt_and_regbase_terminology` 用于核对 `__simd_vf__`、VF 调用和 UB 等术语，不提供本文的逐类延迟测量。没有精确反汇编、独立阶段窗口或下一代 CA 的条目，不应以整窗拟合良好证明指令数正确。
