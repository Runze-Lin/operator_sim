# 指令类别法在三个非 FA 算子上的检验

本试验复用 950DT/PR 合格 CA 窗，检验改变 Vector 阶段的价格能否改善设备时间。三个算子都暴露 `attrs={"vector_model":"instruction_classes"}`，默认保留原法；同一对报价使用相同的核数、每核 tile、首读、写回、缓冲槽和共享带宽。按类别算出的周期进入原流水重新排程。每个完整窗分别调用两种模型后与同一个 CA 设备窗相减。[整窗与局部阶段逐项结果](../results/v2.2/nonfa-vector-comparison.csv) · [复算程序](../calibration/examples/nonfa_vector_study.py)

## 同集结果

| 算子与精度 | DT / PR 整窗数 | 原法 → 类别法整窗 WAPE，DT / PR | DT / PR 阶段数 | 原法 → 类别法阶段 WAPE，DT / PR |
|---|---:|---|---:|---|
| Cast BF16 → FP32 | 33 / 32 | 5.03 → 4.85% / 3.31 → 3.29% | 10 / 10 | 13.40 → 27.71% / 16.47 → 27.89% |
| Sigmoid FP32 → FP32 | 23 / 20 | 5.98 → 6.48% / 16.39 → 16.72% | 10 / 10 | 0.42 → 21.31% / 2.19 → 21.25% |
| DynamicQuantV2 BF16 → E4M3FN、逐行 FP32 scale | 13 / 13 | 6.23 → 6.07% / 6.83 → 6.96% | 13 / 13 | 4.00 → 7.17% / 4.09 → 7.28% |

总计 134 个整窗、66 个阶段。这些都是本轮以前采集的样本。量化 26 窗包含原版 6 个开发窗及 20 个历史留出，当前比较按开发回放处理。每个算子有独立的阶段定义，两个端点的 WAPE 分开用绝对周期差总和除以实测周期总和。算子原生接口依次为 aclnnCast、aclnnSigmoid 和 aclnnDynamicQuantV2；逐窗文件的行号对应原记录中的实际库、kernel 与二进制身份。[分组 WAPE 与最大误差](../results/v2.2/nonfa-vector-summary.json) · [阶段实测与 trace 身份](../results/v2.2/nonfa-stage-evidence.json) · [全窗原记录](../calibration/reference/windows.jsonl)

## 怎样数指令，怎样给周期定价

用一颗 AIV 的本地 tile 表示一次程序。行数记为 r，行宽 w，64 个 FP32 元素占一个 256 B 向量寄存器，因此组数为 g=r×⌈w/64⌉。逐元素程序把本地 tile 展平为一行。处理尾组仍需发起带谓词的指令。在 950 的有效时长事件中可以看到 Cast 各发一条 RV_PLT/VLD/VCVT_F2F/VST；Sigmoid 还发 VMULS、VEXP、VADDS 和 VDIV。逐 token 量化的 9×512 开发窗每 8 个寄存器组出现 24 条转换、16 条载入、8 条除法，并出现一次行归约。两次遍历分别求绝对值的行最大值和用 scale 缩放写出结果；每组计两条谓词、两条读、三条转换、两条算术、一条除法、一条写。每行加归约、乘和 scale 写。编译器的标量准备、广播和辅助写在量化首发/交接项里近似表示。[指令程序及假设](../src/operator_model/operators/vector/instruction_classes.py)

对类别 i，以 Nᵢ 表示指令条数，Iᵢ 表示独立指令连续发起的周期，Sᵢ=NᵢIᵢ。对每个引擎分别计算资源需求，取以下各项最大值作为阶段服务。① 每类的 Sᵢ，② 总指令条数除以发起宽度，③ 读写 Sᵢ 之和除以 UB 端口数，④ 算术、转换、特殊函数与归约 Sᵢ 之和除以共享计算通道数。默认发起宽度、UB 端口、共享计算通道均假设为 2，指数和除法的 Iᵢ 取 4，其余取 1。这些名义值沿用 FA 试点的 arch35 参考价格，由混合程序共用端口的假设合成。芯片可覆盖 `vf_issue_width`、`vf_ub_ports`、`vf_alu_lanes` 及 `vf_<类别>_ii_cycles`。每行只按摘要计数，一次调用只价格化满块和尾块。[arch35 理论 SIMD 吞吐参考](https://asc.gitcode.com/api/appendix/reg_vector_compute_interface_theoretical_perf_summary.html)

## 能否解释实测

- **Cast** 在 [1572864,32] 的 DT 首块中有 21120 个元素、330 组，共计 1320 条载入、转换、存储和谓词指令。双发起给 660 cycles；原价给 582.66，CA 有效 RV 指令区间并集为 512.00。阶段取同一核首个 VF 的正时长执行区间并集。谓词占用独立发起槽与 VCVT/VLD/VST 是否共用同一上限，是现在尚未核实的假设。共享搬运与排队可以遮住这部分差异，所以整窗略有改善而局部反而变差。[Cast 原价与传输排程](examples/cast_bf16_fp32.md)
- **Sigmoid** 在 [524288,12] 的 DT 满块中包含 248 组。每组两条普通算术、一次 exp 和一次 divide，共享双通道的下界为 (2+4+4)/2=5 cycles/组。加上从单组 79 cycles 扣除 5 后保留的 74 cycles 固定准备与依赖，给 1314 cycles。原模型为 79+4×247=1067，实测首笔输入读退休到本块最后向量 store 退休为 1070。分解令瓶颈偏向共享通道，指令数本身正确仍可能错估配对。[Sigmoid 开发时标](examples/sigmoid_fp32.md)
- **逐 token 量化** 将首个 payload 读退休到对应 payload 写发起作为阶段窗口，包含读后唤醒、Vector 执行和写前交接。为隔离程序价，两个方法都保留旧版从开发窗提取的 106.25 cycles 固定成本。DT 的 [9,7168] 每个忙核一行，112 组，原阶段价为 659.25 cycles，类别法为 723.75，CA 为 654。估算与实测逐一按该核实际读写字节对齐；整窗仍使用 host 策略推断的行分配。[量化校准脚本](../calibration/token_quant/fit.py)

Cast 和 Sigmoid 的阶段样本取 950DT/PR 的 vector21 归档，筛选具有有效 core0.veccore0 首块、单次 VST 次数等于由输入字节推导的组数的 shape。区别源于硬件程序的证据口径。Cast 原阶段价解释 SIMD 占用，Sigmoid 原阶段价解释从读完成到输出可写的整个区段。所有阶段都用同样的实际 tile 几何向两种方法定价，整窗仍由各自的完整排程产生。

## 跨芯片推断

通用之处在于把三种程序分开后，可以单独改变目标芯片某种指令的吞吐。扩大 exp 单元影响 Sigmoid；Cast 和这一版逐 token 量化没有指数工作。每 AIV 同时处理的元素数、全片 AIV 数和 UB 大小直接决定组数及 tile 数，频率只把周期换成微秒。实际编译器若改变每组转换次数、合并谓词或使用不同掩码策略，连程序都应替换；单调频估算无法表示这种差异。

DT/PR 的配对 shape 给出一个有限的跨 SKU 检验。Cast、Sigmoid、逐 token 量化的耗时比平均绝对相对误差分别从 8.43% 到 8.47%、8.70% 到 8.47%、3.71% 到 3.63%；三者的快慢方向判对次数均相同。结合阶段误差，结果支持将这三个选项用于新实现的设计情景，当前生产默认保留原法。970DT 缺目标 VF、端口与调度的实测，预计可替换的参数并不能证明误差会减小。[配对跨 SKU 与 970 外推](../results/v2.2/vector-transfer.json)
