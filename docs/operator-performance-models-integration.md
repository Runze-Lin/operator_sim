# 将精细模型并入 operator-performance-models

**建议发布一个含两套后端的 wheel。** 在目标仓纳入精细模型源码、芯片参数、阶段价格和回归测试。调用方继续通过后端名选 `roofline` 或 `operator_sim_full`，两者各自返回同一种系统估计结果。需要严格对照时单独选择一种；需要混用时把回退顺序写进任务。本次只核验接口，迁移和发布是后续工作。

## 核验的版本与调用路径

核验日期为 2026-11-25。目标仓 main `e1cd9525` 声明 Python ≥3.12，无必装运行时第三方包，可选 C++ 事件分配器。其工厂当前内置 `roofline` 和 `communication_v1`。算子仓现有的注入接口足以承载完整设备耗时。系统仓 main `7c5dd530` 中，执行图节点被转换为 `OpFeatures`，预测器由工厂构造或接受注入对象，结果经过校验，再交平台进行资源调度。[目标仓固定版本](https://github.com/ZhanluModelSim/operator-performance-models/tree/e1cd95256d68f6e9073df0ec92e5d247dd4e8b54) · [工厂](https://github.com/ZhanluModelSim/operator-performance-models/blob/e1cd95256d68f6e9073df0ec92e5d247dd4e8b54/src/operator_performance_models/backends.py) · [系统仓工厂](https://github.com/ZhanluModelSim/sim-platform/blob/7c5dd530c64fff7a0732e4362697bf0b44f52120/src/sim_platform/backend/factory.py)

两份源码直接加载的 smoke 使用 BF16 连续 ND 的 MM，M=9、N=256、K=128，返回 589,824 次 Cube 运算。硬件输入采用 950DT，16 位 Cube/Vector 分别为 432/54 TFLOP/s，HBM 字段值为 4，利用率为 1，Roofline 头尾开销为零。目标仓工厂的 Roofline 返回 0.016473 µs；注入当前 FullOperatorSimBackend 后，三种 VF 策略均返回 3.472859 µs，标准的 Estimate.to_dict() 能导出结果。这个纯 Cube 例同时验证了元数据约束：省略 stride/layout 时精细路径拒绝。通过上游工厂直接传入 operator_sim_full 会收到 unknown backend。继续把两个 Estimate 送入系统 main 的 validate_estimate，Roofline 通过，精细后端遇到 `ValueError: Prediction cube_us must be finite and nonnegative`。这复现了下表的接口阻塞。本次验到 Python 后端、结果导出和系统校验器；迁移后的 wheel 与全图 DES 尚待实施，两个微秒值只用于复现接口。

## 四个影响数值与覆盖的兼容点

| 项目 | 已观察的差异 | 迁移处理 |
|---|---|---|
| HBM 单位 | 上游 HBM 的 capacity_gb 和 bandwidth_tb_per_second 分别乘 1024³ 与 1024⁴；现有适配器按十进制把数值传给算子模型。字段值 4 在上游对应 4,398,046,511,104 B/s，在当前精细适配器中对应 4,000,000,000,000 B/s。仅传输阶段可相差约 9.95%。 | 用 capacity_bytes / 10⁹、peak_bytes_per_second / 10⁹ 填写十进制 GB、GB/s。归档比较保留原单位口径，新报表版本单列，不追改旧数据。 |
| 耗时分解 | 完整流水总耗时可定价，独占 Cube、Vector 或 HBM 时间有时不可辨识。上游 PredictionDetails 仍要求若干 float，平台对 cube_us/memory_us 等作数值检查。 | 新增不可变的流水明细类型，配套标准结果投影与系统校验；或仅针对 resource-schedule-v1 允许缺项。未知用 None，显式 structural 才用真实零。总时间含启动，平台不得再加 head_tail。 |
| 零分类与通信 | 系统评估器跳过 control/zero 节点。某些旧捕获将有本地设备工作的节点归入 zero；现有全量后端有 prices_metadata 标志。独立通信后端也能为集合通信定价。 | 将控制节点、已证明的视图和需向精细后端询价的节点区分开。按合同约定由一个后端拥有通信与本地融合工作，保留 target/draft、通信排程与带宽占用的归属。 |
| 精度与硬件 | OpFeatures 有物理张量、属性、布局和集体通信事实；精细路径还依赖 scale、打包轴、逻辑 dtype、芯片核数和缓冲。普通 Hardware 峰值不足以唯一确定这些参数。 | 精细入口复用现有精度合同，缺少决定性证据返回 UnsupportedOperator。芯片模板补齐实施参数并随 wheel 发布；用完整有效资源、VF 策略、页表假设和语义版本区分预测缓存。 |

来源见[上游硬件换算](https://github.com/ZhanluModelSim/operator-performance-models/blob/e1cd95256d68f6e9073df0ec92e5d247dd4e8b54/src/operator_performance_models/hardware.py)、[平台预测校验](https://github.com/ZhanluModelSim/sim-platform/blob/7c5dd530c64fff7a0732e4362697bf0b44f52120/src/sim_platform/backend/prediction.py)、[图评估](https://github.com/ZhanluModelSim/sim-platform/blob/7c5dd530c64fff7a0732e4362697bf0b44f52120/src/sim_platform/backend/evaluation.py)和[当前适配器](../src/operator_model/sim_backend.py)。

## 建议实施顺序

1. 在目标仓的 src 下首先同包交付现有 operator_model 命名空间，让 estimate、CLI、算子模块的绝对导入和 importlib.resources 继续有效。pyproject 显式纳入两个包树，登记 data/chips 与 data/parameters 的 JSON。采用单一发行包提供两个命名空间，升级环境先移除旧的 ascend-operator-model wheel，避免两份发行包争用同一路径。源码、参数及测试均在目标仓维护。CA 工具、报告和最小可重放校准数据可在仓内的独立目录维护，安装预测 wheel 无须 CANN 或大 trace。两仓代码的授权需由项目方确认；本次读到的目标仓为私有，GitHub 未识别 SPDX 许可证。
2. 在 backends.py 增加惰性构造入口，沿用 operator_sim、operator_sim_full 后端名。上游 Roofline 的 adaptor/semantic registry 适合将公共算式交给 Roofline 公式；首次移植应保留精细后端已有的 tiling、资源与依赖调度，并在两端有一致输入语义时抽取共享解析。工厂仍拒绝不匹配的利用率 CSV；VF 策略、芯片资源、通信选项通过带配置的后端对象或目标仓自己的构造接口设置。
3. 系统仓保留通用 predict/validate/DES 责任。以实际 main 的输出类型、特征归一版本 6 和零节点策略做一次小型平台适配；本仓附带的 preview 补丁只用于定位行为差异。回退严格由路由器按 UnsupportedOperator 触发，非法输入和内部错误须保留。显式 fine-only 验收先禁用回退；后续按需设置 fallback 并输出命中计数。
4. 同一 CI 中先测原 Roofline 和 communication_v1，随后用独立环境安装目标 wheel，在源码目录外运行 estimate、工厂预测、JSON/CSV 导出及系统单图 DES。覆盖 BF16、MXFP8/MXFP4、A8W4 展开、未知精度拒绝、布局丢失、通信、纯视图、可选/缺失字段和上下文敏感的缓存键。再以固定合同、同图、同单位复核 27 组及 33 组，并对新增模型逐个检查无漏报/无零时占位。单独报告修正 HBM 换算前后的差异与三芯片证据等级。最后才切换系统默认配置。
