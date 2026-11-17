# 第一版实现：三个算子的 CA 校准闭环

本版是可运行、可测误差的 **tile/stage 半经验模型**。预测输入只有算子和 shape，在线不读取该 shape 的 CA 结果，也不回放指令。成本来自训练点的 CA 区间统计，因此本版没有声称获得独立的 ISA 峰值模型，更不把 A5 校准系数当作可直接迁移的 A7/A8 硬件常数。

模型与评估对应 GitHub 私有仓库 `Runze-Lin/operator_sim`。冻结预测的提交为 `6e005b0`；留出采集在模型冻结之后开始。实测结果见 [结果报告](../results/v0.1/report.md) 和 [逐点 JSON](../results/v0.1/evaluation.json)。

## 支持范围

| 算子 | 固定语义 | shape 输入 | 首轮覆盖 |
| --- | --- | --- | --- |
| dynamicMXquant | ACLNN V1、BF16→E4M3FN、E8M0 scale、axis=-1、block32、rint、scaleAlg=0 | rows, cols；cols 是 64 的倍数 | 4–64 行、256–1024 列，含 384/768 列尾块 |
| matmul | BF16 ND、无转置、cubeMathType=0 | M,N,K | M=1/17/32/48/128，N=64/96/128/256，K=64/128/192/512 的少量组合 |
| FA | 标准非 causal attention、BF16 SBH、B=H=1、D=64，无 dropout/mask/pse | Q,KV,D | Q=1/8/16/17/64、KV=64/96/128/192/256 的少量组合 |

“首轮覆盖”不是这些维度笛卡尔积都已验证。入口会拒绝未实现的算子和明显不支持的 shape；位于训练 shape 包围盒之外返回 `extrapolation`。包围盒内也不等于已验证 kernel 分支：`kernel_selection_verified=false` 明确保留这一点。本版没有支持 MLA/CSA/KDA、GQA、paged KV、grouped GEMM、FP4 MatMul 或通信融合。

## 模型与校准

1. **Shape→分核/工作量**：quant 根据公开尾轴 block32 规则计算 256/512 列 slice 和二维分核；matmul 用 16×64 的基本输出工作块及轮次近似；FA 用 Q64/KV128 分块、1 AIC 配 2 AIV 的小规模分工。后两项是有来源与实验核对的简化假设，不是完整移植 CANN host tiling。每个采样点记录预测核数与实际核数是否一致。
2. **阶段成本**：读入、Vector、Cube、FIXPIPE、写出各有工作量单位。以训练 trace 的非同步、非 wait 指令区间并集估计阶段成本，用非负 `a+b×work` 拟合。每核取最大阶段占用作为校准观测；并不把不同核上的最大值视为同一条实际关键路径。
3. **有限流水**：quant/matmul 保留首块填充和后续轮次重叠；FA 保留 QK→Vector→PV 依赖、Cube 共享和在线 softmax/累加相关的循环约束，用填充、稳态间隔、排空公式求值。不会将 C/V 的整段首尾跨度分别缩放后取 max。
4. **总时延修正**：每个算子仅拟合一个非负固定项和一个非负 body 缩放项。固定项吸收本小范围内未显式描述的标量控制/初始化等成本，不能解释为纯 host launch latency。没有每个 shape 专属的拟合系数。
5. **指令统计**：采样保留每核动态 opcode 次数、按 pipe 次数和 timeline busy 并集；预测器可给出按每核工作量拟合的 opcode 次数诊断。它尚不是经过源码循环展开验证的精确计数器。

公式中的时间单位均为 μs：

```text
phase_j = nonnegative_intercept_j + nonnegative_slope_j × work_j(shape)

quant/matmul body = sum(phase_j) + (waves-1) × max(phase_j)

FA fill = read + cube(QK+PV) + vector + fix
FA II   = max(read, cube+fix, vector+cube/2)
FA body = waves × [fill + (KV_tiles-1) × II + write]

prediction = nonnegative_fixed + nonnegative_body_scale × body
```

阶段成本是从 CA 的占用/延迟观测校准的代理量，仍可能包含上游供数、资源竞争等影响。因此 `stage_services_us` 不应称为跨代不变的纯硬件服务需求；`ideal_compute_efficiency` 返回 null。实际 CA 的 `busy_fraction_by_pipe` 可用于观察，占用区间并集与有用计算效率保持区分。

## 采样和防止泄漏

- [sampling-v1.json](../configs/sampling-v1.json) 先固定 15 个不同 shape：9 个校准点、6 个留出点；另有 3 次相同 shape 复测。每个调用先比较 CPU golden，失败不产生有效时延。
- `fit()` 拒绝任何非 calibration 记录；训练集身份不一致时拒绝拟合。冻结模型、实现 SHA256、样本清单 SHA256 与全部预测。留出采集前再次核对哈希；评估也核对冻结预测可复现。
- 留出结果不参与本版调参。若发现不足，保留 v0.1 数据，下一版另选未用过的 holdout。
- 误差报告保留缺失/失败的分母，分别给各算子和整体的 WAPE、P90 APE、绝对误差及偏差。采用统一权重，尚无生产请求分布权重。
- 比较固定耗时基线、经验 Roofline、只有阶段成本的模型和最终校准模型。经验 Roofline 的有效速率从校准点取得，不冒充理论下界或用户原系统实现。

## 时间、缓存与预算

CA 参考定义为合并 trace 中 `max(ts+dur)-min(ts)` 的跨核指令时间窗。`displayTimeUnit=ns` 只是界面显示单位，`ts/dur` 按 μs 读取；解析器逐核与 profiler 的 `duration_time(us)` 对照，偏差超过容差便拒收。该参考包括所捕获的初始化/标量/等待指令，**不包括 host enqueue、未捕获的设备 dispatch、ACL API 全部时间**。不把未经验证时钟域的日志 TASK 时间戳直接当核周期换算。

CA 使用安装包原有配置，`L2CACHE.cache_enable=0`；本轮没有改变该设置。结论适用于这个 CA 配置，不代表热缓存真机。原始 trace 中仍有未识别配置字段告警和缺失源码行映射，记录在每个样本中。

采集串行运行，目录独立。缓存键包含主机调用二进制、安装包来源清单、SoC 配置、CA 配置、shape、固定属性、输入模式和采集选项；使用缓存前验证 trace SHA256。安装包需保持固定，不能在原前缀手动替换组件后仍认为旧包清单有效。捕获的设备 kernel 二进制 hash 和名称另存于结果中。

总预算：7200 秒累计采集墙钟、28800 秒累计 CPU、10 GiB raw、18 次尝试；单例外层 360 秒，工具内部 timeout=2 分钟。墙钟在启动前预留单例预算；CPU 与 raw 容量在每例结束后统计，因此可能最多超出一个运行中的 case。失败保留并计入预算，不自动无限重试。

预测器无第三方运行依赖、无在线文件读取；模型构造时加载一次小 JSON，使用最大 4096 项 LRU，缓存结果不可变。批量接口在同一进程内执行。报告分别测量模型加载、冷进程、缓存命中/未命中 P50/P95、批量吞吐及缓存内存。尚未接入用户整网模拟器，所以不宣称已经满足“整网额外开销 <5%”。

## 使用与复现

本项目已安装的 Conda Python 可以直接用于在线预测；该命令不激活 CANN：

```bash
toolchains/envs/cann-9.2.1/bin/python -m operator_model --op matmul --shape 17 96 128
toolchains/envs/cann-9.2.1/bin/python -m operator_model --op quant --shape 8 384 --instruction-counts
toolchains/envs/cann-9.2.1/bin/python -m unittest discover -s tests -v
```

在另一台无 CANN 的机器上，仅复制仓库、使用 Python≥3.10 即可预测和运行单测；`python -m pip install -e .` 是可选项。

```python
from operator_model.model import Case, Predictor

model = Predictor("results/v0.1/model.json")
result = model.predict(Case("fa", (17, 96, 64)))
batch = model.predict_many([Case("matmul", (17, 96, 128))] * 100)
```

本次离线闭环执行顺序如下。采集需要已经安装本项目 CANN/Conda 并允许本机 socket；已有 v0.1 冻结文件不会被自动覆盖：

```bash
bash scripts/build-smoke.sh
python3 scripts/collect-v1.py --split calibration
python3 scripts/fit-v1.py
python3 scripts/collect-v1.py --split holdout
python3 scripts/collect-v1.py --split repeat
toolchains/envs/cann-9.2.1/bin/python scripts/evaluate-v1.py
```

采集成功记录可断点复用；失败或未完成目录必须先检查原因。本版脚本针对固定的首轮实验，并非通用大规模调参平台。若只需重新查看结果，无需再次运行 CA，直接读取 `results/v0.1`。
