# 950DT 环境与三个 CA smoke case

2026-11-17：三例均完成仿真、数值检查和 profiling 解析，外层进程均退出 0。每个算子只执行一个输入 shape，没有展开采样或校准。

## 环境

- 主机：aarch64 / openEuler 22.03。工具链在项目 `toolchains/` 下。
- micromamba 2.10.0 管理的 Conda 环境：`toolchains/envs/cann-9.2.1`，Python 3.12.16。不是仅隔离 Python 的 venv。
- CANN Toolkit + 950 Ops：9.2.1，安装时官方索引中最新稳定版；另有 9.3.0.beta.1 预发行版，本次未选择 beta。
- 实际 CANN 前缀：`toolchains/cann/9.2.1/cann-9.2.1`。
- 实际仿真目标、运行时回报均为 `Ascend950DT_9582`。使用官方 `dav_3510/camodel` 后端，没有用重命名方式冒充 DT。
- 该版本 DT 平台配置：32 Cube、64 Vector，Cube/Vector 频率均 1650 MHz。这是 SDK 目标配置，不是物理板卡测量结果。
- 全部操作走 C++ ACLNN 调用安装包的现成算子；不需要 torch、vLLM 或模型权重。

原始安装包大小、SHA256、来源及内置完整性检查记录见 [cann-packages.json](cann-packages.json)。实际组件命令、返回码及平台配置摘要见 [cann-installation.json](cann-installation.json)。

安装存在一个需保留的环境差异：官方复合安装入口受当前用户组和父目录权限检查阻挡，没有完整执行成功；随后使用包内未修改的官方子安装器完成全部 32 个 Toolkit 组件和 8 个 Ops 组件。Ops 中同版本 opbase 已由 Toolkit 安装。按组件要求选择私有或 `--install-for-all` 模式，仅调整新建安装前缀；用户 home 与项目目录仍为 700，没有修改驱动、用户组或系统 CANN。Simulator 使用安装器已有的 `CANN_ECO_SCENE=y` 路径避开缺失的旧 latest_manager。安装记录中保留了实际参数，不能把这次安装视作标准复合入口验收。

环境包装脚本清理继承的 CANN/Python 搜索路径，加载项目版本；检查过实际链接库来自新 CANN 和 Conda。Conda 前缀内 Python 依赖 `pip check` 通过。加载 CANN 自带 Python 包后，完整 `pip check` 仍报告未安装的可选依赖：服务 profiler 的 msguard、两个 OTLP exporter，以及 Python ops 绑定的 torch/torch-npu 和 ninja Python distribution。Conda 的 ninja 可执行文件已安装。本次 C++/CA 链路不使用这些绑定，不能据此宣称整个 PyTorch/服务 profiler 环境可用。

## 结果

| 算子 | 输入 | 数值校验 | 实际记录的核 | profiler 核级 duration |
| --- | --- | --- | --- | --- |
| dynamicMXquant | BF16 `[4,256]` → E4M3FN，block=32 | 1024 输出 + 32 scale 字节全部一致 | 4 个 AIV | 3.86–4.42 μs |
| matmul | BF16 M=32,N=64,K=64 | 2048 输出全部满足 atol=0.01、rtol=0.005；最大绝对误差 0.015625 | 2 个 AIC | 3.30–3.39 μs |
| FA | BF16，SBH，B=1,H=1,Q=16,KV=64,D=64，非 causal | 1024 attention 输出全部满足 atol=0.002、rtol=0.01；最大绝对误差 0.000380433 | 1 个 AIC + 2 个 AIV | AIC 8.73 μs；AIV 8.90/8.98 μs |

表中数值来自工具的核级摘要，包含其统计口径下的等待/启动等影响，**不是端到端算子时间，也不是纯 Cube/Vector 算术耗时**。初次运行未单独计量主机 wall time，不能用这些微秒数衡量 CA 仿真开销。复跑脚本已增加主机计时。小 shape 也不足以判断 FA 在真实工作负载上是否 Vector bound。

取得的数据：

- 每核原始 `instr_log.dump` / `instr_popped_log.dump`，包含 PC、指令、周期及元数据。
- 每核 `*_instr_exe.csv`，包含指令、pipe、call_count、cycles、running_time。
- 每核与合并的 `trace.json`，以及 `visualize_data.bin`。
- profiler 捕获的算子对象 `aicore_binary.o`，其 SHA256 和实际 kernel 名称已记录。
- matmul 的时间线包含 2 条 MMAD；FA 包含 2 条 MMAD 和 316 条 RVECEX 执行事件；quant 包含 328 条 RVECEX 执行事件。这里数的是执行事件，排除了只用于显示的轨道元数据。

完整机器可读摘要见 [smoke-950dt-results.json](smoke-950dt-results.json)，原始文件位于 `runs/smoke-950dt/{quant,matmul,fa}/OPPROF_*/`。安装包二进制是本次执行基线；上游 ops-nn / ops-transformer 的固定 commit 仅用于参考 ACLNN 调用示例，不声称它们与安装包二进制源码完全一致。

## 后续建模的数据边界

1. CSV 行是聚合记录，动态次数看 `call_count`。按 pipe 计数仍包含同步：本例 FA 的 CUBE pipe 有 21 次调用，其中只有 2 次 MMAD，其余为 SET_FLAG/WAIT_FLAG。因此不能把 pipe 总数当作算术指令数。
2. 流水事件存在重叠、等待以及零时长记录。CSV 累加的 latency、raw dump 记录数、时间线事件数口径不同；模型必须区分算术吞吐、同步和访存，不能直接累加或互相替换。未来解析先用 opcode 分类，并验证周期到时间的转换。
3. 预编译 kernel 缺少源码调试行信息，工具报告 code call stack 为空；指令和时间线可用，但本次拿不到可靠的源码行级热图。主机程序加 `-g` 不会给安装包里的设备 kernel 补齐调试信息。
4. 仿真日志包含 `STAT.su_perf_summary_log_enable`、`event_type_4` 未识别配置告警。三个 case 都正常完成，数值和指令数据存在；相关统计字段的完备性尚未验证，不作为校准依据。
5. 初次 FA 采集约 167 MB，quant 约 9.8 MB，matmul 约 4.7 MB。后续采用少量代表 shape 的详细 trace、分支/tiling 缓存与聚合结果；不对所有 shape 长期开启原始 dump，也不在系统仿真的每次预测中调用 CA。

本轮结论是：950DT 上三个代表算子的 CA 调用、正确性检查、指令和流水采集链路可用。利用率模型、采样误差与 A7/A8 外推尚未实现或验证。

## 复跑

在项目根目录执行：

```bash
bash scripts/build-smoke.sh
bash scripts/run-smoke.sh quant
bash scripts/run-smoke.sh matmul
bash scripts/run-smoke.sh fa
```

每条命令只跑一个 case，输出到独立的 `runs/manual-smoke/` 子目录；记录 wall time，外层限制 360 秒，工具内 timeout 为 2 分钟。仿真器需要可创建本机进程间 socket 的运行环境。三个初次采集的摘要可由 `python3 scripts/summarize-smoke.py` 重建；该脚本针对固定初次运行目录，不自动选择后续复跑结果。

Python/Conda 版本见 `environment.yml`、`configs/conda-linux-aarch64.lock.txt` 和 `configs/python-cann-requirements.txt`。这些文件不包含 CANN 安装包本身。模型 config 来源与 shape 推导边界见 [model-configs.md](model-configs.md)。
