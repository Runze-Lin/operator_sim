# 在 SIMPlatform 中切换算子后端

一次推理任务可以给不同节点选不同的耗时模型。在任务 TOML 中把 `operator_sim` 放在 `roofline` 前面，能映射到具体实现的本卡调用使用分块与流水预测，其余节点继续使用原系统的定价和缺失处理。两个后端共享同一份图、通信事实、硬件配置和系统调度。

```toml
[prediction]
backends = ["operator_sim", "roofline"]
```

运行原方案时设 `backends = ["roofline"]`。使用同一任务做比较时也可以在命令行选路由。分别指定输出目录可以保留逐节点的两份账本。

```sh
sim-platform infer task.toml --set 'prediction.backends=["roofline"]' \
  --set 'output.directory="./results-roofline"'
sim-platform infer task.toml --set 'prediction.backends=["operator_sim","roofline"]' \
  --set 'output.directory="./results-operator-sim"'
```

### 安装到其他同版本的平台

本机已在 `preview-0.1.0-preview.1/.venv-runtime` 接好后端。复用其他安装时，激活运行 `sim-platform` 的 Python 3.12 环境，设置 `OPERATOR_SIM` 为本仓库路径，`SIM_WORKSPACE` 为包含 `sim-platform/` 和 `operator-performance-models/` 的平台源码目录。在干净的 preview.1 源码上应用一次补丁，再安装。

```sh
patch -d "$SIM_WORKSPACE" --dry-run -p1 < "$OPERATOR_SIM/integrations/sim-platform/0.1.0-preview.1.patch"
patch -d "$SIM_WORKSPACE" -p1 < "$OPERATOR_SIM/integrations/sim-platform/0.1.0-preview.1.patch"
python -m pip install --no-deps --force-reinstall "$SIM_WORKSPACE/operator-performance-models" "$SIM_WORKSPACE/sim-platform" "$OPERATOR_SIM"
python -c 'from operator_performance_models import create_backend; print(create_backend("operator_sim").version)'
```

补丁对应 `operator-performance-models` 提交 `e6c9b31ab0a0fe66a732c25cd65d86d4f3ddac06` 和 `sim-platform` 提交 `86b3995996eec5043b40ae417ed8c69f3022f675`。它在原工厂中按需加载插件，并为整窗预测引入 `resource-schedule-v1` 详情格式，让平台在校验和缓存时保留未知的分阶段耗时。负载仓、平台 DES 和原有模型继续使用该 preview 的实现。模型版本、映射版本、芯片及参数指纹共同参与缓存身份。[可移植补丁](../integrations/sim-platform/0.1.0-preview.1.patch) · [平台后端](../src/operator_model/sim_backend.py) · [张量合同映射](../src/operator_model/system_adapter.py)

## 图上的时间怎样算

1. 负载仓捕获 target 和 draft 在指定 rank 上的调用。系统把输入输出的本卡维度、数据类型、步长和调用属性转换成算子特征。例如 TP 已经切过 head，后端直接使用捕获的局部形状。再次除以 TP 会低估工作量。
2. 精细后端核对输入与输出是否符合具体库实现的合同。物化为 BF16 的 cache 读取、FP4 原生矩阵乘、普通 BF16 矩阵乘各有自己的数据流。仅凭节点名中的 Matmul、Attention 或 Quant 字样选公式会混淆这些实现。节点在预测过程中可能乱序或命中缓存，无可靠流历史时使用仅依赖当前输入的预测。
3. 命中时用 API 的 `latency_us` 替换该节点的旧价格。设备启动、核内流水与尾波已经计入。该接口只交付设备窗，宿主 dispatch 可由系统单独计入。本次配置的额外首尾开销为零。实现缺失、精度不匹配或缺少必要的路由属性时进入下一后端。原系统处理零成本视图、同步占位、通信及最终缺失；检查 L1 的 `backend_name` 和 `prediction_status` 可还原预算落在哪一条路径。
4. 平台将选中的单节点时间交给原有离散事件调度器（DES），按依赖、资源和并发规则推演，再汇总 target、draft、单轮时间及可用的吞吐指标。差值在同一轮的两个 DES 结果上计算。有重叠的图中，各算子串行时间差之和可以大于或小于轮次时间差。

### 哪份 950DT 配置生效

本次 A/B 两侧都读取用户原任务的 `950DT.jsonc`。其中 Cube 16/8/4 位峰值为 500/1,000/2,000 TOP/s，Vector 32/16/8 位为 16/31/62 TOP/s，片外容量和带宽为 144 GB、4 TB/s。精细后端用这些显式值给工作量定价，采用 9582 模板中的 1.65 GHz、32 AIC、64 AIV、片上缓冲和流水参数。用于 CA 校准的 9582 则是 432/865/1,730 TOP/s 的 Cube 与 103.079 GB 的片外容量。本次精细时间属于指定系统硬件配置上的外推。将硬件改为其他 SKU 时，核数与流水参数要来自对应模板。[标定 SKU 配置](../src/operator_model/data/chips/Ascend950DT_9582.json)

已登记的模板名有 950DT、950PR 和 970DT。研究其他芯片时可以通过平台 Python 入口注入完整模板，键名应与任务的硬件 `name` 相同。模型向目标芯片迁移同类实现，结果标为外推。通信和未覆盖算子的价格仍由任务所选的其他后端提供。

```python
from operator_model.sim_backend import OperatorSimBackend
from sim_platform.api.inference import run_inference

backend = OperatorSimBackend(chips={"980DT": "/path/to/operator-sim-980dt.json"})
result = run_inference(
    "task-980dt.toml",
    overrides={"prediction.backends": ["operator_sim", "roofline"]},
    extra_backends=(backend,),
)
```

### 三个容易混在一起的量

| 量 | 计算与用途 |
| --- | --- |
| 理想 Roofline 耗时 | 对每个资源，用必须完成的工作量除以其峰值吞吐，再取最大值。若张量流量是本次整窗真实经过该级接口的字节，便有 `T_ideal ≤ T_device`。线程数少、启动、依赖、尾波都能使实际耗时更长。 |
| 系统已有 Roofline 预测 | 本次 950DT 预置的 Cube、Vector 和片外带宽利用率分别是 0.6、0.6、0.8。通用基类取受这些效率折减后的计算、访存、通信时间的最大值，再加设定的首尾开销。特定算子可采用专门公式。这个预测可高于或低于精细模型。 |
| 误差表的逻辑系数 U | 用逻辑张量字节和有用算量折算的参考时间除以精细预测。四个实际窗口的复算展示了小 shape、模型低估和片外流量口径对它的影响。[单算子误差、逻辑系数与物理下界](error-analysis.md#为什么有些-u-很小有些超过-100) |

## 九组配置的比较条件

使用 `0.1.0-preview.1` 上的三个 preset，950DT，128 Ki 历史，每请求目标输出 1,024 tokens。每轮 target 校验 9 个位置（8 个候选和 1 个 anchor），draft 也执行一次。平台设定接受 6 个候选，每轮按 7 个输出 token 归一。TP8/EP8、TP8/EP32、DEP64 的 (TP,DP,EP) 依次为 (8,1,8)、(8,4,32)、(1,64,64)。每组单卡 batch 依次取 1、16、64，共 27 个任务。

此前矩阵中的 **114.35、120.63、140.73 等是每卡静态峰值显存 GB**。更换时间模型时可以用它们核对两次捕获是否一致。此次对比的时延取系统 `results/l1/inference.json` 中 rank 0 的 `statistics[0].duration_us`，即一次 target 加一次 draft 的已定价单轮耗时。比较的是两个仿真后端；端到端误差需要独立的整网设备实测。跨 DP 的 EP 同步和服务端采样等未计入，缺少价格的节点在账本保留缺失状态。按系统的 144 GB 容量，有 8 个配置的静态峰值超出容量，依然可以分析捕获图。[原始负载、显存与各算子的覆盖](system-workload-coverage.md)

### 单轮耗时（v1.7 归档）

含 Flash 候选 Indexer 的当前版本另列全部 27 组的三方耗时、覆盖和逐算子差值。[当前端到端对照和回退排名](system-candidate-comparison.md)

每格为 **原 Roofline → 精细优先、Roofline 回退**，单位 ms。括号内是相对原值的变化，正号表示新方案预测耗时更长。两侧均从原任务调用 `run_inference`，重新捕获图并运行 DES。逐组核对 target/draft 的节点、张量和依赖指纹、静态峰值及逐节点回退价格。27 个原后端单轮时间也逐项复现原归档。[全部 27 组，含 target、draft、覆盖、墙钟和输入指纹](../calibration/reference/system-decode-compare.csv)

| 模型 | 并行布局 | 单卡 B=1 | 单卡 B=16 | 单卡 B=64 |
| --- | --- | ---: | ---: | ---: |
| DeepSeek-V4 Pro | TP8/EP8 | 36.94 → 39.05 (+5.7%) | 67.23 → 72.81 (+8.3%) | 167.17 → 184.82 (+10.6%) |
| DeepSeek-V4 Pro | TP8/EP32 | 11.62 → 18.75 (+61.4%) | 41.90 → 52.46 (+25.2%) | 145.38 → 164.61 (+13.2%) |
| DeepSeek-V4 Pro | DEP64 | 12.19 → 20.12 (+65.1%) | 75.45 → 72.95 (−3.3%) | 301.44 → 257.60 (−14.5%) |
| DeepSeek-V4.1 Flash | TP8/EP8 | 21.33 → 25.46 (+19.4%) | 150.07 → 165.96 (+10.6%) | 563.58 → 617.13 (+9.5%) |
| DeepSeek-V4.1 Flash | TP8/EP32 | 12.20 → 18.14 (+48.7%) | 140.95 → 158.60 (+12.5%) | 555.70 → 609.85 (+9.7%) |
| DeepSeek-V4.1 Flash | DEP64 | 12.47 → 18.63 (+49.3%) | 152.08 → 165.01 (+8.5%) | 608.15 → 639.39 (+5.1%) |
| Kimi K3.1 Pro | TP8/EP8 | 63.16 → 68.50 (+8.5%) | 119.42 → 152.61 (+27.8%) | 307.73 → 346.77 (+12.7%) |
| Kimi K3.1 Pro | TP8/EP32 | 20.64 → 34.41 (+66.8%) | 77.00 → 118.66 (+54.1%) | 274.18 → 312.79 (+14.1%) |
| Kimi K3.1 Pro | DEP64 | 26.58 → 43.28 (+62.8%) | 170.30 → 179.82 (+5.6%) | 680.01 → 581.57 (−14.5%) |

24 组预测变长，3 组变短。27 组的 171,232 个节点中，原账本成功定价 98,920 个。精细后端接管 47,820 个，占成功定价节点的 **48.34%**，对应原耗时的 **37.14%**。其余 51,100 个由 Roofline 定价。另有 64,365 个零成本节点、6,078 个占位、1,869 个缺价和 0 个预测失败。本组只在 TP8/EP8 导出 TPOT；另外两种布局的 `tpot_us` 为空，平台记录的原因是跨 DP 的专家同步耗时未汇总。

离线 A/B 每侧用 6 个工作进程。按单任务计，包含重新捕获与 DES 的宿主墙钟中位数为原后端 10.88 s、精细优先 11.76 s。单独检查 BF16 Matmul `[9,128,512]` 时，已构建后端的首次查询为 5.158 ms，随后 99 次直接查询的中位数为 0.201 ms、P95 为 0.217 ms。这个样本的设备时间为 4.228 µs。墙钟会受主机并发影响。[逐任务墙钟](../calibration/reference/system-decode-compare.csv)

### 哪些算子改变了结果

下表把同一原始调用名在所有 27 组中**由精细后端接管的节点**相加。各组是不同工作负载，这些合计用于差值归因。本次逐组验证节点串行时间之和等于单轮 DES 时间。[各配置的逐算子差值](../calibration/reference/system-decode-operators.csv)

| 命中的实现 | 调用数 | 原 Roofline 合计 ms | 精细合计 ms | 差值 ms |
| --- | ---: | ---: | ---: | ---: |
| 连续 DynamicMXQuant，BF16 → MXFP4 | 14,400 | 788.21 | 1,314.75 | +526.54 |
| 路由专家原生 A4W4 GroupedMatmulV5 | 3,582 | 812.78 | 433.38 | −379.40 |
| 逐 token FP8 量化 | 8,253 | 23.60 | 92.98 | +69.38 |
| 物化 MXFP4 cache，AntiMxQuant → BF16 | 1,659 | 11.22 | 44.81 | +33.59 |
| 普通 BF16 RMSNorm | 8,001 | 15.70 | 46.48 | +30.78 |
| vLLM HcPost | 1,926 | 16.72 | 38.93 | +22.22 |
| 稠密原生 A4W4 Matmul | 8,181 | 113.58 | 132.70 | +19.12 |
| FP32 Sigmoid | 684 | 0.02 | 2.08 | +2.06 |
| Cast BF16 → FP32 | 1,134 | 6.47 | 7.11 | +0.64 |

**小 batch 的量化。** Kimi TP8/EP32、B=1 的 target 节点 13 量化 BF16 `[9,7168]`。原账本将读 129,024 B、写 18,144 B 按有效带宽 3.2 TB/s 定价，得到 0.046 µs，首尾取零。按 block32 实现，64,512 个 FP4 值占 32,256 B，另有 2,016 B 的 E8M0 scale，实际所需的逻辑写出是 34,272 B。精细模型首发 2.218 µs，关键核流水与核间到达差 2.024 µs，尾部 0.010 µs，共 4.252 µs。即使把旧公式的写出修正为 34,272 B，其带宽项也只有 0.051 µs；主要差额来自启动和流水。此配置 882 次连续 FP4 量化累计增加 5.90 ms，34 次 cache 展开另增加 5.47 ms。[量化阶段与校准依据](examples/quant-mxfp4.md)

**大 batch 的专家乘。** Pro DEP64、B=64 的 target 节点 112 是 A4W4 gate/up 投影，6 个本地专家各收到 576 行，K=7,168、N=6,144。两侧有效计算量相同，都是 `2×3456×7168×6144 = 304,405,807,104` 次操作。原账本将计算精度记作 BF16，用 `500×0.6 = 300` TOP/s，算出 1,014.69 µs。原生 A4W4 模型按系统配置的 2,000 TOP/s、专家 tile 的分核与搬运排程给出 228.22 µs。精度选择已经改变计算项，单独拿“理想调度与实际调度”解释这里的差值会遗漏主要因素。此配置的 128 个匹配专家乘一共减少 75.31 ms，被量化等节点抵消一部分，整轮减少 43.84 ms。该大形状和系统芯片预置均超出了逐形 CA 校准合同，228.22 µs 是外推。[路由专家乘的实现假设与校准](examples/routed-grouped-matmul.md)

**剩余预算。** Flash TP8/EP32、B=64 的候选过滤 Indexer 仍由 Roofline 定价，单项 386.58 ms，占其原单轮 555.70 ms 的 69.6%。捕获合同目前支持返回工作量，缺少该稀疏实现的完整校准时间。A8W4 的逐 token FP32 激活 scale、非连续 FP4 量化、FP8 cache，以及捕获为其他融合或稀疏签名的 Attention 也沿原路由。系统的匹配说明选择哪一种实现来模拟；捕获图本身没有记录实际运行的 kernel ELF。[各节点的合同与缺口](system-workload-coverage.md)

这 27 组命中了表中的九种实现。适配器还支持双输入连续、BF16 输出的普通 `aten.mm`，这批调用中没有满足条件的实例。对未显式给出 `scale_alg` 的 FP4 量化节点采用 OCP（0）假设，要求末轴 block32、round 及配对 E8M0 输出。改变语义或调用布局时应重新检查命中情况。[捕获图中 FP4 的连续与跨步输入](system-quant-shapes.md)

### 复算 27 组

在 operator-sim 仓库根目录，使用上面准备好的平台环境。令 `MATRIX` 指向带原任务和原 L1 归档的 `experiments/decode-950dt-128k`，`SCRATCH` 指向新的运行目录。原生推理依赖需要项目现有的 CPU PyTorch 环境。本机还使用 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`，并将 preview 的 `compiler-runtime` 放在 `LD_LIBRARY_PATH` 中。

```sh
python -m calibration.system_compare "$MATRIX" "$SCRATCH" roofline --jobs 6
python -m calibration.system_compare "$MATRIX" "$SCRATCH" hybrid --jobs 6
python -m calibration.system_compare "$MATRIX" "$SCRATCH" compare --output "$SCRATCH/tables"
```

比较脚本在返回的系统结果上逐节点核验，并把两种运行的摘要分目录保留。按前文的单任务 CLI 运行时还可以导出完整 `l1/`，其中 `000_verify_target.csv`、`001_verify_draft.csv` 给出节点价格，`inference.json` 给出单轮时间及显存。[复算脚本](../calibration/system_compare.py)
