# 在 SIMPlatform 中切换算子后端

950DT 的长上下文场景已覆盖三个模型的 prefill 与推测 decode。全图后端默认要求冷读写在本节点内付费，原有 Roofline 后端仍可用同一份任务运行。[各配置的两阶段时延、资源约束和复算方式](context-sweep.md)


| 任务 TOML 中的 `prediction.backends` | 计时范围 |
|---|---|
| `["operator_sim_full"]` | 对 27 个 decode 模板的全部原始调用独立定价，含设备、网络和视图。新合同缺失会在系统账本显示 miss。 |
| `["operator_sim","roofline"]` | 以较严格的既有算子合同匹配，剩余调用交给原 Roofline。这是 v1.8 混合比较口径。 |
| `["roofline"]` | 原平台计算方法和通信缺价处理，用于复现基线。 |

```toml
[prediction]
backends = ["operator_sim_full"]
```

三种模式共享单卡捕获、图依赖、并发策略与显存预算。全量比较工具逐节点验证成功状态；新模型接入时也应检查 miss/fail 和节点级来源。跨设备用显式互连情景，芯片 HBM 配置继续用于本地搬运。[27 组同精度 Roofline 与当前完整后端](roofline-audit.md) · [通信推导及调参](examples/system-communication.md)

## 安装到其他同版本的平台

本机已在 `preview-0.1.0-preview.1/.venv-runtime` 接好后端。复用其他安装时，激活运行 `sim-platform` 的 Python 3.12 环境，设置 `OPERATOR_SIM` 为本仓库路径，`SIM_WORKSPACE` 为包含 `sim-platform/` 和 `operator-performance-models/` 的平台源码目录。在干净的 preview.1 源码上应用一次补丁，再安装。

```sh
patch -d "$SIM_WORKSPACE" --dry-run -p1 < "$OPERATOR_SIM/integrations/sim-platform/0.1.0-preview.1.patch"
patch -d "$SIM_WORKSPACE" -p1 < "$OPERATOR_SIM/integrations/sim-platform/0.1.0-preview.1.patch"
python -m pip install --no-deps --force-reinstall "$SIM_WORKSPACE/operator-performance-models" "$SIM_WORKSPACE/sim-platform" "$OPERATOR_SIM"
python -c 'from operator_performance_models import create_backend; print(create_backend("operator_sim_full").version)'
```

补丁对应 `operator-performance-models` 提交 `e6c9b31ab0a0fe66a732c25cd65d86d4f3ddac06` 和 `sim-platform` 提交 `86b3995996eec5043b40ae417ed8c69f3022f675`。它在原工厂中按需加载插件，并为整窗预测引入 `resource-schedule-v1` 详情格式，让平台在校验和缓存时保留未知的分阶段耗时。负载仓、平台 DES 和原有模型继续使用该 preview 的实现。全量后端显式为别名/空分配返回 0，旧后端仍可跳过这些节点。适配器按有序 EP 组补入旧版 Combine 捕获缺少的本地组序号。模型版本、映射版本、互连参数、芯片及参数指纹共同参与缓存身份。[可移植补丁](../integrations/sim-platform/0.1.0-preview.1.patch) · [平台后端](../src/operator_model/sim_backend.py) · [张量合同映射](../src/operator_model/system_adapter.py)

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
    overrides={"prediction.backends": ["operator_sim_full"]},
    extra_backends=(backend,),
)
```

### 三个容易混在一起的量

| 量 | 计算与用途 |
| --- | --- |
| 理想 Roofline 耗时 | 对每个资源，用必须完成的工作量除以其峰值吞吐，再取最大值。若张量流量是本次整窗真实经过该级接口的字节，便有 `T_ideal ≤ T_device`。线程数少、启动、依赖、尾波都能使实际耗时更长。 |
| 系统已有 Roofline 预测 | 本次 950DT 预置的 Cube、Vector 和片外带宽利用率分别是 0.6、0.6、0.8。通用基类取受这些效率折减后的计算、访存、通信时间的最大值，再加设定的首尾开销。特定算子可采用专门公式。这个预测可高于或低于精细模型。 |
| 误差表的逻辑系数 U | 用逻辑张量字节和有用算量折算的参考时间除以精细预测。四个实际窗口的复算展示了小 shape、模型低估和片外流量口径对它的影响。[单算子误差、逻辑系数与物理下界](error-analysis.md#为什么有些-u-很小有些超过-100) |

## 重放 27 个任务

同一比较矩阵使用原预置、128 Ki 历史、目标输出 1,024 tokens、8 个候选、接受 6 个候选，每轮按 7 个输出 token 归一。TP8/EP8、TP8/EP32、DEP64 的 (TP,DP,EP) 分别为 (8,1,8)、(8,4,32)、(1,64,64)。每组单卡 batch 取 1、16、64。复算器逐节点检查调用、原价和完整路由，逐任务核对显存峰值与图指纹，再生成 27 组对比和所有 shape 明细。[v1.9.1 的对照与复算命令](roofline-audit.md) · [原始 27 组账单](system-full-comparison.md) · [v1.8 混合后端归档比较](system-candidate-comparison.md)
