# v0.2：按实现机制扩展的 CA 校准模型

本版在 v0.1 可复现闭环上，补充不同 shape 引起的分核、tile、UB 循环和 FA kernel 变化。在线查询只使用 shape、固定算子属性和冻结的小型参数文件，不调用 CANN、CA 或读取目标 shape 的 trace。

代码版本为 0.2.0；长期设计文档的方案版本独立。实际测试结果见 [本轮报告](../results/v0.2/report.md)，历史小 shape 结果保留在 [v0.1 报告](../results/v0.1/report.md)。

## 使用

在仓库根目录运行；普通 Python 3.10+ 即可，也可使用本地独立 Conda 的 `toolchains/envs/cann-9.2.1/bin/python`。

```bash
python3 -m operator_model --op quant --shape 640 4096 --instruction-counts
python3 -m operator_model --op matmul --shape 129 256 256
python3 -m operator_model --op fa --shape 16 128 64 1 8
```

```python
from operator_model.model_v2 import Case, Predictor

model = Predictor('results/v0.2/model.json')
prediction = model.predict(Case('fa', (16, 128, 64, 1, 8)))
print(prediction['predicted_us'], prediction['coverage'])
batch = model.predict_many([Case('quant', (17, 1024)), Case('matmul', (17, 256, 1024))])
```

服务启动时加载一次模型，复用实例的 4096 项 LRU 缓存。结果不可变；查询超出支持的属性范围应换模型，不能只修改输入字段冒充已校准路径。CLI 默认 v0.2；复现 v0.1 必须显式指定 `--model results/v0.1/model.json`。

## 支持范围与计算方法

| 算子 | 输入与固定属性 | 建模机制 |
| --- | --- | --- |
| dynamicMXquant | `(rows, cols)`；BF16→E4M3FN、E8M0 scale、尾轴 block32、rint、scale_alg=0；cols 为 64 的倍数 | 源码参考的 256/512 列二维分核；253952 字节 UB、双缓冲和临时空间约束；UB 迭代数、主 VF 与 scale VF 循环 |
| matmul | `(M,N,K)`；BF16、ND、不转置、cubeMathType=0 | 16×64 输出粒度；超过 32 核时启发式扩大 tile；K 分段、对齐和尾块；分组 gemv/k_tail/coarse_tile/dense |
| FA | `(Q,KV,D[,B,H])`；BF16、SBH、无 mask/dropout；B=1、D∈{64,128}、H≤32 | Q tile 64/128、KV tile 128/256、head 分核、QK→Vector→PV 依赖、KV 迭代和核间轮次 |

这些是有限范围的分核预测器，尤其 matmul 的 tile 选择是待验证的启发式，不是完整复刻 CANN tiler。`gemv` 只表示 M=1 的模型分组，不表示实际切换到了 Vector kernel。维度的 API 上限 4096 不代表全范围已验证。

分阶段非负成本先由校准 trace 中的忙碌时间拟合，再计算有限流水的填充、稳态和排空。总时延采用非负的常数、流水成本、活跃核数对数项，以及按机制分组的收缩残差。只有 1–2 点的分组不单独拟合全部参数，保持 `provisional_regime`；未见机制输出 `unseen_regime`。校准包围盒内也不保证存在所有 shape 组合。

同一个 kernel 名可能使用不同 tile，需结合实际 MMAD 的 m/n/k 和次数判断。本轮确实观察到 matmul 的 16×64→32×64 tile、K 多轮以及 D=128 FA 的不同 kernel。模型输出 `kernel_selection_verified=false`，候选 kernel 名不等于已执行目标输入并验证其选择。

## 计时与利用率口径

参考时延是 CA trace 中跨核最早指令开始到最晚指令结束的时间窗，单位 μs。它包含窗内的等待和间隙，不包含 host enqueue 或窗外设备调度。不能直接作为完整框架 API 延迟。

- `busy_fraction`：去除同步/等待指令后，流水指令区间并集除以跨核时间窗；不可把各指令 duration 相加代替并集。
- Vector family 包含 RegBase load/store/control；它不是纯算术利用率。Cube/Vector 忙碌比例也不等于吞吐瓶颈判据。
- 活跃 AIC/32、AIV/64 是核覆盖率；MMAD 和 RV_* 动态次数单独保留，用于检查工作量预测。
- 部分 MMAD trace 事件缺少 m/n/k。总次数读取 CSV `call_count`；`matrix_instruction_shapes` 和聚合字段 `executed_cube_flops` 只累计有完整维度的事件，后者是部分计数，不是可直接使用的总执行 FLOPs。评估另列明细完整性。
- 独立的 ISA 发射/吞吐峰值模型尚未校准，故 `ideal_compute_efficiency=null`。当前不输出看似精确的“理论指令时延/CA 时延”。

CANN 9.2.1 的原始 simulator 配置保持不变，记录到 `L2CACHE.cache_enable=0`。拟合系数带有该 CA 环境的访存、依赖和调度影响，不能当作跨代不变的硬件常数。A7/A8 的核数、频率、CV 比例仍仅是规划参数，未验证外推。

## 小预算采样与独立留出

两轮共选择 32 个不同输入：v0.1 的 15 个输入在本轮全部转为校准，新增 10 个校准点和 7 个新留出点。旧留出点用于本轮拟合后，不再算本轮独立测试。预测、参数、实现和采样清单在采集新留出点前冻结并提交 Git。

新增点覆盖极小输入、分核饱和、K 尾块、多轮 K、UB 容量边界、Q=128/129、KV=1025、多 head 和 D=128。它是机制覆盖，尚不是三种整网模型的 workload 加权采样；标准 FA 也不替代 MLA、稀疏注意力或 KDA。

CA 只用于离线校准。大 shape 的策略为：

1. 先用源码分核/容量公式找到边界，再挑能进入相同机制的较小代表 shape，不做全维度笛卡尔积。
2. 串行采集；相同环境、调用器、属性和 shape 走内容哈希缓存，复用时检查 trace 哈希。
3. 常规 profiler 超时 2 分钟，外层进程组另有超时清理。慢例显式进入最多 6 分钟的 profiler 档，外层 480 秒；失败不自动重试。
4. 本轮预算为 18 次尝试、1 小时采集墙钟、8 CPU 小时、8 GiB 原始数据。CPU/磁盘在每个 case 边界检查，可能超出至多一个运行中的 case；墙钟预留完整 case 超时但清理/解析仍有额外开销。不是操作系统级硬配额。
5. `[1024,4096]` quant 的首次 2 分钟超时作为失败保留；仅显式重试一次。大留出点采集前缩到 `[640,4096]`，仍保留两轮 UB，以减少仿真成本。

报告同时列出失败成本、CA 墙钟、累计 CPU、原始数据量和在线预测耗时。CPU 小时不是用户等待时间，在线预测也不是每次重新启动 Python。

## 复现、校验与迭代

```bash
python3 -m unittest discover -s tests -v
python3 scripts/evaluate-v2.py
# 可选图表，依赖 matplotlib；本机 Conda 环境已安装
toolchains/envs/cann-9.2.1/bin/python scripts/plot-v1.py --version 0.2
```

评估从已提交的聚合数据和冻结预测重放，不需重跑 CA。在线开销随评估主机变化。原始大 trace 仅保存在本机 `runs/model-v2/`；重建 CA 需要相同 CANN 安装和 simulator 配置，运行 `scripts/collect-v1.py --version 0.2 --split calibration|holdout`（两种 split 分别调用）。没有 trace 的新 checkout 将实际采集，不会伪造缓存命中。

`fit-v2.py` 拒绝覆盖现有冻结文件；下一轮改模型须使用新版本、新校准快照和未查看的新留出点。目标为新留出 WAPE≤10%、P90 APE≤20%，同时比较固定耗时、经验 Roofline、未校正流水模型。v0.1 的 quant/FA 小 shape 上固定耗时基线曾优于模型，说明必须增加机制覆盖，不能仅凭低误差宣称学到了正确缩放规律。

来源与限制见 [证据清单](evidence-v2.json)。上游源码只作结构参考，不声称与安装包二进制逐行一致；以实际运行得到的 kernel、分核和指令证据检验假设。
