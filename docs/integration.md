# 在系统仿真器里使用 operator-sim

在系统框架中找到原来根据算子及单卡 shape 返回 Roofline 耗时的函数。在这里调用 `operator_model.backend.estimate`，将微秒数交给原有事件调度器。每次查询无需分配设备张量、读取 trace 或启动 CA。目标平台为 `Ascend950DT_9582`。

## 安装及一次查询

在项目根目录使用 Python 3.10 或更新版本安装。源码和 wheel 均带有当前模型参数，运行时依赖仅为 Python 标准库。

```sh
python -m pip install .
operator-sim --op rmsnorm --shape 2 6144
```

部署到其他仿真主机时，可在构建机运行 `python -m pip wheel --no-deps . -w dist`，传输生成的 wheel 后用目标环境的 `python -m pip install 路径/文件名.whl` 安装。命令行返回含 `predicted_us` 的诊断 JSON，适合手动检查。批量系统仿真直接在同一 Python 进程调用下述接口，模型参数在进程内复用。

```python
from operator_model.backend import estimate

result = estimate("rmsnorm", (2, 6144))
print(result.latency_us, result.status)
```

## 接入现有调度器

下例假定系统事件队列以秒计时，原有 `roofline_us` 接口返回微秒。调用方可以自行决定是否采纳外推结果，并记录回退原因。

```python
from operator_model.backend import estimate

def duration_seconds(op, local_shape, attrs, roofline_us, soc="Ascend950DT_9582"):
    result = estimate(op, local_shape, attrs, soc=soc)
    if result.status == "estimated":
        return result.latency_us * 1e-6
    return roofline_us(op, local_shape, attrs) * 1e-6
```

| 返回状态 | 含义 | latency_us |
|---|---|---|
| estimated | 当前模型接受这次调用的属性并给出设备时间估计。新 shape 可能没有逐点 CA；适用范围和独立误差见校准报告。 | µs |
| extrapolated | 已显式标记的外推，例如 KV 超过已完整验证的 128k 的普通 FA，或 T>256 的 mHC 拆分链兼容路径。 | µs |
| unsupported | 算子名、目标 SoC、shape 或调用属性未被该路线接受。 | None |
| unavailable | 有结构模型，缺少可信的完整耗时；例如 KDA chunk。 | None |

`reason` 给出不可用或显式外推的原因。各模型能识别的外推分支不同；需要精确判别某个输入是否已测时，查[逐 shape 校准结果](error-analysis.md)。`latency_us` 是同一 kernel 的跨核设备指令窗口，串行组合取实际 kernel 窗口之和。调度器单独处理主机发射、设备排队、kernel 间空闲、通信和跨流重叠。后端在 CPU 上计算这个估计的墙钟用于衡量仿真效率，请勿加入设备时间线。

## 把框架节点映射到调用参数

传入已经过 TP/DP/EP 切分的每卡输入。shape 使用正整数序列；可选属性使用字典，省略时采用当前已建模的调用合同。相同数学算子的 dtype、布局、融合和可选输出不同时，请先核对[逐算子方法和算例](modeling-strategy.md)，再决定是否路由到该模型。下表列出默认算子名和维度顺序。

| op | 每卡 shape | 关键合同 |
|---|---|---|
| quant | (R,C) | dynamicMXquant，BF16 输入、block32 |
| rmsnorm | (R,C) | BF16，输出含 rstd |
| matmul、addmm | (M,N,K) | BF16；Addmm 的 bias=[N] 且 α=β=1 |
| fa | (Q,KV,D,B,H) | 普通非因果 SBH；Q 是本次校验长度，H 是本卡头数 |
| lightning_indexer | (B,Q,L,G,D) | BSND，G=8、D=128、top-512 |
| mhc_pre、mhc_post、mhc_pre_sinkhorn、mhc | (T,N,D) | N=4；mhc 代表 Pre、Sinkhorn、Post 三段，外部层单独调度 |
| mhc_sinkhorn | (T,N) | N=4，`num_iters` 为实际迭代数 |
| kda_recurrent、kda_chunk | (B,S,Hq,Hv,K,V) | 分别为递推和分块路线，仅递推有完整时延校准 |
| grouped_matmul_swiglu_quant_v2、gmm2、moe_ffn | (M,K,F,E) | `expert_counts` 为 E 个本地专家各自收到的有效 token 数，总和为 M |

例如 `estimate("moe_ffn", (32, 256, 128, 2), {"expert_counts": [1, 31]})` 预测整个两段计算链。调度该节点后不要再把两个 GMM 组件的估计加到同一事件上。EP 大小不能替代本地专家直方图。推测 8 个 token 时，普通 FA 的主模型校验长度常为 Q=9，实际 B、H、打包方式仍由框架提供。

对 RMSNorm 和 mHC Post，默认采用调用上下文未知时的估计，不保存上次预测的设备状态。系统若维护着**同一类 kernel 连续执行时前序调用的最大活跃 AIV 数**，可用 `{"resident_aiv": 32}` 显式传入，取值 0…64；0 表示该隔离序列的首次调用。发生换流或并发干扰而无法确定时，省略该属性。该量用于描述 CA 观察到的启动上下文，硬件缓存状态仍待独立确认。

当调度器知道隔离序列刚开始时，令 `resident = 0`。以下三行只由同一流、同类 kernel 的实际调度事件驱动；同一个待选方案反复试算时保留原状态。返回的 `active_aiv` 是本次预计活跃核数，缺少此信息的路线返回 None。

```python
result = estimate("rmsnorm", (2, 6144), {"resident_aiv": resident})
if result.active_aiv is not None:
    resident = max(resident, result.active_aiv)
```
