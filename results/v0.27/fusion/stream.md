# 从有序派发恢复独立 Sinkhorn 的前序

系统仿真掌握同一卡上一条流的真实派发顺序时，可设置 `OperatorStream(track_sinkhorn_history=True)`。每次传入的仍是 shape、dtype 和迭代次数。流自身保存此前已使用的 AIV 数。默认开关为 `False`，原有的 `estimate()`、RMSNorm、Post 接口与默认数值保持兼容。

```python
from operator_model import OperatorStream

stream = OperatorStream("Ascend950DT_9582", track_sinkhorn_history=True)
for tokens, iters in [(13, 3), (46, 8), (67, 10)]:
    r = stream.advance("mhc_sinkhorn", (tokens, 4), {"num_iters": iters}, dtype="fp32")
    plan = r.projection["target_plan"]
    print(plan["resident_aiv"], plan["active_aiv"], plan["history_state"], r.latency_us)
```

上例依次推得 H=0、13、46。最后一次只用 34 个 AIV，之前的 46 核已经覆盖所需的集合：950DT 得到 3.23485 µs，CA 为 3.26000 µs；950PR 相同派发得到 3.28575 µs，CA 为 3.28545 µs。只调用 `preview()` 可以查看候选派发，成功的 `advance()` 才会写入历史。

## 什么时候能自动使用 H

- 新建流或执行 `reset()` 表示已知冷启动。完整、成功的同族派发按 Hₙ₊₁=max(Hₙ,Aₙ) 更新，其中 Aₙ 是本次从目标芯片分核得到的活动核数。无需读取 CA 的核清单。
- 接入一个已经运行的流时调用 `forget_history()`。成功派发其它家族，包括融合 PreSinkhorn、Matmul 后，独立 Sinkhorn 的过去驻留变成未知。同族后续派发只建立下界 L。若当前活动数 A≤L，则所有可能的 H 都选择热态，使用该预测，并在 `projection["partial_history"]` 返回 H 的下界和上界。若 A>L 则保留既有 shape-only 结果。上界足够、旧 shape-only 缺少价格的单 UB 大形，也可恢复确定的热态预测。
- 调用方显式给出的 `resident_aiv` 优先。显式 `None` 表示这次前序未知，成功派发后的保证覆盖从本次活动核数重新计算。失败、错误 dtype、不同布局、未知实现或仅预览都不会提交状态。若自定义芯片缺 FP32 吞吐或 UB 不满足当前显式机制，自动路径只返回原先可用的 shape-only 结果，这次不能为后续的单 UB 家族证明驻留。
- 970DT 仍使用外推模型。恢复一个派发流的顺序只确定模型应使用哪种状态，目标芯片上实际的指令时延仍需另行验证。

## 与冻结价格和 CA 的核对

直接读取两端原宿主的 `batch.txt`，用每次 `BATCH_BEGIN/BATCH_END` 内实际 `aclnnMhcSinkhorn` 的 shape 和迭代数复核派发顺序。每个 original process 对应一条新流。在线重放只传 shape、dtype、I；预测返回后才对照原冻结的 H、核数、VF 发起时标、整窗及完整 CA 金标。970DT 使用相同输入顺序和目标芯片自身分核。

| 目标与覆盖 | 按顺序重建的价格 | 不同几何 / 实测窗 | 对 CA 的 WAPE / 最大误差 |
|---|---:|---:|---:|
| 950DT，T≤256 | 11 | 10 / 11 | 3.09% / 8.31% |
| 950PR，T≤256 | 11 | 10 / 11 | 3.88% / 12.92% |
| 950DT，T>256、单 UB | 4 | 4 / 4 | 12.96% / 26.37% |
| 950PR，T>256、单 UB | 4 | 4 / 4 | 20.37% / 37.53% |
| 970DT，相同派发顺序 | 15 | 14 / 0 | N/A，外推 |

三芯片共 12 条进程序列、45 次派发，所有 H 与原冻结值一致，每项结果与直接显式传 H 的 API 返回值相同；设备总窗和 VF 发起时标与冻结值的最大差均为 0 µs。两端 30 个实际 CA 窗的核心数和 VF 次数全部匹配。安装 wheel 后从源码树外复验，945 次带状态查询中位 **0.097 ms**，P99 **0.123 ms**，最大 **3.818 ms**，含首次加载。全仓 272 项测试通过，复用了已有采集。[逐派发顺序、宿主记录与实测散列](stream-installed.json) · [可复现校验程序](../../../development/validation/fusion27/stream_check.py)
