# 稠密 RecurrentKda 多链：模型来源与验证

[完整输入、公式、物理量与算例](../../../docs/examples/kda_recurrent.md#kimi-decode1536-条独立状态链v031-冻结外推)。分类：DT/PR 单链完整 CA；多链**冻结条件外推**，不替代公开单链预测接口或把它算进单链历史 WAPE。970DT 只有情景预测。诊断重点是单链窗口中一次性控制和跨链预取有多少能隐藏，并校验轮询分核。B128 不是 1536 次 host dispatch。

| 证据 | 观察/边界 |
|---|---|
| 本机 CANN 9.2.1 `ops_transformer/ascendc/recurrent_kda/op_kernel/arch35/recurrent_kda_arch35.h` | `InitLocalBuffers` 在任务循环外；`Process`/`PrepareNextTask` 轮询 batch×value head；`ProcessHead` 对每个 V 条带串行跑 token，同核能预取后一任务首个状态条带，`CopyOutState` 受输出 final state 条件控制。冻结表记录安装文件 SHA。|
| 公开 `ops-transformer` commit `14d022b1a4c3df263bf697534e3c25c7c992ff5d` 的 `attention/recurrent_kda/op_host/recurrent_kda_tiling_processor.h` | `UpdateDynamicBlockDimByTaskUnits` 里 A5 使用 `min(B×Hv,AIV)`，非 A5 采用不同的均衡规则。`CalcWorkingUbBytes`/`EvaluateBufferProfile`/`FinalizeVStepFromUb` 确定 UB 条带与队列。这个公开宿主是参考实现，不证明安装包二进制与它逐字相同；单链 32 KiB 状态描述符与其 64 列条带吻合。|
| DT/PR v0.17 同形 `[1,1,1,1,128,128]` | 已归档官方完整指令 trace；首个 beta 发起 DT/PR 5650/5502 tick，元数据结束 1218/1329 tick，首个状态读持续 887/871 tick，末状态写结束→END 10/10 tick。模型仅取 DT 数值换算周期。路径、raw 与 strict SHA、实际 `RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0` 标于冻结表。|
| Kimi K3 config / 实际 runner | `configs/models/Kimi-K3.json`: 96 头、head_dim 128；TP8 后 12 个本地头。数值检查复用 `development/validation/large29/cube/recurrent.cpp`, `aclnnRecurrentKda` 实库同步；CPU FP64 序贯 oracle 比对所有 attention 和最终状态元素，按 batch 和 head 给不同初始值/beta。大 B128 没有完成合格实测前，工作量可确定，时间/误差需分开标识。|

[冻结清单与 SHA](freeze.json) · [端点状态及同集对照](score.json) · [现有大形采集状态](../../v0.29/large/cube/state.json) · [新 B6 控制采集状态](state.json)。冻结基线：`W×T₁`，是对 v0.29 的独立状态情景对照，不是经过校准的默认预测。PR/DT 的相对速度需要双端都合格后计算。`score.json` 的 `null` 不是零。

重新读取 B128 完成的收据，无须重跑 CA：`python development/validation/recurrent31/check.py score`。控制任务串行复用 `runs/large29/shared.lock`，每端最多 1 小时 / 3 GiB。B6 计时资格要求正确 SKU/内存模型、主机同步成功、两份数值金标全检通过、内核名与已安装对象 `.text` 对齐，官方时长表、全核 trace 和各核分片、物理 END 一致。另计每个 AIV 实际 `beta` 4 字节 MTE2 命令次数和重复发起间隔；分布与预测对照而不是强制匹配。B128 沿用大形组已有严格校验与预算，不二次采集。新核对程序从登记预测直接评分，不会使用新 trace 重新生成预测。

`estimate("kda_recurrent", shape, attrs, chip=...)` 的 `auto/resource` 也返回这个多链合同时长并标 `extrapolated`；要求 `timing_mode="calibrated"` 时返回 `unavailable`。`estimate_recurrent_chains` 额外允许显示调整 HBM 冷读、写排空和利用率假设。采前[精确模型快照](frozen-model.py)保留 SHA；之后只补齐与实库相同的中性属性（例如 `chunk_size=None`）和未知价格返回值。[ABI 回放](contract-check.json)逐端比较所有已冻结预测字段并检查三个公开模式，数值未变。未知 UB 或不同 stripe/输出队列配置保留工作量并返回无受支持耗时。读/写均命中缓存时外存下界为零；强制落外存时该下界与整个核窗口取 `max`，无实测依据时不额外串行叠加排空。
