# Recurrent KDA 单状态链上界

安装的 CANN 9.2.1 ops-transformer 实现与 BF16 dense BSND 接口下，一个请求只保留一个 Q 头和一个 V 头，一次 kernel 只让一个 AIV 更新状态。该接口规定每条序列最多 8 步，K=128，V 可为 128 或 256。Kimi 的真实头宽 V=128，S=1–8 已有 DT/PR 完整回放；其中 S=8 为 35.396 / 39.450 µs，不能代替 32Ki prefill 的 chunk KDA 模型。

新增 [1,8,1,1,128,256] 是同一实现允许的最大单链状态与最多串行步数，V=256 为通用库边界。初始 FP32 状态 131072 B，每条链 8 次依赖更新。已有回放中 V=256 的 DT 只采 1/2/3/4/6 步，PR 只采 1/2 步。

| 芯片 | 模型 µs | CA µs | 有符号误差 | CA 墙钟 | 口径 |
|---|---:|---:|---:|---:|---|
| Ascend950DT_9582 | 67.203 | N/A | N/A | N/A | 已预注册，待完整窗 |
| Ascend950PR_957b | 68.615 | N/A | N/A | N/A | 已预注册，待完整窗 |
| Ascend970DT | 61.538 | N/A | N/A | N/A | 970DT 外推 |

每端墙钟预算 20 分钟、最多 2 GiB 指令原始数据。按现有串行设备锁逐端执行。核验包括真实 SKU/存储模型、安装包 kernel .text、完整 attention 与最终状态、输入状态未修改、唯一 END、物理核分片与全核指令集合一致，以及官方逐核时长。该几何单独报告为 ABI 压力样例。

[冻结、真实输入和源 SHA](freeze.json) · [排队状态](state.json) · [CANN ABI 的已发布接口说明](https://gitcode.com/cann/ops-transformer/tree/master/attention/recurrent_kda/docs) · [真实 Kimi 单链算例](../../../docs/examples/kda_recurrent.md)
