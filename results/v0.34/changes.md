# v0.34 建模与校准变化

| 项目 | 950DT | 950PR | 数据与含义 |
|---|---:|---:|---|
| Lightning Indexer，旧 → 新 WAPE / 最大 | 12.40 / 31.71% → 5.87 / 15.67% | 11.88 / 26.51% → 4.14 / 10.88% | 同一批 14 / 10 个旧整窗。4 个 DT 形标定首发、页间隔与排序，所有 PR 结果都用同一组价格重算。[逐形](indexer/score.json) |
| mHC Post，公开接口仅 shape，WAPE / 最大 | 27.94 / 63.56% | 28.02 / 67.40% | 239 个旧窗。通道分核的四写候选对首发累计低估 41.89% / 38.34%，对已热复用高估 27.80% / 36.77%。可见相同形处理冷热时需要相反方向的修正。[按真实前序的诊断](post/unknown-state.json) |
| mHC Post，有序流，WAPE / 最大 | 8.38 / 45.13% | 6.83 / 31.78% | 系统提供调用次序；与无状态入口使用同一组物理窗。[全量验收](acceptance.md) |
| AttnResFwd，真实 prefill，512×7168、1 行有效历史 | 45.078 µs，待测 | 55.623 µs，待测 | 当前能按 UB 驻留和每核循环估计；970DT 外推 27.478 µs。旧物化合同的阶段已回放，新合同独立误差为 N/A。[新旧合同与阶段](attn/report.md) |

本轮保留 21 条实现。按用户指定移除 recurrent KDA 多状态链和 chunk 融合 V1；recurrent 单状态链与 FLA chunk V2 继续验收。[范围和大形覆盖](../v0.33/large/report.md)

**mHC Post 的候选选择。** 在已有 v0.33 冷/热/扩核价格之间取受限中点，仍需同时满足模型端点的相对误差限制。回放得到 DT 27.57% / 51.13%、PR 27.05% / 53.07%。两端最大误差回升，未采用。已冻结 [1,4,208] 与 [3,4,240] 的首次及重复请求，仍在原 CA 队列。[候选和真实配对下界](../v0.33/post/report.md)

**大形新端点。** Addmm 的 [65536,128,512] 带 bias、Indexer 的 Q9/KV8192、Kimi AttnRes 的 65 行和 121 行、单状态 KDA 的 S8/V256 均在采集前分别冻结两端。旧队列继续采集多波 Matmul；PR 的 vLLM HcPost 1024-token 达到原始数据预算，记录失败资格。当前未给未完成端点计入 WAPE。Kimi 实际 V=128 的 S=8 已有两端完整控制。[Addmm](addmm/report.md) · [Indexer](indexer/score.json) · [AttnRes](attn/report.md) · [KDA](recurrent/report.md)

**开销。** 安装版 348 项测试通过，本次 3,958 次进程内公开入口查询中位 0.174 ms、P95 3.692 ms、最大 18.381 ms。包含目标资源的重算及结果构造。Matmul 既有大形 8 份回执可复用原 trace 加速复核，不再占用一次新的 CA 采集。耗费数小时的在线 CA 只在通过实现、数值、计时和核退出检查后用于验收。970DT 全部是配置驱动的外推。[查询实测](wheel-check.json) · [测试](wheel-tests.json) · [大 Matmul 重扫开销](../v0.33/matmul/native-replay.json)
