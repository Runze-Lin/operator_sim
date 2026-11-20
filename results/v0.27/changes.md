# v0.27 算子建模与验证

当前精度目标为每条路径、每个实测 SKU 的 WAPE ≤25%、最大绝对相对误差 ≤50%，进程内单次预测 <100 ms。新方法在已有 trace 上建立，用冻结后采集的形状独立检验。970DT 只有基于独立硬件配置的外推。用户日常入口为[工具说明](../../README.md)、[总建模方法](../../docs/modeling-strategy.md)和[逐算子误差](../../docs/error-analysis.md)。

| 本次公开模型与验证 | 950DT | 950PR | 同集比较及适用范围 |
|---|---:|---:|---|
| 内置 mHC Post 紧邻同形重复，WAPE / 最大 | 12 形，2.88% / 8.90% | 10 形，1.93% / 3.85% | 原有序默认分别 43.17% / 64.32%、46.03% / 64.98%。按有序流和库分核入口识别精确重复，普通首次、换形候选仍单列。[冻结结果](post/report.md) |
| 独立 Sinkhorn，给出同族已覆盖 AIV 数 | 10 形、11 窗，3.09% / 8.31% | 10 形、11 窗，3.88% / 12.92% | 同集旧价 16.64% / 37.21%、25.75% / 80.84%。准备、首次、稳态 VF 与写回分别定价。四种扩展形仍单列。[验证与参数](fusion/public-sink.md) |
| 融合 PreSinkhorn，10 个双端匹配输入 | 3.21% / 10.61% | 6.11% / 11.37% | 原 20.56% / 48.53%、24.80% / 44.11%。2:1 联合作业的转换、MMAD、交接与后处理在目标卡重新计价。比值 WAPE 4.68%→6.88%，方向均 7/10。[验证与参数](fusion/public-fused.md) |
| FA，两个 SKU 各十个独立 shape | 4.19% / 15.79% | 6.49% / 13.68% | 32Ki+1 尾页各端 514 条 MMAD，Q9 完整窗为 75.955 / 72.912 µs。新比值 WAPE 4.38%，方向 2/10。[两批实测](../v0.25/coverage/supplement/score.json) |
| Pro B8×256Ki 全部 64 个 indexer 系数，模型 / CA µs | 227.018 / 247.788 | 520.357 / 529.287 | 安装包 `aclnnCast`，末核迭代 100 / 114 轮。两端墙钟各约 51 / 59 分钟；970DT 外推 104.673 µs。[成本与实测](memory/README.md) |

Kimi prefill 新增 `plan_kimi_kda_prefill`，用每条请求本次提交的 token 数、attention TP 和目标芯片选择真实 FLA V1/V2。B8×32Ki、TP8 产生 49,152 个工作项；三个 V2 主核外推为 DT 33.161、PR 39.396、970DT 16.450 ms。前后处理、状态路由和辅助 kernel 尚未全部计时，整个 dispatch 返回 N/A。小形 V1 通过四份逐元素金标和全部设备阶段，DT 实测出现 132 次重复置旗，计时资格失败。宿主对照耗时 331.9 秒、临时目录峰值 1.68 GiB。[分派规格与较大输入](kda/report.md) · [V1 原始失败与对照](kda/host-control/report.md)

B8×32Ki 全 hidden 的 Flash Cast [262144,5120]、Kimi MXQuant [262144,7168] 分别验证了 13.42 亿和 19.38 亿个输出。DT 全核原生包络为 2737.538 / 1778.988 µs，墙钟 4 小时 18 分 / 4 小时 53 分；逐指令全窗 N/A，不纳入主表 WAPE。[Flash 回执](../v0.23/full-hidden/v3/flash-retry.json) · [Kimi 回执](../v0.20/full-quant/native.json) · [逐控制器的双带宽口径与实际瓶颈](quant-envelope.md)

对形状相同而前序不同的 Post，给每个 shape 事后任选一个预测值，绝对误差的最低总和对应 DT WAPE 27.64%、PR 28.81%。这个不可辨识性约束适用于这些已验证的真实窗口。当前仍未全量达标。其余超限组包含 RMSNorm 无前序、PR 的 Cast 容量过渡区、DT 的无前序 Sinkhorn 特定迭代组。156 MiB 的 PR Cast 冻结试验缺一核 END，完整时延 N/A。在剩余 55 核中，单靠冷读＋配置的名义带宽已无法满足描述符区间。进一步的双端 Cast 验证选用 Flash 的 5120 个实际通道，每卡八段、每段 704 token，排在原有单槽之后，每卡最多一次、45 分钟、12 GiB。[形状信息下界](history-bounds.json) · [读区间与设备参数](memory/README.md) · [新输入、双端价格和预算](memory/cast-pair/freeze.json) · [RMSNorm 双端独立验证](rms/report.md)

源码树外安装的 wheel 计时覆盖三种芯片的 1,818 个不同输入，中位 0.106 ms、P95 0.583 ms、最大 5.359 ms。另对 57 次 Kimi 路由、72 次有序 Post 分别计时，最大 2.036 / 25.153 ms。[wheel 核验](wheel-check.json) · [有序 Post](post/published.json) · [模型和冻结来源](model.json) · [评估快照](evaluation.json) · [854 个已去重旧指令窗的公开模型回放](replay.md)
