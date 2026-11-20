# LightningIndexer 的 8192-key 解码与大形缺口

在 950DT 既有原始 CA 数据库上重新得到 **22.709 µs**。本例一个 query 面对 8192 个可用 key，单个 BF16 key 张量是 2 MiB；它能检验单 query 长 KV 排程。此样本此前已用于 DT 阶段开发，本次复核不增加独立样本数，也不增加双端已合格路线数。

| 本地 shape [B,Q,L,G,D] | 950DT 公开 / 实测 µs | 950PR 公开 / 实测 µs | 970DT 外推 µs |
|---|---:|---:|---:|
| [1, 1, 8192, 8, 128] | 21.871 / 22.709 (-3.69%) | 21.926 / N/A | 20.046 |

安装包对应 aclnnLightningIndexer 的稠密 BSND 接口，BF16、top512、sparse_mode=3。查询、key、权重张量分别为 [1,1,8,128]、[1,8192,1,128]、[1,1,8]。CPU 按实际 BF16 输入重建所有候选分数，核对返回的 512 个索引集合。集合内的名次顺序不在此合同中。

SQLite 完整性检查通过。32 个 AIC 与 64 个 AIV 各有一条物理 END，逐核扫描 120,521 条记录。全片首发 tick 5940，全片最后退休 tick 43410，二者差 37470 tick 除以 1650 tick/µs。设备日志确认 Ascend950DT_9582 和 hbm3_8000M；一个 AIC 发出 64 次 MMAD，一个 AIV 有 256 次正时长直方图指令。完整复算与原数据库时长和原数值回执一致。原采集墙钟 121.6 s。

对 AIC 与两类 AIV 已执行的每个 PC/机器码，用一个共同基址匹配官方发布的 LightningIndexer 混合内核。匹配从 .text+0xff30 开始，AIV 入口位于 .text+0x58100。原生库未提供完整加载对象路径；核对覆盖实际执行字节。原生日志报告可选 STAT 汇总项缺失，以上时长直接来自完整的逐核指令记录。[原始计时与正确性回执](../../../runs/v8cv/lightning_indexer-10/result.json) · [原始运行日志](../../../runs/v8cv/lightning_indexer-10/run.log) · [官方内核身份对照](../../v0.34/indexer/identity.json)

8192 个稠密候选位置可在容量上与 32Ki 的 4:1 压缩比较。Pro 的配置含 4:1 层但 top-k 为 1024，Flash 的 top-k 为 512；该库合同不能证明任何一款模型在 32Ki 时实际走相同的稠密 top512 路径。已登记的 Q9×8192 指令样本、Q9×32768 稠密样本均与本例不同。未采集 Q1×8192 的 PR，保留公开预测。[Flash 配置](../../../configs/models/DeepSeek-V4.1-Flash.json) · [Pro 配置](../../../configs/models/DeepSeek-V4-Pro.json) · [Q9 原登记](../../v0.34/indexer/freeze.json)

## 尚未得到完整回执的既有大形

| 已登记的 DT 调用 | 日志实际启动 | 已见 kernel 数 | 最新模拟器进度计数 | 整次运行成功回执 |
|---|---|---:|---:|---|
| Pro MoE 4096 rows, GMM1 / GMM2 | 是 | 1 | 90000 | 未见 |
| Flash built-in MhcPost 1024 rows | 是 | 1 | 40000 | 未见 |
| Flash Addmm 65536 rows | 是 | 1 | 50000 | 未见 |
| Kimi chunk FLA V2 [1024,12] | 未见 | 0 | N/A | 未见 |
| Flash split mHC [512,4,5120] | 未见 | 0 | N/A | 未见 |

审计时间 2026-11-20T09:19:15.416803+00:00。进度计数只用于确认采集进展；没有全核退休和数值回读就不记设备耗时。检查 1291 份原始 run.log，在精确 Q1 合同上命中 1 份，均已写入[机读记录](audit.json)。此次没有发起新 CA。

已有轻量手段各有可用边界。npusim 的 compact SQLite 在这个 DT 同形保留了每条指令的首发与退休；不生成可视报告也可复核。无指令的原生 wrapper 在已归档的纯 AIV Quant/Cast 上可以重建全核起止，但缺少同一混合设备路径上的指令端点偏差验证，因此未将其用作上述混合算子的全核指令耗时。[轻量计时原始约束](../../v0.29/large/vector/report.md) · [采集成本与设备窗差别](../../v0.33/matmul/report.md) · [复核脚本](../../../development/validation/large36/audit.py)
