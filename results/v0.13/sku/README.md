# 950DT 与 950PR 的配对校准

比较 Ascend950DT_9582 与 Ascend950PR_957b。CANN 9.2.1 的完整 SKU 配置分别给出 32/64 与 28/56 个 Cube/Vector 核，两者均为 1.65 GHz。DT 可见外存为 96 GiB，PR 为 112 GiB。模型采用厂商表 3-1 列出的 DT 4 TB/s 和 PR 112 GB 容量档对应的 1.4 TB/s 名义峰值。该 PR 档与具体 SKU 的对应关系由容量推定。16 位 Cube 峰值分别为 432 与 378 TFLOPS，FP32 Vector 分别为 27 与 23 TFLOPS。[昇腾 950 NPU 架构白皮书](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf)

两个静态配置保存软件可用的 UB、L1、L2、L0A/B/C 容量。平台文件还列出 `ddr_rate`、`l2_rate` 等字段，字段本身没有在该文件中标明整片或每核的单位、方向和计量条件，因此本轮保留 L2/MTE 的 GB/s 为未知。离线配置可以由维护者依据补充资料更新。

## 开发对照

首批选取 3 个 950DT 原有 shape，复用原 trace、输入和同一 SHA-256 的 host 可执行文件。用 CANN 官方 `msopprof simulator --soc-version=Ascend950PR_957b` 在全新进程执行相同 ACLNN 调用。每例确认实际 runtime SKU、模拟器成功、非空正确性比较以及每个核的指令窗口与 profiler 时长一致。设备微秒数取所有核首条到末条指令的时间窗。[采样预算](budget.json) · [逐例计时、核函数、文件指纹与成本](pairs.json)

| 算子及每卡 shape | 实际 ACLNN | DT / PR CA µs | DT / PR 活跃核 |
|---|---|---:|---|
| DynamicMXQuant [64,1024] | aclnnDynamicMxQuant | 5.532 / 5.373 | 64 / 44 AIV |
| Matmul [128,256,512] | aclnnMatmul | 4.393 / 4.368 | 32 / 16 AIC |
| FA [16,64,64,1,1] | aclnnFlashAttentionScoreV2 | 9.052 / 8.472 | 各 1 AIC + 2 AIV |

同一行两个 SKU 的 kernel 二进制 SHA-256 均相同。Quant 和 Matmul 用同一内核仍出现了不同的实际分核，表明迁移时需要按目标核数重算分块。对这三点使用 PR 耗时解释模型后，数据角色为机制开发。

原始日志还分别打印 DT 的 `hbm3_8000M.ini`（2 Channels）和 PR 的 `lpddr5_6400M_bailu.ini`（1 Channels）。两个运行选中了不同的外存子模型，同时复用了 DAV_3510 的顶层 CA 配置。这些算例包括标量启动、片上数据周转和有限并行；三个总时延不足以反推出芯片的整片持续外存带宽。[逐次物理内存模型记录](memory-models.json)

三次合格 PR 进程合计 180.6 秒，原始产物 235.7 MiB。沙箱首次执行时本地 profiler 套接字连接失败，运行时返回 9589 且未产出合格时序。三次失败进程另耗 206.3 秒，全部从统计中排除。DT 未重新采集。

冻结后另外选取的两个 PR 新 shape 和各自预测见[开发与独立误差对照](comparison.md)。
