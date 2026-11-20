# v0.28 方法和误差变化

- **Matmul** 增加 20 个新几何、两端共 40 项，含三种模型、普通 decode 的 B=1/9/11/16、MTP8 对应 M=9/81/99/144、full prefill 与 chunk。32Ki Flash [32768,128,512] 已双端完成；[262144,32,5120] 每卡仅输入就达 2.5 GiB。B8×128Ki、B8×256Ki 两个满秩几何分别有 1,048,576 与 2,097,152 行。预测先行冻结，每项真实 CA 才进入误差。此快照的首批 32 项已有 22 个合格、0 个不合格。另登记 Kimi 和 Pro 的整批 B8×32Ki，复用 RMS 完成后的单槽。二级 Stream-K 候选另按登记时是否已开跑区分数据角色。[实时逐形表](matmul/report.md) · [完整 Pro 和 Kimi B8×32Ki](matmul/wide-prefill/report.md) · [长 prefill](matmul/long-prefill/report.md) · [单输出候选将 DT 一例 / PR 两例前瞻 WAPE 减至 5.65% / 2.33%](matmul/streamk-analysis.md)
- **Cast** 升级为按活动核发起与片外写压计算脏输出留驻。新 Flash [5632,5120] 的 DT 旧/新 27.431/27.431 µs，CA 29.062；PR 68.721/54.881 µs，CA 55.220。两次采集墙钟 685/658 s。大 Pro 的 PR 总窗偏差从 −1.69% 变为 −4.30%。[双端回执和旧样本分组](cast/score.json)
- **RMSNorm** 两芯片的十个新形通过了分别检验首遇与复跑的预登记门槛。无前序、最忙核最多两个 UB 块的情况已接入默认。将 gamma/x 就绪、寄存器组归约、输出与 rstd 写回分别排程，公开条件接口可指定哪些核已参与过。独立同形两态已完成 40 个可配对芯片窗。[公式与参数](rms/method.md) · [当前的分组 WAPE、最大误差、DT/PR 比值和成本](rms/report.md)
- **mHC Post / Sinkhorn** 分别按通道与 token 路线、寄存器组固定发起与组内工作定价。Post 提供前序条件接口；Sinkhorn 的宽形已知 H 回放改善，同时保留未知 H 在旧域的退化。[Post 独立与回放](post/report.md) · [Sinkhorn 独立](fusion/forward.md)
- **预测成本与适用域** 安装版普通查询 1371 个不同输入，P95 0.631 ms，最大 2.015 ms。所有 970DT 数字均为外推；缺失的物理退出、未完成的长任务和数值异常均不计入真值。[完整安装版测时](wheel-check.json) · [全量逐形回放](shape-errors.md) · [带时间戳的资格快照](evaluation.json)
