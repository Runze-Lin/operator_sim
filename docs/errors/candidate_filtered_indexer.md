# candidate_filtered_indexer · 逐窗误差

[全部合同](README.md) · [总误差分析](../error-analysis.md) · [原始逐窗记录](../../calibration/reference/windows.jsonl)

BF16 全表融合参考排程在两端均缺完整 CA。A5 的 MXFP4 原生消费层另有固定 metadata 的完整设备窗；包含 AICPU、准备输入和后处理的部署端到端时间仍为外推。

BF16 图按 Hugging Face 参考的全表打分、候选掩码和 top-k 定价。A5 发布的 MXFP4 QSLI 先收集候选再用 Cube 打分。A5 的证据见[融合实现与组件对照](flash-a5-qsli.md)。两种显式分解有双端 CA，逐个 kernel 可查。[分解实测与核身份](../../calibration/candidate_indexer/README.md) · [候选 Indexer 算例](../examples/candidate-filtered-indexer.md)
