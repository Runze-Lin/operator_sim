# candidate_filtered_indexer · 逐窗误差

[全部合同](README.md) · [总误差分析](../error-analysis.md) · [原始逐窗记录](../../calibration/reference/windows.jsonl)

950DT、950PR 当前均有 0 个匹配系统融合排程的实测窗。该排程的 CA 误差为 N/A，三款芯片的资源预测均标为外推。

系统捕获了候选块索引及其形状。系统按官方全表打分、候选掩码和 top-k 的顺序定价。原生候选算子的完整 CA 窗尚缺。两种显式分解有双端 CA，逐个 kernel 可查。[分解实测与核身份](../../calibration/candidate_indexer/README.md) · [候选 Indexer 算例](../examples/candidate-filtered-indexer.md)
