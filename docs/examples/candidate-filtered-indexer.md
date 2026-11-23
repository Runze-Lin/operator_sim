# DeepSeek-V4.1 Flash 的候选索引

取 128Ki decode、单副本 B=1、主模型一次校验 Q=9 的 `inference::candidate_filtered_indexer`。系统捕获了五个输入与一个输出。查询是 BF16 [9,32,128]，key 表是 BF16 [1,131081,128]，头权重是 BF16 [9,32]，位置是 INT64 [9]，候选块 ID 是 INT32 [9,2048]，输出索引是 INT32 [9,512]。其中 131081 包含当轮的 9 个位置。候选生产节点的 `block_size=8`。

```python
from operator_model import estimate

# B, Q, H, D, L, C, K 依次是批量、每请求查询行、头数、头维、
# 每请求 key 表行数、每查询候选块槽位、输出索引槽位。
shape = (1, 9, 32, 128, 131081, 2048, 512)
scan = estimate("candidate_filtered_indexer", shape)
gather = estimate("candidate_filtered_indexer", shape,
                  {"score_policy": "candidate_first", "block_size": 8})
print(scan.work["score_dot_ops"], gather.work["score_dot_ops_upper_bound"])
print(scan.status, scan.latency_us)
```

| 打分方案 | 如何从 shape 复算 | 本例 |
|---|---|---:|
| 系统账单的全表打分 | 所有查询各打 L 个分，每个分含 H 个 D 维点积，乘加计两次。`2×B×Q×H×D×L` | 9,664,339,968 ops |
| 候选优先的条件上界 | 一行 C 个块，每块最多 8 个 key；每请求最多 L 个 key。用 `min(L,C×8)` 替换上式的 L。 | 至多 1,207,959,552 ops |
| 全表方案的接口读字节 | Q、key、头权重分别为 `2×BQHD`、`2×BLD`、`2×BQH` B，另加位置 `8×BQ` B 与候选 `4×BQC` B。 | 33,704,840 B |
| 输出写字节 | 一个入选索引占 4 B，`4×BQ×K`。 | 18,432 B |

全表算式复现系统参考模型使用的打分范围。候选优先是假设另一种实现先读块 ID，再读取有效 key；槽位可重复、越过可见边界或带填充。若取得每行实际参与点积的 key 数，可传 `scored_keys_per_query`（统一整数或长度 B×Q 的序列），此时 `score_dot_ops = 2×H×D×各行实际 key 数之和`。候选路径的读量按逐行 gather 计，跨行复用、L2 命中和物理 HBM 事务仍需单独取证。

系统在 Flash 的这一条 B1 target 节点给出 1006.704 µs，来自系统原有的成本公式。捕获没有提供实现该节点的 950DT/PR 二进制、实际打分范围、排序/归约指令或设备时刻。公开调用返回工作量，`latency_us=None`，950 上的 CA 误差以及 970DT 的设备时间均为 N/A。优先要核实的是实际 kernel 是否在点积前就缩小 key 集；两种方案在这个例子中的点积量约相差八倍。[算子实现](../../src/operator_model/operators/attention/candidate_indexer.py) · [九组系统账单与算子覆盖](../system-workload-coverage.md)

## 950 上能否采到这个前向

在固定版本的 ops-transformer 中，真正接收候选块的是 `aclnnSparseLightningIndexer`。其产品表列明 950PR/DT 不支持；候选 consumer 注册在 arch22。其语义要求逐 query 屏蔽不可达或未入选的块，有效 key 不够时以 −1 填满。950 的 CANN 9.2.1 包内有 `lightning_indexer`、`lightning_indexer_v2` 和量化版本，V2 的 `blockTableOptional` 描述分页地址，签名中没有 `candidateBlockIndices`。包中名字带 sparse 的梯度与 KL loss 也属于其他计算。系统导出的这个节点只创建 meta tensor，未给出 950 自定义 ELF。这个合同的 CA 仍为 N/A。若部署时改用多个 kernel 组合，应验证各 query 的因果边界、重复块和 −1 填充并单列组合的时窗。[候选消费接口与产品表](https://gitcode.com/cann/ops-transformer/blob/14d022b1a4c3df263bf697534e3c25c7c992ff5d/attention/sparse_lightning_indexer/docs/aclnnSparseLightningIndexer.md)
