# v1.11 · Flash A5 候选打分

- 发布 `estimate("candidate_filtered_indexer", ..., dtype="mxfp4", attrs={"implementation":"a5_qsli_projection"})`。固定版本 vLLM Ascend 加载的 PA_BBND 融合内核先按 8 个 key/块收集，再用 Cube 做 32×512×128 的 MX 点积；AIV 负责头归约和位置选择。原 BF16 全表引用保留。SIMPlatform 可以显式选择实现和页表预留大小。
- 资源先验独立快照为 Git `a1106fa`。宽表同一行内相邻 tile 的平均间隔来自强制单团队诊断的 31 个间隔（109,555 cycles）；行首、局部选取和跨团队输出另计。常数 64 周期寻址先验可显式重放。该校准只识别循环服务间隔，芯片迁移时仍检查目标资源约束。[冻结方案](../../calibration/candidate_indexer/qsli-interval-protocol.json) · [训练窗](../../calibration/candidate_indexer/qsli-one-team.json)
- 原生融合小形状走相同发布二进制，两款 950 的 2,048 块满候选单行均通过 512 个正确位置核查。冻结预测为 11.390 / 12.219 µs，CA 为 21.630 / 21.092 µs；单行跨核归并偏低。窄/宽页表对照看到了标量 load 数变化。另有三个普通 QLI 组件几何，两端均取得完整 CA，不能用其窗替代 QSLI。全部实测都按正确性、计时与实现身份单列。[误差明细](../../docs/errors/flash-a5-qsli.md)
- 950DT 另留出 1Mi、32-query 满候选并发单波；一次采集与解析 758.1 s。1,024 条 MXFP4 MMAD、16,384 个正确位置，CA 整窗 84.792 µs。原先验和新冻结模型分别为 52.021、75.856 µs，低估由 38.65% 降到 10.54%。在同图直接按该波次外推，末块为 0.484 / 0.487 s（两种启动假设）。[独立验证](../../calibration/candidate_indexer/prefill-wave.json) · [同图复算](../../calibration/candidate_indexer/wave-chunk.json)
- 沿用原图的 33 个请求配置，重放 Flash 的 105 个捕获点（138 张阶段图），11 组的候选合同改变，另外 22 组逐配置相同。1Mi/B1 末块历史 Roofline 15.176 s，BF16 资源参考 0.871 s，A5 来源校正后的 MX 场景 0.481 s。只改四个消费层，保留原图其他报价。整批请求含超预置的 decode，1Mi/B1 为 93.074 s；端到端未做真机验证。[逐阶段与全部请求](../../calibration/candidate_indexer/context-sensitivity.md)
- 常驻进程从仓外安装 wheel 后，3 个芯片 × 4 个几何 × 21 次公共 `estimate` 调用中位 0.130 ms，含首次载入最大 6.175 ms。普通 QLI 对照每 SKU 约 232 s 采集墙钟；合格原生满候选单行 DT/PR 约 150/153 s，先前缺 profiler IPC 的尝试另占 141/146 s；强制单团队合格诊断 230 s。原始多 GB trace 在 git 外。[在线复算](replay.json) · [采集成本](../../calibration/candidate_indexer/mx-acquisition-cost.json)
- 完整 4,096 行 prefill 仍是系统外推。AICPU 元数据生成、准备 packed 输入与 folded cache、返回位置后处理及请求间关联需要各自的目标运行时证据。970DT 尚无 CA。
