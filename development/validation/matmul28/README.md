# Matmul 双芯片采集

这一组直接调用 CANN 9.2.1 已安装的 `aclnnMatmul`。程序按目标分配完整的 BF16 ND 张量，输入使用随行、列、K 变化的正负值，先对输出填 NaN，再检查每个输出的 BF16 位值。主模型配置、TP 切分和应用权重的逻辑形状来自工作负载合约。独立的算子调用与部署时实际的量化、格式转换和融合需分别核验。

在仓库根目录构建采集程序。已冻结的运行使用登记的二进制 SHA256。

```sh
mkdir -p build/smoke
bash scripts/cann-run.sh bash -c 'g++ -O2 -std=c++17 development/validation/matmul28/runner.cpp -I"$ASCEND_HOME_PATH/include" -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath-link,"$ASCEND_HOME_PATH/lib64" -lopapi -lnnopbase -lascendcl -o build/smoke/matmul28'
```

| 任务 | 工具和结果 |
|---|---|
| 常规 decode/MTP、prefill，合计 16 形 | `campaign.py` 分开监督 decode 和 prefill；每个槽逐一采集芯片，`report.py` 从结果重建逐形表。[进度与结果](../../../results/v0.28/matmul/report.md) |
| Pro 与 Kimi 每卡 8×32768 行 | `wide_prefill.py` 复用 RMSNorm 的单槽，在主机有充足空闲核时执行。[进度与结果](../../../results/v0.28/matmul/wide-prefill/report.md) |
| Flash 每卡 8×131072 或 8×262144 行 | `long_prefill.py` 复用 decode 的采集槽。[进度与结果](../../../results/v0.28/matmul/long-prefill/report.md) |
| 一块输出切分 K | `streamk.py` 登记显式方案；`streamk_audit.py` 从每核输入地址、MMAD 深度、部分和及终端写核核对实际库分支。[同集分析](../../../results/v0.28/matmul/streamk-analysis.md) |
| 另六个窄投影、两个重选块形状 | `narrow.py` 和 `multi/campaign.py` 分别保留公开路由与重选块候选的独立冻结。[窄投影](../../../results/v0.28/matmul/narrow/report.md) · [重选块](../../../results/v0.28/matmul/multi/report.md) |
| 28 个几何的整体计分 | `overview.py watch` 读取冻结价格和双端回执。运行 `refresh_reports.py` 可在新增结果时刷新公开入口同集误差、三个模型的整批张量表和版本快照。采集程序独立运行；汇总进程检查冻结模型散列。[全部数据](../../../results/v0.28/matmul/README.md) · [公开入口的误差与 DT/PR 比值](../../../results/v0.28/matmul/public-report.md) |

计时不使用 host 墙钟。对所有活动核的合并 trace、单核 shard 和官方核表检查时间与核心名册，并检查每核的物理 END。早期检查把标量 END 当作最后可能的退休事件，会漏掉已提交到 FIXP 或跨块队列的长描述符。`drain.py` 保留原始检查回执，重新读取同一批 shard 的全部执行区间，取各核最后退休的事件。每个核的首尾边界必须在共同原点上与合并 trace 对齐，容差沿用原登记的 3 周期。该处理只改变数据资格核验，冻结预测均可用 SHA256 找回。逐形表保留超时、未通过完整核验和异步引擎退休等状态。新芯片不参与 CA 采集。
