# Pro 的 B8×128k、B8×256k indexer Cast

入口是 vLLM Ascend `indexer.prepare_dsa_weights.float`，实验直接调用同 dtype、同连续形状的 CANN `aclnnCast`。每个矩阵含 8S 行、64 个系数；Q/K 头维没有参与这次 Cast。模型来源、算子二进制、实际 rank 的参数、宿主校验和每个芯片的预测均在[最终冻结清单](../../../results/v0.23/pro-prefill/freeze.json)。

同一项先验证全部 FP32 输出位模式、安装库 ELF、真实 runtime SKU 和物理内存配置，再用官方逐核表与合并 trace 建立全核窗口。每一个活动物理核的分片须包含实际 END。某些分片在 END 之后由 DCI 收尾，此外可能还有 END_LABEL。每核真实退休时钟减官方全核相对截止时刻应恢复同一个全局原点。阈值 3 个时钟覆盖 JSON 浮点量化。Flash 256k 两端回放的全核原点极差分别为 0.0124、0.0235 个时钟。[双端历史回放](../../../results/v0.23/pro-prefill/archived-endpoint-controls.txt)

画图使用实测末核并按 GM 连续地址确定每块元素，验证读写字节、两个 UB 地址、转换分组、读完成至写发起的顺序。上栏采用冻结模型给各块分配的等效服务时间，下栏为真实设备时间。每块地址、区间和全部读发起刻度可用 `*-blocks.json` 重画。大批量采集不会向系统仿真后端增加在线开销。

在这个工作区从仓库根目录启动或恢复串行工作：

```bash
toolchains/envs/cann-9.2.1/bin/python development/validation/pro_prefill23/runner.py
```

已有 supervisor 时同目录文件锁避免重复启动。遇到有开始日志而无 cost 的残留实验保留现场。原始行/形状、额度、已完成的磁盘负载、一次是否进入 native 降级口径分别留在状态与每例回执。只有本次进程存在实际完整 native 起止记录、全元素和 ELF 证据时才单列 native 窗；每例始终只发起一个算子。

预检阶段在任何新形状的 CA 开始前保存了两份较早的本地登记。`initial-freeze.json` 之后清理了历史重放辅助脚本的导入，`pre-control-freeze.json` 之后根据已完成的 Flash 256k 原始 trace 明确 DCI 收尾，并把模型合同逐行连回上游快照。这两步未修改新形状的预测、驱动、预算或数值门槛；`freeze.json` 是采集和计分唯一使用的最终登记。
