# 950PR 独立验证：四种预测共用同一实际窗口

`budget.json` 预先登记 30 个主样例和 6 个有序替补。选择依据是输入几何、DT/PR 既有采集身份、宿主 golden 和 CA 资源限制，不使用 PR 实测时延选点。只有数值/采集失败才能启用该算子的下一替补。PR Q=9、KV=32767 是必测，不得以替补代替。

`freeze.json.predictions` / `holdout.json` 中：

| 字段 | 生成方式 |
|---|---|
| `pr_us` | 显式 v0.16 `timing_mode="resource"`、PR 配置，主要考察方法 |
| `legacy_resource_us` | 隔离安装的 v0.15，PR 配置只改名称，让该版本不识别为已校准 SKU |
| `dt_us` | 同形 DT 配置的估计时延直接用作 PR 时间（不是 DT CA） |
| `prior_pr_us` | 隔离安装的 v0.15 + 实名 PR 配置，允许该版本对 PR 的专属校准 |
| `auto_pr_us` / `calibrated_pr_us` | 分别核对新版默认 PR 和显式校准；须与 `pr_us` / `prior_pr_us` 一致 |
| `pr_ca_us` | 冻结后官方 `msopprof simulator`、精确 `Ascend950PR_957b` 的全部核完整指令窗口 |
| `dt_ca_us` | 有历史配对则从既有归档引用，否则 null；不另采集，不影响 DT 预测对照 |

数值输出、唯一实际库 kernel、精确运行时 SKU、PR LPDDR5 模型、逐核跨度和官方摘要均需核验；完整原始 trace 哈希保存在逐点结果中。已列入开发的历史 PR shape 不再当独立样本。冻结时间必须早于首次新 CA，在线查询预算 100ms、离线 CA 预算单列。
