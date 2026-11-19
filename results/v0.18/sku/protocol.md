# 950DT／950PR 独立验证口径

RMSNorm 与 DynamicMXQuant 各 10 个输入，每个输入在两颗芯片各首次单独调用一次。两端均使用 `msopprof simulator`、相同主程序、精度与 shape，分别指定完整 SKU，保留 `runtime_soc`、CA 内存配置、数值比对、二进制和原始 trace 的 SHA-256。RMSNorm 的 y 与反均方根都比对，DynamicMXQuant 的 E4M3FN 输出与 E8M0 scale 都逐字节比对。设备时间是最早一条执行事件到所有核最后一条结束事件之间的窗口；每个活动核还须与官方 profiler 的逐核 duration 表吻合。

`excluded-history.json` 给出选择前的去重集合，包含旧资格失败或已登记过的输入。注册表与宿主哈希见 `registration.json`。初稿把 `[513,3968]` 列入 RMSNorm，旧模型的公开接口上限为 512 行。查看新 CA 前将其换为 `[511,3968]` 并重建注册表，原始注册表保存在 `registration-initial.json`。

`freeze.json` 固定 v0.18 源码与定价参数、v0.17 wheel 及全部案例预测。v0.18 两项在 DT、PR、970DT 均显式使用 `resource`。v0.17 RMSNorm 使用 `auto`，DynamicMXQuant 使用 `resource`，均不传常驻核提示。970DT 仅输出配置预测，没有实测。每组 `speedup_ca = dt_ca_us / pr_ca_us`，`speedup_pred = dt_us / pr_us`。`pr_us_reuse_dt` 是直接把新 DT 预测用作 PR 预测，因此它的预期加速比始终为 1。

`pairs.json` 保留逐组冻结预测与双端完整测量，`records.json` 为单端长表；只有双端均合格的行进入加速比统计。`trace-evidence.json` 在冻结后从原始 trace 读取终止核上的流水区间并集、MTE3 和 SIMD 的重叠区间、按 PC 区分的 DMA 发起间隔以及全核描述符字节数。描述符报告的字节数可能经过缓存，不能直接认作芯片实际 HBM 事务流量。

`query-wall.json` 的单位为毫秒，记录同一 Python 进程中每个新 op/shape/SKU 初次查询的墙钟；设备估计及 CA 时间单位为微秒。`all-costs.json` 与 `campaign-completion.json` 单列离线 CA 开销，包含新 DT 和 PR；对 40 次实际启动使用最多四并发，普通输入单次限 200 秒，大输入限 360 秒，总预算 60 分钟和 24 GiB。原始文件位于 `runs/sku18/`。

RMSNorm `[511,3968]` 的首次 PR 捕获在解析期间触及 3200 MiB 的临时输出上限，进程退出后没有完整的解析成功标志。经单独登记，同一输入允许在原预测、宿主、精度和 360 秒墙钟下补采一次，将此例临时上限调整为 4800 MiB。`recovery-registration.json` 保存批准原因与原成本哈希，`primary-summary.json` 保留原 40 次的结果。`final-completion.json` 是包含补采及流水证据的最终摘要。两种上限均在 24 GiB 总盘预算内执行。
