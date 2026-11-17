# FA 轻量采集：复现与证据边界

本轮不改变v0.4预测或CANN kernel。交付 [当前建模公式](../results/long-fa/modeling-strategy.md)、[逐shape误差与成本](../results/long-fa/error-analysis.md)。

## 可复用路径

官方 npusim 0.8.0 helper / contract7 直接生成SQLite；只做SQL归约，不运行 `npusim report`，不生成完整JSON时间线。相同旧FA `[17,1153,64,1,1]` 的单次探索：msopprof全trace保留约120.9MiB、84.7秒；compact约7.2MiB、68.2秒。保留量约降低94%，墙钟约降低19%；这是单次、同shape不同运行比较，不是重复统计或峰值磁盘测量。

原生summary尝试未通过指令时间窗等价性门槛，8线程/1线程没有更好墙钟结果，均保留失败/拒绝记录。现用默认32线程。

```bash
# 构建仅用于本地仿真的显式SKU适配器
bash scripts/build-capture.sh

# 每个name只能执行一次；禁止覆盖旧尝试
bash scripts/cann-run.sh python scripts/probe-fa-capture.py \
  --name my-bridge-run --backend npusim --local-sku-adapter \
  --q 17 --kv 1153 --budget-seconds 120

# 新32k全程仿真；本轮已执行，不需要再运行
# --name my-32k-run --backend npusim --local-sku-adapter --kv 32768 --budget-seconds 360
```

脚本记录host/model/config/adapter SHA、执行命令、数值验证、墙钟和CPU口径。数据库解析验证：正确SoC、进程/应用成功、唯一kernel DB、helper版本、数据库完整性、指令数量、tick/ns换算、活跃核数。缺完整成功证据时拒绝计入误差。

在线预测无需CANN。重绘本轮报告需要本地保留的runs/long-fa数据库和Conda matplotlib：

```bash
toolchains/envs/cann-9.2.1/bin/python scripts/report-long-fa.py
python3 -m unittest discover -s tests -v
```

提交的聚合 `evaluation.json` 已包含逐点结果、流水并集、SHA与32k的127个稳态诊断间隔，可以直接检查，不需要下载原始DB。

## SKU适配器及限制

本机 `npusim -s Ascend950DT` 选择dav_3510物理CA，但默认runtime返回 `Ascend950PR_9589`，不满足项目固定的DT_9582目标。设置CAMODEL_SOC_VERSION本身没有纠正它。标准msopprof注入库也拦截rtGetSocVersion/halGetSocVersion，但依赖自己的采集服务，不能直接拿给npusim。

`capture/sim_soc_adapter.cpp` 是小型替代适配器：先调用真实函数、保留失败；仅当 `soc_version=Ascend950DT` 且 `OPERATOR_SIM_TARGET_SOC=Ascend950DT_9582` 时改写返回SKU。它改变SDK平台查询的目标配置，不只是更改日志标签；不修改物理CA、指令、kernel或计时器。桥接case检验了正确性、entry/tiling key、核数、MMAD和计时，不能证明所有算子/shape等价。没有此适配器的错误SKU采集全部排除。

本轮compact bridge=12.552727µs，旧v0.3同shape=12.486668µs，差+0.529%；本轮标准msopprof重复=12.149697µs。存在运行间差异，不能用单次0.529%作为跨shape误差上界。compact没有导出kernel ELF，身份限于DB entry、blockDim、环境与host哈希。

同一次native+helper桥接：helper指令窗12.555758µs；native summary的起止tick为2534–33386。即使按SDK内嵌main_clk_frequency=2.5换算，也为12.3408µs，不能假定与指令窗等价。因此summary不能直接替换参考值，native日志中的block_start/end也未当计时数据。

## 证据来源

- 已完整读取的 CANNBOT 卡片：`ops/catlass-dsl/guides/profiling/simulator_analysis`，950平台；卡片SHA `42dcf1b1a23c83b4a90cb56db71c9dab3e914e164a72dd8fe9226019d65f3dd7`。原始文档来源为卡片固定的Ascend/msopprof revision `b362f30e7a49ccc5fb80f93f2026332f6001bb82`。它支持“core-id只限制解析、超时部分trace不是完整参考”等采集语义，不为自研适配器背书。
- 知识仓commit `92dc260756e91e72e6dcd60f3996a4d54f43c8f5`；查询及ready/outcome审计见本地runs/long-fa，哈希摘要保存在 [evidence.json](../results/long-fa/evidence.json)。
- 本机CANN9.2.1 primary实现：`python/site-packages/cannsim/core/record/runtime.py`、`paths.py`、`record.py`、`prof/src/common/utils.py`；它们说明注入helper、DB路径及频率换算。`msopprof simulator --help` 用于核对dump、core-id、timeout语义。
- `CAMODEL_CONFIG_PATH` 和 sysconf.esl_top_thread_cnt 来自本机libcommon.so配置加载信息与实际运行验证；其效果属于工程实验，不伪装成知识卡已覆盖的结论。

没有修改SDK或全局CANN配置。本轮新增配置文件仅通过单进程环境变量加载；原始库配置保留哈希。
