# 950DT 上的完整 prefill Cast

DeepSeek V4 Flash 输入按单卡 8 个请求 × 32,768 token、hidden size 5,120 整段调度推导。单次 `aclnnCast` 将 [262144,5120] 从 BF16 转成 FP32，共转换 1,342,177,280 个元素，逻辑读量为 2.5 GiB，逻辑写量为 5 GiB。如果系统将 prefill 切成 chunk，应按各次实际调用输入后端。

![上部为模型两条并行资源约束，下部为 31 个完成核的原生窗口，使用同一微秒轴](cast-full-hidden-partial.png)

| 上一次全尺寸采集 | 结果 |
|---|---:|
| 启动 / 有配对结束的 AIV | 64 / 31 |
| 已完成核最早开始 / 最晚结束，wrapper tick | 11,479 / 3,476,654 |
| 完整设备窗口可证实的下界 | ≥ 2,100.106 µs |
| 全量数值校验、全核耗时、模型相对误差 | N/A |
| 宿主墙钟 | 6 小时 11 分 33 秒 |
| 128 个周期状态输出占总落盘比例 | 96.08% |

原记录缺少 AIV 19 和 32–63 的 wrapper done。下界只使用 1.65 GHz 同域 wrapper tick。上图的模型给出最长核本地流水 1,081.88 µs、共享存储约束 2,514.82 µs，取较大者。下图的原点是已完成核中最早可见的 wrapper 开始。[逐核原始 tick 及摘要哈希](previous-partial.json)

两项独立短控依次使用同一 [32,4096] 输入及同一 CPU 集合，对照保留所有周期日志与将 128 个已知日志文件重定向到 /dev/null。两次均要求完整数值校验、64 个核的相同标识、全核窗口对照和系统调用文件模式核验。短控实际观测到安装包 `ops_math/cast_apt/Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance.o`。已预登记长任务 24 小时，短控达标时采用 16 GiB 上限，日志保留时采用 48 GiB。完整 Flash 重采排在已启动的 [262144,7168] Kimi DynamicMxQuant（12 小时 / 24 GiB）后。[冻结合同](v2/registration.json) · [实时队列状态](v2/queue-status.json)
