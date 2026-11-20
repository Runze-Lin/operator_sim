# V1 的 DT 宿主对照

原样调用 FLA A5 26.9.1 `aclnnChunkKdaFwd`，输入 `[1,2,1,1,128,128]`，两条请求各一 token，保留四个逐元素金标。这个实验检验同一正式库的四次阶段发起能否越过辅助转置。

|记录|原 DT|单次 DT 宿主对照|950PR|
|---|---|---|---|
|采集进度|超时|完成|未采集|
|墙钟 s|3600.9|331.896|—|
|可观察的 V1 主核发起|0 / 4|4 / 4|—|
|完成且可核验的 V1 指令窗|0 / 4|4 / 4|—|
|四组逐项超差/元素数|N/A|attention 0/256, aqk 0/128, akk 0/128, state_vk 0/32768|—|
|重复置旗|目标尚未发起|132|—|
|合格 V1 主核总时间 µs|N/A|N/A|N/A|

原 DT 只捕获第一个 FP32 Transpose 的 `.text`。STARS tick 490 是任务开始，500 是 AIV0 的 block start，542 是发往 AIV0 的 kick。原生 task 文件只有 BEGIN，无 END，官方逐核表为零。超时退出时整机日志虽报 cycle count 10000，无法把它归属 KDA。该次测到的是前置辅助核进入设备，无法判断一次 V1 重复置旗是否发生。

本次沿用原 host、ELF、每个输入/金标 SHA、标准 msopprof 及资格代码，与原队列使用同一互斥锁。原 8 CPU 条件曾临时增加 16 个 CPU 约 50 秒，恢复时整机剩余空闲量为 41.3 个等值 CPU；旧控制器同时检查物理核冲突和余量，原日志没有保存触发时逐线程位置。此次在连续四个 2 秒窗口检查空闲物理核，只使用 112 个，每个实测忙碌率 ≤20%，采前整机空闲 248.9 CPU、1 分钟负载 14.37。给其它作业预留至少 96 个空闲等值 CPU。墙钟限 1,440 s，目录限 8 GiB，仅 DT 一次。采前宿主环境与原运行时段不同，实际机时差不能全归于一个数值参数。

[上次原生日志指纹和首个真实 block](original-native-evidence.json) · [冻结及预留资源](register.json) · [对照状态](state.json)

## 实际窗口

|设备核、ordinal|参与核数|诊断窗 µs|正式组合资格|
|---|---:|---:|---|
|`Transpose_float32_int64_high_performance_10000` #0|1|2.995|N/A|
|`Transpose_float32_int64_high_performance_10001` #0|64|5.729|N/A|
|`ChunkKdaFwd_5e864108d5688e456abb78f238c67b16_2_mix_aic` #0|96|37.439|N/A|
|`ChunkKdaFwd_5e864108d5688e456abb78f238c67b16_2_mix_aic` #1|96|5.364|N/A|
|`ChunkKdaFwd_5e864108d5688e456abb78f238c67b16_2_mix_aic` #2|96|50.190|N/A|
|`ChunkKdaFwd_5e864108d5688e456abb78f238c67b16_2_mix_aic` #3|96|11.785|N/A|
|`Transpose_float32_int64_high_performance_10001` #1|64|2.738|N/A|

该宿主配置到达了正式 V1 主核，实际回调重现重复 SET。此结果排除了“所有失败都发生在辅助启动阶段”的解释。逐段窗口只用于定位，不能用来标定正常硬件耗时。

[实测、ELF 和全核数值/时序回执](dt.json) · [原生任务与回调归属依据](native-evidence.json)

## 将错误缩小到官方 V1 的 Prepare 和 Finalize

|同族物理 ordinal · 宿主阶段|重复 SET|不同故障 PC|与单序列 FULL 故障 PC 相同数|首个匹配 WAIT 之前已二次 SET 的 AIC / 全部 AIC|
|---|---:|---:|---:|---:|
|0 · gate_prepare|114|25|0|2 / 32|
|1 · post_wu|0|0|0|0 / 32|
|2 · fwd_h|0|0|0|0 / 32|
|3 · finalize|18|9|0|2 / 32|

132 次回调的 PC 均在相应的实际载入 `.text` 中反汇编为 `SET_FLAG.M.MTE1`，四个 stage 与单序列控制的融合 `.text` 相同。原单序列 FULL 的故障 PC 与切成四段后的故障 PC 没有重合；同一二进制实际走了不同的代码段。对每个阶段重新从各 AIC 窗口起点计数，在 Prepare 和 Finalize 内各找到两个物理 AIC、各三个事件 ID 的严格前缀证据。同一阶段内第一次匹配 WAIT 发起前已有两次 SET，跨阶段重置事件无法消除这类错误。不能把整阶段的 132 次回调都唯一定位到单行 C++。

例如 Prepare 的 `core0.cubecore0` 对 ID 0 在 tick 42,398 / 56,413 两次置旗，对应相对加载基址 `0x11316c` / `0x141568`，后一个位置正是模拟器报告的错误 PC。供应商可以分别检查 phase 0 和 phase 3 的 TPipe 与 Catlass 本地资源生命周期，然后复用本页同一公开调用和四组原始数组验证修复。两个中间阶段的零回调仅描述这次执行，不能推出其它输入都同步正确。[逐 PC 指令、每核首 WAIT 前证据和原始 trace SHA](fault-sites.json)
