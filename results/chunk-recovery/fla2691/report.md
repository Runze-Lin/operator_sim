# Kimi K3 使用的 FLA A5 发行包与 Chunk KDA 分派

Kimi 的 prefill 会收集初始 VK 状态、在调用侧对 Q/K 做 L2 归一化，再调用 FLA 的 `chunk_kda_fwd` 并写回最终 VK 状态。项目固定的 vLLM Ascend 快照 `c09d3e20` 的 A5 镜像为 CANN 9.2.1，FLA A5 固定 26.9.1。这个版本通过独立的厂商 OPP 和 `libcust_opapi.so` 发布。其 8 个融合 KDA 设备对象有 4 种不同的 `.text`，与 CANN 9.2.1 内建目录中 8 个包的 4 种 `.text` 完全不相交。这为同名 `aclnnChunkKdaFwd` 提供了真正不同的实现。宿主项目镜像的下载路径在核查时返回 404，正式 release 给出了同名 A5/aarch64 wheel；下载大小 121,538,023 B，SHA256 与发布方元数据一致。wheel 内的 OPP 标记编译器为 9.1.0。[目标 vLLM 镜像](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/Dockerfile.a5) · [Kimi 调用路径](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/ops/kimi_kda.py) · [FLA 26.9.1 正式发布](https://github.com/flashserve/flash-linear-attention-npu/releases/tag/v26.9.1-release) · [冻结输入与二进制哈希](register.json)

## 同一模型有两种 Chunk KDA 实现

在该版本的默认 Kimi 合同下，令每次调用包含的各序列 token 数分别为 $S_i$，每个注意力 TP 分片有 $H_v$ 个 value 头，分块长度为 64。FLA 的工作项数为 $W=H_v\sum_i\lceil S_i/64\rceil$。当 Q/K/V 是 BF16、$K=V=128$、所有变长区间严格递增且 $W\ge4096$ 时，调用 `aclnnChunkKdaFwdV2` 的组合实现，顺序执行 Prepare、FwdH、Finalize。较小工作量走融合入口 `aclnnChunkKdaFwd`。在库的通用接口中，内核内另做 QK L2norm、beta sigmoid、允许负特征值或关闭 exp2 等非默认开关可强制 V2；Kimi 的 raw-gate 融合开关本身不会强制 V2。4096 来自 FLA 发行包的实际 host 分派，即便其注释把阈值的原始调优数据归于 A2，在该版 A5 wrapper 上也会采用该阈值。[vLLM 对 FLA 的实际参数](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/ops/kda.py) · [FLA 固定源码](https://github.com/flashserve/flash-linear-attention-npu/tree/5cc699069c7f)

| 单卡同次调用，假设本地 12 个头 | 每次调用的 $W$ | FLA 26.9.1 路由 |
|---|---:|---|
| 8 个请求，各 32,768 token 整段 | 49,152 | V2 三段 |
| 8 个请求，各 131,072 token 整段 | 196,608 | V2 三段 |
| 8 个请求，各 262,144 token 整段 | 393,216 | V2 三段 |
| 8 个请求，各只调度 512 token | 768 | V1 融合 |
| 8 个请求，各只调度 2,048 token | 3,072 | V1 融合 |

表里的 $S_i$ 是这一批实际送入 KDA 的新 token 数。系统仿真可以直接用调度后的分段长度和本地头数复现分派。表中这些生产输入尚无合格的整组 CA 窗口，数值仅用于验证选择规则。公开 V2 入口的两 token 实测见下方独立记录。

## 一次小型复核

950DT 单流运行 BF16 Q/K/V 和 raw gate、FP32 beta，$B=1,S=64,H_q=H_v=2,K=V=128$，$W=2$。两个头用于缩小本地开销。保持 gate、真实宽度、布局、显式变长索引、初态及全部 4 组独立 CPU 参考检查。宿主通过已发布 FLA OPP 执行，设备代码没有修改。注册上限 900 s、1 GiB、1 次 CA；大小与耗时分开核验。使用 loader 的只读重定位诊断确认两个 ACLNN 符号实际解析到 wheel 的库。宿主打印出的 `dladdr` 地址落在可执行程序的 PLT，不能单独识别设备 ELF。实际主库的身份由捕获的设备代码 `.text` 与发布包逐字节哈希共同确定。[采前约束](register.json)

终态与逐核数据见[回执](receipt.json)及[重复 SET 对应的机器指令](faults.json)。这套验证对四组数值、输入状态保全、物理设备、主 kernel 身份、全部串行 launch 的逐核指令/官方 duration 和 mem-fence 错误分别记账。缺少完整输出时，保留已经跑到的确切阶段；辅助 transpose 的独立窗口只用于核查数据采集链。

外部厂商若提供修复，能够直接复验的对象是这份 wheel 和输入对应的 BF16 V1 MIX。复核还需重新比较该代 950 TPipe 的事件分配、释放和嵌套资源的生命周期。源码审查器的候选逐条保留在 `faults.json`；这些候选覆盖跨分支文本，动态部分用实际运行 PC 判断发生的方向。[CANN TPipe 约束](https://gitcode.com/cann/asc-devkit/blob/56500ea0dc219380f50f4a17ce73ef5264f830fc/docs/zh/api/SIMD-API/basic_api/resource_management/TPipe/TPipe_constructor.md)

## 实测终态

| 新采 950DT | 观测 |
|---|---|
| 实际主 kernel | `ChunkKdaFwd_5e864108d5688e456abb78f238c67b16_2_mix_aic`，设备 `.text` 匹配 wheel |
| 主核执行到达位置 | 96 份 shard 均有真实 END，64 份有 CA 终点标签，2 个 AIC 发出 26 条 MMAD，日志出现主任务 TASK_DONE |
| 同一 AIC 的同步异常 | 240 次，49 个不同 PC，全部解码为 `SET_FLAG.M.MTE1` |
| 整个公开接口的数值 / 计时 | 四组 golden 返回 0/4；全调用耗时 N/A，主核耗时 N/A |
| 辅助 transpose | 前两次完整窗分别为 3.3691 和 4.7861 µs；主核结束后还启动了第三次 transpose |
| 采集成本 | 311.23 s；2 s 轮询检测到超出 1 GiB 阈值时为 1.226 GiB，退出码 −9 |

`core0.cubecore0` 在 tick 31,345、31,347、31,353 为事件 0、1、2 分别发出 M→MTE1 SET。同一核在 tick 77,143、77,147、77,152 再次设置这三面旗，中间没有消费同旗的 WAIT。随后在 tick 79,503 发起首条 MMAD，最终完成内核任务。这既确认第一组错误发生在哪个生命周期，也表明本例的重复 SET 可与后续矩阵运算并存。V1 的外层 `TPipe` 和 CATLASS 含 `TPipe` 的局部资源是应向该 FLA 版本追查的具体源码位置。缺少行表、补丁差分和输出，仍无法把一个 C++ 内联变量或输出错误作为已证实根因。[逐核真实指令](partial-main.json) · [每个故障 PC](faults.json)

源文件从 vLLM Ascend 上述完整 commit 的 GitHub raw 地址重取，Dockerfile、Kimi 适配和 FLA 调用层均与缓存逐字节一致。[文件与官方原文哈希](vllm-source.json)

另一次独立、直接调用 V2 的 950DT 两 token 诊断已全数值和全流水通过。三段实测合计 23.2406 µs，该诊断保存为单独数据集。[V2 正式里程碑与真实流水图](v2/report.md)
