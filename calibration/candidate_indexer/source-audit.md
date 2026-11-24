# DeepSeek-V4.1-Flash 的候选 Indexer 到底扫多少 key

**当某个请求最后一行正好看到 1,048,576 个位置，发布配置的第 20 层会给全部 1,048,576 个 C1 index key 打分。950 的候选 consumer 在第 24、28、32、36 层，每层用自己的 Q 和权重，只给入选的 2,048 个八位置块打分，最多 16,384 个 key，最后各留 512 个。** 因而同一个模型既有随上下文线性增长的选块生产者，也有打分条数封顶的消费者。以下按主模型的 0 起层号计算，批内各请求分别处理。这是从固定版本的配置、源码和 950 软件包复算的工作数。用于 CA 的普通 LI 或分解组件的耗时另见实测记录。[发布配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/12b5c3a462cdaf73dfced279a50adc7c2bb4a9c1/config.json#L59) · [950 选择入口](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/models/deepseek_v41/mixed_quant_attention.py#L178-L232) · [CA 口径](README.md)

## 哪个模型，哪一层，哪一份 cache

发布 checkpoint 的 `text_config` 与系统预置 `deepseek-ai--DeepSeek-V4.1-Flash.json` 按字节相同，SHA-256 为 `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`。配置的主干为 40 层，最大位置数 1,048,576、滑窗 128。主 attention 的 64 头、维度 512 与 **Indexer 自己的 32 头、维度 128** 分别计量。压缩比数组共有 43 项，此表只计 40 层主干。[发布配置第 27–43、59–127 行](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/12b5c3a462cdaf73dfced279a50adc7c2bb4a9c1/config.json)

| 主干层 | 索引 K cache 的来源 | 到达 1Mi 的一行上，950 后端执行的索引工作 |
|---|---|---|
| 0、1 | 滑窗 | 无长上下文 Indexer |
| 2–19，压缩比 2 | 2、8、14 各自写，后续层复用最近的 source | 2、8、14 各给 524,288 个已产生的 C2 key 打分一次，其余层复用选中的位置 |
| 20–39，压缩比 1 | 20 写 C1 的普通 K/scale，同时写 folded K/scale 副本 | 20 全表打分，生成位置 top-512，以及每行候选块 ID 和有效长度 |
| 24、28、32、36，属于上一组 | 从 20 读 folded K/scale，复用该层给出的候选块 | 每层独立重打分，上限为 16,384 个 key；单层输出位置 top-512 |
| 21–39 中的其他层 | 复用 20 的 KV 以及最近索引源的选中位置 | 继续执行稀疏 attention，无新增候选打分 |

HF 的参考模型会更新共享的索引 K，在 `end_pos // compress_ratio` 个压缩位置上计算 QK、ReLU、乘头权重、对头求和。在参考代码中，consumer 随后才用共享的布尔掩码过滤，再按分数取位置、按位置排序。若按头切分，参考实现有分数 all-reduce。当前审阅的 vLLM Ascend 实现复制全部 index heads，因此 TP8 系统图里的单 rank 可以仍有 H=32；不能从主 attention 的 TP 直接把 Indexer 工作量除 8。滑窗 KV、用于索引的 K、最终 sparse attention 读取的 KV 应分别计数。[HF 参考模型第 517–611 行](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/12b5c3a462cdaf73dfced279a50adc7c2bb4a9c1/inference/model.py#L517) · [共享索引与 KV](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/12b5c3a462cdaf73dfced279a50adc7c2bb4a9c1/inference/model.py#L721) · [Ascend 索引头复制](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/models/deepseek_v41/indexer.py#L33-L50)

## 从行有效长度到 Cube 的 tile 数

令 `v` 为本行**因果可见的已压缩位置数**。满一行 C1 为 1Mi，C2 为 512Ki。对 C1 请求，最终有效长度为 `L`、本次实际输入 `q` 行，第 `j` 行（0 起）可见 `v = L − q + j + 1` 个位置。带压缩余数、多请求和图补齐时，kernel 会用当前请求的 `seqused_k`、`used_q`、压缩余数及行号做夹取，padding 行清零。对每个有数据的 query 行：

1. **生产者**在所有可见位置上算分，每 8 个位置取最大值，再取分数最高的至多 2,048 个块。wheel 在 `ops/quant_lightning_indexer_dsl.py` 第 2757–2790、2793–2853、2898–2917 行给出有效长度、块最大值和写回，第四个返回值明确为 `c = min(2048, ceil(v / 8))`；空行返回零（第 2649–2679、3894–3898 行）。第三个返回值存候选块号。HF 参考对最新块一律置正无穷，wheel 只对**未满 8 个位置的最新块**强制入选。两处的选中块身份可能不同，长度公式不受影响。生产者直接从位置分数同时产生两种排行，仅凭位置 top-512 无法还原块 top-2048。
2. **消费者**按这 `c` 个块号做 PA 查表和 gather，`TILE_N=512`，每 tile 装 64 个候选块，故行上执行 `ceil(c / 64)` 个候选 tile，最多 32 个。Cube tile 的 M=32 对应一行的 32 个 index heads，K=128 对应 head dim。实际发给 Cube 的 padded key 槽位为 `512 × ceil(c/64)`；逻辑有效 key 为各选中块中有效位置数之和。在该生产者及连续可见前缀下，非空行的有效数为 `8c − ((8 − v % 8) % 8)`。对于外部提供、可能重复或含无效 ID 的候选表，必须使用其真实长度及索引有效性。计算发射量、数学有效量、最终保留的 top-512 各自不同。
3. **流水**在每个 tile 先把每个候选八位置记录寻址到真实物理页，再 unpack 到四槽 GM staging 的 key 和 scale，AIC 以 MXFP4 QK 乘加、FP32 累加、转换到 BF16 交接给 AIV 评分，随后做 top-k 和需要时的跨分片归并。候选 QK 的 tile 算术量为 `2 × 32 × 512 × 128` ops，每候选块的原始 K/scale 记录为 `8 × (64+4) = 544` 字节。缓存命中、重复读、GM staging 往返、各级 buffer 和阶段重叠单列计算。wheel 的可执行 ABI 最多支持 36 个 Cube worker，实际 grid 由设备 Cube 数目提供；未来芯片外推时可迁移工作分解，原 binary 的 36 worker 上限不代表新芯片规格。可核查 `quant_sparse_lightning_indexer_dsl.py` 第 51–77、579–643、1388–1423、1466–1521 行以及 `quant_sparse_lightning_indexer_metadata_dsl.py` 第 108–132 行。[950 官方算子 wheel](https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/cannbot_arena_net_ops-0.1.0-cp312-cp312-linux_aarch64.whl)

例如在 1Mi 处结束的连续 Q=9，单个 consumer 有 147,428 个**有效** Q×key 对，启动 288 个 512-key tile，对应 147,456 个 Cube key 槽位，不含跨头乘数。实际系统长上下文捕获的一条 B1、TP8/EP32 的末块为历史 1,044,480、Q=4096、L=1,048,576、H=32、D=128。此处一个 consumer 有 67,108,864 个 padded Q×key 槽位，四个 consumer 各执行一次。C1 层 20 对全部因果可见位置的**算法工作**为 4,286,580,736 个 Q×key 对（4,096 行等差数列）；这不是对该层 native 打分实际发射 tile 数的测量。若 1Mi 指的是完整 prompt，再生成新的 token，会超过上述发布配置上限。[逐锚点系统输入](context-sensitivity-flash-anchors.csv)

## 如何确定 950 后端和测量边界

| 调用或数据 | 能核实的实现边界 |
|---|---|
| 950 后端选择 | 硬件能力 `DSV41_PACKED_CACHE` 与 A5 的 `get_dsv41_packed_cache_ops` 选择 `MixedQuantPackedCacheOps`。普通 index cache 每位置 64 B 打包数据及 4 B E8M0 scale，候选源还维护每 8 个位置一条 544 B 的 folded twin。层 20 用 `quant_lightning_indexer`，消费层用 `quant_sparse_lightning_indexer`。 |
| Q 转换 | `attention/dsa_v41.py` 第 388、413–430、513、529、584–594 行可在辅流做独立的 MXFP4 Q 量化并在选择前等事件；没有预先打包的 Q 时，`mixed_quant_attention.py` 第 24–48 行单独调用 Triton 转换。QLI/QSLI 接口的实际 Q 已打包。输入投影、RoPE、Q 量化、K 写缓存与折叠副本、随后按位置排序，都有各自归属；评估消费节点的边界时要对齐这些成本。 |
| consumer 入参与入口 | `mixed_quant_attention.py` 第 106–175 行使用 Q(T,32,64) uint8、Q scale(T,32,2,2)、FP32 头权重(T,32)、folded K(P,page/8,544)、候选 ID(T,1,2048)、候选有效长度(T,1)、请求行起点/有效 K 长度与 PA block table。先建 QSLI metadata，再调用 `vllm_ascend::quant_sparse_lightning_indexer`，最后单独调用 `prepare_indexer_indices`。metadata 与后处理均在 fused QSLI 设备窗外。 |
| Torch → AOT | `vllm_ascend/ops/packaged_attention.py` 第 22–28、320–376 行将 PrivateUse1 实现绑定到 wheel 的 `ops.quant_sparse_lightning_indexer_dsl`。wheel 的 `ops/__init__.py` 注册 `_native`，`operators/qsli/operator.json` 有 36 个 DAV-3510 AOT variants，记录分页与 TND 两个源函数。此路径匹配的是 vLLM 插件和配套 wheel；CANN 标准同名/近名 API 是否安装是另一条实现链。 |

[950 硬件能力](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/device/hardware_profile.py#L333-L362) · [设备算子类](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/device/device_op.py#L1058-L1061) · [缓存规格](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/models/deepseek_v41/cache_config.py#L85-L122) · [Q 量化及同步](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/attention/dsa_v41.py#L413-L430) · [Torch 分派](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/ops/packaged_attention.py#L320-L376)

**页表宽度影响地址计算。** 官方 950DT 部署示例传 `--block-size 128`，C1 folded cache 每物理页有 16 条八位置记录。kernel 由 `k.shape[1] × 8` 求出真实页大小。wheel 第 673–693、799–857 行将 `block_table.shape[1] <= 2048` 的表缓存在 UB，过宽则逐候选标量取块号和页表项，再读取记录，Cube 工作条数仍由候选长度决定。vLLM 构造 metadata 时仅裁请求行，保留页表的列宽。按 page=128、每个请求整表保存在一处的条件，128Ki、256Ki、512Ki、1Mi 上下文至少需要 1,024、2,048、4,096、8,192 列。预分配余量、最大上下文容量或 CP 划分会改变实际列宽；跨过阈值的依据是传入的表，最少需要的列数只构成下界。源码没有给出每个地址依赖的延迟周期。[950 部署命令](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/docs/source/tutorials/models/DeepSeek-V4.1-Flash.md#L264-L299) · [vLLM 构造缓存 metadata](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/attention/dsa_v41.py#L1387-L1404)

### 默认软件包的 AOT 身份

在上述 wrapper 上，`mask_mode=3, cmp_ratio=1, layout_k=PA_BBND, has_cu=True, has_seqused_q=False, has_seqused_k=True, has_block_table=True, has_output_offset=False, pa_block_size=128, splits=8, auto_metadata=False`。wheel 的第 2050–2064、2104–2121 行产生该 profile，第 2171–2215 行列出默认导出顺序。配套 runtime 的 `cannbotdsl/package/native/publisher.py` 第 75–80、123 行保留**首次收集顺序**。按默认导出集合，并与各顺位 PA/TND、页宽和可选入参的 sidecar 契约交叉检查，`qsli/operator.json` 的 `artifacts[16]` 是这一配置的候选 provider：

- `binary_key = 400228d98408fad5510dc0ff2580bb878c30b3f8dd4457e2ee349b2300d4e3de`
- `provider.so` SHA-256 为 `6f071b48df79d0410a146a482938c4cee44771c85cd3defcea003b963d384a1d`，sidecar 含动态维度 `B, K_S0, MAX_BLOCKS, PA, T1, WORKERS`。相同 binary 可接收不同的页表宽度。
- `runtime.py` 第 134–137、239–270、299–320 行显示默认模式为 `prefer`，实际请求按 IR/契约与目标形成的 binary key 匹配已注册资源。`CANNBOTDSL_NATIVE_BINARY_REPORT` 可记录 `installed_hit` 和 key。自定义导出文件或进程策略可能改变行为。这里的身份是软件包静态推定；原部署进程的加载收据、该融合 kernel 的生产 CA 时窗应单独记录。[固定版本 Dockerfile 的配套二进制来源](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/Dockerfile.a5#L89-L92) · [配套 runtime wheel](https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/cannbotdsl-0.6.0%2Bg91e7c8f-cp312-cp312-manylinux_2_28_aarch64.whl)

### 单行满候选的 native CA 核对

在 Q=1、可见 L=1Mi、C=2,048、PA 页 128、表宽 8,201 的冻结样例中，两端 msOpProf 的 `installed_hit` 均命中上述 AOT。逐字节在 provider 中查得所加载的设备 ELF，偏移 443,904，SHA-256 为 `b3a854470fe46368ac752ccccd7ed0bb7ffb087b658373c2ac97bfce027ae4ff`。用官方 metadata 程序的宿主解释器生成调度表，32 份局部结果交给一个 AIV 归并。两端都发出 32 条 M=32、N=512、K=128 的 `MMAD_MX`，512 个返回全局位置全部通过独立 CPU golden。950DT 的 32 个 AIC 各执行一次，950PR 的 28 个物理 AIC 中 24 个执行一次、4 个各执行两次。本机 CA 环境的宿主属性在两个 SKU 均返回 32 个逻辑 worker。此次整窗包括实际分片归并，未计设备侧 AICPU metadata、上游量化与末端按位置排序。[原生逐端 1Mi 单行收据](long-row-results.json) · [冻结输入及精度判据](long-row-fixture-freeze.json)

| 设备 | 冻结预测 µs | 单行融合 CA µs | 有符号误差 | 最后一条 MMAD 退休到整窗结束 µs |
|---|---:|---:|---:|---:|
| 950DT | 11.390 | 21.630 | −47.34% | 10.273 |
| 950PR | 12.219 | 21.092 | −42.07% | 8.893 |

两端末核均为第 0 核的第 0 个 AIV。源码 `LdMergeStage` 第 480–526 行中，单行 32 个分片的局部 top-512 需要五轮串行归并，每轮最多处理 7 份新输入，后续轮带入上轮的 512 个赢家。各轮处理 3,584、4,096、4,096、4,096、2,560 个索引/分数；CA 中最后一次直方图指令在距离整窗起点 21.153 / 20.541 µs 时退休。末次 MMAD 至整窗结束的差值也包含局部选择、映射、屏障与写出。按真实单行证明的分片尾部不能直接乘 4,096 来代表整个末块，实际 prefill 会有不同的行间并行与归并安排。[逐端流水和选中位置收据](long-row-results.json)

## 与系统捕获、校准的对应

本地系统源码 `preview-0.1.0-preview.1/sim-workspace/inference-models-simulator/src/inference_models/models/text/deepseek_v41/attention.py` 第 34–66、69–105 行按相同的 KV source / index source 建图，四层各发一个 `inference::candidate_filtered_indexer`。它接收逻辑 Q/K、权重、位置和候选张量，元设备建图仅给形状，`candidate_blocks` 的输入只有 512 个赢家，输出是按 2048 容量分配的占位。它无法提供具体块号、物理页布局、动态长度或打包转接的运行轨迹。大模型报告里的 BF16 全扫、MXFP4 候选优先，以及用户实际 950 集群运行，应各按自己的接口和 dtype 解读。全表 block max/排行属于第 20 层生产者，不能在每个 consumer 或只负责物化的辅助节点重复收费。[长上下文对比及系统锚点](context-sensitivity.md)

检索范围包含发布 HF 的 V4.1-Flash 文本配置和 inference 源码、vLLM Ascend 提交 `c09d3e20b9c45c80b34ae3781440d90904a44023` 的 V4.1 模型/设备路由/教程及 Dockerfile、教程对应两个 aarch64 wheel、当前系统预置和捕获、CANNBot 的 950 普通 QuantLightningIndexer 运算符资料，以及已有 CANN 9.2.1 OPP 接口审计。上文引用的本地插件主要文件与 Git tree 的 blob hash 已核对。未将相邻产品 DeepSeek-V4-Flash、V4 Pro、A2/A3 的候选 op 或 CANN 的普通 LightningIndexer 替代此模型的融合 consumer。教程给出 950DT 原始 MXFP checkpoint 单节点 DP8/TP1/EP8 参考部署；本文用它确认硬件软件路径，系统实验的 TP/EP 另取自捕获。[950 支持范围及 checkpoint](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/docs/source/tutorials/models/DeepSeek-V4.1-Flash.md#L32-L57)

| 可复核的上游文件 | SHA-256 |
|---|---|
| HF `inference/model.py` | `e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d` |
| 算子 `cannbot_arena_net_ops-0.1.0-cp312-cp312-linux_aarch64.whl` | `4f2caa0f52cbe6234b90843d4e83e543cfb7e58a89f44567dd343ecea79cb9ca` |
| runtime `cannbotdsl-0.6.0+g91e7c8f-cp312-cp312-manylinux_2_28_aarch64.whl` | `65b0957dd5462ac97e9318c688d95e9e4e71815832ef3b8c73caf0211998c51f` |
| wheel 内 `ops/quant_lightning_indexer_dsl.py` | `c4bdf5431e66a8deb659965dba91236e55ebdfb3946bf150830f701bb331d744` |
| wheel 内 `ops/quant_sparse_lightning_indexer_dsl.py` | `a1a8983368599d765c7f7ac1fc303972e3d0bc7e17c1abed502536dc9f17a1e7` |
| wheel 内 `ops/quant_sparse_lightning_indexer_metadata_dsl.py` | `16a75f7b2bb79ecccf3c2caa93527078d313ad1ebd381623170db475fd088e95` |
