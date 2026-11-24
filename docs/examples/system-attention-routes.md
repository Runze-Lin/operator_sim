# Decode 图中的索引、注意力与序列状态

本页覆盖三模型 128 Ki decode 捕获中的 13 种设备节点。输入使用每个 rank 的物理张量，MXFP4 每字节两个值，scale 单独占用空间。计算使用本地逻辑维度、查询长度和有效 KV 长度。以下算例使用随包的 950DT。端到端配置可以覆盖芯片资源，结果会随之改变。

## 调度与计价

一张芯片若有 $C_c$ 个 AIC、$C_v$ 个 AIV、频率 $f$ GHz 和对应精度整芯片峰值 $P$ TOP/s，单核速度为 $r=1000P/(Cf)$ ops/cycle。$r_c$ 使用乘加对应的精度、$r_v$ 使用 FP32 累加精度。乘加计两次运算。一个 AIC 可搭配的 AIV 数取 $p=\min(2,\max(1,\lfloor C_v/C_c\rfloor))$；无跨核同步实测的新代际沿用这一实现假设。

分核先数独立 tile，$W=\lceil N_{tile}/C_{live}\rceil$ 给出最慢核的波数。每个阶段按本核的计算次数或所搬字节求周期，同一资源串行，双缓冲允许的重叠取最大值。模型时间为

$$
T_{\mu s} = L_0/(1000f)+\max\left(WC_{path}/(1000f),\ (R+S)/(1000B_{HBM})\right).
$$

$L_0$ 为首发和未分解的启动控制周期；$C_{path}$ 是关键波的片内流水长度；$R,S$ 为实现估算的逻辑读写字节；$B_{HBM}$ 为芯片 GB/s。最后一项是按该搬运方案算的共享端口预算，重复行能否命中 L2 仍依赖实际地址；API 的 `physical_hbm_bytes` 返回空值。AIV 本地搬运用 $n f/B_{MTE2}$ 周期，$n$ 以字节计；缺少端口实测时，索引用已校准的 416 cycle / 32 KiB 发起间隔，普通状态更新假设每核 78 bytes/cycle。尾波不足一整块时保留关键核的时间。返回的核数是模型分到有用工作的核数，CA 的全核启动名单另计。

## 按图节点复核

以下 $B,Q,H,L,D$ 分别指 rank 本地请求、每请求的校验步、查询头、每请求有效键数和每头维度；$E$ 是图给出的有效 query–key 对数。表中的“外推”覆盖有已测组件但缺少该图完整执行的情况。

| 图节点 | 具体算例与内部流水 | 950 证据 |
|---|---|---|
| `vllm_ascend::quant_lightning_indexer_reference` | Pro 的 BF16 Q[9,64,128] 与 K[1,32770,128] 计算每个头的点积，再做正值截断、加权求和和 top1024。三组查询块沿 K 扫描，每页 128 键，每个 8192 键的分段与前段赢家归并。前缀掩码会使最后几行的有效位置略有差异，预算按有效银行上界。 | BF16 普通 LightningIndexer 是可比的运算组件；名字中的 quant 未改变本图 Q/K 的实际 BF16 类型。 |
| `inference::lightning_indexer` | Flash 第 2、8、14 层取 H32、L65540、top512。第 20 层全表 L131081 时，9 行约 118 万对，另做每 8 个位置一次块最大、最新有效块置顶和块 top2048。该成本计在完整 logits 还在生产者手中时。 | 复用普通 BF16 LightningIndexer 的 H8 和 H32 组件记录。全表长循环及融合侧输出外推。 |
| `inference::candidate_blocks` | Flash 接口表面输入 [9,512]，输出 [9,2048] I32。使用源层生成的块表，按行分核发布，逻辑读写各 $9\times2048\times4=73728$ B。两组 logits 即使有相同的前 512 个赢家，也可以有不同的前 2048 个块。系统契约假定有该侧输出。 | 暂无该侧输出的整窗 CA。若换图使生产者不生成侧输出，需要传入全量 scores 或块分数。 |
| `inference::kimi_dsa_indexer_score` | Kimi 的 FP4 查询存 [9,32,64] U8，另有每头 4 个 scale；K 的每行 68 B 包含 64 B 值和 4 B scale。扫描 **131081** 个有效位置，top2048 的 int32 缓冲不参与点积。$2Q L H D$ 次 MXFP4 Cube 运算，分头加权/排序上 Vector。 | 950 的 QuantLightningIndexerV2 模式 5 可执行 MXFP4，并要求独立 E8M0 scale 与分核 metadata。图上的 68 B 混排直接读取路线外推。 |
| `vllm_ascend::sparse_attn_sharedkv_reference` | Pro 每查询 16 个本地头，$D=512$。例如压缩库容量 32770，图给 $E=10368$，计算 $2EH(512+512)$，每次 gather 的 latent 同时供 K 与 V。FP4 block32 或 FP8 行 scale 从对应张量读取。 | 已有 MixQuantSparseFlashMLA 的 Q 精度和 KV 布局与此图有差别，整窗 N/A。 |
| `inference::kimi_gated_dsa_attention` | Kimi TP8 的查询含 latent 512 和 RoPE 64，先 gather top2048，再做 QK（576 维）、softmax、P×V（512 维）；$E=9\times2048$。图上 `gate_inside=false`，输出是 BF16[9,12,512]。 | FP4 吸收式 MLA 设备实现外推。 |
| `npu::npu_fused_infer_attention_score` | Kimi draft Q[9,12,64]，2 个 KV 头共享给 12 个查询头，历史和临时 K/V 合计 L131090。读跨段数据流，每块 128 键，用 split-KV 并行，再按 LSE 合并相同查询的局部结果。输出 BF16 和 FP32 LSE。 | 下面给出双端原生 GQA 数值与检查。图中历史为交错 KV，假设能直接读取两段。若实际先拼接历史，则另有写读和调用成本。 |
| `_C_ascend::compressor` | Pro $M=9,K=7168,N=512$，同一输入有 KV 和 gate 两个 A8W4 投影。AIV 内联扩 W、量化 X，Cube 分 16×64×128 做 $4MNK$ ops，片内交接后写 FP32 状态。压缩率 128 时本步输出可为 [0,512]，这时投影和状态写仍执行。完成组才计算权重归一化、池化、RMSNorm 和 RoPE。 | 低精度张量和附加 MX scale 不符合原生 compressor 的接口；融合外推。 |
| `_C_ascend::compressor_metadata` | 从两张 [1048576,64] BF16 的 sin/cos 表各 gather **2** 行，写相应行和 slot。若返回零行只保留工作提交/元数据。 | 逻辑未分页 metadata 投影，整窗 N/A。 |
| `inference::compressor_source_positions` | 输入 9 个 i64 位置，压缩率 2、history=131072 时输出 4 个位置。按已完成压缩组计算源 token，读写各 $4\times8$ B。 | AIV 元数据实现假设。 |
| `inference::c2_ring_compressor` | 9 个位置×512 通道、32 行环形状态、输出 4 行。输入已有 FP32 KV 和 gate 结果。按 128 通道分片，在同一请求内按序写环，每两行在线归一化后池化；不重复前置投影。 | FP32 ring 布局与外部投影的组合外推。 |
| `vllm_ascend::causal_conv1d_update_reference` | Kimi BF16 X[9,4608]、W[4,4608]、状态 [1,11,4608]。各核读取独立通道 stripe，按时序做 4 点卷积、SiLU、写 Y 及状态。下节有逐项计算。 | 双端 CANN 9.2.1 `aclnnCausalConv1dUpdate`。 |
| `vllm_ascend::recurrent_kda_reference` | Kimi TP8 的状态 [1,12,128,128] FP32，Q/K/V 为 MXFP4。将 V 维切成 32 行 stripe，48 个 AIV 任务。每个 stripe 常驻 UB，沿 Q 的 9 步顺序执行 Q/K 归一化、衰减、状态读出、delta 外积更新、Q×新状态；全状态读写各一次。 | 组合的 FP4 校验状态链外推；历史 chunk BF16 实现的整窗不适用。 |

### 每页计算和公式来源

索引的 key 页有 $S=128$ 个位置，每核处理至多 $r=4$ 个 query，投影 Cube 的物理 M 用 $m=2H\lceil r/2\rceil$，尾头组对齐采用 950 组件的双行设定。H64 的四查询块拆成两条 M128 MMAD；两芯片实际的三个工作核分别有 32、32、16 条 128×128×128 指令。每页发起间隔取
$\max(S b_K f/B_{MTE2},2mSD/r_c,\lceil r/p\rceil S(3H-1)/r_v)$。
$b_K$ 是一行 key 的字节数；后三项分别来自每头点积、ReLU、乘权重及跨头累加。对第一个分段做 key histogram，此后将上一段的 top-k 一并归并，每 256 keys 一个 histogram 块。540 cycle/次、48.75 cycle/块、末次 drain 与段间交接取自普通索引的指令时间戳，按目标 $r_v$ 换算。第 20 层独有的块 ranking 对 $\lceil L/8\rceil$ 个额外候选做 histogram；现有 CA 没有这段融合。 [索引参数](../../src/operator_model/data/parameters/indexer.json) · [候选源与后续重索引](candidate-filtered-indexer.md)

注意力按查询块、KV 头、查询头组和 KV 分片分核。每页的 QK 和 PV 时间（cycle）分别为 $2mSD_K/r_c$、$2mSD_V/r_c$，其中 $m$ 是有效查询与头数的乘积向 16 对齐。对 $u$ 个有效查询头，在线 softmax 和跨页重缩放约需 $[uS(6+3\,I_{low})+3uD_V]/(pr_v)$。六个等效操作覆盖极值、偏移、指数、归约、缩放和累计，低精度额外三项估计 cast/概率量化。这是设计级预算，exp/div 的真实吞吐可能不同。首个页按 fetch→QK→softmax→PV 顺序，之后的页用 $\max(C_{fetch},C_{QK}+C_{PV},C_{softmax})$ 发起。$n$ 个 KV 分片的额外合并预算 $2(n-1)u(D_V+2)/(pr_v)$，来自向量累计和两个标量统计量。

状态类算子按独立通道或状态行切分。卷积每输出标量 $2K+4$ 次等效运算，前 $2K$ 是乘加，4 是 SiLU 等效预算。每核搬运包含输入、权重、过去 $K-1$ 行、Y 和本步产生的状态行。KDA 对 $V\times K$ 的每个 FP32 状态元素用约 7 个操作/步（衰减、两次读出点积、一次 delta 外积更新），另收 Q/K 的预处理及行归约。每个 stripe 宽度取 $\min(V,32,\lfloor(UB-8192)/(3\times4K)\rfloor)$，8192 B 预留给标量和队列，三个 FP32 缓冲供状态/暂存/交接。不同核数、UB 或频率重新排程，上限 32 是本设计选项。

### 双端 CA 的计价与边界

校准取两芯片同一算例中“实测 cycles 减模型流水 cycles”的均值。该剩余量将首发、未拆出的控制与等待合并为一个架构假设，不能从单个开发点辨识各部分。设备时间使用首条核指令到最后退休的窗口；离线墙钟另存。 [原生探针](../../calibration/system_attention_ca/probe.cpp) · [开发点、ELF 与核名单](../../calibration/system_attention_ca/development.json) · [采样前冻结值](../../calibration/system_attention_ca/frozen.json)

- 卷积开发点 $B=1,Q=9,D=4608,K=4$。每核 512 通道，9 个 stripe。950DT 单 stripe 逻辑 MTE 预算 472.62 cycles，SIMD 216.27 cycles，加两组宽 4 的控制预算 $2\times4\times32=256$ cycles，得到 728.62 cycles。DT 与 PR 实测 17406 和 16888 cycles，共同剩余 16418 cycles。默认预测 **10.392 / 10.392 µs**，CA 为 **10.549 / 10.235 µs**。该点用每通道固定输入，核验所有 Y 与更新后的整块 state。该输入对时序移位不敏感。
- FIA 开发点 $B=1,Q=9,H/J=12/2,D=64,L=512$。950DT 的 QK/PV 各 128.16 cycles，softmax 101.38、fetch 416、merge 41.82，共 815.51；PR 为 819.40。整窗分别 21477 和 20403 cycles，得到共同剩余 20123 cycles。预测 **12.690 / 12.692 µs**，CA 为 **13.016 / 12.365 µs**。TND contiguous K/V 的 Y 按头和通道变化、LSE 与 $\ln L$ 逐值核对。LSE 是常量，本探针不能检验不同 LSE 排列；图上长历史及临时 KV 的拼接也没有在此 CA 测量。

### 冻结后比较与误差

所有数值为设备 µs；误差为 $(T_{model}/T_{CA}-1)\times100\%$。每个留出只测一个新 shape 的两个 SKU，这里不能估计整条路由的误差分布。 [卷积、FIA 独立样本](../../calibration/system_attention_ca/independent.json) · [H64 组件与各核 MMAD](../../calibration/system_attention_ca/indexer-h64.json)

| 角色和新输入 | DT 预测 / CA | DT 误差 | PR 预测 / CA | PR 误差 | 离线 CA 墙钟 DT / PR |
|---|---:|---:|---:|---:|---:|
| 卷积独立，B1 Q9 D9216 K4 | 10.392 / 7.718 | +34.65% | 10.424 / 8.514 | +22.44% | 163 / 377 秒 |
| 连续 BF16 GQA 组件独立，B1 Q9 H12 J2 D64 L1024 | 12.724 / 14.255 | −10.74% | 12.727 / 13.729 | −7.30% | 282 / 116 秒 |
| 普通 BF16 索引组件机制复核，B1 Q9 L2048 H64 D128 top1024 | 14.978 / 17.410 | −13.97% | 15.009 / 17.058 | −12.01% | 145 / 240 秒 |

卷积的通道数加倍，实测有退休窗口的 AIV 从 9 增至 54（DT）与 45（PR），冻结的 512 通道分片规则预报 18 个。头开销和通道 stripe 的策略变化使设备时间下降；不能据此确定每个 AIV 的实际通道数或把 54/45 直接用作新参数。FIA 在 L1024 时模型将 KV 分给更多核，实测增长更多；split-KV 的调度和局部归并仍是预测不确定项。H64 两端每条 M128 MMAD 分布符合两子块预算，PR 的间隔可见 514 cycles，每次提交的非计算部分、排序交接与末端 drain 解释总耗时仍有约两微秒差距。这些检查均保留长上下文和系统部署外推标记。

**Kimi 的 V2 脱离原生元数据前处理时的适用边界。** 在两个 SKU 上实际调用 `aclnnQuantLightningIndexerV2Metadata` 时，安装包均要求 1024 个 int32，CA 日志报 `unsupported task type: AICPU_D`，检查输出为全零。官方测试做法依赖已物化的 metadata。系统模型直接按 68 B/行读取、在线拆 MX scale 和值；如独立使用 V2 算力内核，需要准备分离的 K 及 scale。若每次先物化一次，覆盖有效银行的额外逻辑读加写为 $2\times68\times B L$ 字节；若缓存已维护为分离布局，则无此重排。前处理和自动分核端到端的 CA 栏为 N/A。[V2 类型、量化模式与 metadata 契约](https://gitcode.com/cann/ops-transformer/blob/14d022b1a4c3df263bf697534e3c25c7c992ff5d/attention/quant_lightning_indexer_v2/docs/aclnnQuantLightningIndexerV2.md)

可以单独验证 V2 算力组件。按 **安装包自带** 的 1024-word 表协议将一个 AIC 的工作域设成完整的单请求区间，关闭跨核 split-K，得到已知正确的静态分核。用分离的 Q、K、MX scales 和 FP32 weights，B1 Q1 H32 D128 L2176 的 2048 个选中 ID 在 DT/PR 均逐个核验通过。真实 MMAD_MX 为单核 17 次 [64,128,128]；两端首发分别 8568、7211 cycles，中位发起间隔 315、311 cycles，第 4→5 条之间单次间隔达到 4097、4201 cycles。末 MMAD 退休后到整窗结束还需 4614、3864 cycles。组件整窗 **14.211 / 12.909 µs**，离线各 81 秒。第一条、间隔、退休和尾段分别取自原始指令时间戳，可以用来校准同一静态分核和分离存储的替代实现。生产图的 Q9、长循环和 68 B 混排实现维持前述估算，直接叠加这一整窗会重复收费。[MXFP4 静态组件记录](../../calibration/system_attention_ca/mxindexer-static.json) · [自动 metadata 失败证据](../../calibration/system_attention_ca/metadata-eligibility.json)

其余资源假设的首发使用 4587 cycles（LI 首 MMAD；仅作为 CV 的代理），3750 / 2200 / 1200 cycles 分别是 KDA、ring、短元数据 kernel 的工程预算，没有整窗 CA 分别拟合。所有 970DT 数字均为外推。低精度设计共用目标芯片峰值与存储层级，对未来代际的软件实现、实际分核和稀疏地址局部性尚无观测。

## 三套芯片配置上的完整图节点示例

下面采用每个模型 TP8/EP8 中首个匹配的 rank0 实例，形状是首个输入与首个输出的物理存储维度，FP4 的最后一维可只有逻辑宽度的一半。该表列 **模型预测**微秒数，组件 CA 见上表；970DT 全部是外推。

| 图节点 | 捕获 | 输入 → 输出 | 950DT 预测 | 950PR 预测 | 970DT 外推 |
|---|---|---|---:|---:|---:|
| `compressor` | v4-pro target | 9×7168 → 0×512 | 5.032 | 5.684 | 4.610 |
| `sparse_attn_sharedkv_reference` | v4-pro target | 9×16×512 → 9×16×512 | 4.737 | 6.760 | 5.098 |
| `compressor_metadata` | v4-pro target | 1048576×64 → 2×1×1×64 | 0.770 | 0.770 | 0.706 |
| `quant_lightning_indexer_reference` | v4-pro target | 9×64×128 → 9×1024 | 186.129 | 186.741 | 190.990 |
| `c2_ring_compressor` | v41-flash target | 9×512 → 4×512 | 1.479 | 1.479 | 1.355 |
| `compressor_source_positions` | v41-flash target | 9 → 4 | 0.766 | 0.766 | 0.702 |
| `lightning_indexer` | v41-flash target | 9×32×128 → 9×512 | 207.880 | 209.091 | 231.101 |
| `candidate_blocks` | v41-flash target | 9×512 → 9×2048 | 0.893 | 0.893 | 0.819 |
| `causal_conv1d_update_reference` | kimi-k31 target | 9×4608 → 9×4608 | 10.392 | 10.392 | 9.526 |
| `recurrent_kda_reference` | kimi-k31 target | 1×9×12×64 → 1×9×12×128 | 3.338 | 3.456 | 3.059 |
| `kimi_dsa_indexer_score` | kimi-k31 target | 9×32×64 → 9×2048 | 184.559 | 187.347 | 296.949 |
| `kimi_gated_dsa_attention` | kimi-k31 target | 9×12×256 → 9×12×512 | 4.309 | 6.964 | 4.824 |
| `npu_fused_infer_attention_score` | kimi-k31 draft | 9×12×64 → 9×12×64 | 28.982 | 60.157 | 16.507 |
