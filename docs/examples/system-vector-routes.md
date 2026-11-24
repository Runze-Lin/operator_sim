# 系统图中的向量、路由与状态更新

27 个任务里有 42 种调用名落入本页。输入来自单 rank 的实际捕获，包含 dtype、物理 shape、stride、归约轴、写入位置和量化格式。可以在形状清单里按调用名筛选每种输入、节点出处、重复次数、两款 950 的预测以及 970DT 外推。例表中的 950 数字也是模型耗时，独立的 CA 观测列在后文。在线调用只读取芯片配置。[模型源码](../../src/operator_model/system_routes/vector.py) · [逐合同与形状 CSV.gz](../../calibration/system_vector_ca/shapes.csv.gz)

## 流水与芯片资源

首先按语义挑出实际读写的行，计算读取字节 $R$、写出字节 $W$、运算量 $O$ 和可独立处理的元素数 $N$。例如 index、embedding 的输入表有数亿行，设备只读取传入 ID 指向的行。CacheStore、Kimi StateUpdate 的输出是整个 cache 的别名，设备只更新本轮新增位置。FP4 的物理字节槽存放两个数，加上 E8M0 scale 的字节数才是编码后的数据量。

设 $g$ 为不可拆或计划共同发起的一组元素数。按 head 旋转可采用两个 head 的活跃切片，逐行散写取行宽。9×7168 BF16 加法在 DT 报告 36 个负载核，以 $64512/36=1792$ 个元素为一组，是用于稠密逐点算子的分核近似。逐行广播乘使用 2048 个元素分组。拉取少量 embedding ID 按至多八行一组、上限 1024 元素。字节填充按 32 B 一组。分组归约默认每组输出 256 元素。

$$
A=\min\!\left(C_V,\left\lceil N/g\right\rceil\right),\qquad
n_c=g\left\lceil\left\lceil N/g\right\rceil/A\right\rceil .
$$

$C_V$ 是芯片的 AIV 核数，$A$ 是预测有负载的核数，$n_c$ 是最慢核的元素预算。组宽取 $\min(g,N)$。令 $R_d=R+R_{repeat}+R_{line}$。后两项分别是 UB 放不下导致的重复加载和不规则地址的一次 32 B 事务补足。每个活跃元素的平均 live 数据预算为 $(R_d+W)/N+4$ 字节，最后 4 字节留给 FP32 临时值。用 UB 的一半装一页，向下对齐 16 个元素，另一半供下一页预取。取 UB 页宽和 $n_c$ 中较小者为 $P$。第一个满页独占的三个阶段是

$$
t_r=P\frac{R_d}{N b_r},\qquad t_v=P\frac{O}{N u},\qquad t_w=P\frac{W}{N b_w}.
$$

三个阶段以设备周期计。$b_r,b_w$ 是单个 AIV 读、写的 B/cycle，优先读取芯片 MTE 带宽再除以 GHz。缺失时 $b_r=48.36$ 来源于 AttnRes 的 950 读服务标定，$b_w=32$ 是按一条 32 B 传输/周期的假设；小窗口无法同时确定独立读写速率和启动成本。设目标芯片总 Vector 算力为 $F$ TOP/s、主频为 $f$ GHz，则 $u=1000F/(2C_Vf)$。除以 2 将峰值 FMA 折算为普通逐元素服务的每核发起能力。整数逻辑、转换和非线性使用所选 16/32 位算术的服务等效量；该近似在整数 64 位、多条特殊函数指令上误差较大。

$$
T_{core}=t_r+t_v+t_w+\frac{n_c-P}{P}\max(t_r,t_v,t_w),
\qquad
T_{\mu s}=\frac{L+\max(T_{core},(R_d+W)f/B_{HBM})}{1000f}.
$$

$L$ 是启动、指令准备、尚未拆开的首次响应和收尾的合计周期。$B_{HBM}$ 为芯片 GB/s。后续页采用双缓冲的最大资源间隔，尾页按有效长度近似，端口总流量和芯片带宽限制并行生效。共用缓存中的输入、在 L2 退休的输出可能使这里的冷流量假设偏保守。在线复杂度与 tensor 秩、返回的阶段数有关，大 shape 使用波数公式直接求解。

**如何得到控制周期。** Add、行广播 Mul、FP8 CacheLoad、无裁剪 SwiGLU、局部 RoPE 的 DT 短形状提供启动参考。扣除上式的负载周期后分别近似取 4400、6100、7380、5560、7260 cycles。分离式 Slice 的 4500 cycles 来自已有 9×128 非连续量化调用。使用相似前端的合成算子借用这些控制预算。对 15 个实跑的 ATen 原语，则在各 SKU 的 9 行试验上取 $L=T_{CA}\,1000f-\max(T_{core},(R_d+W)f/B_{HBM})$。输入 shape 的算术和字节都由实现计算，启动剩余量仅依赖原语和已校准的 SKU。其他芯片继承 DT 控制周期再按目标频率、核数、UB、HBM 和 Vector 峰值计算。9 行误差近零是解方程的结果；mask/math 的 144 行在拟合前已采集但不参与拟合，属于预留尺寸回放。相同 API 的不同精度、布局、广播方式或调用队列前序仍会改变固定部分。[标定和复算脚本](../../calibration/system_vector_ca/primitives.py) · [逐窗比较](../../calibration/system_vector_ca/primitives.csv)

## 分支算法

| 类型或调用 | 具体的工作分配和计数 |
|---|---|
| ATen 点运算、cast、clone、cat、pad | 加、减、乘、负、比较、掩码、转换各计 1 次/输出；本组 pow 是平方；sigmoid 拆负、exp、加、倒数。数据重排只计实际读写，stride=0 的展开轴读取独立源元素一次，完整写出目标。独立 FP32 sigmoid 与 BF16→FP32 cast 优先使用已校准专用模型。 |
| sum、mean、gather/embedding | 将归约放在 UB 寄存器中，对 $n_i$ 个输入、$n_o$ 个输出执行 $n_i-n_o$ 次加，求平均再加 $n_o$ 次缩放。选行读取 IDs 与所选负载，行尾补齐 32 B 事务。 |
| RMS 和分组门控 RMS | 普通 BF16 加权 RMS 用主模型。融合版本保持组行在一个 AIV 上，以平方、求和、rsqrt、gamma、sigmoid 和输入相乘定价，取 $9N+3\lceil N/D_g\rceil$ 个等效操作，$D_g$ 是组宽。 |
| 局部与整头 RoPE | 两个输入 lane 交叉，乘 cos、sin，再加，计 $3N$；$N$ 只含实际旋转的 head 子区间。两张正余弦表按独立行计。原位结果写回相同 storage。 |
| SwiGLU、SiTU、MLA gate | SwiGLU 拆成两半，sigmoid 和两次乘合计约 6 次/输出，gate 上限与 up 双边裁剪再加 3 次比较。SiTU 将 gate 计算为 $\beta\tanh(g/\beta)\sigma(g)$，可选 $u'=\beta_L\tanh(u/\beta_L)$，再乘 $u'$，两半均参与，按约 10 次等效操作/输出。MLA gate 取 $v\sigma(g)$，按 5 次/输出。专用 transcendental 的吞吐未标定。 |
| cache 读/写、量化、Kimi 状态 | BF16→MXFP4 沿最后一维每 32 元素归约、缩放和打包；不连续输入显式串接 Slice 和量化的两次派发。per-token FP8 为各行一个 FP32 scale。K/V cache 读物化有效行。写 $T$ 行、逻辑行宽 $D$：FP4 的实际写字节是 $T(\lceil D/2\rceil+2\lceil D/64\rceil)$，FP8 是 $T(D+4)$。Kimi indexer 状态先逐行 LayerNorm，再编码，写 $T\,rowBytes$；query 量化还写 head weight 和每 32 元素的 scale。 |
| MoE 路由 | sigmoid 加偏置、单专家组、无完整 score 输出时调用已验证的原生路由模型。其他非 hash 组按专家行 score 和 $E\lceil\log_2(K+1)\rceil$ 比较预算。Hash 先读一整行 $E$ 个 logit 做 sqrtsoftplus，在 UB 中保留分数；从 tid2eid 取 $K$ 个 ID，再 gather 并归一化，写 $8TK$ B。折为 $7TE+2TK$ 次向量运算，查表读另计输入 ID 与所选表行。孤立的 K 列不代表只计算 K 次得分。若同一 AIV 轮转多个 token，每个后续 token 还串行经过 ID 与映射行两次读响应，默认采用该芯片 quant_read_latency_cycles。raw 名相同而仅传 bias 的草稿分支走全行排名，共用已经验证的 sigmoid 排名成本，sqrtsoftplus 排名整窗缺测。 |
| mHC、Attention residual | mHC Post 用已测 vLLM 适配。Pre V2 先将 $N$ 条 stream 展平，对 $K=ND$、$C=N^2+2N$ 的投影计 $2TKC$ Cube ops；复用 4096 宽的 split-K、转换、MMAD 发起周期和逐行 Sinkhorn 标定，在更宽 K 上按分片、至多 128 行 M 页和 20 轮 Sinkhorn 重排。HF32 资源沿用 BF16 相对吞吐。AttnRes 按有效 bank 行和 UB 大小算 resident/reload，两次访存分开；输出 norm 已是图中独立节点。 |
| Engram hash、lookup、反量化、gate | hash 每个 token 探一次 vocab 重映射，然后按 n-gram 长和 head 数做混合与模运算；每槽约 $3g+2$ 操作，$g$ 是 n-gram 长。lookup 读取本 shard 的 INT8 子段与每 32 值一个 FP32 scale。反量化按转换与乘法每值 2 次。gate 对 hidden 与 key 归一化、加权点积、符号平方根和 sigmoid 后做残差加权，估计 $11N+4\lceil N/D\rceil$ 次。 |

这些设计使用了 CANNBot RegBase 向量寄存器装载/算术/单次写出卡片 OL-245，和本机 vLLM Ascend 的 `ops/fused_moe/moe_mlp.py`、`models/deepseek_v41/engram/common.py`、`csrc/moe/moe_gating_top_k_hash/op_kernel/arch35`。开源 kernel 提供算法及循环边界，确认本次安装包实际派发的依据是每个 CA 回执中的 ELF SHA-256 和完成核 roster。

## 42 种调用的实际输入例子

以下 42 个名字在 27 个 native 图对中共有 70,624 次调用。表取实际遇到的一个形状，列出第一个输入及第一个输出的物理 shape；FP4 每个槽有两个值。每个例子的其他输入、步长、属性见对应 CSV。DT、PR、970DT 数字依次为模型设备微秒，其中 970DT 全部为外推。

| 调用名 | 示例输入 → 输出 | DT / PR / 970DT µs | 计时依据 |
|---|---|---:|---|
| `_C_ascend::inplace_partial_rotary_mul` | BF16[9,1,16,512] → 原位 | 4.419 / 4.428 / 4.042 | partial RoPE，双端 9×1×8×512 整窗 |
| `_C_ascend::moe_gating_top_k` | F32[2,896] → F32[2,16] | 5.340 / 5.391 / 4.893 | sigmoid 单组原生回放；其他打分按投影 |
| `_C_ascend::moe_gating_top_k_hash` | F32[2,384] → F32[2,6] | 4.330 / 4.353 / 3.968 | 本行是仅传 bias 的草稿排名投影；带 tid2eid 的目标路径用 V2 双端验证 9×384、K=6 |
| `_C_ascend::npu_hc_post` | BF16[1,9,7168] → BF16[1,9,4,7168] | 5.849 / 5.915 / 5.359 | 匹配 vLLM 专用 BF16 Post |
| `_C_ascend::npu_hc_pre_v2` | BF16[9,4,7168] → BF16[9,7168] | 10.390 / 10.347 / 9.224 | 原生 PreSinkhorn 等价输出 ABI，不同扩展派发未识别 |
| `aten._to_copy.default` | BF16[9,7168] → F32[9,7168] | 2.416 / 2.417 / 2.214 | 连续 BF16→FP32 有专用模型；其他转换沿用 Cast 前端 |
| `aten.add.Tensor` | BF16[9,7168] → BF16[9,7168] | 2.829 / 2.943 / 2.593 | BF16 同形 [9/144,7168] 的 CA |
| `aten.bitwise_or.Tensor` | bool[9] → bool[9] | 0.983 / 0.964 / 0.901 | 对应 ACLNN 原语，9 行开启/收尾校准，异形或异 dtype 仍属投影 |
| `aten.cat.default` | BF16[9,1,448] → BF16[9,1,512] | 2.069 / 2.274 / 1.897 | 对应 ACLNN 原语，9 行开启/收尾校准，异形或异 dtype 仍属投影 |
| `aten.clone.default` | BF16[9,4,7168] → BF16[9,4,7168] | 2.904 / 3.127 / 2.589 | 物化含零步长视图；拷贝专用 CA 无合格整窗 |
| `aten.constant_pad_nd.default` | BF16[9,7168] → BF16[16,7168] | 3.052 / 3.049 / 2.758 | 对应 ACLNN 原语，9 行开启/收尾校准，异形或异 dtype 仍属投影 |
| `aten.embedding.default` | BF16[16160,7168] → BF16[9,7168] | 2.681 / 2.433 / 2.458 | 对应 ACLNN 原语，9 行开启/收尾校准，异形或异 dtype 仍属投影 |
| `aten.ge.Scalar` | I64[9] → bool[9] | 1.399 / 1.288 / 1.282 | 对应 ACLNN 原语，9 行开启/收尾校准，异形或异 dtype 仍属投影 |
| `aten.index.Tensor` | BF16[1048576,64] → BF16[9,64] | 1.973 / 1.773 / 1.809 | 沿第 0 维选行，原生 GatherV2 组件 |
| `aten.lt.Scalar` | I64[9] → bool[9] | 2.921 / 2.639 / 2.678 | 对应 ACLNN 原语，9 行开启/收尾校准，异形或异 dtype 仍属投影 |
| `aten.masked_fill.Scalar` | I64[9] → I64[9] | 1.337 / 1.221 / 1.226 | INT64 原位 MaskedFill 的组件校准；这里是非原位 |
| `aten.masked_fill_.Scalar` | BF16[9,7168] → BF16[9,7168] | 1.452 / 1.403 / 1.331 | 原位 MaskedFill 只实测 INT64；BF16 广播投影 |
| `aten.mean.dim` | F32[9,28672] → F32[9,1] | 15.136 / 15.110 / 13.874 | 双端 Mean 采用 FP32×64 末轴；移植到 BF16 与跨行归约 |
| `aten.mul.Tensor` | F32[9,4] → F32[9,4] | 3.702 / 3.702 / 3.394 | FP32 行广播 [9/144,4] 的 CA |
| `aten.neg.default` | BF16[9,64] → BF16[9,64] | 2.082 / 1.942 / 1.908 | 对应 ACLNN 原语，9 行开启/收尾校准，异形或异 dtype 仍属投影 |
| `aten.pow.Tensor_Scalar` | F32[9,28672] → F32[9,28672] | 2.180 / 2.765 / 1.784 | 对应 ACLNN 原语，9 行开启/收尾校准，异形或异 dtype 仍属投影 |
| `aten.rsqrt.default` | F32[9,1] → F32[9,1] | 1.161 / 1.080 / 1.064 | 对应 ACLNN 原语，9 行开启/收尾校准，异形或异 dtype 仍属投影 |
| `aten.sigmoid.default` | F32[9,4] → F32[9,4] | 2.691 / 2.693 / 2.467 | 连续 FP32 的原生校准；其他布局投影 |
| `aten.sub.Tensor` | I64[9] → I64[9] | 1.159 / 1.086 / 1.063 | 对应 ACLNN 原语，9 行开启/收尾校准，异形或异 dtype 仍属投影 |
| `aten.sum.dim_IntList` | F32[9,4,7168] → F32[9,7168] | 3.158 / 3.477 / 2.804 | 双端 ReduceSum 采用 FP32×64 末轴；归约轴变化属外推 |
| `inference::attention_residual` | BF16[9,7168] → BF16[9,7168] | 3.389 / 3.389 / 3.107 | AttnRes 源码的 resident/reload 分支；当前输出边界无整窗 |
| `inference::cache_load` | FP8[1,128,512] → BF16[1,128,512] | 4.534 / 4.614 / 4.156 | FP4 连续已校准；FP8 128/1024 行有整窗；其他布局投影 |
| `inference::cache_store` | BF16[9,512] → FP8[1,128,512] | 4.505 / 4.505 / 4.130 | 量化及增量散写融合，原生整窗 N/A |
| `inference::dynamic_mx_quant_fp8_reference` | BF16[9,7168] → FP8[9,7168] | 4.051 / 4.060 / 3.713 | per-token scale，匹配 DynamicQuantV2 组件 |
| `inference::engram_dequantize` | U8[9,864] → BF16[9,768] | 4.481 / 4.488 / 4.104 | 带 FP32 scale 的 INT8 行解包，整窗 N/A |
| `inference::engram_gate` | BF16[9,4,5120] → BF16[9,4,5120] | 3.445 / 3.581 / 3.158 | unrotated Engram gate 源码，整窗 N/A |
| `inference::engram_hash` | I64[1,9] → I64[9,2,24] | 2.673 / 2.673 / 2.450 | hash 及 vocab remap，整窗 N/A |
| `inference::engram_lookup` | I64[9,24] → U8[9,864] | 2.696 / 2.696 / 2.472 | 按 ID 读取与每组 scale，整窗 N/A |
| `inference::grouped_gated_rms_norm_reference` | BF16[9,12,128] → BF16[9,12,128] | 2.700 / 2.726 / 2.460 | 带 gate 的按组 RMS，整窗 N/A |
| `inference::kimi_dsa_indexer_query_quant` | BF16[9,32,128] → U8[9,32,64] | 4.508 / 4.541 / 4.119 | 带 FP32 head weight 的打包量化，整窗 N/A |
| `inference::mla_output_gate` | BF16[9,1536] → BF16[9,1536] | 2.781 / 2.782 / 2.549 | Sigmoid 组件，单派发乘入 value 的整窗 N/A |
| `inference::situ` | BF16[9,8448] → BF16[9,4224] | 3.647 / 3.650 / 3.343 | SiTU 参考源码；仅 SwiGLU 共用前端，SiTU 整窗 N/A |
| `inference::state_update` | BF16[9,128] → U8[1,132114,68] | 4.485 / 4.486 / 4.112 | LayerNorm 加增量写，整窗 N/A |
| `npu::npu_dynamic_mx_quant` | BF16[9,7168] → FP4x2[9,3584] | 4.218 / 3.869 / 3.866 | 连续 MXFP4 专用模型；非连续 Slice+Quant 两次派发已实跑 |
| `npu::npu_rms_norm` | BF16[9,7168] → BF16[9,7168] | 3.288 / 3.294 / 3.014 | 普通 BF16 RMSNorm 有完整原生双端校准；布局变型投影 |
| `npu::npu_rotary_mul` | BF16[1,9,1,64] → BF16[1,9,1,64] | 4.415 / 4.415 / 4.047 | 参考同族 partial RoPE；该非原位整窗 N/A |
| `npu::npu_swiglu` | BF16[9,768] → BF16[9,384] | 3.821 / 3.825 / 3.503 | SwiGlu 无裁剪有双端 CA；clamp 融合整窗 N/A |

## CA 对照与误差

**真实例子，BF16 Add。** DT 的 $9\times7168$ 输入有 $N=64{,}512$ 个输出。读取两个 BF16 张量 $R=258{,}048$ B，写 $W=129{,}024$ B，算术 $O=64{,}512$。$g=1792$ 得 36 AIV，最慢核一次处理 1792 个元素。读、算、写分别约为 $1792\times4/48.36=148.22$、$1792/(54000/(64\times1.65\times2))=7.01$、$1792\times2/32=112$ cycles；未分第二个 UB 页。芯片带宽约束为 $(R+W)\times1.65/4000=159.67$ cycles，取 267.23 cycles 的较大值，再加 4400 cycles，预测 **2.829 µs**；CA 为 **2.841 µs**，36 个核。相同几何在 PR 的 CA 实际只启 28 核，模型仍选 36 核。长到 144 行时 PR 预测 7.090 对 CA 4.825 µs，偏高 47.0%。预测将两个输入和全部输出都按冷 HBM 流量收费；这个下界已压过核内成本。现有回执没有 HBM 实读写计数，不能把这 47.0% 全部分配给缓存驻留。

**实际 hash。** $T=2,E=384,K=6$ 时，读取全部 $2\times384\times4=3072$ B 的 logits，另读 8 B token id、48 B 映射。UB 评分后依赖 ID 读取一行，输出 48 B 权重和 48 B 索引。从关键 AIV 的 MTE2 时间戳可见多次搬运，随后 MTE3 有权重、ID 两次写。aclnnMoeGatingTopKV2 的 950 内置 MoeGatingTopK ELF 和 vLLM arch35 HashCompute 的“整行评分、再从 UB gather”一致；图本身没有记录实际 ELF。原先只评分 $K$ 个、沿用 Add 前端，预测 DT/PR 2.671/2.671 µs，对 CA 5.265/4.895 µs 低估 49.3%/45.4%。新模型用这两个开发窗口扣减芯片服务预算，得到 DT/PR 前端剩余 8632.87/8021.30 cycles。增至**未用于拟合且冻结后采集的 9 行**，每行一个 AIV，模型 5.265/4.895 µs，CA 5.565/5.200 µs，低估 5.4%/5.9%。前端启动的核间差异还未建出。后续多波的读取响应沿用量化读延迟，在本次 2/9 行窗口上还不可验证。同名 bias 路径没有 tid2eid，模型独立标为 sqrtsoftplus 排名投影。

**ATen 基元。** 每个 SKU、每个原语的 9 行开发窗口各一条，30 条共用于求控制周期。下表展示同一批实现的 144 行结果，数字为“预测 / CA”设备微秒。mask/math 的 20 个点为预先采到、未参与拟合的回放；memory 的 10 个点为冻结后复测。后者每 SKU 五点 WAPE 为 DT **17.6%**、PR **18.2%**；前者每 SKU 十点分别为 **16.2%**、**18.2%**。单一方向每原语只有一个留出 shape，这些数值不能代表各算子全范围误差。[全部 60 个预测、实测、误差、分核](../../calibration/system_vector_ca/primitives.csv) · [CANN 接口与 ELF 证据](../../calibration/system_primitives_ca/measured.json)

| API | DT µs | DT 误差 | PR µs | PR 误差 | DT / PR 的 CA 核数 |
|---|---:|---:|---:|---:|---:|
| aclnnBitwiseOrTensor | 0.990 / 1.005 | -1.5% | 0.971 / 0.944 | +2.8% | 1 / 1 |
| aclnnCast | 1.277 / 1.393 | -8.3% | 1.269 / 1.525 | -16.8% | 5 / 5 |
| aclnnCat | 2.069 / 2.810 | -26.4% | 2.274 / 2.941 | -22.7% | 2 / 2 |
| aclnnConstantPadNd | 2.969 / 2.444 | +21.5% | 2.820 / 2.347 | +20.2% | 4 / 4 |
| aclnnEmbedding | 2.650 / 3.068 | -13.6% | 2.282 / 2.647 | -13.8% | 18 / 18 |
| aclnnGatherV2 | 1.973 / 2.282 | -13.5% | 1.773 / 2.124 | -16.5% | 18 / 18 |
| aclnnGeScalar | 1.416 / 1.407 | +0.6% | 1.305 / 1.342 | -2.8% | 1 / 1 |
| aclnnInplaceMaskedFillScalar | 1.373 / 1.385 | -0.9% | 1.257 / 1.344 | -6.5% | 1 / 1 |
| aclnnLtScalar | 2.938 / 3.219 | -8.7% | 2.656 / 2.656 | -0.0% | 1 / 1 |
| aclnnMean | 2.581 / 2.202 | +17.2% | 2.552 / 2.165 | +17.9% | 3 / 3 |
| aclnnNeg | 2.161 / 2.387 | -9.5% | 2.022 / 2.100 | -3.7% | 5 / 5 |
| aclnnPowTensorScalar | 1.712 / 2.079 | -17.7% | 1.524 / 2.033 | -25.0% | 9 / 9 |
| aclnnReduceSum | 3.352 / 3.172 | +5.7% | 3.072 / 2.864 | +7.3% | 3 / 3 |
| aclnnRsqrt | 1.394 / 1.621 | -14.0% | 1.313 / 1.666 | -21.2% | 9 / 9 |
| aclnnSubs | 1.194 / 2.968 | -59.8% | 1.121 / 3.117 | -64.0% | 2 / 2 |

INT64 aclnnSubs 的 9→144 行从 1 核变为 2 核，模型仍是 1 核，双端低估 59.8%/64.0%。校准的短单核前端没有覆盖这个分支；核数变化已在两端窗口证实，新增控制耗时的确切指令边界未辨识。aclnnCat 在 144 行实际只启 2 核，而模型给 11 核，DT/PR 分别低估 26.4%/22.7%；Pad 则从 9 行的全核派发转成 4 核，模型保留全核。Pow/Rsqrt 的特殊函数、跨行归约与小数据量控制开销也没有等价到实际 VF 周期。aclnnCast 表中是同一基元控制模型的复算；系统的连续 BF16→FP32 合同优先走已有独立 Cast 模型，其正式误差见 Cast 专页。[Cast 校准](cast_bf16_fp32.md)

**采集范围。** 原生 probe 对实际 ACLNN 调用做数值 golden，再逐核核验唯一 END 与报告 roster，匹配 dump 与本机 OPP 对象的 SHA-256 后才报 CA。FP8 缓存读、无裁剪 SwiGLU、局部 RoPE、hash、宽 PreSinkhorn 回执保留核数、整窗及 trace 摘要。失败的调度和纯 DMA Copy 不进入基元的 60 窗统计。普通 MoE sigmoid [2,896,16] 直接复用原测试，DT 为 5.340 / 5.472 µs，PR 为 5.391 / 5.446 µs。其他未取得完整 CA 的 fused 方案在例表标明，有些只有一种组成算子的校准。[逐窗原始摘要与预测](../../calibration/system_vector_ca/windows.json) · [原生 router 逐形状数据](../../calibration/reference/windows.jsonl)

**宽 mHC Pre。** [9,4,7168] 的关键 K 是 28672，投影列为 24，Cube 工作为 $2\times9\times28672\times24=12{,}386{,}304$ ops。冻结后使用 CANN 9.2.1 **aclnnMhcPreSinkhorn** 执行 20 轮，phi/bias 为零、输入 BF16 随行和通道变化，逐元素核验三组有效输出共 64,692 个值。DT 模型 **10.390 µs**、CA **10.771 µs**（−3.53%），PR 模型 **10.347 µs**、CA **10.678 µs**（−3.10%）。双端均实用 28 AIC 和 56 AIV。该库的 h_res 写成 [T,16]，可作零代价视图交付图上的 [T,4,4]；这份证据支持相同数值合同的公开库实现。DT 第一次 490 秒主机超时没有进入结果，单独重试后完整出窗。此次发布的向量及 mHC 专用回执共 22 窗，含复用 6 窗、新增 16 个成功窗口；新增成功窗口的串行等效 CA 墙钟合计 1,622 秒。原语另外采集的 60 窗单列。在线查询不读取任何 trace。[复现宽 Pre 原生 fixture](../../calibration/system_vector_ca/mhc.cpp) · [Hash 原生 fixture](../../calibration/system_vector_ca/hash.cpp) · [采集与复核脚本](../../calibration/system_vector_ca/capture.py)

重算误差时，可对已解包原始 950 窗口运行 `python -m calibration.system_vector_ca.receipt DIR... --opp KERNEL_ROOT/ascend950 --out receipts.json`，再运行 `python -m calibration.system_vector_ca.report receipts.json --output predictions.json`；15 个原语对照可用 `python -m calibration.system_vector_ca.primitives calibration/system_primitives_ca/measured.json --out primitives.csv`。CA 输出均采用设备微秒，文件中的 ca_wall_s 才是仿真器墙钟秒。
