# 系统模型目录中的 MLA、QSA 和门控状态注意力

下列算例使用系统入口的 rank-local 张量和语义字段、`Ascend950DT_9582` 芯片模板、`vector_model="auto"`，返回调度耗时（µs）、按精度区分的 Cube/Vector 工作、逻辑读写字节、任务与核数。新设计的 VF 阶段默认按指令类别计时。已有 BF16 GQA 的完整 CA 窗口对应特定的 Q/K/V 布局及输出，FA、KDA 与 LightningIndexer 的同族证据只用于提出流水假设。下面新增合同的整窗 CA 误差均为 N/A。

实现边界来自系统源码的 `ops/attention.py`、`layers/linear_attention.py`、DeepSeek V3.2 的 `attention.py`、GLM-5.3 的 `indexer.py` 和 Qwen3.8 的 `attention.py`。CANNBot 950 知识卡 `ops/triton/optimizations/patterns/attention_online_softmax_fusion` 描述了可见键、FP32 在线归一化及 PV 递推；页长、启动周期与跨核划分仍为本模型的假设。[在线 softmax 机制来源](https://gitcode.com/AgenticCompiler/BiShengKernelAgent/blob/21053d2fd6e8d7bb885dec66d887dd9e75014c5f/skills/triton/triton-npu-optimize-knowledge/references/patterns/attention-online-softmax-fusion.md)

## DeepSeek 与 GLM 的 MLA

从查询内容部分和 RoPE 部分得到点积宽度 Dk，输出最后一维给出读出宽度 Dv。对于按实际输入精度计算的 QK 和 PV，操作量为 **2 × P × H × (Dk + Dv)**，H 为本地查询头数，P 是各请求各 query 的可见 key 数之和。稠密路径读取捕获的掩码统计。稀疏 MLA 使用系统给出的有效选择数上界；Meta 图没有真实的索引值。吸收投影后的解码复用同一 latent cache 作为 K 和 V，前置投影得到的独立 K、V 各计一次。低精度路径单列输入、scale、概率转换与输出字节。系统端所称 FP8 在这些签名中可能为 E4M3 加 FP32 行 scale，不能将其视作 block32 E8M0 MXFP8。

**可复算的例子。** DeepSeek-V3.2，BF16，B=16，每请求 decode Q=1，历史 131,056，16 个本地头。查询内容和 RoPE 宽 512、64，输出宽 512。每行最多选择 2,048 个历史位置，P=32,768，Cube 为 1,140,850,688 ops。128 key 一页，每行 16 页，两个团队分摊，32 个 AIC 各配一个 AIV，单波结束。每个团队 8 页的 Cube 总服务为 4,357.416 周期，指令类别法得 VF 服务 3,840 周期。两级重叠后的单波为 **max(4357.416,3840) + min(4357.416,3840)/8 + 321.5 = 5158.916 周期**。第二项表示第一页串行，321.5 周期是两团队的局部 softmax 状态归并。在 1.65 GHz 下本地约 3.127 µs。按稀疏选取的行估算 38,174,720 B 读、262,144 B 写，在 4 TB/s 下为 9.609 µs。模型给出 **2.780 + max(3.127,9.609) = 12.389 µs**。2.780 µs 来自同族首发先验 4,587 周期；128-key 页和对称分工属于候选实现设计。相邻 query 的访存复用、索引填充值和跨核冲突仍可改变真实结果。

## Qwen QSA 与 GLM 池化 Indexer

QSA 第 j 行可见长度为 v，在索引复用时额外截断到生成该索引时的上下文长度。压缩比 r、token 预算 K 对应该行 **min(r × floor(v/r), K) + v mod r** 个候选位置。后一项保留未成组的尾部。实现用整除与算术级数求和，每个请求只需常数次计算。Qwen3.8 Flash Next，BF16，首块 Q=512，r=4、K=2048，全部前缀都在预算内，P=512×513/2=131,328。3 个本地头、Dk=Dv=256 的 QK/PV 为 403,439,616 ops，32 个查询块分布到 32 个 AIC，模型估计 4.489 µs。

选择器本身是另一项工作。QSA 与 KPool 对压缩行做点积、跨头归约和排名；候选池按 256 行分片，每团队处理至多 4 行 query。GLM-5.3 Flash 的单行示例，压缩比 4，32 个头、128 维、分配 8,192 个池行时，工作为 2×8192×32×128 = 67,108,864 ops。核数和池行数量按输入推导；该配置模型返回 3.309 µs。缺少逐 query 的有效池长度时，模型以分配的池容量为上界，并在输出中标 `history_bound=capacity`。这会高估长预留、短有效历史，且排名的跨核合并开销仍属设计估计。

## Qwen 的 Gated Delta Rule

状态布局为 [B,Hv,V,K]。核沿请求、value 头和 V 条带切分，同一条带的时间步串行。UB 容量减去控制保留后，为 3 份 FP32 状态工作区分配条带，条带上限 64。decode 按每步 **7VK + 8K + 4V + 8** 项 FP32 等价工作估计状态衰减、投影、误差、外积写回、输出及门控。prefill 使用源程序的 64-token chunk，令 S2 为各块真实长度平方和乘以 Hv，R 为 token 数乘 Hv。输入精度下的相关矩阵工作为 **2K × S2**。假设状态参与的转换及乘法走 BF16，另计 **6RVK + 2(K+2V)S2** 次运算，FP32 存储读写另列。模型核对 Q/K/V 的逻辑宽度和物理打包宽度，输入降位不会自动改变 FP32 状态容量。

Qwen3.5-27B 首块 BF16、B=1、512 token、6 个 value 头、K=V=128，状态形状为 [1,6,128,128]。每头 8 个满块，S2=196,608，R=3,072。矩阵工作为 503,316,480 ops，12 个独立条带各推进 8 个块，返回 10.454 µs。跨块临时张量能否驻留片上、求逆/三角递推的具体指令及 kernel 的融合程度还需实际实现检验。Kimi/GLM 的 chunk KDA 复用已有三阶段模型，输入 QK 的精度从捕获中读取，状态与输出阶段仍独立使用 BF16 工作量。

这些预测的首发常量、条带数量和指令发起有独立含义。它们允许替换芯片频率、资源数与吞吐，但无法替代同实现、同精度的设备校准。[新合同的推导实现](../../src/operator_model/system_routes/catalog_attention.py) · [原注意力合同](../../src/operator_model/system_routes/attention.py) · [已有 CA 证据](../error-analysis.md)
