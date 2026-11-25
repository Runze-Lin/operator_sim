# 文本模型目录的 Vector、状态和布局算例

系统传入本卡输入、输出、stride、精度、语义属性和芯片配置。同名算子在普通 decode 中可能只更新一小段缓存，在多 token 校验中则需要保留每个前缀的状态。可以用实际任务的特征键回查两种边界。[逐模型、逐算子的 shape 索引](../../results/v2.4/catalog/shape-index.md)

## Qwen AddRmsNorm

例如 BF16 输入和残差均为 [9,512]，gamma 为 [512]。记 R 为 token 行数、D 为末轴通道数、E=RD。逐行将输入与残差相加得到 z，计算 1/sqrt(mean(z²)+ε)，再乘 z 和 gamma。输出包括归一化 BF16、FP32 的行逆标准差及 BF16 残差和。冷读字节为 4E+2D，写字节为 4E+4R，本例分别为 **19,456 B 和 18,468 B**。权重预加一的 Gemma 路径另计加法；其语义差别保留在调用属性中。

每个 AIV 先领取一行。总行数超过 AIV 数时，最忙核领取 ceil(R / AIV数) 行。一行分为能装入 UB 的面板，上限按读写张量、FP32 临时量和双缓冲计算。首个面板把读、算、写时间相加，后续面板每个元素的推进时间取三者最大值。取单核流水和共享 HBM 时间中较大的值，加上固定前段服务周期，再除以每微秒的时钟周期数。原法用每核吞吐为 Vector 算术定时；指令法按 256 字节寄存器和 FP32 运算位宽折算加载、算术、归约、rsqrt 和写回的发起次数，由可并行发射槽和各端口约束决定时间。[共享面板模型](../../src/operator_model/system_routes/vector.py) · [指令类别换算](../../src/operator_model/system_routes/vf.py)

以 [9,512] 完整 CA 窗的末指令退休时间减去首指令发起时间，在两个 SKU 上分别得到 **4.0364 µs 和 3.9279 µs**。用芯片频率换回周期，扣除指令法算出的有效载荷流水，剩下约 **6,506 和 6,327 周期**，作为同一程序的固定控制项。该残差还吸收了此开发样例未单独辨识的争用与等待。其它宽度和核数不再重新拟合。没有等价测量的新架构继承 DT 控制周期，并按目标资源为工作量定时。[开发窗与 ELF 身份](../../results/v2.4/add-rms-development.json) · [校准与留出结果](catalog-ca.md)

## GLM、Qwen 的状态与局部工作

| 调用 | 工作量和分块依据 | 算例或边界 |
|---|---|---|
| Indexer LayerNorm | 对每行做均值和方差两次归约，仿射操作按元素计；按 FP32 运算、实际输入和权重的存储字节计价。 | [1,128] 和 [512,128] 每行归约同样宽，后者会有多轮。同一宽度的步长和 gamma dtype 仍要核对。[CA 对照](catalog-layer-norm-ca.md) |
| HC gate mix、HC combine | 对 S 路残差，gate mix 遍历 S 份输入，逐元素门控，再沿 S 归约；combine 向每路注入加权激活。 | S=4，宽 2560 时，每个 token 读 10,240 个混合元素。局部投影由图中独立的矩阵乘计算。 |
| n-gram hash、lookup、prepare context | 哈希按 token、head、n-gram 阶数计整数运算。查表按选中行读取参数，按输出行写满 masked 结果。 | TP8、16 个 head、每 head 160 个 INT8 值和五个 FP32 scale，一 token 产生 2,880 B；均匀哈希假设下本卡读取约 360 B 有效表载荷，再读 128 B ID。词表大小影响容量，单次查询不遍历词表。 |
| PLE gate + causal conv | 对每 token 的 S×D 个通道进行归一化和门控；深度卷积按 tap 数 K 计约 2KSD 个算术操作。dilation 决定环形历史的跨度。 | S=4、D=2560、K=4、dilation=3 时，每个通道至多保存 9 个先前位置；校验图若返回前缀快照，另计输出张量字节。 |
| MLA cache write、state update | 只写新行的普通更新按活跃行字节计费；若图生成每个 verify 前缀的完整状态，按对应输出形状计写带宽。 | 写 9 个 BF16、D=512 的新行共 9,216 B，分配 32Ki 或 128Ki 缓存不改变这次写回。显式输出 [1,9,C,128] FP8 和 [1,9,C] FP32 快照时，写 9C×132 B，只量化新值。 |
| QKV parallel layout、reuse sparse indices、ATen split | 分头布局描述、复用已有索引、split 视图按原捕获语义返回已有存储；物化、稀疏选择和通信由相应调用计费。 | 同一 [512,2048] 的 QKV 缓冲返回自身，新增设备时间为 0。 |
| SiLU、sin/cos、scalar pow、div、arange、softmax | SiLU 分解为 exp、算术和倒数；三角函数采用范围归约和多项式的设计假设；连续范围按序号生成；softmax 按行最大值与和两次归约。 | FP32 [4,4,16,2064] 的末轴 softmax 有 256 行，每行处理 2064 个有效分数。捕获的块扩散源模型采用 FP32。 |

除列出实测的等价子合同外，上述程序的完整设备误差为 N/A。推理方实际使用的融合算法、寄存器发射吞吐、缓存在各 rank 的命中率以及哈希行在各分片的分布，都会改变模型与设备的差距。[实现](../../src/operator_model/system_routes/catalog_vector.py)
