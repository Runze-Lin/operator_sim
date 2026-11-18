# 单卡算子的逻辑工作量

输入是完成并行切分后该卡的 shape 与张量属性。每个矩阵乘加计 2 次标量运算。Vector 按元素计数，一次 exp、比较、除法或 rsqrt 各计 1 次。不同类别每周期可完成的数量不同。字段 `cube_ops_by_dtype` 按矩阵输入精度归类，`vec_ops_by_dtype` 按所列代数步骤的计算精度或显式注明的源精度归类，`vec_ops_by_class` 进一步区分算术与特殊函数。部分可辨识的路线只累计已列步骤。

`logical_read_bytes`、`logical_write_bytes` 单位为字节，每个 kernel 的每个有效输入或输出张量按一次完整读或写计。组合算子的生产者写和消费者读分别计入。实际缓存复用、重复搬运、对齐填充和 workspace 需要各层事务证据，所以全部路线返回 `physical_hbm_bytes=null`。逻辑字节也不等于程序峰值显存占用。

| 路线 | Cube、Vector 工作量口径 | 读写范围与遗漏 |
|---|---|---|
| DynamicMXQuant | BF16 输入上逐元素取绝对值、每 32 元素取最大，后者每组 31 次比较 | 读 BF16 输入，写 FP8 数据及 E8M0 尺度。指数生成、缩放和编码工作未展开 |
| RMSNorm | FP32 平方、求和、乘均值系数、加 epsilon、rsqrt、归一化和乘 gamma | x 与 gamma 各读一次，写 BF16 y 和逐行 FP32 rstd。类型转换未计 |
| Matmul | BF16 Cube 为 2MNK | 读 A、B，写 Y。无分核间重复读 |
| Addmm | BF16 Cube 为 2MNK，融合 bias 的 MN 次加法单独记 `fused_bias_additions` | 增加 N 个 BF16 bias 输入 |
| FA | 稠密非因果 QK 和 PV 的 Cube 共 4BHQ·KV·D。FP32 逻辑 softmax 的缩放、比较、减、exp、求和、除法 | 读 BF16 Q/K/V，写 BF16 O 和两路 [B,H,Q,8] 的 FP32 统计量。在线分块归并未展开 |
| Lightning Indexer | 按各 query 可见 key 数累计 BF16 点积、FP32 ReLU、逐头加权与归约 | 读 query、key、weights，写每行 top-512 的 int32 索引。TopK 比较数随实现变化，未计 |
| mHC Pre | 宽度 W=N²+2N 的投影为 2TNDW。HF32 按 32-bit 输入算力代理。Vector 含 RMS、仿射、sigmoid 和 residual 混合 | 读 BF16 x 与 FP32 phi/alpha/bias，写 BF16 h_in 和五个 FP32 输出（h_post、h_res、inv_rms、h_mix、h_pre） |
| mHC Sinkhorn | 按实际 I 次迭代计 FP32 最大值、减法、exp、归约、epsilon 加法及 2I 轮逐元素除法 | 读、写各一个 FP32 [T,N,N]。推理合同没有辅助输出 |
| mHC Post | FP32 residual 矩阵混合、post 支路乘加，另按 FP32 源元素数计输出 cast | 读 x/h_out/h_post/h_res，写 BF16 [T,N,D]。切核和驻留状态不改变逻辑张量 |
| mHC PreSinkhorn | 融合 Pre 与 Sinkhorn 的代数步骤 | 写三个有效推理张量。need_backward=false 的辅助描述符不计有效写，片内中间结果不重复计 |
| mHC | 实际 split Pre、Sinkhorn、Post 的工作之和 | 分别累计三次 kernel 输入和输出，外部 attention/MLP 单独调度 |
| KDA recurrent | FP32 核心递推为 B·S·Hv·(7KV+2V)。门处理与缩放未完全展开 | 含 Q/K/V、gate、beta、初始状态的读以及输出、最终状态的写 |
| KDA chunk | 按分块累加可见三角点积、稠密等价矩阵项和 state decay。三角求解等项未知 | 按选择的导出计状态、两路注意力和 gate 前缀。原始 gate 或未知附加输出会令对应方向的总字节为 null。计时未验证 |
| GMM1–SwiGLU–MX | 活跃 token 数 M 的 MXFP8 Cube 为 4MKF，FP32 SwiGLU 的五个标量步骤各 MF | 读活跃专家权重、输入及两者 MX 尺度、int64 专家直方图，写 FP8 与尺度。重定标与转码工作未展开 |
| GMM2 | MXFP8 Cube 为 2MFK | 同样计有效专家、尺度及直方图。输出 BF16 [M,K]，MX 缩放并入矩阵乘 |
| MoE FFN | 上述 GMM1 与 GMM2 合计 | 组件间 FP8 激活及尺度均计一次写和一次读，通信及路由由系统层处理 |

### 可手算核对的例子

- RMSNorm [2,6144]，令元素数 X=12288。平方、乘逆均方根、乘 gamma 共 3X 次乘法，每行另乘一次 1/D。每行求和用 D−1 次加法，加 epsilon 一次，总加法数为 X，再求两次 rsqrt。共 49,156 次 FP32 标量工作，读 BF16 x 与 gamma 共 36,864 B，写 24,576 B 的 y 和 8 B 的 rstd。
- mHC Sinkhorn [1,4]、I=1。row softmax 取最大值比较 12 次、减法 16 次、exp 16 次，行、列各归约 12 次并对每个元素做除法 16 次。行 softmax 后每元素加 epsilon，列归约分母每列加 epsilon，合计 20 次 epsilon 加法。因此 Vector 共 120 次，FP32 输入、输出各 64 B。
- BF16 FA [Q=9,KV=512,D=64,B=2,H=4]。两个矩阵乘合计 9,437,184 次 Cube 运算。读 1,057,792 B，写 13,824 B，其中统计输出 4,608 B。二维得分矩阵在 kernel 内部计算，无对外完整张量读写。
- mHC 的 T=257、N=4、D=16384 仍可按 split 三段计算工作量。该形状的耗时随原有超出 256-token 范围的估时路线标为外推，张量公式自身与 950DT 核数无关。
