# A4W4 路由专家的单次分组矩阵乘

**DeepSeek-V4-Pro，TP8/EP8，rank 0，B=1，校验 Q=9。** 第 0 层 target 的 dispatch 为本卡专家保留 54 行容量，实际送入 gate/up 投影的有 48 行。本例计算这 48 行的 MXFP4 乘加数、激活和专家权重读量、BF16 输出写量。950DT、950PR、970DT 都返回相同的数学工作量，三者的设备时延均为 N/A。

## 从原始图取得输入

`v4-pro/native/000_rank0_target.json.gz` 节点 118 的原始名为 `npu::npu_grouped_matmul`，组件为 `model.layers.0.mlp.experts.gate_up_proj`。其 `semantic.group_counts` 长度 48，每个元素为 1。前驱节点 117 的 `expert-routing.csv` 行独立给出相同的 48 个计数、`valid_rows=48`、`capacity_rows=54`。一个 token 可被发送给多个专家，多个 rank 又可向本卡发送，因而不能用本卡的 9 个查询 token 代替 48 条已路由行。[系统盘点与原始数据复核命令](../system-workload-coverage.md#复核与接入)

| 角色 | 本卡 GMM 逻辑张量 | 数据类型和布局依据 |
|---|---|---|
| A | [48,7168] | E2M1，两元素一个字节，K 内连续打包 |
| W | [48,7168,6144] | E2M1；捕获的 [G,K,N] 是底层按 [G,N,K] 每行打包的转置视图，逻辑 `logical_weight_shape=[48,6144,7168]` |
| W 的 scale | [48,112,6144,2] | E8M0，每 32 个 K 共用一个，存储以两个相邻 scale 为一对 |
| A 的 scale | [48,112,2] | 同样 E8M0/block32，按行持有 |
| Y | [48,6144] | FP32 累加后物化 BF16；gate 和 up 两半都在这个输出内 |

```python
from operator_model import estimate

result = estimate(
    "routed_grouped_matmul",
    (48, 6144, 7168),  # R, N, K
    {"expert_counts": [1] * 48, "dispatch_capacity_rows": 54},
    dtype="mxfp4", chip="Ascend950DT_9582",
)
print(result.work["cube_ops"])           # 4227858432
print(result.work["logical_read_bytes"]) # 1123207680
print(result.work["logical_write_bytes"])  # 589824
print(result.status, result.latency_us)  # unavailable None
```

## 分专家推导

设 `mₑ` 为本卡第 e 个专家收到的有效行数，`G=len(expert_counts)` 是本卡专家数，`E=count(mₑ>0)` 是实际用到权重的专家数，`R=Σmₑ`。令 N 为输出宽度，K 为缩并宽度。本例 G=E=R=48、K=7168、N=6144。输入 (R,N,K) 指已经打包并去掉未用路由容量的矩阵，未用行只在 `projection.unused_dispatch_capacity_rows=54−48=6` 中出现。没有上游容量时该字段为 null。

各专家做 `[mₑ,K] × [K,N] → [mₑ,N]`。一次乘加计两次有用运算，因此第 e 个专家工作量为 `2mₑNK`，整次投影为 `2RNK=2×48×6144×7168=4,227,858,432 ops`。FP32 是累加精度，计数归在执行这次乘加的 `cube_ops_by_dtype["mxfp4"]`。边界没有独立的 Vector 原语；实际 scale 处理、转换或调度所发出的指令数未观测。

编码后的 K 行载荷占 `L=K/2=3584 B`。该接口支持正偶数 K 的双值打包和无显式 K 补零合同。每行有 `g=⌈K/32⌉=224` 个有效 scale，成对保存占用 `S=2⌈g/2⌉=224 B`。若 g 为奇数，保留的最后一个配对槽位计入接口字节。每行输入读取一次；同一专家的所有行共用其一套权重及 scale，空专家的权重只属于模型存储容量。

| 输出字段 | 计算与本例代入 | 结果 |
|---|---|---:|
| `activation_bytes` | R×L = 48×3584 | 172,032 B |
| `activation_scale_bytes` | R×S = 48×224 | 10,752 B |
| `active_weight_bytes` | E×N×L = 48×6144×3584 | 1,056,964,608 B |
| `active_weight_scale_bytes` | E×N×S = 48×6144×224 | 66,060,288 B |
| `logical_read_bytes` | 以上四项之和 | **1,123,207,680 B** |
| `logical_write_bytes` | BF16 两字节，2×R×N = 2×48×6144 | **589,824 B** |
| `all_expert_weight_and_scale_storage_bytes` | G×N×(L+S) = 48×6144×3808 | 1,123,024,896 B |

最后一项用来报告本卡所有专家的**编码张量容量**，已含本例的权重和权重 scale，不加到读量上。对固定 R、N、K，专家行数从均匀变成集中时，数学乘加数保持不变；非空专家数若减少，边界上的唯一权重字节随之减少。即使非空专家数相同，直方图不同也可能改变核间调度，因而入口会保留整份 `expert_counts`。实际 HBM 访问还受分核、重读、L2 命中影响，`physical_hbm_bytes` 为 null。[工作量实现](../../src/operator_model/operators/moe/routed_grouped_matmul.py)

## 计时与覆盖边界

在系统旧基线的同一份 L1 账本中，节点 118 的预算是 **351.18672 µs**。之后节点 119 为独立 SwiGLU，120 为独立动态量化，121 为 down 投影，预算分别为 0.27648、0.10512、175.71216 µs。每个投影只接管自己的节点。跨卡 dispatch/combine、路由元数据、bias、SwiGLU、输出量化和全局专家权重均须另算。直方图在这个捕获合同中是主机属性，实际设备若读取 group list，还应由其实现模型补上该流量。

27 个常规 run 有 3,582 个 gate/up 或 down 的双 MXFP4 节点。其四个输入、BF16 输出、零 K 补齐、group32、FP32 累加、每专家计数与前驱 dispatch 的有效行和容量逐一核对；这批模拟路由恰好全部专家非空、每专家行数相同，且每次 dispatch 容量均大于有效行。通过公共 `estimate` 回放这 3,582 个节点，Cube 工作量均与逐专家求和一致，逻辑读写均与每个输入输出张量按编码宽度独立计算的字节数一致。本机每节点查询墙钟中位数 0.063 ms、P95 0.084 ms、最大 3.172 ms；读取和解压系统账本的时间另计。空组、热点和奇数个 scale 的配对尾槽只通过公式与单元测试检查。另有 963 个输出头分组的 A8W4 调用，使用逐 token FP32 激活 scale，接入时另行匹配。[系统分组统计](../system-workload-coverage.md)

`aclnnGroupedMatmulV5` 是可考察的 950 实现，真实派发、groupList 类型、存储布局、计算和搬运重叠需进一步验证。当前账本没有单投影 CA 整窗，`latency_us`、实际用核、kernel 数及 HBM 物理流量均为 N/A，无法从 351.18672 µs 计算本模型的 CA 误差。[CANN GroupedMatmulV5 接口](https://gitcode.com/cann/ops-transformer/blob/14d022b1a4c3df263bf697534e3c25c7c992ff5d/gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md)
