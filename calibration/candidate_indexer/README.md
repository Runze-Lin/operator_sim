# Flash 候选过滤 Indexer 的 950 CA 证据

A5 vLLM 的部署包另含按候选 gather 的 MXFP4 QSLI。下面的 ACLNN 盘点指 CANN 标准安装包，历史 dense/compact 指 BF16 组件实验。发布 wheel 的原生合同与新 MXFP4 计时可从[部署源码审计](source-audit.md)、[组件和融合验证](../../docs/errors/flash-a5-qsli.md)及[同图系统重放](context-sensitivity.md)复核。


## 哪个库实现与模型相符

Flash 的共享 source 层先对压缩位置打分，分别取 8 个压缩位置内的最大分作为块分，强制纳入最新可见块，按配置再取至多 2048 个块。consumer 层复用这些块号，用**本层**的 Q、K 和头权重计算位置分，选位置级 top-512 并按位置升序输出。参考实现先遍历完整可见的压缩 K，再用候选掩码筛分；当 index heads 跨卡时，各卡的局部 head 加权分还需要先做 all-reduce。这里测量的输入从 consumer 的 Q/K/权重/候选块号开始。上游投影、旋转、假量化、块选择及 HCCL 都在探针边界之外。[DeepSeek 参考实现，固定版本](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/12b5c3a462cdaf73dfced279a50adc7c2bb4a9c1/inference/model.py)

| 入口 | 与系统节点的关系 | 950DT / 950PR |
|---|---|---|
| `aclnnSparseLightningIndexer`、`aclnnQuantSparseLightningIndexer` | 明确消费每个 query 的候选块 ID；非候选位置不能进入输出 | ops-transformer 文档明确仅支持 A2/A3。9.2.1 的 950 OPP 和已安装 ACLNN 头文件均无这两个推理入口。[BF16/FP16 接口](https://gitcode.com/cann/ops-transformer/blob/14d022b1a4c3df263bf697534e3c25c7c992ff5d/attention/sparse_lightning_indexer/docs/aclnnSparseLightningIndexer.md) · [量化接口](https://gitcode.com/cann/ops-transformer/blob/14d022b1a4c3df263bf697534e3c25c7c992ff5d/attention/quant_sparse_lightning_indexer/docs/aclnnQuantSparseLightningIndexer.md) |
| `aclnnLightningIndexerV3` 候选 source | 生产块号，同时返回普通位置 TopK | 文档产品表包含 950，但开启 candidate 的规则只支持 DAV_2201。已装 9.2.1 头文件止于 V2。[V3 选路与样例](https://gitcode.com/cann/ops-transformer/blob/14d022b1a4c3df263bf697534e3c25c7c992ff5d/attention/lightning_indexer_v2/docs/aclnnLightningIndexerV3.md) |
| `aclnnLightningIndexer` / `V2` | 支持普通加权 ReLU 打分和 top-k。`blockTable` 按 batch 映射 paged KV 物理页；`metadata` 为运行元信息 | 950 具备库实现。本次紧凑实验调用 V1，对已经打包的 K 排序。[V2 接口](https://gitcode.com/cann/ops-transformer/blob/14d022b1a4c3df263bf697534e3c25c7c992ff5d/attention/lightning_indexer_v2/docs/aclnnLightningIndexerV2.md) |
| `aclnnPoolKeyIndexer` / `aclnnMsaIndexScore` | 前者从 pooled key 等输入选块；后者计算 MiniMax 每个 KV 块的 maxpool score | 两者的调用边界及输出均与本节点不同。[MSA 计算定义](https://gitcode.com/cann/ops-transformer/blob/14d022b1a4c3df263bf697534e3c25c7c992ff5d/attention/msa_index_score/docs/aclnnMsaIndexScore.md) |

## 可复核的小样例

单卡单请求、末位 query，一个本地头分组有 32 个 query heads，D=128，单个压缩 KV head，BF16 Q/K/weights。800 个可见压缩位置组成 100 个块。输入为 75 个不同的候选块，包含最新块，一共保留 600 个位置。本批 Flash 源层 20 和消费层 24/28/32/36 的配置压缩比均为 1；128 Ki（131,072 token）加 9 个当轮位置，对应 target 的 131,081 行 key。最多 2048 个候选块、每块 8 个压缩位置，合计上限 16384 个候选 key，最终有序输出仍是 512 个位置。本样例选择更小的候选集以控制 CA 开销，并保留“合法位置多于 512、候选有空洞、需要淘汰、输出顺序回到全局位置”的性质。

第 j 个输入块号为 `floor((74−j) × 99 / 74)`，逆序送入。每个位置 p 的正向 head 点积分为 `floor(p/128) + (p mod 128)/128`，反向 head 得到其相反数。两者权重均为 0.5，其余 heads 权重为零。对浮点数学参考，ReLU 与加权和后分数随 p 严格递增。CPU 从给定块号独立构造目标集合、取最大的 512 个全局位置并升序排列。设备端必须逐项对上这 512 个位置，包括最近块和被排除块之间的空洞。该构造用于识别 mask、局部到全局 ID 还原、排序的错误；精度相近的并列分数、跨 rank 累加、padding、部分尾块和多 query 各自持有不同候选等情况需单独验证。

**全表打分**。探针将 BF16 Q/K/W 显式转换到 FP32，用 Matmul 算出全部 `32 × 800` 个点积，ReLU、权重广播和沿头 ReduceSum 得到 800 个分数。将 `[100,1]` 块号和 `[1,75]` 候选做“不等于”比较，对候选维取 all，得到 100 个“该块所有 ID 都不命中”的排除标志。标志按块广播到 8 个位置，填负无穷。最后做 TopK，按全局位置排序并转 INT32。FP32 是这个探针的实验选择。参考代码对 Q 做 FP4 假量化后调用 einsum，其实际运行精度还受框架输入 dtype 和后端影响。二维广播 membership 为这个可复跑实验的实现选择，复杂度为块数乘候选数，系统模型可以采用线性 bitmap 的假设。

**先收集 key 再打分**。两次 Gather 分别打包 600 个 BF16 key 和对应原位置；普通 LightningIndexer 只对 600 个位置计算，输出局部 top-512。再一次 Gather 回到全局位置，Sort 得到升序。这里的 `B=Q=1` 且整段都可见，紧凑缓冲区的普通 causal 限制在该例中恰好正确。多 query、部分块或不足 512 个有效位置需要不同的数据处理。该路径用于量化“确实少读、少算”时的实现差异。

设备数值在每个 kernel 的所有发射物理核上取“最早首条指令到最晚退休”，把串行 kernel 的这些窗口相加。每一段均检查 trace 唯一 END、与厂商逐核时长一致以及真实加载 ELF 和安装包一致。组件之间的派发/排队空隙、ACL 分配与同步耗时不计入该和。同一小形状在两端、每种分解各一窗，属于开发 smoke；下文另列冻结后的新增组件检查。用于估计它们各自实现开销；Flash 生产融合算子的直接 CA 延迟与独立 WAPE 均为 N/A。

## 实测与适用范围

以下均为上面的同一个 Q=1 开发样例，时间单位 µs。两种顺序的算力需求、算术精度及取/存中间结果的频率也不同，二者时间差不能单独归因于少算 200 个 key。

| 路径 / 设备 | 打分输入 key | 完整 kernel 数 | 各 kernel 设备窗之和 | 其中普通 LI 子窗 | 匹配/检查输出 | 离线墙钟 |
|---|---:|---:|---:|---:|---:|---:|
| 全扫 950DT | 800 | 13 | 37.0570 | — | 512/512 | 166.86 s |
| 全扫 950PR | 800 | 13 | 36.4024 | — | 512/512 | 170.86 s |
| 紧凑 950DT | 600 | 5 | 17.0927 | 5.4818 | 512/512 | 113.84 s |
| 紧凑 950PR | 600 | 5 | 16.9315 | 5.4988 | 512/512 | 129.22 s |

全扫实验中，FP32 MatMulV3 子窗为 DT 3.2152 / PR 3.2861，位置 Sort（INT64）为 7.4188 / 7.5503。紧凑路径使用 INT32 位置 Sort，五段分别为 DT 3.6685、2.1661、5.4818、1.8297、3.9467，PR 3.4303、2.1309、5.4988、1.8170、4.0545。保留的紧凑证据列出每段的运行时 kernel、安装包相对路径、ELF 与物理核 trace 摘要、发射核数及 cycles。[全扫精确值](dense.json) · [紧凑精确值](compact.json)

把普通 LightningIndexer 的 H=8 物理排程参数直接用于 H=32、600 keys，在 DT 得到 5.4586 µs（−0.42%），在 PR 得到 5.4719 µs（−0.49%）。这次完整时间接近，但 H=8 排程把物理 MMAD 高度算成 32。两个设备的同一批 H=32 trace 都显示实际 M=64，四个 N=128 页及一个 N=88 页，全都 K=128。四个发起间隔 DT 为 627、740、300、665 cycles，PR 为 585、750、300、785；预测给出四个 416 cycles。首发、读、计算、输出可部分抵消，短序列单点不足以检验长循环。[普通 LI 资源排程](../../src/operator_model/operators/attention/indexer.py) · [逐核 MMAD 证据](compact.json)

### H32 分片规则与第二个独立组件样例

CANN 9.2.1 安装包自带的普通 BF16 LightningIndexer arch35 源文件给出了补齐规则。对本例的 TopK=512、H≤32，kernel 的 s1Base=4、mBase=4H。每个 query tile 的实际头行数为 tH；Cube 先向上取到 2H 的倍数，再把每段 L0 M 向上取到 16 的倍数。单 query 时，H=8 得 M=16，H=32 得 M=64。三或四个 query 组成完整大组时，分别得 M=32 或 128。H=64 且 topK≤2048 有单独的 split-M 分支。该结论取自安装包的 `impl/ops_transformer/ascendc/lightning_indexer/op_kernel/arch35/lightning_indexer_kernel_arch35.h` 第 177–186 行与 `lightning_indexer_service_cube_arch35.h` 第 182、205、378 行，两份文件 SHA-256 分别为 `4854403a5316fa06aac1d3ffa910ffb9f8d0888eca29d7db222ff042fc34edfe` 和 `d97e554802646d842ec63bb6014e220930383585ca2ac3150af909d5d1db69be`。两端此前留存的 Q=9、H=8、8192-key trace 中，每个设备前两个活跃 AIC 分别发 64 条 M=32 的 MMAD，最后一个活跃 AIC 发 64 条 M=16 的 MMAD，与 4+4+1 行分组一致。旧 H8 与新 H32 实际加载的同一 ELF SHA-256 为 `4940c25dd5f72cce57bf18c30df8568ecce22d0f472c513235a0dc03db905ca6`。[H8 留存 trace 摘要](reused-h8.json) · [H32 首个 trace](compact.json)

新增 Q=1、H=32、D=128、4096 源压缩 key 的检查。512 个源块中分散选取 256 个块，供普通 LI 的连续缓冲区为 2048 keys。通过给截止分数以上的全局位置加一个可精确表示的偏置，分离 BF16 排序在第 512 名附近的并列；不改变数学 TopK 的集合。在采集前冻结了**普通 LI 子窗**预测和代码指纹。整条紧凑链另外记录组件窗和，只用来观察这组库调用的代价。

| SKU | 采前 H8 锚点推 H32 普通 LI µs | 普通 LI CA µs | 有符号误差 | 紧凑五段窗和 µs | 完整输出 | 离线墙钟 |
|---|---:|---:|---:|---:|---:|---:|
| 950DT | 8.5237 | 8.0164 | +6.33% | 19.7727 | 512/512 | 106.55 s |
| 950PR | 8.5443 | 7.9055 | +8.08% | 20.0376 | 512/512 | 113.68 s |

两端都发了 16 条 (64,128,128) 的 MMAD。普通 LI 子窗在 PR 上快了 0.111 µs，而采前模型认为 PR 慢 0.021 µs；五段链则是 DT 快 0.265 µs。新增形状检验了 H32 普通打分组件的跨端外推，尚未测到部署时的融合候选 consumer。分核、搬运和头对齐如果改变，组件与整链的快慢方向可能不同。[冻结预测](frozen-component-check.json) · [第二组逐段实测](compact-component-check.json)

## 复跑

使用 CANN 9.2.1、950 算子包及可通过本地 Unix socket 连接 msopprof 的环境。源码编译及单个 capture 如下；若 CANN 已激活，可省略 `--launcher`。首个样例的四个组合为 `dense/compact` 和 `dt/pr`。第二个样例在 `compact` 后增加 `--blocks 512 --selected 256`。每次使用新的记录目录。

```sh
cann="$ASCEND_HOME_PATH/aarch64-linux"
c++ -std=c++17 -O2 calibration/candidate_indexer/probe.cpp -I"$cann/include" \
    -L"$cann/lib64" -Wl,--allow-shlib-undefined -lopapi -lnnopbase -lascendcl \
    -o /tmp/candidate-indexer
python -m calibration.candidate_indexer.capture dense dt /tmp/candidate-indexer \
    runs/candidate-indexer/dense-dt --launcher /path/to/activate-cann.sh
python -m calibration.candidate_indexer.receipt runs/candidate-indexer/dense-dt \
    --opp "$ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/kernel/ascend950" \
    --out /tmp/candidate-dense-dt.json
```

原始指令 trace 存放在忽略的 `runs/`。采集设置 `--launch-count=32`，msopprof 的缺省值只记录首个 kernel。`receipt.py` 拒绝设备型号/内存模型不符、索引错误、物理核时长不一致、ELF 不同或缺少流水的结果。全表 membership 早期试验采用布尔 Scatter，在小样例上未及时退休，已停止；最终实现使用显式块号 membership。

六次合格采集在两种 SKU 上累计约 801 秒进程墙钟，DT/PR 可并行。在此之外，两次套接字受限诊断约 204 秒，两次缺省仅导出首个 kernel 的复核约 192 秒，两次提前停止的 Scatter 试验约 640 秒。这些尝试均未作为新增的完整 CA 窗或独立样本。
