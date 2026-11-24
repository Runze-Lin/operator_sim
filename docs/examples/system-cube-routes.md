# Decode 系统图中的矩阵投影

同一个 `npu_grouped_matmul` 名称在这批模型中有两种工作方式。MoE 用两侧 E8M0 分组比例的 A4W4 原生矩阵乘；DeepSeek 输出头用一行一个 FP32 比例的 FP8 激活和分组 MXFP4 权重。两种操作的解码、展开、累加以及量化后处理开销不同。选择路线前先核对本卡实际张量的 dtype、物理存储、stride、逻辑维度、scale 轴和分组计数。[系统矩阵模型](../../src/operator_model/system_routes/cube.py)

| 捕获合同 | 本地计算 | 计时与依据 |
|---|---|---|
| `aten.mm.default`，FP32、W 转置视图 | 严格 FP32 MMAD，窄矩阵可沿 K 分核，单个输出列块跨核合并 | 950DT/PR 的 `aclnnMm`，两种开发投影 |
| `aten.mm.default`，BF16、W 转置视图 | BF16 Cube，FP32 累加，FIX 写 BF16 | 950DT/PR 的 `aclnnMm`，开发与留出；连续 W 继续走原 Matmul |
| `aten.addmm.default`，FP32 | FP32 MMAD，单位 alpha、beta，沿 N 广播 bias 后写 FP32 | 950DT/PR 的 `aclnnAddmm` |
| `inference::matmul_rhs_transposed` | BF16 A 和连续 [N,K] 权重，BF16 MMAD 保留 FP32 累加值，FIX 写 FP32 | 950DT/PR 实跑 BF16→FP32 的 `aclnnMm`，使用 Stream-K 与 FP32 工作区，见下方开发与独立验证 |
| `npu::npu_quant_matmul`，A4W4 | 两侧沿 K 的分组 E8M0，原生 MX Cube，输出 BF16 | 复用 [aclnnQuantMatmulV5 校准](mx-native-tail.md) |
| `npu::npu_grouped_matmul`，A4W4 专家 | 每个本地专家按 histogram 分配 M 行，使用两侧原生 MX | 复用 [aclnnGroupedMatmulV5 校准](routed-grouped-matmul.md) |
| `inference::mixed_precision_matmul` 及 `npu::npu_grouped_matmul` 的 FP8 token-scale 输出头 | AIV 将 E2M1 展开成 E4M3，保留 W 的分组 E8M0，Cube 使用单位 A MX scale；FP32 累加结果在 AIV 乘本行 FP32 scale，再转 BF16 | 两种合同均为设计外推，完整 CA 为 N/A |
| `inference::quantized_batched_matmul` | Kimi 两侧均沿各自末轴分组；A 的比例沿 K，W 的比例沿 N。两侧解码及应用各自 E8M0，然后用 BF16 Cube 批量乘、FP32 累加，输出 BF16 | 设计外推，完整 CA 为 N/A。设计假设还原后的输入在 BF16 的表示范围内 |

安装包的 A8W4 MX 路线要求两侧都是沿 K 的 E8M0。FP32 token scale 和 Kimi 的 W 沿 N 分组都改变了这个接口。这里为系统缺失实现估计了一个量化布局正确的方案；原生 A8W4 双侧 MX 的校准表仍仅适用于原接口。[CANNBot 950 量化 Matmul 知识卡的 R-008 路线](https://gitcode.com/cann/cannbot-knowledge/blob/92dc260756e91e72e6dcd60f3996a4d54f43c8f5/knowledge/ops/ascendc/operators/aclnn/nn/matmul/quant_matmul_950.md) · [双侧 MX 原生算例](mx-handoff.md)

## 从 tile 算到设备时间

输入为本地行数（分组时是各组的行数）、N、K、各个 dtype 与 scale，以及芯片核数、主频、各精度峰值、HBM 带宽和片上容量。本节为表中新增加的路线给出公式。可从返回的 `work.schedule` 读取分块、K 页数、分核和各资源周期。设备窗口从首条指令到最晚参与核退休。

1. **输出分块。** M 块取有效行上限与 128 的较小者并对齐 16。FP32 原生薄矩阵先按 256 列分组，再将 N 均分给各组、向上对齐 16；其余路线按可并行的行块/组数量，把 N 分给空闲 AIC，列块最低 16、最高 128。最终行块也受 L0C 容量约束。记块的名义尺寸为 `a,b`，所有组的输出任务数为 `J = Σ ceil(Mg/a) × ceil(N/b)`，其中 `Mg` 为组内的行数。FP32 将剩余核用于分摊 K，选 `s=min(max(1,floor(C/J)),max(1,floor(K/16)))`，其余路线 `s=1`。最多用 `min(C,J×s)` 个 AIC，波数 `w=ceil(J×s/C)`，每分片最多 `h=ceil(K/s)`。此规则受 MatMulV3 的 Stream-K 与实测 MMAD 几何启发。其他小矩阵、图编译器和后代芯片可以选不同分块。[ops-nn 950 Stream-K 分块参考](https://gitcode.com/cann/ops-nn/blob/602e4ad985da00aad28c16432841e9c5069e10de/matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_streamk_tiling.cpp)
2. **K 页。** 令 `e` 为展开后 Cube 操作数的字节数。两个 L0 输入缓冲和两个 L1 缓冲分别能容纳 `L0A/(2ae)`、`L0B/(2be)`、`L1/(2(a+b)e)` 个 K 位置。带 AIV 展开时还需满足 `UB/(4(a+b)e)`。取能同时满足这些约束的 K 深度并向下对齐，FP32 对齐 16，其余路线对齐 32，再截到 K。页深记为 `d`，每分片页数 `p=ceil(h/d)`。双缓冲允许当前页计算时读取后一页。颗粒度和 UB 中的 4 份工作区为设计假设。
3. **MMAD。** 令 `R` 为目标每 AIC 每 cycle 的有效对齐运算数。对 BF16/FP8，`R=1000×峰值(TOPS)/(C×主频(GHz))`。对严格 FP32，开发窗中物理 `16×192×32` MMAD 约 409 cycles，扣 25 cycles 描述符后得 `2×16×192×32/(409−25)=512` ops/cycle。在 N=4 上，K512 与 K128 的描述符分别约 1603 和 416 cycles，两者差分给出 3.09 cycles/K 的窄块约束。这是 MMAD 描述符的有效吞吐，非系统标称 TF32 峰值。未来设备先按每核 BF16 能力比例外推 512，若提供严格 FP32 峰值则同时受该峰值限制。`T_cube=w×(h×max(2ab/R,窄块约束)+p×描述符固定价)`，其中窄块约束仅在 FP32 中启用；BF16/FP8 固定价取 32 cycles 作设计先验。FP32 分块结果留在 FP32 并合并。
4. **传输。** 冷启动时按有效输入及输出一次收发计算 `T_hbm=(B_in+B_out)×f/min(BW_hbm,active_AIC×128×f)`。这里带宽单位 GB/s，f 为 GHz，输出是 cycles。各 AIC 重读的输入 panel 另算 `T_panel=w×(有效块高+有效块宽)×h×e/B_mte`，其中 `B_mte` 单位 B/cycle，使用芯片 MTE2 设定并以每核 128 B/cycle 为上限，未知时继承此上限。128 为端口设计先验。重复搬运可来自片上数据；这批 trace 没有逐次读的 HBM 命中归属。取 `T_move=max(T_hbm,T_panel)`。
5. **Vector 和 CV。** 实际参与的 AIV 按活跃 AIC 占比计算。展开 W 每个元素一次、行比例后乘与输出转换每个输出元素各一次；batched 路线的 A、W 元素各需解码和比例应用。分片重复读取/展开的部分计入服务时间。分精度相加，`T_vec = Σ_p W_p × f × C_V / (1000 × P_p × V)` cycles。W_p 是含重复面板的该类操作数，f 是 GHz，C_V 是整片 AIV 数，V 是活跃 AIV 数，P_p 是对应位宽的整片 Vector TOP/s。token-FP8 路线按 8 位吞吐折算 W 展开，按 FP32 吞吐折算 token 缩放和输出转换；末轴双 MX 解码路线用 BF16 等价服务。INT8 峰值是展开吞吐的代理，真实 unpack/cast 指令服务率仍待独立测量。CV 由 `w×p` 个重叠机会约束，取 `D=max(T_move,T_cube,T_vec)`，`T_body=D+(T_move+T_cube+T_vec−D)/(w×p)`。一页只有一次机会就串行支付各段；长流水趋于其中最长服务。纯 Cube 与融合 bias 使用 `T_body=D`。此项将两种缓冲与多波负载近似成相同服务，未逐条实现跨核事件及尾波内部重排。
6. **整窗。** `t_us=(T_start+T_body+T_end)/(1000f)`。开发窗以首个 MMAD 发起为前界，以最后 MMAD 结束到最后退休为尾界。双端 BF16 观测前界为 1809/1780 cycles、尾界 522/518，取 1794/520。FP32 Addmm 两个窗前界 2004/1861、尾界 2270/2210，取约 1932/2240。FP32 Mm 两种几何观测前界 3605–4998、尾界 2579–4259，取范围内的代表预算 4273/3335。此处包含到达错位、启动和部分跨核收尾，不能唯一分解为分别独立的硬件常数。新芯片复用周期先验，按目标主频换算。因只有少量开发几何，分 K 时 AIV 的具体合并指令、访存命中、FP32 输出及 CV 换型的附加同步仍有不确定度。

## 从模型输入到输出的算例

以下时间使用内置 SKU。DT/PR 列为设备预测；970DT 列均为外推。实际的 27 组系统配置使用系统提供的资源覆盖值。

| 模型与本地操作 | 本地 M,N,K 或批组 | 950DT µs | 950PR µs | 970DT µs | DT 分核与工作 |
|---|---|---:|---:|---:|---|
| V4 Pro，FP32 router | 9,384,7168 | 8.081 | 12.669 | 5.081 | 32 AIC，块 16×192×32，16 路 K，14 页 |
| Kimi draft，FP32 Addmm | 8,1,7680 | 2.993 | 3.059 | 2.442 | 32 路 K，每核最多 240 个 K；bias 广播 8 次 |
| V4 Pro draft，BF16 转置权重 | 9,896,21504 | 11.137 | 29.216 | 5.952 | 28 AIC，块 16×32×512，42 页，读 38,922,240 B |
| Kimi router，BF16 输入 FP32 输出 | 9,896,7168 | 10.280 | 13.312 | 7.722 | 32 AIC，沿 N 分 4 块，再作 8 路 K，读 12,974,080 B，写 32,256 B |
| V4 Pro，token-FP8/MXFP4 投影 | 9,1536,7168 | 3.412 | 5.724 | 2.143 | 32 AIC，块 16×48×672，11 页 |
| V4 Pro，token-FP8/MXFP4 头分组 | 2 组，每组 9,1024,4096 | 2.896 | 4.737 | 2.104 | 32 AIC，块 16×64×512，8 页 |
| Kimi，末轴 MXFP4 batched | 12 组，每组 9,512,128 | 1.808 | 1.861 | 1.422 | 32 AIC，块 16×128×128，48 个输出任务、2 波，BF16 MMAD |

以 V4 Pro 的 token-FP8 投影为例。A 的形状为 [9,7168]，W 是 [1536,7168] 的半字节存储，W 的每行还有 224 个 E8M0 比例值。输入读取 `9×7168 + 1536×(7168/2+224) + 4×9 = 5,913,636 B`，输出写 `2×9×1536=27,648 B`。DT 总峰值 865 TOPS、32 AIC、1.65 GHz 得 `R=16382.6` ops/cycle；每个 AIC 负责 48 列，切 11 页，故 Cube 约 `2×16×48×7168/R+11×32=1024.1` cycles。全片冷读写为 2450.8 cycles，单个面板约 `(9+48)×7168/128=3192` cycles。对 11,010,048 个 W 元素作展开，对 13,824 个输出各做比例乘与转换，64 个 AIV 全部计入时，W 展开以 8 位代理 54 TOPS 得 `11,010,048×1.65/(1000×54)=336.42` cycles，缩放和转换按 FP32 27 TOPS 得 `2×13,824×1.65/(1000×27)=1.69` cycles。体段用 `3192+(1024.1+338.11)/11=3315.8` cycles，加前后 1794 和 520，DT 得 3.412 µs。多出的存储若要物化到 GM，或必须分两个 kernel，实际耗时将相应增加。

Kimi batched 例的 W 以 [12,128,256] 个物理字节保存 [12,128,512] 的值，每个 K 行内沿 N 分组。直接把这些 E8M0 视作沿 K 的比例会改变乘积。模型为 `12×128×(9+512)` 个输入元素各做 unpack 和 scale，并按 `2×12×9×512×128 = 14,155,776` BF16 Cube ops 计时。DT 的两个工作波给出 Cube 192.2、面板 548.0、Vector 51.4 cycles；加上首尾得到 1.808 µs。CA 整窗 N/A，端到端覆盖率中的该行是方案估计。

## Kimi 的 BF16 输入、FP32 输出为何要另选流水

`aclnnMm` 在 950 上支持 BF16 输入、FP32 累加与输出。实际 Kimi [9,896,7168] 的原生 kernel 给每个 N 块 224 列，两端都沿 K 分片，不同 AIC 的部分和经 AIV 归并。DT 用 8 路 K，32 个 AIC，每核 14 片 K64；PR 用 7 路 K，28 个 AIC，每核 16 片 K64。两端有 36 个 AIV 执行向量加。朴素 BF16 输出的启动开销和分核不足以描述这条路线。[ops-nn 的 950 aclnnMm 精度契约](https://gitcode.com/cann/ops-nn/blob/602e4ad985da00aad28c16432841e9c5069e10de/matmul/mat_mul_v3/docs/aclnnMm.md)

- 输出选择 `ceil(N/256)` 个均衡 N 条带，再向上对齐 16；M 块不超过 128 并满足 L0C。可用核心多于输出块时，将剩余并行度分给 K。局部内存双缓冲容量约束沿用上述 K 页规则，采用 BF16 输入字节数。第一个 GM→L1 payload 在 DT/PR 的 2881/2920 cycles 开始，首次 MMAD 分别在 5432/6142 cycles。两组差分的均值取 2886 cycles 作为“首个 payload 到 MMAD”的参考；启动到首读取 2900 cycles。
- 同一波从最早 AIC 到最晚 AIC 首发 MMAD 的跨度分别为 3194/3333 cycles。参考满核跨度取 3264 cycles，按 `(active−1)/(C−1)` 缩放，单核时为零。DT 每 AIC 的局部 MMAD 流从首发到末次完成，中位 4160.5 cycles，`(9+224)×896×2/4160.5=100.36` B/cycle。取 100 为整条面板流水的有效每核速率上限，芯片提供更低的 MTE2 速率时使用后者。它同时反映换页和发起间隔，不能分辨端口本身的带宽。PR 每核观察到更长等待，由目标全片带宽约束覆盖。
- 从首读起，全部逻辑冷数据传输与各核 MMAD 有重叠。若 `T_local` 是最忙核各波的面板字节除以上述每核速率，`T_body=max(T_hbm,2886+核首发跨度+max(T_local,T_cube))`。这样早已在读的数据不会在 MMAD 启动后再付一次。分 K 的部分和按 FP32 物化，工作区约 `4×s×M×N` B。归并尾取三者最大值：实测末 MMAD 后退休计数 3906/3565 的中值 3736 cycles、工作区写入与读回按 L2 带宽折算、`(s−1)MN` 个 FP32 加按 Vector 算力折算。L2 带宽缺失时，以两倍目标 HBM 带宽作假设。
- 在该例，DT 的面板预算为 `(9+224)×896×2/100=4175.36` cycles，全片带宽预算为 5365.11 cycles。总时间为 `(2900+max(5365.11,2886+3264+4175.36)+3736)/1650 = 10.280` µs。PR 的较低外存带宽使共享读进入临界路径，得 13.312 µs。970DT 无实测，使用目标资源外推为 7.722 µs。[模型](../../src/operator_model/system_routes/cube.py) · [开发窗证据](../../calibration/system_cube_ca/router.json)

## 复现

在源码根目录编译探针，使用同一个 CANN 9.2.1 的 OPP 和模拟器。以下命令生成开发数据；选择留出集时，将探针最后一个参数改成 `holdout` 并给出新的 `run_dir`。Kimi 的 BF16 输入 FP32 输出单例使用 `router`，新留出使用 `router-holdout`。隔离环境由 `scripts/cann-run.sh` 装载。

```bash
cann=toolchains/cann/9.2.1/cann-9.2.1
c++ -O2 -std=c++17 -I"$cann/aarch64-linux/include" calibration/system_cube_ca/probe.cpp \
  -L"$cann/aarch64-linux/lib64" -Wl,--allow-shlib-undefined -lopapi -lnnopbase -lascendcl -o /tmp/cube-ca
run_dir="$PWD/runs/system-cube-dev"
for sku in Ascend950DT_9582 Ascend950PR_957b
do
  mkdir -p "$run_dir/$sku"
  bash scripts/cann-run.sh msopprof simulator --soc-version="$sku" --launch-count=24 \
    --dump=false --timeout=12 --output="$run_dir/$sku/profile" /tmp/cube-ca development > "$run_dir/$sku.log" 2>&1
done
python3 -m calibration.system_cube_ca.receipt "$run_dir" \
  "$cann/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950" /tmp/cube-ca-dev.json --role development
```

`predictions.py` 从内置芯片配置重算设备预测。已签收的冻结结果保存在 `frozen.json`，含模型 hash；`replay.py` 可在已安装系统框架的 Python 环境中对 27 个存量捕获重放所有矩阵合同。大 trace 和 OPP 不需要分发给在线调用者。

## 同形状双端 CA 对照

下表的误差为 (模型 − CA)/CA。同一组行使用相同的逻辑 M,N,K、转置视图和输出精度。两端的芯片时钟都为 1.65 GHz；最后一列始终为无目标 CA 的外推。

| 数据角色、接口、输入→输出 | M,N,K | DT 模型 / CA µs | DT 误差 | PR 模型 / CA µs | PR 误差 | DT / PR 实际工作 AIC | 等效逻辑 Roofline U，DT / PR | 970DT 外推 µs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 开发 · aclnnMm FP32 | 9,4,20480 | 5.840 / 7.535 | -22.5% | 6.012 / 7.659 | -21.5% | 32 / 28 | N/A | 4.534 |
| 开发 · aclnnMm FP32 | 9,384,5120 | 7.090 / 7.162 | -1.0% | 10.370 / 8.995 | +15.3% | 32 / 28 | N/A | 4.841 |
| 开发 · aclnnAddmm FP32 | 8,1,7680 | 2.993 / 3.665 | -18.3% | 3.059 / 3.691 | -17.1% | 32 / 28 | N/A | 2.442 |
| 开发 · aclnnMm BF16 | 9,128,512 | 1.524 / 1.807 | -15.7% | 1.524 / 1.815 | -16.0% | 8 / 8 | 2.3% / 6.7% | 1.397 |
| 冻结留出 · aclnnMm FP32 | 9,4,28672 | 6.319 / 8.175 | -22.7% | 6.559 / 8.455 | -22.4% | 32 / 28 | N/A | 4.651 |
| 冻结留出 · aclnnMm FP32 | 9,384,7168 | 8.081 / 7.891 | +2.4% | 12.669 / 10.802 | +17.3% | 32 / 28 | N/A | 5.081 |
| 冻结留出 · aclnnAddmm FP32 | 128,1,5376 | 3.388 / 5.013 | -32.4% | 4.510 / 4.785 | -5.7% | 32 / 28 | N/A | 2.531 |
| 冻结留出 · aclnnMm BF16 | 144,128,512 | 2.101 / 2.385 | -11.9% | 2.101 / 2.392 | -12.2% | 24 / 24 | 3.8% / 10.7% | 1.926 |
| Stream-K 开发 · aclnnMm BF16→FP32 | 9,896,7168 | 10.280 / 9.955 | +3.3% | 13.312 / 13.191 | +0.9% | 32 / 28 | 31.6% / 69.8% | 7.722 |
| Stream-K 新留出 · aclnnMm BF16→FP32 | 144,896,7168 | 15.904 / 14.136 | +12.5% | 18.345 / 17.575 | +4.4% | 32 / 28 | 26.9% / 60.1% | 9.104 |

U 取逻辑 Cube 理想时间与输入输出字节在峰值 HBM 带宽下时间的最大值，再除以模型设备时间。内置两款 950 缺少严格 FP32 峰值，该精度的 U 为 N/A。逻辑输入字节数并非实测 HBM 流量。

| 数据组 | 双端完整窗数 | DT WAPE / 最大绝对相对误差 | PR WAPE / 最大绝对相对误差 | CA 采集跨度，DT / PR 秒 |
|---|---:|---:|---:|---:|
| 开发 | 8 | 13.5% / 22.5% | 17.8% / 21.5% | 93 / 111 |
| 冻结留出 | 8 | 16.9% / 32.4% | 16.4% / 22.4% | 507 / 563 |
| Stream-K 开发 | 2 | 3.3% / 3.3% | 0.9% / 0.9% | 496 / 512 |
| Stream-K 新留出 | 2 | 12.5% / 12.5% | 4.4% / 4.4% | 173 / 185 |

27 配置的 54 张 target/draft 图含 20,538 个矩阵节点。去除模块路径与分区标签的重复后，共 336 个归一化输入，覆盖 9 种合同，均返回有价窗口。核对使用系统的实际 OpFeatures 转换；FP4 的物理字节槽、逻辑 shape 和 stride 都参与检查。此回放检验覆盖，耗时误差只来自上面的同合同校准。冻结时每次 Python 排程测得 21–275 µs（已载入芯片配置，不含图规范化和 CA）。[逐配置、逐阶段、逐合同回放计数](../../calibration/system_cube_ca/replay.json) · [回放程序](../../calibration/system_cube_ca/replay.py)

两种 FP32 Mm 开发投影在 DT/PR 均观测到跨核分 K。以 DT 的 [9,4,20480] 为例，32 个 AIC 各计算 K640，实测每核一片 K512、一片 K128。7.535 µs 整窗中最早 MMAD 发起在 4967 cycles，最后 MMAD 结束在 8173 cycles，随后还有 4259 cycles。模型用统一启动和退休预算，低估该窄列分核调用。N384 的 [9,384,5120] 在 DT 的物理 MMAD 为 [9,192,32]，PR 上共享存储时间对体阶段的影响更大。实际共享层命中、各核到达和 MMAD 之间的间隔仍能产生误差。

验证程序对每个输出元素逐位检查。A 的每一行选择不同的 K，W 的数据同时依赖列和 K；Addmm 另外加入非零 bias。探针记录实际 SKU、内存模型、每核首指令与唯一 END 退休点，校验 dump ELF 与 OPP 中文件的 SHA-256，并统计真正执行 MMAD 的 AIC。只计作一次稀疏输入正确性检查，尚未覆盖一般数值分布、累加顺序和特殊浮点数。宿主采集跨度含模拟及解析，独立留出期间机器并发负载较高。

CANN 9.2.1 的 FP32 两条 API 在开发窗均调到 ops-nn 的 `MatMulV3_ND_ND_ND_ND_FP32_FP32_FP32_FP32_high_performance_4162_mix_aic`。BF16→BF16 开发窗为 `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_66`。各行的 kernel、ELF、MMAD 维度与计数、时间戳摘要都在对应 receipt 中。冻结后对系统张量角色归一化的修复和外推状态标记不改变表中的排程或冻结价格。[开发 CA](../../calibration/system_cube_ca/measured.json) · [冻结预测、工作量与分核](../../calibration/system_cube_ca/frozen.json) · [CA 探针](../../calibration/system_cube_ca/probe.cpp) · [窗口与 ELF 核查](../../calibration/system_cube_ca/receipt.py)

留出集中，DT 的 Addmm [128,1,5376] 低估 32.4%。该次首个 MMAD 与末次 MMAD 后退休分别为 2786/2575 cycles，高于模型继承的小行数 bias 启动/退休 1932/2240；每 AIC 三次 MMAD，核间错位与发起间隔也未完全暴露在各段预算里。实测该 shape 为 PR 更快，模型判成 DT 更快。BF16 [144,128,512] 两端都用 24 个 AIC，各 MMAD [48,16,256]；当前启发式选择 16 个 AIC 并给出相同的双端时间。留出误差保留原冻结结果。

独立 CA 明细保留各 SKU 的内核、启动核数、生产核数和整窗。单次分核规则可能在新几何选错，样本数未达到每合同十个独立 shape。[冻结留出 CA](../../calibration/system_cube_ca/independent.json)

Kimi [9,896,7168] 第一次冻结采用 BF16 输出的分核近似，DT/PR 为 4.654/10.693 µs，低估 53.2%/18.9%。两端 kernel 实际均为 `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_FP32_high_performance_4162_mix_aic`，有 36 个 AIV 执行 RV_VADD。这个样本用于识别并校准正式版的 FP32 输出 Stream-K，表中“Stream-K 开发”为当前方法在原样本上的回放。独立数据另取 27 组里尚未使用的最小 M=144。[首次冻结，历史对照](../../calibration/system_cube_ca/router-frozen.json) · [当前开发和留出冻结价格](../../calibration/system_cube_ca/router-current.json) · [开发原生 CA](../../calibration/system_cube_ca/router.json)

M144 按冻结结果评估，两端各 129,024 个 FP32 输出逐位正确，DT/PR 均正确预测 32/28 个 AIC。两端实际 MMAD 是 [144,224,64]，模型把 M 按 128+16 两块分工，从而给每个 M,N 任务配置更少的 K 分片，代价偏高。实测 64/56 个 AIV 上有 RV_VADD；在线模型保留归并操作量和时序预算，生产 AIV 数未知。DT/PR 的跨芯片加速比按 PR 时间除以 DT 时间计算，预测 1.154、实测 1.243。[Stream-K 新留出 CA](../../calibration/system_cube_ca/router-independent.json)
