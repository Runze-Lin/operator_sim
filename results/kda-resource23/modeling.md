# KDA V2 的分块、资源与设备调度

系统模拟器可以显式选择这个实验实现。它与已冻结候选共用同一组工作计数、阶段预算和芯片资源假设。输入为芯片 JSON、BF16 的 `[B,S,Hq,Hv,K,V]`，以及通过宿主合同固定的布局和可选项。输出含每段活动核、分块、MMAD 维度和运算数、DMA 指令数及分层字节、估计 RVEC 子流水记录数、关键核阶段预算和设备微秒数。它按工作量和目标芯片给每个阶段定价。代码从固定的本地参数读取资源价格，预测时无需 CANN。当前合同为等长 packed BSND、Kimi 等头、`K=V=128,C=64`、原始 BF16 gate、FP32 VK 初态，输出 final state、Aqk、Akk。小样例经正式 FLA A5 26.9.1 的 `aclnnChunkKdaFwdV2` 强制进入三段，大规模 Kimi 还需满足其默认 V2 分派条件。[可运行模型](../../development/validation/kda_resource23/model.py) · [冻结与输入](freeze.json) · [完整来源清单](source-evidence.json)

## 从 shape 推导 tile 和分核

每条序列有 $J=\lceil S/64\rceil$ 个块，全部序列共有 $C=BJ$ 块。每个头的当前块有 $n=\min(64,S-64j)$ 个有效 token，其中 $j$ 从零起。令 $H=H_q=H_v$，$A$ 为阶段可用 AIC 数，950 常规混合组占 1 AIC+2 AIV。

| 阶段 | 发布实现如何分核 | 每个头每块的计算 |
|---|---|---|
| Prepare | $C<A$ 时按 chunk×Q 头组分核，当前合同每组一头。$C\ge A$ 时按 chunk 分核，同一核将局部 head 按最多 4 个一组处理。 | $j'=1..\lceil n/16\rceil$ 逐次发 $(32,16j',128)$ MMAD，同时完成 QK 与 KK 的 16 行。Vector 处理两个 32 行三角逆，两个半块各有 $31\cdot32/2=496$ 次源行依赖，短尾也执行。$n>32$ 时追加两个 $(32,32,32)$ FP32。最后有两条 $(p,128,p)$ BF16 算 W、U，$p=32$ 或 $64$，取能盖住 $n$ 的最小值。 |
| FwdH | 分核单位为整条 sequence×head。$D=BH$，$h=\lceil D/A\rceil$，启用 $\lceil D/h\rceil$ 组。一个头的相邻状态递推串行。 | 令 $a=16\lceil n/16\rceil$，先算 $(a,128,128)$，再算 $(128,128,a)$。Vector 对首块转换初态，逐块修正值和状态。中间状态留 UB，下一块保存 BF16 H，最终写回 FP32 VK。 |
| Finalize | 小 $C$ 按块和头分，充足时按块分。varlen 且整头映射且最少每组 8 个块时，正式 host 切到 AIV 搬输入。 | 先发 $(n,128,128)$ 初始化 L0C，再发 $(n,128,n)$ 直接累加 AqkV。同一源形的 $n=2$ 在此阶段按 2 发，FwdH 则按 16 发。AIV 搬运分支把 Q/H/A/V 搬入 UB 后按 16 列条带写 L1。 |

上述 MMAD 元组依次为 $(M,N,K)$。实际发起工作为 $2MNK$；硬件计价另用对齐到 16 的维度估算占用。950DT 和 PR 单核 FP16/BF16 Cube 峰值相同，小样本只开一核，因此不能用 32/28 当作 S2 的算力加速比。970 的情景假设保留双 AIV 协作合同，以 $\min(N_{AIC},\lfloor N_{AIV}/2\rfloor)$ 限制混合组，纯 Cube 路径仍可用全部 AIC。正式 970 实现可能重新分核或编译出 1:1 混合组。[发布 host/kernel 源文件及 Git blob/文件 SHA](source-evidence.json)

## 搬运与时间的计算

1. 对每一条源代码中的 DMA 加入搬运方向、描述符数和操作字节。块内的 $r=256n$ 是 BF16 的 $n\times128$ 字节。比如 Finalize 的 AIC 输入分支共读 $r+32768+2n^2+r$ 字节，分别对应 Q、128×128 的状态、有效 Aqk 子矩阵和值。状态向量阶段在首块从 GM 读取两次 65,536 字节 FP32 初态，分别服务初态转 BF16 和首次 FP32 更新。FIXP 公开的 `processed_bytes` 取转换前 FP32 字节，因此 BF16 输出 $r$ 的描述符可记作 $2r$。这些是各层描述符字节；HBM 真实计数受缓存与地址复用影响。
2. 同资源一组描述符的服务周期为 $L_d+\sum_i b_i/R_d+2(k-1)$。$L_d$ 为该方向响应开销，$b_i$ 为每条描述符字节，$k$ 是该组描述符数。$R_d$ 来自芯片配置的 MTE GB/s 除以 GHz，缺省时借用既有 Matmul 单核读、入 L0、写服务率。共享访问限于单参与组分得的 L2 带宽。$L_d$ 取合格 DT S2 的每条描述符服务窗扣除本地字节服务后的中位数，含队列等待，2 cycles/额外描述符是近同时发起小批的近似。大场景另比较 $(B_{GM,read}-C_{L2})_+/BW_{HBM}$ 得到的一种外存压力情景。它假设超出 L2 的读量进入外存，未观测逐行冷热或所有回写。跨核 HBM 利用率曲线和多头拥塞仍缺数据。
3. Cube 单核 BF16 服务率 $P_{16}=1000\cdot\mathrm{TOPS}_{16}/(f_{GHz}N_{AIC})$，单位 ops/cycle。每发 MMAD 按 $L_c+2\bar M\bar N\bar K/P$ 定价，$\bar d=16\lceil d/16\rceil$。$L_c$ 为七条合格小例 MMAD 的耗时扣除工作量后的中位数。C4/C5 在源码中明确关闭 HF32，trace 报 FP32。冻结代码的 `f32_tf32` 键需读作“普通 FP32，假设每周期吞吐为 BF16 的一半”。真实 FP32 吞吐仍待测。[冻结元数据勘误](frozen-annotations.json)
4. Vector 按每个 VF 分成 RVECEX、RVECLD、RVECST、RVECSU、RVECLP 计数。V0/V6 随有效行数变，V1 的有用前缀工作单位为 $n+\sum_{j=1}^{\lceil n/16\rceil}\min(n,16j)$，V3 含固定双三角逆。每 VF 在两种 tile 的逐队列计数间插值，以 S2 实发 VF 的起止差标定依赖开销，再按每周期开销和目标每 AIV 的 FP32 吞吐给新增执行记录定价。参考 issue 近似每周期 4 个 RVECEX 记录。S64 仅提供结构性计数，该次严格数值资格未通过，其设备或 VF 时长均未作为参数。快慢时延的来源可从参数中逐条核查，这是近似，缺少非线性 branch 的逐 PC 动态映射。[逐条资源观测和用途标签](dt-primitive-services.json)
5. 同一阶段按源码先后依赖串接，能并发的分支取 max。例如 Prepare 的 C4/C5 与 V6 可并发。C2 的三个队列满足 $E_{i,q}=\max(E_{i-1,q},E_{i,q-1})+c_{i,q}$，$i$ 是当前 band，$q$ 是 load、MMAD、FIXP。每个混合组补一次从该段首条指令到第一笔载荷 DMA 的前端周期，分别为 9,952、895、3,121；薄样本只约束这个启动事件。最终取最慢核的依赖时间，与共享带宽情景取 max，再除相应频率转成微秒。三段时长求和作为连续设备服务，宿主 dispatch 间隔与额外 kernel 另计。多头并发带来的交错、跨 tile 开始新工作的速率，以及部分 FP32 数据进入 MTE 前的等待仍缺定标。

## 具体算例

950DT 的 `[1,2,1,1,128,128]` 有一个两行尾块。Prepare/FwdH/Finalize 分别发 3/2/2 条 MMAD，FwdH 两次 ALU 维度是 $(16,128,128)$ 和 $(128,128,16)$，合计 1,048,576 ops。Finalize 读 $2\times512+32768+8=33,800$ 字节至 L1，FIXP 描述符处理 1,024 字节换成 512 字节 BF16 输出。预测三段 13.669、5.373、2.934 µs，完整 CA 15.281、4.739、3.221 µs。下图中 FwdH 的一部分 W 搬运提前到初态转换期间，直接按顺序估计会积累高估。Prepare 的 V0 在模型中结束较早，后续 V1/C2 也较早发起。小例本身不足以确定跨芯片启动差。[同轴图和误差分析](report.md) · [冻结记录](freeze.json)

在仓库根目录运行 `python3 development/validation/kda_resource23/audit.py --shape 1 2 1 1 128 128 --chip Ascend950PR_957b` 可查看全部中间值。芯片参数既可指定注册名，也可传本地芯片 JSON 路径。此命令复核历史候选。


### 在系统仿真器中调用

```python
from operator_model import estimate

out = estimate(
    "kda_chunk", [8, 32768, 12, 12, 128, 128],
    {"implementation": "fla_26_9_1_v2"}, dtype="bf16", chip="Ascend970DT",
)
print(out.status, out.latency_us, out.projection["qualification"])
for stage in out.projection["target_plan"]["stages"]:
    print(stage["family"], stage["device_us"], stage["active_aic"],
          stage["launched_aiv"], stage["max_core_head_chunks"],
          stage["issued_cube_ops_by_dtype"], stage["dma_bytes"])
```

这里的 B、S 指等长的逻辑请求数和每请求 token。相同物理 packed BSND 也可传 `[1,262144,12,12,128,128]` 并用 `cu_seqlens=[i*32768 for i in range(9)]` 指定序列边界。不同长度、未建模的 gate、状态、额外输出选项会返回 `unsupported`。CLI 使用同样的 `--attrs-json '{"implementation":"fla_26_9_1_v2"}'`。选定该合同时，DT、PR 和 970 的状态均为 `extrapolated`，在 `projection.qualification` 中标为 `experimental`；显式要求已校准策略会返回 `unsupported`。安装包只含 4,357 字节的 KDA 价格资料，运行时无需读取历史 trace。每阶段给出按格式计算的 Cube ops、可确认的 Vector lane ops 子集、RVEC 发起记录、各层传输计数与字节、最大核上的块数及设备微秒。芯片若提供严格 FP32 Cube 峰值，C4/C5 可使用它定价。[已安装 wheel 的仓外验证](installed-check.json) · [公共资源模型](../../operator_model/kda_fla_v2.py)
