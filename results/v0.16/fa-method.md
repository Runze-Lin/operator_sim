# BF16 稠密 FA 的绝对资源模型

接口输入每卡 `[Q,KV,D,B,H]`。实际校准对象是 CANN 9.2.1 ops-transformer 的 `aclnnFlashAttentionScoreV2`，采用 SBH、非因果、无 mask/dropout、D=64/128、Q/K/V 头数相同。预测按目标芯片重新决定分核、每个阶段的服务和暴露时间。参数集的十个开发数据库均来自 950DT。所有设备走相同计算图，在线不读取 CANN、trace 或其他 SKU 的时间记录。[资源模型](../../operator_model/fa_resource.py) · [DT 输入文件、哈希与提取的阶段](fa-dt-evidence.json) · [校准脚本](../../development/validation/sku16/fa_dt_calibrate.py) · [冻结参数](fa.json)

## Tile 和波次

设 $C,V$ 为整卡 Cube 和 Vector 核数，初始任务数 $G_0=BH\lceil Q/128\rceil$。对于小于 $C$ 的初始网格，基本 Q 块 $M=64$，其余 $M=128$。满足前述小网格且 $KV>1024$，或满足 $D=64$、$Q$ 是 128 的倍数、$KV$ 是 256 的倍数时，基本 KV 块 $N=256$，其余 $N=128$。后一个条件用模运算表示为 $Q\bmod128=KV\bmod256=0$。这是 950 host regular basic 路由在所列输入子域上的规则。工作单元数 $G=BH\lceil Q/M\rceil$，同时占用 $A=\min(C,V,G)$ 个 AIC 和 $\min(2A,V)$ 个 AIV，需 $\lceil G/A\rceil$ 波。相同的满波合并计价，每波用剩余真实 Q 行的最长任务计时。对 C:V 改变的芯片，假设实现能重新分配最多两个 Vector worker 给一个 AIC，并取得平衡。在 1:1 芯片上执行同一算法属于重编译情景。

## 工作与资源

以下计算单位是目标时钟周期，最后除以 $1000 f$ 得到微秒，$f$ 是 GHz。令 $m,n$ 是当前 tile 的实际 Q/KV 长度，$d=D$，带帽表示向上对齐 16。此处 $u=\lceil n/128\rceil$ 是一个 KV tile 的 MMAD 分段数，两个乘法各需要 $u$ 段。$r=\lceil m/\min(2,V/a)\rceil$ 是本波每个繁忙 AIV 的保守行数，$a$ 为本波实际 AIC 数。

| 服务 | 每 tile 的工作与价格 |
|---|---|
| QK 与 PV | 各 $2mnd$ 有效 op，各 $F=2\hat m\hat n\hat d$ 填充后算术 op，每次乘加计两次。$t_Q=F/P_C+u h_Q$，$t_P=F/P_C+u h_P$。$P_C$ 为整芯片 BF16 TOPS 除以核数和时钟换算成 op/cycle/core。$h_Q,h_P$ 来自第一块 QK、PV 各自 MMAD 起止跨度，扣去该块峰值服务再除指令分段数。|
| score 与 PV 更新 | 每块 score 暂计缩放、求最大、指数差、累加、归一相关的五项等价工作，输出累计暂计每行每通道三项，暴露服务 $s_1=5rn/P_V$、$s_2=3rd/P_V$。$P_V$ 由目标每核 FP32 峰值乘以 DT 校准的效率。DT 记录的 SIMD 忙时含特殊函数、归约及发射成本，等价工作量不等于实际 ISA 条数。|
| GM/L2 到核内 | Q 每 Q 任务只读 $2md$ B，K、V 每 KV tile 各读 $b=2nd$ B。单核流式速度 $R=\min(R_{local},B_{MTE2},B_{L2}/a,B_{HBM}/a)$，单位 B/cycle。未知 L2 或显式 MTE2 端口限额时，继承 $R_{local}$（80.07 B/cycle）的上限。HBM 使用目标显存介质的整卡规格。在多个 Q 任务间默认 K/V 各自冷读，跨核命中会减小真实外存流量。|
| L1→L0 与 FIX | 每块近似传 $b_1=2[2\hat n d+\hat m(d+\hat n)]$ B 的操作数，包括 Q/K 与 P/V。FIX 近似处理 $b_F=4\hat m(\hat n+d)$ B 的 FP32 结果。带宽 $R_1,R_F$ 来自 DT 长循环相应流水的非排除指令忙时。可用每核 MTE1 GB/s 显式覆盖 $R_1$。|
| 输出写回 | 最后一块写出 BF16 输出 $2md$ B，两份 FP32 [m,8] 统计量共 $64m$ B。$R_W$ 计可用 MTE3 每核及 HBM/L2 全卡带宽，未知 MTE3 时使用 DT 局部流速。 |

## 首轮、稳态与尾块

首轮先等待 Q 与 K，随后做 QK、Vec1，最后等待 V 后执行 PV。以 Q 首次发起为起点，$L$ 表示首读固定延迟，$g_K,g_V$ 表示 Q→K、K→V 首发差值，$g_0$ 表示读就绪至首 MMAD 开始的间隙，$h$ 表示 QK/Vec1 与 V 都就绪后仍暴露的控制及交接周期。定义

$$t_{QK,start}=\max(L+2md/R,\ g_K+L+b/R,\ (2md+b)/R)+g_0,$$
$$t_{PV,start}=\max(g_K+g_V+L+b/R,\ t_{QK,start}+t_Q+s_1)+h,\qquad T_{fill}=t_{PV,start}+t_P.$$

两个初始请求可以重叠，同一端口的累计 payload 仍受带宽约束。填充时的等待和后续发起两个 tile 的间隔可以不同。稳态使用

$$I(n)=\max(t_Q+t_P,\ s_1+s_2,\ 2b/R,\ b_1/R_1,\ b_F/R_F,\ Ju/2).$$

$J$ 是 DT 32k 的两条 256-token K/V 流中段较大间隔，约 853.40 周期。物理上它含当时的发起、缓冲复用和同步节拍，这三个份额不能由同一条 trace 唯一确定。外推采用继承该有效间隔的假设，128-token 分段按一半计价。对有目标证据的芯片，可分别设置 `details.fa_issue_cycles`、`details.fa_read_latency_cycles`、`details.fa_mte_bytes_per_cycle`。这些字段覆盖相应的绝对单位，当一个带宽或算力项成为更慢的资源时，$I$ 会按该资源重新定价。

令 $K=\lceil KV/N\rceil$，末轮有效长度 $n_t=1+(KV-1)\bmod N$。第一块用 $n=\min(KV,N)$，当 $K>1$ 时加上 $(K-2)I(N)+I(n_t)$。排空取 $s_2+h_D+(2md+64m)/R_W$，$h_D$ 为末条 PV 完成到全核结束的跨度扣除上述向量和写回后的剩余周期。所有波累加，首次 Q 发起前的 launch 周期整次调用只加一次。读延迟、发起间隔、计算服务和暴露控制残差在资源模型中分别可见。预测只处理有限种首尾 tile 和工作波，复杂度不随 KV 或头数线性增长。

## 算例与证据边界

DT 的 [9,32768,64,1,1] 使用 1 AIC、2 AIV，Q 只搬 1152 B，128 轮 K/V 各搬 32768 B，共 512 条 MMAD。每个满块两次乘法各有 294912 有效 op；按 16 行填充后，各定价 524288 op。预测分为启动 6.479 µs、首轮 3.877 µs、127 次续发 65.686 µs、排空 0.334 µs，合计 76.377 µs。该 DT 开发库实测 75.980 µs。若改为 KV=16385，模型计 64 个满块和 1 个单 token 块，末次 K/V 各搬 128 B，末块每种乘法各一段。在 [64,2048,128,1,64] 上分别降低 Vector 峰值、Cube 峰值和显存带宽，独立测试检查瓶颈资源变化确实提高预测时长。[资源敏感性与尾块测试](../../tests/test_fa_resource.py)

校准分支由 host 的 KV 块模板（128 或 256）选择。首尾参数来自相应 DT 模板下各条指令边界的中位数；256 模板只有 4k 与 32k 两条开发记录，不能推断控制周期在其他微架构完全相同。由短期最快 8 回合 K/V 发起窗口得到的局部流速包含实现影响，它只代表当前代码观察到的有效速度。多个 wave 间同一 kernel 的 overlap、不同 head 的 cache 命中、GQA 共享 KV、因果裁剪、专用 FIA/KDA、同一 wave 不均匀向量绑定及模型外的 KV 存储格式均未在本校准中测定。若目标缺 BF16 Cube 或 FP32 Vector TOPS，沿用 DT 的每周期每核能力假设。

本机复用已解析的 ChipConfig，通过 resource_fa 直接查询各一千次。[9,32768,64,1,1] 中位数 0.034 ms、P95 0.039 ms、最大 0.337 ms；[2048,524288,128,1,144] 中位数 0.049 ms、P95 0.055 ms、最大 0.175 ms。此口径不含公共 API 的输入校验、工作量统计与返回对象构建。
