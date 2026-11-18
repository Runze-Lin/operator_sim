# 用前序核覆盖解释暴露在关键路径上的时间

**适用范围。** 模型用于 Ascend950DT_9582 / CANN 9.2.1 的 BF16 RMSNorm（ops-nn，`aclnnRmsNorm`，`entry_kernel_1388`）与 N=4 带残差映射的 mHC Post（ops-transformer，`aclnnMhcPost`，少 token 的 `entry_kernel_5`、按 token 的 `entry_kernel_6`）。指令来源是本机安装包实际调用后的数据库；开放源码仅用于解释。目标是跨所有活跃 AIV 的首末指令时间差，不含 CPU 发射、等待及多算子间隙。本文不把 SCALAR 区间长度等同于硬件指令吞吐，不把 CA 结果称为实卡性能。

**从 shape 到工作量。** 令 T 为 token 数，D 为末维宽，W=ceil(T/64) 为最忙核行数。RMSNorm 取 J=1（整行），A=ceil(T/W)，SIMD 工作单位 U=W·ceil(D/128) 个向量分组，全芯片主输出 B=2TD/1024 KiB，2 是 BF16 字节数。Post 的 N=4，T≥32 取 J=1；T<32 从 J=1 开始，当 ceil(D/J)>256 且 T·J≤32 时翻倍。A=J·ceil(T/W)，d=ceil(D/J)，U=W·ceil(d/64)·N(N+1)：64 为每向量组列数，N(N+1) 为 N 路残差的组合及外层贡献。A 衡量本批工作分派范围；U 说明计算随最忙核中的工作而变，不能拿全芯片运算量直接当成串行时间。

**把同样的工作区分为不同的启动上下文。** H 表示该独立执行序列上，前序同类算子预测的已激活 AIV 高水位（整数核数，不读 CA）。H=0 对应首次发射；0<H<A 对应扩展到新核；H≥A 对应复用已有核覆盖。H 是对上下文的代理，不等于缓存内容，不追踪跨流、地址复用或逐核替换。系统仿真在同样的受控序列中可传 `attrs['resident_aiv']=H`，并在调用后更新 H=max(H,A)；换流、混合负载或可能驱逐时不应沿用该假设。**默认输入仅为 shape 和实现属性**，按前两批旧 trace（每族各 2×10 次发射）的前序核占用经验分布 P(H) 作边际化，不根据当前 shape/总时延查表。该分布不是用户线上负载的统计。

对每个阶段 j，在详细上下文 z 下估计 t_j(z)=b_(j,z)+s_j·X_j+e_j·(W−1)；模型时间为各阶段之和，默认返回 ∑_z P(z|A)∑_j t_j(z)。z 取上述首次、扩展、复用三类，由 A 和 H 判定。b 是暴露开销（µs），s 是每工作单位暴露时间，e 是多一行未掩盖的写回开销（µs/行）。X_SIMD=U，RMSNorm 的 X_MTE3=B；其余斜率设零。e 只在 RMSNorm 写回启用。模型预测 arrival、SCALAR、MTE2、SIMD、MTE3、other 六段。arrival 是全芯片首指令到最后完成核开始的偏移，不能再叠加整个发射延迟。提取时重叠区间按 SIMD→MTE3→MTE2→SCALAR→other 归入唯一阶段（arrival 在局部之前），使六段严格相加等于 CA makespan；这些暴露时间不是各队列原始 busy 时间。`profile.json` 的 `service` 单位随 X 而变；post_channel 的 SIMD 斜率接近零，只表示开发样本的每核工作集中，**不能据此声称更宽通道没有计算成本**。

**校准来源与边界。** 每族 23 条 v0.7–v0.9 开发 trace，队列分别拟合非负截距/斜率，用稳健残差 `soft_l1`（尺度 0.12 µs）；SCALAR、MTE2、SIMD、arrival 允许在三类上下文使用不同截距，MTE3/other 共享截距。系数是实现、重叠与 CA 共同决定的有效成本，非 ISA 峰值。在线只需小数组加权与标量运算；没有 trace 解析或 shape 缓存。RMSNorm 输入要求 T≤512、D≤8192 且 D 为 16 倍；Post 额外限 T≤256、N=4。未经过 A7/A8 或冷/热跨代验证。

**算例（冻结值）。** RMSNorm [1,5120]：A=W=1，U=40，B=10 KiB。未知上下文默认预测约 2.35 µs，H≥1 时约 2.19 µs。Post [1,4,4096]：列分 16 片，各 256 列，A=16，U=80。默认估计约 3.09 µs；已覆盖至少 16 核时约 2.27 µs。对应逐 shape CA 时间在 `rows.json`，两个模式均在 CA 之前冻结。
