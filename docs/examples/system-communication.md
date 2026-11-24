# 从捕获的 rank 路由计算通信时间

27 个任务只提供 950DT 芯片配置和 rank 0 的原生图。TP 节点记录了有序通信组与张量。EP 节点还记录每个目的 rank 的有效行、payload 和 scale 字节。模型用这些事实推导工作量；带宽、步延迟、网络并发和算法选择属于场景假设。单芯片 CA 无法给出跨设备整窗，DT/PR 两端的误差均为 N/A。

## TP 的 ring

记通信组大小为 P，本卡输入为 S 字节，输入元素数为 E。AllGather 环上轮转 P−1 次，本卡发送和接收各 (P−1)S 字节。SUM AllReduce 先 reduce-scatter 再 all-gather，步数 R=2(P−1)，单向网络字节 X=ceil(2S(P−1)/P)，本地归约工作 V=ceil(E(P−1)/P)。两个方向可以同时传。每次启动算 α 微秒，单向每卡可注入 B GB/s，网络有效服务时间为 X/(1000B) 微秒。ring 不是已识别的 HCCL 实机算法；分层拓扑或算法切换会改变这个时间。

每块 UB 留四个缓冲区域，分段可用字节 C=32 floor(UB/128)。AllGather 的本地并行度按 ceil(S/C) 个任务估计，AllReduce 按 P ceil(S/(PC)) 个任务估计，活跃 AIV 取芯片上限。给定片外字节读写 L 和每卡 HBM 带宽 H，算本地内存服务 L/(1000 H a/A)，其中 a/A 是活跃核数占比。用相同占比和对应精度的 Vector 峰值计算归约时间。总时间为 Rα 加上网络、本地内存和归约服务三者的最大值，再加调用方显式配置的 kernel 开销。用数据分段流水允许三个服务重叠；收到第一段的延迟和拥塞仍会影响真实网络。

以 Flash TP8 的 BF16 [9,5120] 为例，P=8，S=92160，E=46080，X=161280 B，V=40320 次加法。默认每卡 200 GB/s、每步 1 µs，网络部分是 14+0.8064=14.8064 µs；模型还比较此时的本卡读写、归约限制。带宽是用户可替换的有效注入假设，不是 950DT HBM 带宽。

## EP 的 prequantized Dispatch 和 Combine

按捕获的 EP 组定位本地序号 j。第 i 个目的 rank 的包字节数是 payload[i]+scale[i]。远端总发送与接收分别取所有 i≠j 的和，网络载荷 X 取较大者。把每个有流量的远端 peer 分配给一个 AIV，控制轮数为 ceil(活跃 peer 数/AIV 数)。这一近似允许多个 peer 同时交换。读取或写回的数据按有效路由行计价，预留 expert capacity 由系统显存预算单独管理。

Dispatch 的 FP4 激活已经带分组 scale，打包并根据 expert id 写到有效目标行。Combine 反向发送 token-FP8 payload 和 FP32 scale，随后按 gate 权重反量化并累加。对于收到的有效行数 r 和隐藏宽度 D，模型算 3rD 个 Vector 操作，分别对应反量化、乘权和加和。分核单位是有效行和行内 UB 分块。最终使用与 TP 相同的多服务重叠规则。

Flash TP8/EP8、B=1 的首层示例有 2 个本地有效 token，top-6。rank 0 的 12 个 dispatch 行全部发往本地，收到 4 个 rank 各 12 行。每行 FP4 payload 为 2560 B，scale 为 160 B。去掉自发自收，远端为 36×2720=97920 B。反向 Combine 的单行 FP8 payload 为 5120 B，scale 为 4 B，远端发送 36×5124=184464 B；本地收到的 12 行参与 3×12×5120=184320 个操作。实图显式声明预量化传输 ABI，这两个模型采用该边界，有库同名接口的单卡启动时间不足以校准端到端交换。

Python 注入时可使用 `FullOperatorSimBackend(fabric=FabricScenario(injection_gbs=200, step_us=1))`。两个参数分别改变通信带宽和控制步成本，并进入预测缓存签名。若任务显式提供链路拓扑，应为通信节点选择系统的拓扑后端。新的多机部署需要输入跨机带宽、拥塞和集体算法；本页参数仅用于这组缺少网络参数的 27 个配置。[模型源码](../../src/operator_model/system_routes/communication.py)

实现边界来自 CANNBot 的 Ascend 950 MC2 Dispatch/Combine 卡，以及卡对应的 ops-transformer 接口文档。它们描述专家分发/合并、有效 token 与 buffer 约束；本任务的每步 1 µs、200 GB/s 和 rank-local 并行假设由本版本单列。 [950 Dispatch 参考实现](https://gitcode.com/cann/ops-transformer/tree/be33d95bb755cedec3108f70805c1874d1792540/mc2/moe_distribute_dispatch_v2) · [950 Combine 参考实现](https://gitcode.com/cann/ops-transformer/tree/be33d95bb755cedec3108f70805c1874d1792540/mc2/moe_distribute_combine_v2)
