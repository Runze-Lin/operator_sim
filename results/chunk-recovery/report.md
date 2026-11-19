# Chunk KDA 的 CA 计时与事件诊断

| 公开 ACLNN 接口 | 950DT 完整数值且可计时 | 950PR 完整数值且可计时 | 双端耗时 / WAPE / 最大误差 | 970DT |
|---|---:|---:|---|---|
| `aclnnChunkKdaFwd` | 0 | 0 | N/A | 仅列结构工作量，时延 N/A |

## 为什么 Cube 在第一次 MMAD 之前就报错

复用 BF16 `[B,S,Hq,Hv,K,V]=[1,64,2,2,128,128]`、`chunkSize=64` 的 PR 旧 trace。安装包、PR 捕获二进制 `.text` 哈希一致，设备实际执行的是 `ChunkKdaFwd_c7537eac07b2e805494bc8b22c2da1a4_2_mix_aic`。在 `core0.cubecore0` 上，三种 event ID 各有两次连续的 Cube → MTE1 SET。第二次发旗前，同一 AIC 上没有对应消费该旗的 WAIT，已有的 577 条退休记录里没有 MMAD。同合同 DT 也报告了相同的三个 PC。[设备观测及六个 PC](../v0.21/chunk/pr-flag-audit.json) · [符号及发布源码核对](source-audit.json)

| 本地事件 ID | 第一次 SET 周期 | 第二次 SET 周期 | 两条指令在 `.text` 的偏移 |
|---:|---:|---:|---|
| 0 | 15,781 | 32,080 | `0x97018` → `0x99688` |
| 1 | 15,784 | 32,081 | `0x97030` → `0x99690` |
| 2 | 15,787 | 32,085 | `0x97044` → `0x996b0` |

设备代码支持如下解释。950 的 `TPipe::Init()` 在 AIC 上动态分配并 SET 0、1、2 三个 M→MTE1 事件，分别用于矩阵宏及 bias 数据搬入。`DestroyWithoutPipeAll()` 才会对三个事件逐个 WAIT 和释放。KDA 专用分支先创建外层 `TPipe`，传给 `RunFrontEnd` 和 `RunChunkKdaPrepare`。prepare 在多处 Cube score 路径创建局部 `Catlass::Arch::Resource`。这个类的首个成员又是 `TPipe`，C++ 会先构造成员，再进入构造函数体调用 `pipe.Destroy()`。外层的事件仍然有效，内层成员会在任何释放动作之前再次 SET 同组事件。库的 debug 版 `TPipe::Init()` 自带“当前 TPipe 指针必须为空”的断言。[发布实现](https://gitcode.com/cann/ops-transformer/tree/8fbf1ffea0cf0c401407fdf202743a75e0a427b7/attention/chunk_kda_fwd/op_kernel) · [CATLASS 资源定义](https://gitcode.com/cann/catlass/blob/769cd40a8716b28650b6bebb08db4834eea4462f/include/catlass/arch/resource.hpp) · [DAV_3510 TPipe](https://gitcode.com/cann/asc-devkit/blob/56500ea0dc219380f50f4a17ce73ef5264f830fc/impl/basic_api/dav_3510/kernel_tpipe_impl_c310.h)

CANNbot 的 950 稳定版 `TPipe`、`Destroy` 两张 API 卡明确给出寿命约束，在同一时刻只能有一个有效的 TPipe。若还需创建一个，应先释放前一个。同步状态异常符合这条公开约束。[Ascend C 构造函数约束](https://gitcode.com/cann/asc-devkit/blob/56500ea0dc219380f50f4a17ce73ef5264f830fc/docs/zh/api/SIMD-API/basic_api/resource_management/TPipe/TPipe_constructor.md)

两组指令都包含分配器的 `SFF0`、`SBITSET1` 和寄存器驱动的三次 `SET_FLAG.M.MTE1`。在程序顺序上，第二组后面的 `0x99b04 / 0x99b08 / 0x99b0c` 正是连续三条 `WAIT_FLAG.M.MTE1`。退休流只证明报错前的执行；后面三条是静态反汇编。二进制缺少 `.debug_line`，无法把第二组唯一定位到某个内联的 `Resource` 变量。事件重复和所属设备二进制已实证，嵌套 `TPipe` 是与指令模式、生命周期都吻合的源码归因。

## 核数、双缓冲与布局对照

当前配置的 DT 为 32 AIC/64 AIV，PR 为 28/56，两者都能满足该安装包 MIX 要求的 1 AIC 配 2 AIV。宿主按 `min(可用 AIC数, max(1, 请求数 × V头数))` 分块，固定长度用 batch，变长用序列数。旧样例在两端都发出 2 个块。相邻 head 配对、score 双缓冲和跨核生产消费会改变 MMAD 发起顺序与等待时长；本次确认的 0、1、2 是同一个物理 AIC 的 M→MTE1 本地事件。AIV 上的 `gatePipe` 只在 `ASCEND_IS_AIV` 的分支内构造，无法直接在 AIC 写这组本地事件。970DT 的 1:1 微架构需要独立实现和分块假设。

另测 `[1,3,1,1,16,16]`，Q/K/V 改用 FP16，gate、beta 保持 FP32，`chunkSize=128` 进入 generic key 1。保持 BNSD 连续布局，省去前置序列转置和 VK 状态转置，不传初态、不请求最终态。只申请 `attnOut`、`Aqk`、`Akk`，分别与独立递推和显式三角计算逐元素比较。登记早于采集。DT 核名元数据、AIC 和两个 AIV 的 wrapper 入口偏移 `0x127800` 相符。前六个新报错 PC 在 FP16 库 ELF 上均解码为 `SET_FLAG.M.MTE1`，与旧 BF16 不同地址，印证新分派也存在本地事件异常。[注册](generic-fp16-c128-3x16-registration.json) · [运行时入口核对](generic-fp16-c128-3x16-dispatch.json)

**FP16 探针终态。** 运行时确认为 950DT_9582。到登记的 25 分钟上限后停止，实际墙钟 1,502.408 秒，原始文件共 167,279 B。模拟器在 11 分 25 秒首次报错，此后继续执行，累计 116 条重复 SET，覆盖 25 个不同 PC；全部落在 AIC 入口段，25 处均解码为 `SET_FLAG.M.MTE1`。唯一的主 kernel DB 为 `0000_entry_kernel_1.db`，终态 `KernelInfo` 为空、已提交指令为 0。宿主没有返回逐元素结果，48 个注意力输出及各 384 个三角输出的正确性均未证实。该次采集不计入耗时样本或独立样本，不用 1,502 秒的墙钟代设备时间。没有追加 PR 长采集。[终态回执](generic-fp16-c128-3x16-receipt.json) · [25 个 PC 的逐项反汇编](generic-fp16-c128-3x16-dispatch.json)

## CANN 兼容版本

比较 9.2.1 与 9.3.0-beta.1 对应标签下 37 个 KDA host/kernel C++ 源文件。31 个文件字节有变化，保留预处理指令界限、字符串和运算符后，唯一的代码 token 差异是 host 的一条报错文案。安装包自带的 24 份 KDA 设备源码与 9.2.1 tag 对应文件逐字节一致。9.2.0 的整个 KDA 子树也与 9.2.1 哈希一致。源码默认的 CATLASS 依赖提交都为 `769cd40a8716b28650b6bebb08db4834eea4462f`。9.3-beta 的 DAV_3510 `TPipe` 整份头文件与已安装 9.2.1 字节相同。源码对比未覆盖 beta 的编译器、CA 引擎、发布二进制或构建时外部依赖覆盖；目前证据不足以认定切换 beta 包会修复本例。正式校准需同合同在可识别的发布 MIX 二进制上数值通过，并取得完整窗口。如使用供应商修复版本，应单列其库、CA、编译器版本和 `.text` 哈希。[9.2.1 发布树](https://gitcode.com/cann/ops-transformer/tree/v9.2.1/attention/chunk_kda_fwd) · [9.3-beta 发布树](https://gitcode.com/cann/ops-transformer/tree/v9.3.0-beta.1/attention/chunk_kda_fwd) · [复核数据](source-audit.json)

还检查了 2026‑11‑19 官方主线 `6f665150`。主线包含第二页 K 分块的 L1 B 双缓冲修复，但修改位于 `chunk_kda_fwd_prepare.h`；950 的 `__CCE_AICORE__ == 310` 分支直接包含另一个 `arch35/chunk_kda_fwd_prepare_arch35.h`，和 beta 的对应文件哈希相同。已核对的主线修改不能解释本例第一次 MMAD 之前的三个本地事件。其余相关改动包括 host 报错和空指针处理。[主线差异](https://gitcode.com/cann/ops-transformer/tree/6f66515097634b873bed26f8c8ea7cbe180f8c9b/attention/chunk_kda_fwd/op_kernel)

## 复核和使用边界

在同一 CANN 环境可使用下面的工具重读历史 trace 和两个已抓取的标签。在新 checkout 中，需先把 `ops-transformer` 的 v9.2.0、v9.2.1、v9.3.0-beta.1 标签和上述主线快照（ref 名 `upstream/master`）放在 `runs/chunk-recovery/upstream`，`asc-devkit` beta 的对应头文件放在 `runs/chunk-recovery/asc-devkit`，并恢复机内的原始 PR dump。`dispatch.py` 另需新探针的 `core0.wrapper_log.dump`。系统仿真的结构预测完全离线。[逐算子算例及诊断图](../../docs/examples/kda_chunk.md)

```bash
python development/validation/chunk_recovery/source_audit.py
python development/validation/chunk_recovery/dispatch.py
```
