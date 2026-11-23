# DynamicMxQuant · BF16 → packed MXFP4 E2M1

Kimi-K3.1 的 128k decode 模拟账单中，单卡 batch=1、TP8/EP32、主模型每次校验 9 个 token，`NpuDynamicMxQuant` 的 `[9,128]` 输入在一轮 target 中出现 69 次。捕获的该张量是行步长 6,288 的视图。下面用同一维度的连续输入展示模型和两端独立 CA；给实际视图传 `attrs={"input_stride":[6288,1]}` 会返回工作量、设备时间 N/A。入口固定连续 BF16、末轴 32 元素一组、`round`、OCP (`scale_alg=0`)。账单给出了两张输出的形状与 dtype，系统的 `quantize_mxfp4` 设置了分组和舍入。OCP 是本例对系统未记录属性的显式选择。系统账单的时间来自系统模拟器；下文写作 CA 的时间均来自独立原生派发。

```python
from operator_model import estimate

e = estimate("quant_mxfp4", [9, 128], {"group_size": 32, "round_mode": "round"},
             dtype="bf16", chip="Ascend950DT_9582")
print(e.status, e.latency_us, e.active_aiv)      # extrapolated, 4.129040…, 9
print(e.work["packed_shape"], e.work["scale_shape"])  # [9, 64], [9, 2, 2]
```

此入口接受 1–7 维，末维为 ≥32 的偶数。系统如提前 padding，传本次算子实际收到的 shape，padding 自身另算。原 `quant` 仍为 BF16 → MXFP8 E4M3FN。FP4 输出精度与循环独立核算，当前阶段价格的一部分迁自 950 的 MXFP8 数据。三种芯片均返回 `extrapolated`，`timing_mode="calibrated"` 返回 `unavailable` 并保留工作量。

## 从一行数据算到整窗

1. **接口字节与数学工作**。将前导维相乘得行数 R，保留实际行宽 C，X=RC，G=R×ceil(C/32)。读 2X 字节 BF16，写 X/2 字节 packed E2M1 和 2R×ceil(C/64) 字节 E8M0。最后一组仅有半对时，scale 张量仍保留补齐槽。对本例，这三项为 2304、576、36 B，G=36。按单元素归属统计，指数提取、组内最大值比较、缩放乘法、转换四项合计 X+(X−G)+X+X=4572 次。位域和转换归入 BF16 输入宽度；每类速度可以不同。
2. **分核与 UB**。host 先按 512 列片计任务；若 R×ceil(C/512)≤0.75×AIV 核数，则改用 256 列片，再枚举行列切分、优先减小尾块空洞。本例 9≤48，一个 AIV 分到一行 128 列，9 个 AIV、0 个 AIC。UB 容纳的 block32 组数为 8×floor((UB−2048)/(256×5+8×2×5))。分母首项来自双槽 BF16 输入和 packed 输出，每个逻辑元素合计 5 B；第二项是最大指数、scale、倒数缓冲的预算。950DT 的 253952 B UB 可放 1480 组，本例每核只需一块。列块按组数补偶，小组和大行均以此规划。
3. **Vector 工作**。某 UB tile 的行数 h、有效宽 w，对应补偶组数 g=h×2×ceil(w/64)。BF16 OCP 源码每处理 256 个输入，求块内最大指数的内层有 6 个 Reg 调用，缩放、cast 并打包的内层有 9 个；每处理 128 个 scale，生成 E8M0 和倒数的内层有 16 个。因此内层源码调用数为 15×ceil(g/8)+16×ceil(g/128)。本例 h=1、w=128、g=4，得到 31 个源码调用。发起成本假设为 950DT 每周期 1 调用，其他芯片按「16 bit Vector 峰值 / AIV 核数 / 频率」相对 DT 伸缩，峰值未知则保留 1。加 23 周期依赖，共 54 周期。本口径排除了循环外初始化、标量控制，也没有假设一个源码调用固定编译为一条指令。
4. **搬运与调度**。DT 单块的 MTE2 服务为 1029.740+256/48.363=1035.033 周期，发起间隔单列为 256/68.913=3.715 周期。MTE3 合并价格为 242.510+81.000+(64+4)/92.027=324.249 周期。上述「固定延迟 + 字节数 / 每核字节吞吐」来自 950 参考 quant 的读响应、发起、写服务和 scale 暴露尾部，在这里是迁移假设。多块按读→Vector→写排序，同资源与双槽复用施加等待。实现用 max-plus 的重复平方法处理长循环。全芯片还计算共享外存和 L2 约束，只有超出核内流水的部分增加总时长。此例共享时间均可隐藏。[原始价格](../../src/operator_model/data/parameters/quant_resources.json) · [目标芯片覆盖值](../../src/operator_model/data/chips/Ascend950DT_9582.json) · [多核流量规则](../../src/operator_model/scheduling/memory.py)
5. **合成设备时间**。从所有核的最早指令计时，套用同 SKU quant 首读参数 4686.5 周期，加各核到达跨度 219.921×log₂(9)=697.134 周期，末尾另有 16 周期。总计 (4686.5+697.134+1035.033+54+324.249+16)/1650=4.129040 µs；1650 是 1.65 GHz 对应的 cycles/µs。多块使用该芯片的 prefetch 首发价。系统配置的 `kernel_overhead_us` 只进入 `scheduled_us`。[FP4 调度](../../src/operator_model/operators/quant/mxfp4.py) · [固定版本 host 划分](https://gitcode.com/cann/ops-nn/blob/602e4ad985da00aad28c16432841e9c5069e10de/quant/dynamic_mx_quant/op_host/arch35/dynamic_mx_quant_tail_axis_tiling_arch35.cpp) · [FP4 内层循环](https://gitcode.com/cann/ops-nn/blob/602e4ad985da00aad28c16432841e9c5069e10de/quant/dynamic_mx_quant/op_kernel/arch35/dynamic_mx_quant_tail_axis.h)

## 同一设备零点下的实测

安装包 CANN 9.2.1 的 `aclnnDynamicMxQuant` 发起一个 `DynamicMxQuant_bf16toe2m1_high_performance_33_mix_aiv`。两端加载的 ELF 均与各自安装目录的 FP4 对象 SHA256 一致，均有 9 个核和完整 END。输入包含逐组正负 FP4 值与可变 2 的幂；独立 golden 对比了全部 1152 个 nibble 和 36 个 scale，均通过。以下每个区间表示最后完成核 `core17.veccore0` 上的首发与末退休，原点取所有核最早指令，单位 µs。[紧凑冻结及收据](../../calibration/reference/quant-mxfp4-smoke.json)

| 端点、阶段 | 950DT | 950PR |
|---|---:|---:|
| 256 B 输入 MTE2 描述符 | 2.563–3.026 | 2.571–2.999 |
| RVECEX 区间，均为 64 个事件 | 3.040–3.121 | 3.013–3.094 |
| 64 B packed 数据 MTE3 描述符 | 3.136–3.461 | 3.109–3.368 |
| 4 B scale MTE3 描述符 | 3.137–3.902 | 3.110–3.853 |
| 全核 CA / 冻结模型 | 3.912 / 4.129 | 3.862 / 3.836 |
| 有符号相对误差 | +5.54% | −0.69% |

模型在 DT 将末核输入发起置于 (4686.5+697.134)/1650=3.263 µs，比 CA 晚 0.700 µs。模型写服务为 0.197 µs，CA 两个输出的描述符从首次发起到最后退休跨 0.766 µs。两种偏差部分抵消。系统调用者若只看整窗，会高估当前模型区分读、算、写的能力；描述符区间本身含等待，不能用 4 B 除以整个窗口推断写口带宽。本例两端首轮 CA 分别耗费 54.0 s 与 51.7 s。离线解析另计。

## 冻结后的三组追加验证

在相同模型和探测器上预先冻结了六个新窗口。各端分别报告冻结预测 / CA（µs）、测到的 AIV；四种 shape 的分核数在两端均命中。下面不含首轮 [9,128]。[追加冻结、数值与全核收据、逐端终核事件和预算](../../calibration/reference/quant-mxfp4-followup.json.gz)

| BF16 shape | 为什么选 | DT，预测 / CA，核数 | PR，预测 / CA，核数 | 970DT 外推 µs |
|---|---|---|---|---:|
| [9,7168] | Kimi B1/TP8；宽行被多个核平分 | 4.556 / 4.453，63 | 4.225 / 3.760，35 | 4.176 |
| [9,288] | V4.1-Flash B1/TP8；256+32 列，尾片只有一个 block32 | 4.266 / 5.034，18 | 3.977 / 4.890，18 | 3.910 |
| [576,128] | Kimi B64；多行在有限核上划分 | 4.572 / 4.390，64 | 4.271 / 4.193，53 | 4.266 |

最大误差出现在 [9,288]，DT 低估 15.26%，PR 低估 18.66%。实际最后完成的是 32 列尾核。仅 2 B 的 scale 描述符在 DT 的窗口是 3.279–5.024 µs，在 PR 为 3.201–4.880 µs；另一份 16 B packed 写几乎同时发起，早约 1.6 µs 退休。尾核各有 808 个 RVECEX 事件；源码中奇数个 block32 时还需将整个输入 UB 缓冲区填零并等 V→MTE2。模型目前用主块调度和聚合写回定价，遗漏了尾核清零与晚退休。这些观察尚不能从描述符窗口分离存储事务、排队与同步的各自占比。

首轮与补充合计，每端 4 个独立 shape；DT WAPE 7.14%、PR 8.87%。未按这些结果重新调参。追加实验同时最多两路，从首例启动至末例完成共 272.2 s，六个进程墙钟相加 360.0 s，均在各自 120 s 限时内成功。真实场景的输入规模仍大于此组 CA 覆盖。

四个独立 CA 几何中，[9,7168] 与 [9,288] 在捕获中确有连续输入。Kimi 的 [9,128] 与 [576,128] 在这些 run 中均为非连续视图，连续 CA 只验证同形的紧凑布局。系统可传元素步长 `input_stride` 核查，省略时默认连续。[全部捕获 shape 和匹配结果](../system-quant-shapes.md)

## 系统中更大的实际输入

| 真实模拟账单的单卡 BF16 输入 | 每轮 target 次数 | 逻辑读 / 写（B） | 950DT / 950PR / 970DT 分析外推（µs） |
|---|---:|---:|---:|
| DeepSeek-V4-Pro B16 · 16×32770×512 | 30 | 536903680 / 142615040 | 207.88 / 469.81 / 100.48 |
| Kimi-K3.1 B16 · 16×131081×576 | 24 | 2416084992 / 641772576 | 1290.22 / 2168.63 / 632.77 |

两行均为系统模拟的 128k decode 配置。量化的输入包括该调用完整处理的张量；只把当前校验 9 个 token 代入会丢失缓存物化产生的大张量。此处缺大规模 FP4 CA，瓶颈还受 L2 驻留、读写竞争、每核轮次影响；970DT 的片内成本沿用 950 默认场景。

## 重放检查

紧凑档内保存全部冻结输入、定价、核 roster、END 与退休 tick、各物理 shard 指纹、加载 ELF 指纹和逐输出校验数。以下代码在仓根重算 8 个全核窗口并验证其收据与冻结关联。

```python
import gzip, json
from pathlib import Path
from calibration.__main__ import ingest

root = Path("calibration/reference")
for name in ("quant-mxfp4-smoke.json", "quant-mxfp4-followup.json.gz"):
    data = (root / name).read_bytes()
    doc = json.loads(gzip.decompress(data) if name.endswith(".gz") else data)
    for row in doc["observations"]:
        assert ingest(doc["freeze"], row["receipt"]) == row["measurement"]
```

离线探测器使用 CANN 的标准逻辑 FP4 descriptor，底层缓冲按两值一字节分配。输入每组最大绝对值设置为 6×2 的已知次幂，对比正负两条 nibble 与有符号组指数。使用现有校准清单冻结 rank-2 输入及全部 `Y`、`scale` 个数后，调用 `python -m calibration.quant_mxfp4 --help` 查看采集命令。探测器按 Matmul 编译示例将源文件换成 `calibration/quant_mxfp4.cpp`；仅有模拟器的开发机可给链接器补 `-Wl,-rpath-link,"$ASCEND_HOME_PATH/aarch64-linux/devlib/linux/aarch64"`。[数值探测器](../../calibration/quant_mxfp4.cpp) · [全核计时校验](../../calibration/quant_mxfp4.py) · [构建示例](../../calibration/mx_matmul/README.md)
