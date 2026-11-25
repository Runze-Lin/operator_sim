# FA · 冻结后 Vector 高需求样例

数据合同为 CANN 9.2.1 的 aclnnFlashAttentionScoreV2，稠密非因果 SBH，BF16 Q/K/V 和输出，FP32 累积，B=H=1。这三个几何在 v2.1 的归档窗和新增验证中未出现。以旧版预测的中间满页 Vector 需求为选择标准，取 Q=60、63、64 三种行数、D=64/128 和半页尾或整页尾。所有 shape 和两种 950 的全部预测应先写入同一个冻结文件，再执行采集。970DT 同形预测列为外推。

```sh
# 只复算已提交的预测和收据（无需 CANN）
python3 calibration/examples/fa_vector_holdout.py evaluate
python3 calibration/examples/fa_vector_holdout.py evaluate --long

# 如需重新采集，先准备 v2.1 的独立 import 目录，在新目录写入冻结数据。
fa_run_dir=$(mktemp -d)
mkdir "$fa_run_dir/old"
git archive 651cd72 src | tar -x -C "$fa_run_dir/old"
fa_args=(--output-dir "$fa_run_dir/records" --raw-dir "$fa_run_dir/raw")
PYTHONPATH=src python3 calibration/examples/fa_vector_holdout.py freeze "${fa_args[@]}" --v21-src "$fa_run_dir/old/src"
python3 calibration/examples/fa_vector_holdout.py collect "${fa_args[@]}"
python3 calibration/examples/fa_vector_holdout.py evaluate "${fa_args[@]}"
```

收集器使用本机既有的 CANN 9.2.1 环境入口 scripts/cann-run.sh 与 build/smoke/three_ops。其他机器可预先激活 CANN 环境；脚本在本地包装器缺失时直接调用 PATH 上的 msopprof。可按[上版冻结运行说明](../../v2.1/new-ca/README.md)从归档源码编译同一个 BF16 host。原始 trace 使用 runs/fa-vector-v22/ 下的每窗独立目录，每个进程预算 360 秒、1100 MiB，同时最多两个进程。窗口资格要求目标 SKU 与内存模型一致、一个实际 FA kernel、attentionOut 的每一个元素均对上 host 参考、各核 END 齐全、每核起止与 profiler 的 duration 表相符。核对 trace 的 MMAD 精度、计数和实际核数，保留资产哈希。

VF 的观测口径是对每个物理 AIV 合并正时长 RVEC/RV_* 指令执行区间，然后取所有 AIV 中最大的并集时长，按 1.65 GHz 换算周期。向量阶段估计取最忙 AIV 的首张、续张和末张页 Vec1 与 Vec2 服务的和。这个对照不包括 SIMD 等待和跨引擎空闲。整算子另取所有核最早有效时刻到最晚退休的跨度。冻结报价见 [predictions.json](predictions.json)，原法、v2.1 类别法和本轮类别法逐形结果见 [comparison.json](comparison.json)。

## 补充的长稳态试验

首批两组 DT 完整窗的忙核 RV 并集占整窗约 12% 和 16%。为减少固定启动成本的遮蔽，按同一冻结模型追加 [64,3072,64,1,1]，两端各 1 窗，采集之前独立保存[补充预测](long/predictions.json)。选取时 DT 的新版预测忙核 VF 服务 / 整窗为 41.46%。这两条是根据首两条的观测自适应选择的独立机制核验，统计在[独立补充表](long/comparison.json)。复算用相同命令加 `--long`，每个 CA 的预算相同。首批采集与补充采集各自最多使用两个进程。

本链路的时间线包含正时长 RVEC* 指令，也包含 PUSHQ 队列中的 VF 描述符。每个核的 VF 描述符区间和 RV 指令区间分别合并保存；各核同一指令类型的零时长发起标记不计占用。若现场导出中缺少 SIMD 队列标识，其原口径给 N/A。原 BF16 等效 VF 在 DT32Ki 的参考是更宽的 SIMD 并集；直接相对 RV 执行并集计算的差值度量相同软件阶段的服务偏差，也包含观察边界差异。

## 冻结结果

下表均为实际库 kernel，前六行为首批，最后两行为独立登记的补充几何。整窗时间为 µs，忙核执行与描述符并集为 cycles。

| 芯片 | Q×KV×D | CA 整窗 | 原方法 | v2.1 类别 | v2.2 类别 | RV 执行并集 | VF 描述符并集 | RV / 整窗 | 970DT 类别法外推 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DT | 60×448×64 | 9.885 | 10.959 | 10.080 | 9.877 | 1962 | 2831 | 12.0% | 9.871 |
| DT | 63×640×128 | 11.121 | 13.481 | 11.315 | 11.232 | 2931 | 3424 | 16.0% | 11.489 |
| DT | 64×768×64 | 10.662 | 13.019 | 11.217 | 10.806 | 3196 | 4162 | 18.2% | 11.470 |
| PR | 60×448×64 | 9.512 | 11.017 | 10.080 | 9.877 | 1962 | 2767 | 12.5% | 9.871 |
| PR | 63×640×128 | 10.945 | 13.614 | 11.315 | 11.232 | 2931 | 3378 | 16.2% | 11.489 |
| PR | 64×768×64 | 10.335 | 13.149 | 11.217 | 10.806 | 3196 | 4086 | 18.7% | 11.470 |
| DT | 64×3072×64 | 16.963 | 26.972 | 18.809 | 17.693 | 9549 | 10976 | 34.1% | 22.068 |
| PR | 64×3072×64 | 16.673 | 27.433 | 18.809 | 17.693 | 9549 | 10929 | 34.7% | 22.068 |

8 次采集的进程墙钟相加为 712.73 s，原始目录合计 1304.66 MiB。首批 DT/PR 新类别法的整窗 WAPE 为 0.83% / 3.65%，对忙核 RV 并集的差异为 15.56% / 15.56%。较长稳态的新形在两端都高估 RV 26.74%。
