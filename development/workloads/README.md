# Decoder 假跑与 MoE 路由校准

模型假跑使用项目锁定的 HF config 以及 vLLM-Ascend Python 源。叶调用只接受符号张量。真实路由 CA 使用 CANN 9.2.1 的单算子 API。两套证据的重现方法如下。[模型覆盖分析](../../docs/workload-coverage.md) · [路由算例](../../docs/examples/moe_gating_top_k.md)

## Decoder 调用路径

```bash
python3 development/workloads/decoder_probe.py --fetch
```

有网络时 `--fetch` 将固定源码放入 `runs/workloads19/sources`；已下载时可离线重放。自选缓存目录用 `--source-cache`，输出目录用 `--output`。项目不加载外部 checkpoint 或执行远端仓库的初始化逻辑。

## 数值与性能

从项目根目录在已安装 CANN 的主机编译数值检查器。运行时对 FP32 logits+偏置独立算 sigmoid、排序、归一化，逐项检查专家 ID 和权重。每个 case 一次隔离进程。

```bash
bash scripts/cann-run.sh bash -c 'g++ -O2 -std=c++17 -I"$ASCEND_HOME_PATH/include" -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath-link,"$ASCEND_HOME_PATH/lib64" development/workloads/moe_router_smoke.cpp -lopapi -lnnopbase -lascendcl -o build/smoke/moe_router19'
```

`router_validate.py` 的 `register calibration|holdout` 固定 host 二进制 digest、输入、芯片和单例限额，holdout 还写入冻结预测及模型 digest。已存在的登记文件会阻止意外覆盖。子命令 `collect calibration|holdout` 读取这些合同并检查启动数、实际 SKU、内存类型、两组 golden、每核 trace 时间和 profiler duration 表。该步骤要求本机进程间套接字可用。两路并发仅用于离线采集。已存的独立样本继续保留原有数据角色。新一轮开发应选择新的目录、版本和样本。

若 msopprof 外层收尾超限，router_windows.py 会复用内部完整 trace。它逐核要求首末指令、完整 duration 表、binary SHA、两组 golden 以及同 SKU 的正常导出对照；主进程超限记录保留。仍缺证据时可用 router_recover.py 登记同一形状的单并发重试；失败及重试成本分别保留。

采集结束后程序从合格行和冻结预测复算控制、读、Vector 和写回的校准窗口及独立 WAPE。用以重建报告的原始指令 trace 位于 `runs/workloads19`，版本化结果保存其哈希、实际窗口、校验与最小复查证据。

```bash
python3 development/workloads/router_windows.py
toolchains/envs/cann-9.2.1/bin/python development/workloads/router_report.py
```


## 两个模型叶算子的复现证据

DeepSeek 的 [BF16→FP32 Cast](../../docs/examples/cast_bf16_fp32.md) 使用 `cast_bf16_fp32.cpp` 对实际 aclnnCast 输出与 CPU 参照逐位核对。Kimi 融合残差的 [原版构建、四组 golden 与双端验证](../../results/v0.19/workloads/attention-residual.md) 使用 `attention_residual_smoke.cpp` 和由固定 upstream blob 生成的 vendor OPP。后者分别保存最初两组和补充八组冻结预测，设备耗时只覆盖已校准的驻留分支。[补充形状及逐阶段统计](../../results/v0.19/workloads/attn-res-extension-evaluation.json)
