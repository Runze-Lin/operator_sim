# Kimi K3 beta Sigmoid 的独立 CA 重现

在线算子为 `operator_model.sigmoid_resource.resource_sigmoid`；平台来源为本机安装的 CANN 9.2.1 ops-nn，FP32 RegBase 实现位于 `opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/sigmoid/arch35/sigmoid_f32.h`。开源 host 参考 ops-nn `602e4ad985da00aad28c16432841e9c5069e10de` 的 `activation/sigmoid/op_host/arch35/sigmoid_tiling_arch35.cpp`。它向 ATVOSS 传入 FP32=32 bits、bufferDivisor=128 bits 和目标 AIV/UB；模板策略可核对项目内 CANNBot 的 `ops/ascendc-tiling-design/references/elewise/tiling.md`。公开源码用于解释，实测 binary 的 SHA256 记录在各个完整窗口下。

从仓库根目录进入已安装的 CANN 隔离环境构建 host，然后逐阶段运行。若已有冻结文件，直接 collect 可继续未完成的记录。同一阶段同时只运行一个 collector，后续 CA 使用本地 IPC。多个 agent 同时跑计算时保持本 campaign 并发上限为 2。

```bash
mkdir -p build/smoke results/v0.20/newop
bash scripts/cann-run.sh bash -c 'g++ -O2 -std=c++17 -I"$ASCEND_HOME_PATH/include" -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath-link,"$ASCEND_HOME_PATH/lib64" development/validation/sigmoid20/smoke.cpp -lopapi -lnnopbase -lascendcl -o build/smoke/sigmoid20'
python3 development/validation/sigmoid20/campaign.py register development
python3 development/validation/sigmoid20/campaign.py collect development
python3 development/validation/sigmoid20/campaign.py register holdout
python3 development/validation/sigmoid20/campaign.py collect holdout
python3 development/validation/sigmoid20/report.py
toolchains/envs/cann-9.2.1/bin/python development/validation/sigmoid20/figure.py development 3072 96 development
```

`report.py watch` 可跟随原始进度并更新校准账本。`figure.py holdout 262144 12 prefill32k` 选取已通过核验的 DT 大样本。`runs/sigmoid20` 存大体积 profiler/trace。版本化结果保存每个窗口、核数、哈希、数值、冻结预测、误差和采集成本。

小时级运行可由独立 tmux 会话接管，自动维护 10 个形状的完整账本、控制器快照、长上下文曲线和终止核图。若已有前台 collector，监督程序先跟踪其实际 PID。若 collector 提前退出，但原生 CA 仍活着，等它完成后按主机数值检查、profiler 结束标志和 `/usr/bin/time` 恢复状态。恢复计时会标明基于文件起止时间，保留体积仅作峰值下界。原生采集未完整时，其目录保存在 `runs/sigmoid20-supervisor/interrupted`，冻结预测保持原样。

```bash
tmux new-session -d -s operator-sim20-kimi-sigmoid -c "$PWD" 'toolchains/envs/cann-9.2.1/bin/python development/validation/sigmoid20/durable.py'
cat runs/sigmoid20-supervisor/status.json
cat runs/sigmoid20-supervisor/complete.json
```

`long_figures.py` 输出每个合格 DT/PR 长 shape 的同轴模型/CA PNG 及完整 tile 时间戳；`controllers.py` 备份 msopprof 自动清理前的各通道文本；`sensitivity.py` 做冻结后的 L2 驻留敏感性（与独立模型评分分开）；`bench.py` 创建全新进程测首次公共查询。
