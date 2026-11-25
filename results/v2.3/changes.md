# v2.3　从 FA 推广到各算子家族

原生 API、流式 API 及系统后端都接受 Vector 方法选择。已知库实现默认保留原法；登记为新设计的系统 VF 程序、FP16 稠密 FA 和按行稠密 FA 默认按指令类别计价。两法复用实现匹配、tile、每核任务及非 VF 依赖。全新 dtype 仍需登记程序；无 VF 和无合格整窗的路径分别保留原报价和 N/A。增加了可选的、按芯片和描述符记录的搬运间隔/延迟，以及独立端口有限缓冲子流水的在线闭式与离线事件重放。[总方法](../../docs/modeling-strategy.md) · [端口和流水示例](../../docs/examples/transfer-stream.md)

本轮新增 CA 次数为 0，采集预算为 0。主档案 36 个合同，2,180 个实测窗，另回放此前两轮的 11 个合格 FA 窗。2190 个窗口有配对预测；FP16 另有一窗仅可打分指令法。分角色汇总，旧留出仍属于本轮回放。原 BF16 FA 普通档案 WAPE 在 DT/PR 上从 1.84%/5.05% 变为 1.62%/4.72%；专项 BF16 的 5+5 个窗口从 24.58%/30.04% 变为 2.40%/5.87%。Recurrent KDA 则从 6.53%/4.35% 变为 47.13%/49.82%。A8W4 历史留出 WAPE 双端上升而最大误差下降。所有已有原生库实现保留原默认，970DT 仅列外推。主档案以外的原生 QSLI 有 8 个 DT 窗和 4 个 PR 窗，另 6 条只保留组件之和。[全量比较与表头释义](../../docs/error-analysis.md) · [同窗明细](native-windows.csv.gz) · [候选 Indexer 记录](candidate-methods.csv)

86 个系统名以 293 份实际签名在 DT/PR 上运行三种选择；35 个名字有匹配的计时合同，共 684 个测量窗，包含主档案复用的观测。其余报告两法报价变化。系统数据按签名重放，本轮没有重算端到端 DES。查询 CPU 开销按每次原生 API/流式调用单列，模型与 CA 使用设备微秒；此次原生查询的原法/指令法 P95 约 3.25/6.16 ms，实测最大约 10.58/22.39 ms。系统墙钟另采用热进程重复查询，方法和原始值均随表给出。[系统签名表](../../docs/system-vector-methods.md) · [研究报告附录 G](../../docs/paper/15-vector-methods.md)

重算：仓根设置 `PYTHONPATH=src:.`，运行 `python -m calibration.examples.all_vector_methods`、`python -m calibration.examples.system_vector_methods` 及 `python -m calibration.candidate_indexer.compare_vf_methods`。同一程序的设备报价可复核；查询墙钟取决于进程状态与主机负载。[模型与输入、结果的 SHA-256 清单](replay-manifest.json)
