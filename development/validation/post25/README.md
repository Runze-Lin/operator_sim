# 复核内置 MhcPost 的逐 token 流水

本目录只针对 CANN 9.2.1 安装包的 BF16、四条残差流、含 hRes 的 aclnnMhcPost。拟合文件和采集注册文件都有不可变的来源哈希，全部金额按周期或每 KiB 周期记录。参考源码、安装库和实际 vLLM 的 HcPost 分别见[内置算子介绍](../../../docs/examples/mhc_post.md)与[vLLM 分支](../../../docs/examples/hc_post_vllm.md)。

- [逐行调度函数](../../../operator_model/post_token_pipeline.py)只需要目标芯片、该形状的分核方案、已知前序状态和已装载参数。普通查询通过有限次 max 与乘法计算首行、稳态和末行。可选 `inspect_rows` 只生成请求的几个行序号。
- [开发数据选择和价格估计](calibrate.py)读取已有 31 个全窗末核的可观测起止，分别估计端口发起间隔、有效载荷响应和输入就绪后的向量/交接周期。开发留出每次同时剔除整个待测进程和该形状在所有其他进程的样本。
- [输入与模型冻结](freeze.py)记录两端各十个全新形状的四组同流顺序、立即重复及未知前序对照，以及 970DT 外推。仅当生成一个新的试验时才运行。已有 freeze.json 会阻止覆盖。
- [采集](collect.py)使用单个队列锁，重复使用上一轮已验证的 aclnnMhcPost 全张量金标宿主、msOpProf 仿真以及 ELF/逐核 END/输出写回检查。每个组、SKU 独立进程，每个目标输出必须全部通过。已有原始运行目录会阻止覆盖。
- [数值误差和跨芯片汇总](report.py)、[分阶段对照](milestones.py)、[真实输出地址到 token 时间线](figure.py)均基于留存回执。可额外运行 [physical.py](physical.py)，按全物理指令单独验证包括所有参与 AIV 的入口、END、输出和官方时长。[verify_schedule.py](verify_schedule.py)验算闭式与逐行递推相等，并检查三档整段形状的在线开销。

复画 950DT 的两轮独立形状。命令从仓库根目录执行，需要原始 `runs/post25/` 目录与结果内的 prices.json。

```bash
toolchains/envs/cann-9.2.1/bin/python development/validation/post25/figure.py boundary 67 384 two-token-repeat
toolchains/envs/cann-9.2.1/bin/python development/validation/post25/report.py
```

[冻结的输入、模型与预算](../../../results/v0.25/post/freeze.json) · [两端误差与 970DT 外推](../../../results/v0.25/post/report.md)
