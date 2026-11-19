# 复验 FLA A5 的 Kimi 融合 Chunk KDA

在项目根目录运行以下命令。线上算子查询不依赖本目录。必须先有原来冻结的 FLA A5 26.9.1 发行包与 9.2.1 CANN 隔离环境，run 会逐个核对冻结清单的 SHA256，包括主机程序和参考数组。恢复官方参考代码时，可以用 `fetch_sources.py` 对照已固定的 Git tree 逐文件验证 Git blob。原始运行和输出获取不需要安装 Torch 或改设备内核。[发行包与先前探针](../chunk_recovery/fla2691/README.md) · [采前清单](../../../results/v0.24/kda-v1/register.json)

```bash
python3 development/validation/kda_v1_24/fetch_sources.py
python3 development/validation/kda_v1_24/investigate.py run
python3 development/validation/kda_v1_24/inspect_sync.py
toolchains/envs/cann-9.2.1/bin/python development/validation/kda_v1_24/numerical_details.py
toolchains/envs/cann-9.2.1/bin/python development/validation/kda_v1_24/operand_replay.py
python3 development/validation/kda_v1_24/compare_pair.py
python3 development/validation/kda_v1_24/report.py
```

`investigate.py register` 仅在首次登记空白结果目录时使用。它编译调用宿主、生成两组 host CPU oracle，锁定三组开发测试及一项有条件的 PR dense 配对。采集时按真实 SoC 选择 HBM3 或 LPDDR5 模型，使用单独的 FLA OPP 和动态符号核验；结果里保留整个公开接口，包括辅助 Transpose。一个已经启动的 raw capture 即使中断也不会被同一脚本覆盖。独立的 `watch.py` 可以在采集期间运行，按 receipt 到达次序发布同步证据、实际数组和汇总。[总表与判据](../../../results/v0.24/kda-v1/README.md)
