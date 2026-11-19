# FLA 26.9.1 的 Chunk KDA 复核

本组验证使用官方 A5/aarch64 wheel 的主机动态库、tiling 和原样设备 ELF。用户目录内的项目 CANN 9.2.1、950DT 的 CA 与隔离环境通过仓库的 `scripts/cann-run.sh` 加载。输出目录位于 `results/chunk-recovery/fla2691`，`raw/` 由 Git 忽略。发布版数据与本次采集分别保留 SHA256。

在具有原版本原始数据的 checkout 中，以下命令只重读文件：

```bash
env PYTHONDONTWRITEBYTECODE=1 python development/validation/chunk_recovery/fla2691/score.py
env PYTHONDONTWRITEBYTECODE=1 python development/validation/chunk_recovery/fla2691/postmortem.py
env PYTHONDONTWRITEBYTECODE=1 python development/validation/chunk_recovery/fla2691/v2/score_v2.py
```

重新开展实验使用新 checkout 和新的采集/注册目录。领取官方 `v26.9.1-release` 的 [release JSON](https://api.github.com/repos/flashserve/flash-linear-attention-npu/releases/tags/v26.9.1-release) 及其 `flash_linear_attention_npu_a5-26.9.1-py3-none-manylinux_2_34_aarch64.whl`，在 `raw/` 中命名为 `release.json` 和 `fla-a5.whl`，运行同目录 `prepare.py`。该脚本核对 GitHub 发布方 SHA256，仅提取 wheel 的本地运行时。两类调用使用由 [Kimi 数值驱动](../../chunk21/production_probe.py) 生成的 `build/smoke/chunk21-kimi.cpp`。V1 注册还需与 [vLLM 源文件哈希](../../../../results/chunk-recovery/fla2691/vllm-source.json) 相同的项目源缓存。

V2 注册还会核对 FLA commit `5cc699069c7f91a35aabd583032c8720a0b626fa` 中三份宿主源码的 Git blob。将[提交的递归文件树](https://api.github.com/repos/flashserve/flash-linear-attention-npu/git/trees/5cc699069c7f91a35aabd583032c8720a0b626fa?recursive=1) 保存为 `raw/fla-source-tree.json`，按 [V2 注册](../../../../results/chunk-recovery/fla2691/v2/register.json) 的 `verified_host_source` 将三份原文以各自 basename 放入 `raw/`。为当前 checkout 生成独立注册后才可采集。V1 的 `probe.py` 和 V2 的 `v2/probe_v2.py` 均接受 `register` / `run`；注册与日志使用一次性路径，避免无意重复。V2 新启动进程内用 `dlsym` 和 `dladdr` 检查两个真实符号来自 wheel 的 `libcust_opapi.so`。Linux 本机通信、CA 运行资源和约 3 GiB 可用空间是该 V2 采集的运行条件。

V1 的 `score.py` 另外读取 `raw/loader-bindings.txt`。在同一环境中对已经冻结的宿主运行 `LD_BIND_NOW=1 LD_DEBUG=bindings kimi-fla`，省略位置参数让宿主在创建 ACL 设备前退出。保留其中从 `kimi-fla` 解析出的 `aclnnChunkKdaFwd`、`aclnnChunkKdaFwdGetWorkspaceSize` 两行。由于 V1 的开发宿主通过可执行文件的 PLT 取函数地址，不能只看其 `dladdr` 日志。

`faults.py` 对每个真实重复 SET 的 PC 单独核对反汇编。其静态审查输入为 wheel 内 `chunk_kda_fwd.cpp`、`arch35/chunk_kda_fwd_impl.h`、`arch35/chunk_kda_fwd_prepare.h` 三份原文；输入同一组文件分别运行项目技能中的 `sync_audit.py`、`ascendc_flow_analyzer.py`，通过 `--format json` 将输出保留到 `raw/sync_audit.txt`、`raw/ascendc_flow_analyzer.txt`。先运行 `faults.py`，再运行 `postmortem.py`。最终的合格时间必须同时经过独立数值、实际 kernel 身份及全使用核指令窗口核查。

V2 的原始合并 Chrome trace 把部分核间同步显示为配对的 B/E。运行下面的离线步骤可在保留首次 X-only 回执的基础上核对全部物理核并重绘实测图。

```bash
env PYTHONDONTWRITEBYTECODE=1 python development/validation/chunk_recovery/fla2691/v2/verify_chrome.py
env PYTHONDONTWRITEBYTECODE=1 toolchains/envs/cann-9.2.1/bin/python development/validation/chunk_recovery/fla2691/v2/timeline.py
```
