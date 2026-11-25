# 950 MXFP8 × MXFP8 稠密投影的首片校准

在 CANN 9.2.1 环境按[构建说明](../mx_matmul/README.md)编译 `calibration/mx_matmul/native.cpp`。本批使用它的 `a8w8` 模式。原始采集目录可以放到仓库之外，每次采集使用新目录。每次调用只执行一个安装的 ops-nn MX 内核，内存模型随 SoC 选择。

```sh
export PYTHONPATH=src:.
scratch=/tmp/mx8-example
mkdir -p "$scratch"
python -m calibration.mx8_refinement.study freeze "$scratch/frozen.json"
python -m calibration.mx8_refinement.study freeze --prefill "$scratch/prefill-frozen.json"
bash calibration/mx8_refinement/launch.sh "$scratch/Ascend950DT_9582-1x128x64" /tmp/mx-matmul-ca Ascend950DT_9582 1 128 64
```

对两个冻结文件中的 DT 和 PR 几何重复采集即可重算总表。默认直接使用已激活环境的 `msopprof`，也可设 `CANN_LAUNCHER=/absolute/path/to/environment-wrapper.sh`。以下使用跟随版本保存的预测进行对照。`--opp` 指向实际安装的 OPP 根目录，可通过 `ASCEND_OPP_PATH` 提供。

```sh
python -m calibration.mx8_refinement.study report "$scratch/report.json" --scratch "$scratch" \
    --freeze results/v2.5/mx8-frozen.json --baseline results/v2.5/mx8-v24.json \
    --extra-freeze results/v2.5/mx8-prefill-frozen.json --extra-baseline results/v2.5/mx8-v24-prefill.json
```

在开发样例 [9,256,128] 的 OPPROF 目录上执行 `python -m calibration.mx8_refinement.phase_evidence "$profile/simulator" "$scratch/phases.json"` 可以重新提取每核首片设置、首 DMA、依赖和退休。这个提取器要求每核有四次 scale/值搬运和单条 MMAD_MX。独立的 `study report` 逐核检查 END、跨度、实际 MMAD 数和形状，验证全输出及内核 ELF；缺少结束标记或超时时 `qualified=false` 且 `ca_us=null`。逐次成本 `wall_s` 是 profiler 建立结果目录到输出日志结束的墙钟近似值，秒级精度。

第二批六种独立几何是在读完前四种的结果后选择。同一模型哈希继续冻结。第三份预测以 `freeze --extension` 生成，报告命令尾部再加 `--extra-freeze results/v2.5/mx8-extension-frozen.json --extra-baseline results/v2.5/mx8-v24-extension.json` 即可合并三个清单。接口校验各清单内 shape × SKU 唯一且预测源文件哈希一致。真实系统中 Kimi 路由器投影包含 [9,896,7168]；这里保持其矩阵尺寸并使用双 MXFP8 E8M0 合同。实际路由器捕获是 BF16→FP32，两者计时列分别属于各自的精度与输出。[系统图签名](../../docs/system-workload-coverage.md)
