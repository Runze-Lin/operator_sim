# 15 个基础调用的 950 双端 CA

mask 使用 INT64 token ID 和 BOOL mask；math 覆盖 BF16 取负、FP32 平方/归约/rsqrt；memory 覆盖 BF16 的 Embedding、GatherV2、Cast、Cat 和 Pad。每组取 9、144 行，向量宽度 64。设备窗取所有已报告活动核最早指令到最后退休，命令成功、输出核查、SKU 内存模型、各核 trace 以及已安装 OPP ELF 的哈希必须同时通过。[完整回执](measured.json) · [预测及逐窗误差](../system_vector_ca/primitives.csv)

CANN 9.2.1 环境中构建小型探针。SDK 指向含 include 和 lib64 的 Toolkit 目录，launcher 为加载其运行环境的脚本。

```sh
g++ -std=c++17 -O2 calibration/system_primitives_ca/probe.cpp -I"$SDK/include" \
    -L"$SDK/lib64" -Wl,--allow-shlib-undefined -lopapi -lascendcl -lnnopbase -o /tmp/opsim-primitive
python -m calibration.system_primitives_ca.collect "$RAW" "$LAUNCHER" /tmp/opsim-primitive
python -m calibration.system_primitives_ca.receipt "$RAW"/* --opp "$OPP_ASCEND950" --out "$REPORT"
```

复现时使用新输出路径。旧版 smoke 在 InplaceCopy 触发模拟 DMA 信号 11，未给 Copy/Clone 记 CA 成绩；移除该调用后两种尺寸的其余访存程序分别重跑成功。144 行第一次合并作业超时，完整双端重新采集的日志和二进制摘要在回执中。原始 FP32 golden 用绝对误差比较；现行探针另外显式拒绝非有限输出。回执的 probe_sha256 对应各次真正运行的二进制。
