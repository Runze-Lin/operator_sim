# 独立 Sinkhorn 的寄存器组排程与缺失前序

分析、正式模型与发布时间见 [v0.28 方法和回放](../../../results/v0.28/fusion/report.md)。这套代码运行在验证目录，发布包继续提供 v0.27 的默认和显式前序入口。

在仓库根目录重算开发与留出摘要：

```bash
python3 development/validation/fusion28/landmarks.py
python3 development/validation/fusion28/calibrate.py
python3 development/validation/fusion28/score28.py
python3 development/validation/fusion28/recent28.py
python3 development/validation/fusion28/checks28.py
python3 development/validation/fusion28/issue28.py
toolchains/envs/cann-9.2.1/bin/python development/validation/fusion28/figure28.py --pr
toolchains/envs/cann-9.2.1/bin/python development/validation/fusion28/figure28.py --forward --group first --t 69
toolchains/envs/cann-9.2.1/bin/python development/validation/fusion28/publish28.py
```

既有首次登记哈希为 `fbdb476d649246d40e7e41e2a9ee26147380dbe2174fd32b7931aa691399f55f`。单槽 `sink28-frozen` 保留原始采集和成功回执，资源不足时等待。单端首次预算 600 s，后续五个有序窗 1800 s。若会话消失，在核实无活跃实例后可执行 `nice -n 15 python3 development/validation/fusion28/campaign28.py collect` 接续已有冻结。原始采集目录已生成但没有成本回执时，程序会停下保留信息。后台 `sink28-publish` 每次等完整合格的实测回执落盘后出图，最后按末核重核向量发起间隔。
