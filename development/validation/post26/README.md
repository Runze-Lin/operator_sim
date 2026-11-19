# 复核内置 Post 的同流重复价格

可选实验入口为 `OperatorStream(..., experimental_post_repeats=True)`。其状态在当前 rank/设备/流内由成功的 advance 更新。两个调用都通过内置 Post 的 dtype、残差、布局合同，shape 相同并且已知驻留、走 RegBase token 路线时才启用重复价格。其他前序和无状态入口继续使用原估计。

- [注册](freeze.py)将 v0.25 五个前瞻独立形与本轮新五形、两种备用形连接。所有端点预测和价格在 CA 前冻结，独立数据不会回写价格。
- [串行采集](collect.py)沿用前代实际 ACLNN 宿主、独立全输出金标、ELF 与全芯片窗口校验。已有队列运行期间持久等待，备用只看完整重复形状数量。
- [评分](score.py)通过正式 OperatorStream 逐一验证三芯片冻结值，再按冷态、扩核、普通热态、立即重复分别评分，估计双端加速比与开销。
- [新独立时间线](figure.py)逐字节核实输出 GM 分片、系数、hOut 与残差的读地址。上栏整行预测与下栏两半片实际进度使用同一时标。
- [小 token 通道路线](channel.py)从原有 cold/repeat 配对计算 shape-only 的不可约最大误差，保留采集与数据角色。

从仓库根目录运行 `toolchains/envs/cann-9.2.1/bin/python development/validation/post26/score.py` 可重新核对已返回的独立窗口。[当前报告](../../../results/v0.26/post/report.md)
