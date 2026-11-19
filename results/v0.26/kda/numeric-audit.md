# KDA Prepare 的两个 BF16 超限点

**原发布算法口径下两个样本仍然失败。** 改为在 CPU 上逐步执行安装包写明的 FP32 gate、BF16 操作数和 FP32 逆三角更新后，重现了 950DT 和 950PR 已存的两个辅助张量。按不同 shape 去重，每端共有 64,640 个 BF16 元素逐位一致。额外用一个未参与本次定位的已合格形状 [193,2] 复核，每端又有 49,408 个元素逐位一致。这三项 CPU 重放没有新增 CA。[原始双端超限回执](failed-257.json) · [另一原始回执](failed-248.json) · [诊断和数组 SHA](numeric-audit.json) · [独立 CPU 复核结果](numeric-audit-control.json)

| T / H | 辅助量和零基坐标 [头,全局行,块内列] | 原发布算法 FP64→BF16 | 安装包精度重放 | DT / PR 原始 CA | 原绝对误差 / 容限 |
|---|---|---:|---:|---:|---:|
| 257 / 1 | Aqk [0,243,44] | 0.000072479248046875 | 0.00007772445678710938 | 0.00007772445678710938 | 5.24521e−6 / 3.13249e−6 |
| 248 / 1 | Akk [0,242,40] | 0.000274658203125 | 0.0002651214599609375 | 0.0002651214599609375 | 9.53674e−6 / 6.29153e−6 |

这两例其余 Aqk / Akk 元素均通过原逐元素门槛。冻结协议要求四组输出全部合格，原“数值失败、不可计时”结论继续保留。这里的逐位比较在宿主已经保存的 BF16 数据上进行。[双参考 CPU 程序](../../../development/validation/cross26/numeric_audit.py)

## 发布参考与安装实现分别计算什么

上游提交 `5cc699069c7f91a35aabd583032c8720a0b626fa` 的 Prepare ATK 脚本 Git blob 为 `46286c0a61d19341c7ca9ffd020c56c77467c336`，与本地下载副本一致。它将 Q、K、gate 计算转为 FP64，做连续行的 gate 累加后再乘 1/ln2。每个最多 16 行的带取实际行范围中点作为参考，用 `exp2(ΔG)` 调整 Q、K 并舍入到 BF16；点积与两块 32×32 下三角求解采用 FP64，最终结果再转 BF16。历史本地 golden 以一次完整下三角求解替代两块求解。将原发布的两块求解和左下角乘法重新移植到 NumPy/SciPy 后，两个失败形状的全部 Aqk/Akk 与冻结 golden 完全相同。这里核对的是同一算法的 CPU 实现；本机没有运行 PyTorch ATK 执行器。[对应的发布参考源文件](https://github.com/flashserve/flash-linear-attention-npu/blob/5cc699069c7f91a35aabd583032c8720a0b626fa/tests/atk/chunk_kda_fwd_prepare/executor_chunk_kda_fwd_prepare.py)

wheel 的 arch35 `StageV0Vf` 对每步 gate 在 FP32 内依次做加偏置、乘以 exp(A_log)、取反、Exp、加一、求倒数和乘以负下界。它**先按 FP32 乘以 1/ln2，再逐行累加 FP32 carry**。`ExpPair` 截断差值、乘 FP32 ln2，调用自然指数 Exp；`Store128FromFp32` 使用 BF16 RINT。KDA Cube 接受上述 BF16 操作数，FP32 接收分数。`StageV3Vf` 的两个 32 行对角逆按 source 行递增依次执行独立浮点乘、减。CPU 将同样的计算顺序用于这两个失败点所在的对角块。左下块采用 NumPy FP32 矩阵乘法，不声称复刻硬件每一级乘加树。[设备 Vector 源码](../../chunk-recovery/fla2691/raw/fla_npu/opp/vendors/fla_npu_transformer/op_impl/ai_core/tbe/fla_npu_transformer_impl/ascendc/chunk_kda_fwd_prepare/arch35/chunk_kda_fwd_prepare_vec.h) · [设备指数常数与策略](../../chunk-recovery/fla2691/raw/fla_npu/opp/vendors/fla_npu_transformer/op_impl/ai_core/tbe/fla_npu_transformer_impl/ascendc/chunk_kda_fwd_prepare/chunk_kda_fwd_prepare_policy.h)

## 哪个中间数改变了结果

**Aqk 的第 62 通道。** T257 的坐标落在全局起始行 192 的完整块内。第 243 行属于最后一个带，参考行是 248。唯一改变的点积操作数是 K⁻ 的第 62 通道。发布公式得到 −0.0535888678703，设备精度重放为 −0.0535888634622。两者位于相邻 BF16 数的中点 −0.0535888671875 两侧，分别舍入为 −0.0537109375 和 −0.053466796875。相乘的 Q⁺ 保持 0.2353515625，因此整条点积从 0.00082077086 变为 0.00087822974。各乘积绝对值之和为 0.6785801，正负相消使一个操作数的末位改变足以超过原容限。

**Akk 的第 111 通道。** T248 的最后一个带只有 8 行，参考行改为 244。K⁻ 第 111 通道越过正数中点 +0.0535888671875，从 +0.0537109375 变成 +0.053466796875；K⁺ 为 −0.1171875。目标属于第二个 32 行对角逆的第 18 行、第 8 列。保持整个求逆为 FP64，只将发布参考的 Lkk[50,40] 换成设备精度值，输出便从 0.000274658203125 翻为 CA 的 0.0002651214599609375；反过来只恢复这一项又翻回发布值。递推依赖区域另有两项变化，但无需它们也能复现目标 BF16。[通道值、点积与反事实实验](numeric-audit.json)

这两处单独把 sigmoid 步换成 FP32，或单独把缩放后的前缀累加换成 FP32，都没有翻转最终的 BF16。完整前端换为设备顺序、仍以 FP64 点积和求解，两个目标即已翻转到 CA。再改用 FP32 点积和求解后，所有辅助元素逐位相同。两张卡的已保存辅助数组 SHA 分别全等，运行的是同一已发布 Prepare 实现；核数差异没有改变这批样本的输出位模式。[输入/实际/参考 SHA 及逐阶段消融](numeric-audit.json)

## 前向检查

在读取 [193,2] 的实际辅助数组之前，按复用的数值合同封存了完整带两个 head 的参考函数，以及该控制与拟测 [259,2] 两套参考数组的 SHA。T193 每端的 Aqk/Akk 各 24,704 元素，设备精度参考逐位全等。原发布参考只有一个 Akk 值与实际不同，仍在原容差内。[冻结时间与完整数组指纹](numeric-audit-freeze.json) · [旧合格窗的独立 CPU 检查](numeric-audit-control.json)

T259/H2 则在新 CA 之前就产生了可检验的预期：两套参考的 Aqk 有六个 BF16 不同，其中一个超过原容限；Akk 有五个不同，全部在容限内。该预测已经封存为 `b514c87a15afc4bf4025dd752a435c9f2100909c7bb3a77b36417eb2a46d6e64`。DT、PR 的两次原库调用排在已有 T192/H4 之后，仍使用同一单槽锁，每端最多 12 分钟、3 GiB。正式校验仍由原 host 对四组输出执行；离线精度一致性另打分。两端回执登记为 `numeric26-*-259-2-official.json`，精度一致性登记为 `numeric-audit-prospective.json`。[采集进度](numeric-audit-state.json)

**前向验证已完成。** T259/H2 的 950DT、950PR 均得到预先登记的结果：每端 Aqk 与 Akk 各 33,152 个 BF16 值逐位匹配中间精度参考；两端相对原发布算法参考的 Aqk 都有一个值超过容差，Akk 全部在容差内。attention/最终状态均通过。原计时资格仍为 N/A。[两套参考的实际偏差、逐位计数及完整 SHA](numeric-audit-prospective.json)
