# v0.7 迭代变化

保持v0.6 quant/Matmul/FA模型和原始trace；新增真实库算子的小shape局部回归对照；通用默认要求算子结构/硬件资源分离，未校准机制预测保持N/A，按路由/属性限定预测域。MoE两阶段和mHC组合按真实串行kernel净窗口描述，不混入host开销。公式及每参数解释维护在全量 modeling-strategy.md。

模型先用calibration拟合，再冻结全部holdout预测。旧版新扩展没有可比较时延模型，v0.6同集误差为N/A；原三族同集预测没有变化。新增独立与固定基线比较：

| 算子 | 开发/独立n | 机制独立WAPE | 局部回归开发WAPE | 局部回归独立WAPE / max | 固定同集WAPE |
|---|---|---|---|---|---|
| RMSNorm | 2/1 | N/A | 0.00% | 32.12% / 32.12% | 42.22% |
| Addmm | 2/1 | N/A | 0.00% | 4.13% / 4.13% | 3.68% |
| Lightning Indexer | 2/1 | N/A | 0.00% | 5.90% / 5.90% | 0.81% |
| mHC Pre | 2/1 | N/A | 1.61% | 1.14% / 1.14% | 1.14% |
| mHC Sinkhorn | 2/1 | N/A | 0.00% | 8.96% / 8.96% | 6.35% |
| mHC PreSinkhorn | 2/1 | N/A | 1.22% | 7.78% / 7.78% | 7.78% |
| mHC Post | 2/1 | N/A | 15.36% | 12.70% / 12.70% | 12.70% |
| mHC组合 | 2/1 | N/A | 2.50% | 2.70% / 2.70% | 2.70% |
| KDA recurrent | 2/1 | N/A | 0.00% | 1.84% / 1.84% | 33.77% |
| KDA chunk | 0/0 | N/A | N/A | N/A / N/A | N/A |
| GMM1–SwiGLU–MX量化 | 3/1 | N/A | 1.27% | 0.35% / 0.35% | 2.35% |
| MoE GMM2 | 3/1 | N/A | 4.37% | 4.10% / 4.10% | 5.33% |
| MoE FFN组合 | 3/1 | N/A | 2.28% | 1.60% / 1.60% | 3.34% |


新增CA 36 次（成功 29），墙钟 43.28 分钟，记录CPU 7.96 小时（部分终止进程为下界），保留 893.67 MiB；DB归约 16.44 秒。复用记录不再次计费。预测基准逐算子见 evaluation.json，旧族cache和未缓存口径分开。

未完成范围：真实整网dispatch形状/路由、全面大shape、独立指令吞吐识别、跨代A7/A8准确性；Q9 recurrent单调用不支持；256k/512k FA未实测部分仍为外推。失败/未采集项保持N/A，不能用本轮小样本误差代替全域准确率。

产物：model.json、frozen-predictions.json、evaluation.json、accuracy-by-operator.png/.svg、validation-and-cost.png/.svg；全量 docs/error-analysis.md 覆盖旧与新算子。
