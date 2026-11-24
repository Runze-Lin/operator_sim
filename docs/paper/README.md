# 昇腾算子性能建模：论文正文与复算指南

[完整中文 PDF](../operator-sim-report-zh.pdf) 以同一个 FA 长循环贯穿阶段定价、流水合成、指令校准和跨芯片检验。模型版本 v1.13.1。微秒表示设备时间，CPU 查询开销单列。970DT 的所有数字均为假设资源配置下的外推。

| 章节 | 在这里可以复核什么 |
|---|---|
| [中英文摘要](00-abstract.md) | 问题、方法、主要结果和证据范围 |
| [1　系统问题与理论参照](01-foundations.md) | 什么条件下 Roofline 耗时是下界 |
| [2　从张量到完成时间](02-method.md) | DynamicMXQuant 和 Matmul 的每核指令、字节、分块和重叠 |
| [3　FA 的阶段与整窗](03-fa-model.md) | 手算 PR 128Ki 的六个资源周期及 512 页总时间 |
| [4　CA 指令校准](04-calibration.md) | 从 DT 32Ki 的 256 个原始发起点估计价格，检查残差 |
| [5　跨芯片迁移](05-transfer.md) | 950 双端检验、970DT 假设、瓶颈翻转与灵敏度 |
| [6　误差及查询开销](06-validation.md) | 开发与留出的区别、mHC 配对、A8W4 最差点、Flash 可控布局 |
| [7　系统结果](07-system.md) | 三个模型与 33 组请求、1Mi 末块及重放误差 |
| [8　讨论与文献](08-discussion.md) | 可证实的机制、主要不确定性与最有区分力的测量 |
| [附录 A](09-contracts.md) · [B](10-evidence.md) · [C](11-requests.md) · [D](12-reproduce.md) | 逐算子公式与算例、逐合同误差、33 组明细、复现方法 |

## 构建 PDF

在仓库根目录使用含中文字体（如 Droid Sans Fallback）、Pango 与 HarfBuzz 的环境。下列 Python 依赖只供排版、绘图，在线预测器可独立安装。PDF 的证据链接相对于仓库中的 PDF 所在目录解析，单独分发 PDF 时仍可阅读全部正文、图表与附录。PDF 编译使用已归档图表，无需原始大 trace 或 CANN。可用自有虚拟环境执行：

~~~bash
python -m pip install -e .
python -m pip install 'weasyprint==66.0' 'markdown-it-py==3.0.0' matplotlib numpy
python -m calibration.performance_report
~~~

更新算例数据时，可先运行[附录 D 的重绘命令](12-reproduce.md)。系统图与附录 B/C 由同一脚本从发布清单生成。FA 图可仅用压缩时间戳重绘；持有原始开发窗数据库时，可用脚本的可选参数对照 SHA256 和每一条发起时刻。[FA 重绘与原始校验](../../calibration/examples/paper_fa_figures.py) · [迁移情景重算](../../calibration/examples/paper_transfer_sensitivity.py) · [系统图和结果表](../../calibration/examples/paper_system_results.py)
