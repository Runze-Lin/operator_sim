# 昇腾算子性能建模：论文正文与复算指南

[完整中文 PDF](../operator-sim-report-zh.pdf) 以同一个 FA 长循环贯穿阶段定价、流水合成、指令校准和跨芯片检验。主体对应 v2.0.0，校准与系统结果沿用 v1.13.1 冻结记录；附录 E 和 F 分别报告 v2.1 FA 试点与 v2.2 多算子检验。微秒表示设备时间，CPU 查询开销单列。970DT 的所有数字均为假设资源配置下的外推。

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
| [附录 E　FA 指令类别试验](13-fa-vector-pilot.md) | 同窗比较两种 VF、独立搬运探针、忙核指令和新精度 |
| [附录 F　指令建模与跨芯片迁移](14-vector-transfer.md) | FA 新窗、三个 Vector 算子、查询成本与 970DT 敏感性 |

## 构建 PDF

在仓库根目录使用具备 Pango、HarfBuzz 和 Noto CJK 简体中文字体的环境。中文正文使用 Noto Serif CJK SC，标题和表格使用 Noto Sans CJK SC。若字体未注册到系统，可把 NotoSerifCJKsc-Regular.otf 与 NotoSansCJKsc-Regular.otf 放在同一目录，再用下方的参数指定该目录。PDF 会嵌入用到的字体，证据链接相对于 PDF 所在目录解析。构建只需要归档图表与下列排版依赖。

~~~bash
python -m pip install -e .
python -m pip install 'weasyprint==66.0' 'markdown-it-py==3.0.0' matplotlib numpy
python -m calibration.performance_report --font-dir /path/to/fonts
~~~

正文行内公式用单个美元符号标记，独立公式用成对的双美元符号各占一行。排版脚本用 Matplotlib MathText 将 TeX 语法转为矢量公式，并为独立公式按章编号。PDF 中的公式可以缩放，TeX 源式仍保留在 Markdown 中。

更新算例数据时，可先运行[附录 D 的重绘命令](12-reproduce.md)。系统图与附录 B/C 由同一脚本从发布清单生成。FA 图可仅用压缩时间戳重绘；持有原始开发窗数据库时，可用脚本的可选参数对照 SHA256 和每一条发起时刻。[FA 重绘与原始校验](../../calibration/examples/paper_fa_figures.py) · [迁移情景重算](../../calibration/examples/paper_transfer_sensitivity.py) · [系统图和结果表](../../calibration/examples/paper_system_results.py)
