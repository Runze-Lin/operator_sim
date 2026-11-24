# 附录 D　如何复算一次预测

## D.1　先确认单位与输入合同

| 符号或输出 | 单位 | 读法 |
|---|---|---|
| M、N、K；Q、KV、D | 元素数或位置数 | 矩阵例使用 [M,N,K]；FA 入口按 [Q,KV,D,B,H] 传本卡值。 |
| L、J、I | cycles | L 是发起到相应完成或就绪的响应，J 是再次发起的最小距离，I 是可持续的一页或一块的间隔。 |
| R、W | B/cycle 或 B/s | 先确定是单核端口、共享链路还是整片 HBM，再与同一层级的字节数相除。 |
| f | GHz | 1 GHz = 1000 cycles/µs。周期除以 1000f 得到设备微秒数。 |
| active_aic、active_aiv | 核 | 记录有任务的核数；每个核能处理多少块仍取决于分配结果。 |
| latency_us；查询墙钟 | µs；ms | 前者进入系统调度，后者是运行预测函数的 CPU 成本。 |

安装当前项目后，从任意目录运行下面的 Python 片段。固定精度、shape 与实现，分别改变芯片。保留状态字段一起看，970DT 的数值为架构外推。

~~~python
from operator_model import estimate, load_chip

for sku in ("Ascend950DT_9582", "Ascend950PR_957b", "Ascend970DT"):
    result = estimate("fa", [9, 131072, 64, 1, 1],
                      chip=load_chip(sku), dtype="bf16")
    print(sku, result.status, round(result.latency_us, 3),
          result.active_aic, result.active_aiv)
~~~

该输入的 DT、PR、970DT 返回时间分别为 274.987、274.988、252.093 µs；活动 AIC/AIV 分别为 1/2、1/2、1/1。公共 API 对 PR 的目标资源路线也返回 extrapolated；本算例有 PR 同形状 CA，能额外报告该点的预测偏差。970DT 尚无这种对照。

## D.2　沿证据反向复算

1. 在仓库根目录运行下面的复算命令。FA 脚本只读版本化时标，给出中段斜率和其它阶段参数；第 4 章的图生成脚本可加 `--db /path/to/32ki.sqlite`，在本地核对原始数据库的 SHA-256 和 256 个发起时刻。数据库无需随发布包分发。
2. 调查新算子时，先在逐窗页确认 kernel 身份、输入属性、是否有序流及完整结束标记，再区分用于标定与用于揭盲的输入。得到一个总时间后不能唯一回推出所有阶段参数。
3. 重放系统情景时，先确定配置采用哪一套峰值、分片和冷读写政策。附录 C 的 CSV 同时含基线、两次冻结、不同页布局及 phase 分解，方便把差异定位到具体阶段。

~~~bash
python calibration/examples/replay_fa.py
python calibration/examples/paper_fa_figures.py
python calibration/examples/paper_transfer_sensitivity.py
python calibration/examples/paper_system_results.py
python -m calibration.performance_report
~~~

生成报告需要可用的中文字体、Python 排版依赖和绘图库，具体版本与入口见[论文源稿索引](README.md)。运行在线估计不需要上述 PDF 工具。原始大 trace 可单独保存；版本化数据和图已足够重建本报告的图表。[逐窗档案](../errors/README.md) · [FA 压缩时标](../../calibration/examples/fa-950dt.json) · [运行时公共接口](../integration.md)
