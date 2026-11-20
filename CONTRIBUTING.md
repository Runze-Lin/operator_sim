# 维护 operator-sim

## 本地开发

在独立的 Python 3.10+ 环境中，从仓库根目录执行：

```sh
python -m pip install -e .
python -m unittest discover -s tests
python -m unittest discover -s calibration/tests
python -m calibration.verify --reprice
```

第一组测试覆盖计算量、分核、流水和系统入口；第二组检查离线收据与误差统计。最后一条命令将已安装模型与当前参考数据逐窗对照。发布前可用 `python -m pip install build` 和 `python -m build` 生成 wheel 及源码包，然后在新环境、仓库外安装 wheel 并运行 README 的例子。部署环境只需 wheel。

## 增加算子或扩展已有实现

1. 明确调用合同。记录公开名称、本卡 shape 的维度顺序、dtype、布局、属性、包内实际 API 和 kernel。不同实现复用同一前端名称时，通过显式属性分派。
2. 在 [operators/](src/operator_model/operators/) 中按算子族实现资源调度。分别计算算法工作、tile 和尾块、所占核数、搬运与指令、依赖以及最终设备窗。共享的排程和计价函数在 [scheduling/](src/operator_model/scheduling/)，通用的张量工作量在 [workload.py](src/operator_model/workload.py)。仅有结构证据时允许返回工作量和空耗时。
3. 将新路径接入 [backend.py](src/operator_model/backend.py) 的分派、默认 dtype 和 work 统计。如需识别库实现的结构计划，维护 [extensions/](src/operator_model/extensions/) 的对应登记；已有参考路径和别名见 [reference.py](src/operator_model/scheduling/reference.py)。同流状态归 [stream.py](src/operator_model/stream.py) 管理。
4. 让目标芯片通过配置提供核数、频率、吞吐和容量；芯片级证据写入 [data/chips/](src/operator_model/data/chips/)，按路线复用的价格及来源写入 [data/parameters/](src/operator_model/data/parameters/)。给每个新增自由参数写明物理含义、单位、如何识别、跨芯片继承的假设和影响。
5. 从公开 API 增加测试，覆盖普通形状、尾块/分核边界、目标参数变化和不支持输入。补全 [建模方法](docs/modeling-strategy.md)、一个从输入到设备时间的[算例](docs/examples/)及[误差分析](docs/error-analysis.md)。

## 重新校准

先冻结旧预测、硬件配置、输入合同和预算，选择能区分假设的小形状、边界和多波大形状。尽量复用现有 trace。开发回放和采前冻结的新形状分别统计。比较双方耗时、工作量、核数、关键阶段以及预测墙钟。缺少核退休、全输出正确性或实际实现身份时，保留缺口说明。

原始 CA 数据库、trace、OPP 二进制、日志和临时产物放在仓库外或被忽略的本地目录。发布只收录必要参数、可重算的逐窗摘要和少量可讲解的证据。不同观测协议和重用同一 trace 的合同分别统计；新增实测用 SHA-256 绑定到原始证据。具体收据、导入工具和可运行示例见[校准指南](calibration/README.md)。
