# 使用 CA 校准算子模型

校准在仓库的 `calibration/` 中离线执行。系统仿真只调用已安装的 `operator_model`。以下命令从仓库根目录运行，仅使用 Python 标准库和项目本身。

## 先验证一条完整路径

示例采用真实 `aclnnMatmul` 的 BF16 ND 调用，设备为 950DT，形状为 [32768,128,512]。算子返回 4,194,304 个数，32 个 AIC 全部退休。已归档收据体积约 5 KiB，复现这个例子无需重跑 CA。

```sh
python -m pip install .
cal_dir=$(mktemp -d)
python -m calibration freeze calibration/examples/cases.json "$cal_dir/freeze.json"
python -m calibration import "$cal_dir/freeze.json" calibration/examples/matmul-ca.json "$cal_dir/measurement.json"
python -m calibration score "$cal_dir/freeze.json" "$cal_dir/measurement.json"
```

预期模型为 15.467022 µs，CA 为 15.561818 µs，有符号误差 −0.6092%。两次公开查询分别对应 950DT 与 970DT，后者 7.777500 µs 是外推。本例采集早于当前冻结，数据角色是 `replay`。[输入清单](examples/cases.json) · [真实逐核收据](examples/matmul-ca.json)

## 从 CA 事件复算 FA 参数

在仓库根目录运行 `python3 calibration/examples/replay_fa.py`。它用 950DT 的 10 个开发窗复算 7 项通用价格或基准量、短页与宽页各 9 项首尾参数，并与发布参数逐项核对。输入节选含原始数据库指纹、按页排序的 K/V 发起时刻、按核合并后的队列忙周期，以及首轮和尾轮的事件差。整窗误差、970DT 外推和目标形状的预测都没有参与这次复算。[参数与事件的逐项推导](../docs/examples/fa.md#这些速率和固定周期从哪里来) · [事件节选](examples/fa-950dt.json) · [复算脚本](examples/replay_fa.py)

## 给新 shape 采前冻结

复制并修改输入清单的 `cases` 数组，按芯片分别登记。同一组 case ID 必须唯一。每次冻结都用一个新文件名。每个 case 含以下字段。

| 字段 | 含义 |
|---|---|
| `id, op, chip` | 样例 ID、公开算子名、目标设备 SKU。自定义 SKU 用 `freeze ... --chip-config chip.json` 加载，可重复传入。 |
| `shape, dtype, attrs` | 芯片本地实际输入维度、数据类型及调用属性，和系统仿真使用同一个接口。 |
| `contract, api` | 误差分组名与采集时必须调用的原生接口。合同应区分计时协议、实现和是否传入前序。 |
| `outputs` | 每个输出张量的名字和需要校验的元素总数。 |
| `prefix` | 普通查询用 null；有序流从 `[]` 开始，按顺序加入 `{"op":...,"shape":[...],"attrs":{...}}`。遇到未知前序加入 `{"forget":true}`。 |

`freeze` 保存输入、软件指纹、实际解析过的芯片配置、预测和 UTC 时间，并对文档计算 SHA-256。请在启动采集前将该指纹写入对应 raw 目录，例如执行 `python -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' "$cal_dir/freeze.json" > "$raw/freeze.sha256"`。采集器要另存开始/结束 UTC 时间、完整返回码、真正的输出检查和核级时序。

## 从原始 CA 转换

项目自带的 `msopprof` 适配器能读取 CANN 9.2.1、950DT/PR、BF16 ND、无 bias 的单次 `aclnnMatmul`（一次内核启动）。输入目录包含宿主 `run.log`、`cost.json` 和一次 `profile/OPPROF_*/` 导出。外部目录可用绝对路径。安装包目录参数应指向同一套 CANN 的独立 OPP 对象。

```sh
python -m calibration.msopprof --freeze "$cal_dir/freeze.json" --case bf16-flash-projection \
  --raw "$raw" --library-object "$opp_object" --output "$cal_dir/ca.json"
python -m calibration import "$cal_dir/freeze.json" "$cal_dir/ca.json" "$cal_dir/measured.json"
python -m calibration score "$cal_dir/freeze.json" "$cal_dir/measured.json"
```

具体导出和日志约定如下。

- `cost.json` 提供 `started_utc`、`finished_utc`、整数 `exit_code` 和 `stop_reason`（成功为 null）。
- 宿主在调用处打印 `runtime_soc=...`、`case=matmul api=aclnnMatmul M=... N=... K=... dtype=BF16 layout=ND_ND workspace=...`。独立计算全张量 golden，比较后打印 `correctness count=M*N bad=0`（count 是实际整数）和 `SMOKE_PASS`。同一日志保留 msopprof 的外存配置、内核名称、逐核 duration 表和导出完成记录。转换器读回整份独立输出检查结果。
- 从 `dump/aicore_binary.o` 与用户指定同名系列的 OPP ELF 比较整个对象的 SHA-256；从合并的 `simulator/trace.json` 扫描每条 X 事件，逐核要求恰有一个物理 END、完整名册和与厂商 duration 相符的退休窗。这个适配器绑定该导出的紧凑 JSON 格式与 1.65 GHz 设备时钟；其他格式应按其真实时钟、记录布局增加适配器。

## 给其他算子添加适配器

复用 `freeze/import/score`。新的离线适配器验证实际 shape、dtype、输入属性、芯片、数值和可执行对象，并生成 `kind="ca-receipt", schema=1`。可以参照 Matmul 适配器和例子。按执行顺序列出 stages；单个 stage 取所有核中最早发起至最晚退休，多 stage 的设备时间取各段之和，排除宿主发起间隔。涉及并发重叠的复合调用需提供整次含重叠的统一设备窗，避免重复求和。

- `case` 为完整输入合同。`complete=true`、`timing="all-core-instruction"`、`provider`、`run_log_sha256` 标明收据和计时来源。
- `correctness` 记录逐输出的 `checked` 和 `bad`。每项必须覆盖预登记元素数。特殊算子的 golden 定义由采集程序和合同约定，例如 Indexer 的 top-k 集合相等。
- 各 stage 保存运行时 `kernel`、实际 `kernel_binary_sha256`、`trace_sha256`、`ticks_per_us`、独立获取的 `launched_cores` 和逐核 `cores`。每核必有 `first_tick, end_tick, retire_tick, end_count`。对异步搬运，一条核的 `retire_tick` 可晚于标量 END。通用导入检查字段、范围及冻结绑定，原始身份与 golden 的真实性由可信适配器验证。

采前冻结绑定的收据标 `prospective`。在同一输入上再次采集依然是重复观测；评价形状泛化时，应另外预先划定未见几何和互斥的开发集。晚于采集生成的冻结用于 `replay`。每次 `score` 可传入多份 measurement JSON，同批重复 case/物理 stage 会被拒绝，未收到的 case 列在 `pending`。单次、组合、多个输入协议的物理窗复用应分别评分。

## 查看参考数据

```sh
python -m calibration.lookup matmul --shape 32768 128 512
python -m calibration.lookup kda_chunk --contract fla_kda_v2
python -m calibration.verify
python -m calibration.verify --reprice
```

`verify` 从逐窗口重新计算全部合同的统计值。加上 `--reprice` 会调用已安装后端，为每个窗口分别运行该 950 SKU 和 970DT，并验证与记录中的预测同值。稠密 MX 两条精度路线的验证另有同一 V5 接口的[采集驱动与用法](mx_matmul/README.md)。239 个有序 mHC Post 样例带有真实启动前缀。设备真值、模型墙钟和 970 外推各自单列。原始 trace 可以留在外部证据存储，通过每个 stage 的 SHA-256 定位。[文件指纹与证据快照](reference/manifest.json) · [逐窗 JSONL](reference/windows.jsonl) · [统计](reference/summary.json) · [大形、部分证据与外推](reference/large-shapes.json) · [整体误差手册](../docs/error-analysis.md)
