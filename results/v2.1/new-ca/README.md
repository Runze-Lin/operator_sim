# 三次冻结后 FA 验证的复现方法

[冻结预测](predictions.json) 保存预测时刻、各模型输出、运行程序与参数哈希。[实测记录](receipts.json) 包含输出校验、实际 kernel、各核时间窗和 RVEC 明细。三次采集均为稠密非因果 SBH，维度顺序为 Q、KV、D、B、H。本次新增的是两个几何形状、三次 CA。BF16 用同一份 host 和相同输入分别采 DT、PR。

## 1. 准备输入和环境

以下命令在发布仓根目录运行，使用 Linux AArch64、GCC、CANN 9.2.1 toolkit、同版算子包以及目标 SKU 的 CA simulator。用本机安装前缀替换第一行的示例路径。当前工作机使用独立 conda 环境启动，亦可在已激活的 CANN 环境中执行后续命令。

```bash
source /path/to/cann-9.2.1/set_env.sh
project_dir="$PWD"
export fa_raw="$(mktemp -d)"
mkdir "$fa_raw/bin"
g++ -std=c++17 -O2 -g -Wall -Wextra results/v2.1/new-ca/three_ops.cpp \
  -I"$ASCEND_HOME_PATH/include" -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath-link,"$ASCEND_HOME_PATH/lib64" \
  -lopapi -lnnopbase -lascendcl -o "$fa_raw/bin/three_ops"
g++ -std=c++17 -O2 calibration/examples/fa_fp16_smoke.cpp \
  -I"$ASCEND_HOME_PATH/include" -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath-link,"$ASCEND_HOME_PATH/lib64" \
  -lopapi -lnnopbase -lascendcl -o "$fa_raw/bin/fa_fp16"
```

[BF16 调用方](three_ops.cpp) 是冻结时 `smoke/three_ops.cpp` 的原始文件，其 SHA256 完全相同。[FP16 调用方](../../../calibration/examples/fa_fp16_smoke.cpp) 单独创建 FP16 的 Q/K/V 和输出。两个调用方都通过 `aclnnFlashAttentionScoreV2` 执行安装包中的实现，逐一检查 attentionOut 的 576 个元素。校验界为绝对误差 ≤ 0.002 + 0.01 × |参考值|。辅助统计量没有数值 golden。

## 2. 核对模型冻结并保存本轮运行的身份

在本目录之外另外安装依赖时，Python 应可导入源码仓的 `operator_model`。以下命令验证模型输入文件、已经发布的冻结快照与 CA 记录之间的内容绑定，逐项复算已冻结值，并在开始新的 CA 运行前写入本机 host 哈希。发布版在公共 API 增加 implementation 别名，下面对该文件接受冻结版和发布版两个确切哈希，并对每个 shape 和策略重新运行预测。采用 `-g` 编译的二进制会嵌入源码路径，其他机器可以用源码哈希确认输入身份。

```bash
PYTHONPATH=src python3 - <<'PY'
import hashlib, json, math, os
from pathlib import Path
from operator_model import estimate
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
base = Path('results/v2.1/new-ca')
f = json.loads((base/'predictions.json').read_text())
r = json.loads((base/'receipts.json').read_text())
assert sha(base/'predictions.json') == r['freeze_sha256']
for path, expected in f['hashes'].items():
    if path == 'src/operator_model/backend.py':
        release_hash = 'efd02b6dd324eb29a02a1b929af660e44c9605cdfe3f50396df65327e124d2f3'
        assert sha(path) in (expected, release_hash), path
    elif path.startswith('src/'):
        assert sha(path) == expected, path
assert sha(base/'three_ops.cpp') == f['hashes']['smoke/three_ops.cpp']
assert sha('calibration/examples/fa_fp16_smoke.cpp') == f['fp16_source_sha256']
arms = {'baseline': {}, 'classes': {'vector_model': 'instruction_classes'},
        'probe': {'fa_port_model': 'probe'},
        'both': {'vector_model': 'instruction_classes', 'fa_port_model': 'probe'}}
for sku, forecasts in f['forecasts'].items():
    for arm, attrs in arms.items():
        prediction = estimate('fa', f['shape'], attrs, chip=sku)
        expected = forecasts[arm]
        assert prediction.status == expected['status']
        if expected['us'] is None:
            assert prediction.latency_us is None
        else:
            assert math.isclose(prediction.latency_us, expected['us'], rel_tol=1e-12)
        assert (prediction.active_aic, prediction.active_aiv) == (expected['aic'], expected['aiv'])
for sku, expected in f['fp16_forecasts'].items():
    assert math.isclose(estimate('fa',f['fp16_shape'],dtype='fp16',chip=sku).latency_us, expected)
raw = Path(os.environ['fa_raw'])
snapshot = {k:sha(raw/'bin'/k) for k in ['three_ops','fa_fp16']}
snapshot['published_freeze'] = r['freeze_sha256']
with (raw/'local-freeze.json').open('x') as dest:
    json.dump(snapshot, dest, indent=2)
PY
```

## 3. 采集三个完整时间窗

启动主机需要支持 CANN simulator 的本地 Unix IPC。下面每条命令启动一次 CA，在独立目录写入原始 trace。发布证据的每次预算为 300 秒和 750 MiB，分别实际用时 96.09、100.03、70.07 秒。以下复现命令用 GNU timeout 限时，每 2 秒检查目录大小。相同 shape 或精度的额外重复实验需使用新的输出目录。

```bash
export OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
run_fa() (
    set -e
    local sku="$1" dtype="$2"
    shift 2
    mkdir "$fa_raw/$sku-$dtype"
    cd "$fa_raw/$sku-$dtype"
    # 单独的进程组方便达到磁盘预算时停止模拟器子进程。
    setsid timeout --kill-after=5s 300s msopprof simulator \
        --soc-version="$sku" --dump=true --launch-count=20 --timeout=5 \
        --output="$PWD/profile" "$@" >run.log 2>&1 &
    local job=$!
    while kill -0 "$job" 2>/dev/null; do
        if (( $(du -sm . | cut -f1) > 750 )); then
            kill -TERM -- "-$job" 2>/dev/null || true
            wait "$job" || true
            return 1
        fi
        sleep 2
    done
    wait "$job"
)
run_fa Ascend950DT_9582 bf16 "$fa_raw/bin/three_ops" fa 9 4097 64 1 1
run_fa Ascend950PR_957b bf16 "$fa_raw/bin/three_ops" fa 9 4097 64 1 1
run_fa Ascend950DT_9582 fp16 "$fa_raw/bin/fa_fp16"
```

## 4. 核对整窗和指令类别

用下面的命令对任一采集目录重算整窗和部分指令数。原始 trace 留在采集机，路径模式见实测记录的 `collection.raw_path`。本机原始目录前缀为 `runs/fa-vector-v21-local-ipc/`。记录中包含每个 trace 和实际 kernel 二进制的 SHA256。发布时使用 runtime SKU、内存模型、唯一 kernel、golden、厂商逐核 duration 和各核 END 联合验收。两个 SKU 的 CA 时钟均为 1.65 GHz；trace 的 ts 和 dur 以 µs 表示，核级时钟换算为 1650 cycle/µs。

```bash
python3 calibration/examples/inspect_fa_vector_ca.py "$fa_raw/Ascend950DT_9582-bf16" Ascend950DT_9582 bf16
python3 calibration/examples/inspect_fa_vector_ca.py "$fa_raw/Ascend950PR_957b-bf16" Ascend950PR_957b bf16
python3 calibration/examples/inspect_fa_vector_ca.py "$fa_raw/Ascend950DT_9582-fp16" Ascend950DT_9582 fp16
```

RVEC 选择正时长的 X 事件统计已执行指令，有同一 PC 的零时长标记时另计。DT 的 9 个 query 行共见 9 个正时长 VDIV，分布为 5+4，另有 9 个零时长标记。每个模型指令类的比较映射列在 `receipts.json → rows → class_check`。没有 MMAD 参数详情的指令保留为 unreported。FP16 的 6 个 MMAD 均有元数据，两个输入为 f16，累加为 f32。970DT 仅复算预测。
