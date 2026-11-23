# 用 CANN 9.2.1 核对双 MXFP4 分组矩阵乘

选择已经激活 CANN、含 950DT/PR 模拟器的环境。驱动只派发一个 `aclnnGroupedMatmulV5`，给 A/W 分别填可变符号和 E8M0 scale，并按专家独立计算全输出 BF16 golden。W 底层 [G,N,K]，传给 grouped API 时采用 [G,K,N] 转置视图，scale 同步变为 [G,⌈K/64⌉,N,2]。groupList 是 INT64 的每专家**行数**，接口属性 `splitItem=3, groupType=0, groupListType=1, actType=0`。scale 与每个专家的权重分别共享，全部 G 组都用同一个设备 kernel。启动时核对各 SoC 的内存配置和完整设备窗。

```bash
c++ -std=c++17 -O2 calibration/routed_gmm/probe.cpp -I"$ASCEND_HOME_PATH/include" \
    -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath-link,"$ASCEND_HOME_PATH/lib64" \
    -lopapi -lnnopbase -lascendcl -o /tmp/routed-probe

export OPP_GMM="$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/grouped_matmul_apt"
export RAW=/tmp/new-routed-ca
python -m calibration.routed_gmm.campaign dev --raw "$RAW" --probe /tmp/routed-probe --opp "$OPP_GMM"
python -m calibration.routed_gmm.campaign freeze --raw "$RAW" --probe /tmp/routed-probe --opp "$OPP_GMM" --freeze "$RAW/freeze.json"
python -m calibration.routed_gmm.campaign hold --raw "$RAW" --probe /tmp/routed-probe --opp "$OPP_GMM" --freeze "$RAW/freeze.json"
python -m calibration.routed_gmm.bundle --raw "$RAW" --archive /tmp/new-routed.json.gz
python -m calibration.routed_gmm.bundle --archive /tmp/new-routed.json.gz --windows
```

`--launcher /path/to/activate-cann.sh` 可把 CANN 与调用 Python 的环境隔离。该 shell 脚本应执行传入的命令及其参数。稳态复核直接运行 `python -m calibration.routed_gmm.bundle`。开发、独立样本取自 `campaign.py` 不同的列表，hold 验证源文件和驱动哈希，并重算每个已封存预测。每次只运行一个模拟器；msopprof 单样本预算 15 分钟，启动和清理含在 16 分钟墙钟上限中。运行中可用空间低于 75 GiB 时停止。标准 CA 记录含真实 ELF 与已装 OPP 匹配、完整全核时钟边界、逐核指令文件的 SHA-256 与数值覆盖；归档只带紧凑证据。失败或超限单列。同一物理窗可复用，但开发数据不会转换为独立样本。

算例图使用归档的设备时间区间，在装有 matplotlib 的环境运行 `python -m calibration.routed_gmm.plot --output /tmp/routed-example.png` 即可重绘。若有开发原始 trace，可以追加 `--raw "$RAW" --data /tmp/routed-example.json` 重新提取两端末核。
