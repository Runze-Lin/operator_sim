# 系统目录的 950 CA 对照

AddRmsNorm 直接调用已安装的 aclnnAddRmsNorm；输出 BF16 y、BF16 残差和、FP32 逆标准差均有 CPU golden。开发窗按 SKU 核验内存配置、已安装 ELF、所有活动核最早发起及 END。十个独立 shape 使用同一份事前冻结文件。验证输入、两端分核及 970DT 假设外推见[逐窗结果](../../docs/examples/catalog-ca.md)。

在已激活 CANN 9.2.1、能运行 msopprof simulator 的环境下，SDK 指向具有 include 和 lib64 的平台目录，OPP 指向已安装 kernel/ascend950 目录。RAW 为新的本地目录，保留足够空间安放 trace。使用 CANN Python 环境之外的控制进程时，可用可选的 --launcher 指定加载该环境并执行其参数的 shell 脚本。

~~~sh
g++ -std=c++17 -O2 calibration/catalog_ca/add_rms_norm.cpp -I"$SDK/include" \
    -L"$SDK/lib64" -Wl,--allow-shlib-undefined -lopapi -lascendcl -lnnopbase -o /tmp/add-rms
python -m calibration.catalog_ca.campaign collect --raw "$RAW" --probe /tmp/add-rms --workers 1
python -m calibration.catalog_ca.campaign score --raw "$RAW" --opp "$OPP" --workers 1
python -m calibration.catalog_ca.report
~~~

collect 复用发布的冻结预测；freeze 只用于新建试验。每个 shape、SKU 有独立的固定预测、运行日志和模拟器墙钟。超时后可指定新的 --retry-raw，只重试已结束的失败样本，score 传入相同两处路径。失败数据不能参与 WAPE。原始 dump、数据库和指令 trace 留在本机，仓内仅保留核验与复算所需的摘要。[LayerNorm 分段核验](../../docs/examples/catalog-layer-norm-ca.md) · [BatchMatMul 与 MXFP8 试验](../../docs/examples/catalog-cube.md)
