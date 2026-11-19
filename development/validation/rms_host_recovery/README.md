# v0.21 RMSNorm 采集宿主修复

原冻结含 10 个 BF16 RMSNorm shape。smoke/large_vector.cpp 在调用 aclInit 前为 quant 和 RMSNorm 共同限制 C≤7168 且 C 能被 64 整除。7 个 shape 的宽度超出该宿主入口，另 3 个 shape 保留原采集。修复后 RMSNorm 入口接受本工具支持的 0<C≤8192 且 C 能被 16 整除。R 的范围、量化入口、张量、输入值、设备代码、金标和采集命令均与原来相同。

1. 用项目的 Python 运行 build.py。在 build/rms-host-recovery 生成独立的源码副本和 host。脚本核对原源码 SHA 后仅替换参数检查。不要覆盖旧的 build/smoke/large_vector。
2. 首次采集前运行 recover.py register。results/v0.21/rms/host-recovery.json 固定新/旧文件指纹、原预测、候选 14 个端点、两个 SKU 和原预算。已有登记直接用于续跑。
3. 运行 recover.py run。它等待历史大 RMS 队列释放槽位，以单槽串行先测 950PR [32,1280]，与相同旧宿主的历史合法宽度、同输入数值和完整窗口比较。随后核验每个原回执、日志及“尚未发出任何设备 kernel”后补采。完整成本和 trace 留在 runs/rms21-host-recovery。重启遇到未结束的旧尝试会报告原目录，防止覆盖。
4. 得分程序从 results/v0.21/rms/receipts 读取原合格端点，并从 retries 根目录读取仅有一次的合格补采。control 子目录不属于独立形状。每个补采回执都有原始失败、两输出逐元素金标、精确 SKU/内存类型、包内 ELF、全核设备窗与墙钟成本。每端预算沿用原登记。

控制例子从首次设备指令到全核最后一条指令计时。按供应商各核 duration 表复核每个核上下界。从 CA 控制面的设备零点和输出文件创建时间得不到这个窗口。
