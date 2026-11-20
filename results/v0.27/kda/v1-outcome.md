# Packed V1 四段原生执行

两条各一 token，总 T=2、H=1、K=V=128。原始四组输出的逐元素超差数顺序为 O / Aqk / Akk / 最终 VK。正式 µs 是同一公开 V1 申请的全部四个主设备 kernel 窗之和。辅助转置、提交间隔及 Kimi 外部 Q/K 归一化和状态路由另计。

|芯片|状态|物理 V1 主核|四组数值超差元素|同步检查|正式主核 µs|依据|
|---|---|---:|---|---:|---:|---|
|950DT|未合格|0/4|—, —, —, —|目标未发起|N/A|[v1-Ascend950DT_9582.json](v1-Ascend950DT_9582.json)|
|950PR|DT 全合同条件未触发 PR|—|—|—|N/A|[v1-Ascend950PR_957b.json](v1-Ascend950PR_957b.json)|

只有两组公开符号绑定、四组数值、未改初态、四份正式融合设备代码、全部主辅全核窗、时间顺序、正常退出与零同步错误同时成立才有正式设备时间。采中的零个错误须等程序结束才能判定。上游 host 中同一 stage 参数依次取 0、1、2、3；实际 ELF ordinal、原始窗口、官表与错误由逐卡回执留存。[同一 ABI 的发布版代码](https://github.com/flashserve/flash-linear-attention-npu/blob/5cc699069c7f91a35aabd583032c8720a0b626fa/fla/ops/ascendc/kda/chunk_kda_fwd/op_host/op_api/aclnn_chunk_kda_fwd.cpp#L879-L898) · [采集成本与宿主调度](host-scheduling.json)

同一组两请求输入的独立宿主对照在原运行结束后登记，仍由同一 DT 正式库和四组金标核验。两次运行各有独立回执。[对照结果](host-control/report.md)
