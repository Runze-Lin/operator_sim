<!-- cannbot:ops-perf-optimize:start -->

---
name: perf-optimize
description: AscendC / CANNBot-DSL 算子性能调优 Agent 团队入口。仅在用户明确要求对某个具体 AscendC 算子或代码目录、CANNBot-DSL 程序做性能优化/性能调优时，编排 ops-perf-optimize Skill；一般咨询、报告分析、精度排障或仅提及 cases.csv、msprof、profiling、加速比时不触发。
mode: primary
skills:
  - ops-perf-optimize-workflow
permission:
  external_directory: allow
---

# PerformanceOptimizationAgent

本文件只负责注册主 Agent 和触发 Skill，不承载工作流规则。

只有同时满足以下两项，才是本工作流的触发条件：

1. 用户有明确的执行意图：要求“性能优化”“性能调优”“降低耗时”“提升某算子性能”等，而不是询问、解释或分析已有材料；
2. 用户给出了具体 AscendC 算子、CANNBot-DSL 程序、算子代码目录，或上下文中已有唯一明确的待优化算子。

满足触发条件时：

1. 必须调用并完整读取 `ops-perf-optimize-workflow` Skill；其源码入口位于本插件的 `/home/l00925865/operator_sim/.agents/skills/ops-perf-optimize-workflow/SKILL.md`。
2. 把用户完整请求原样交给该 Skill，按 Skill 指向的统一工作流契约执行。
3. 禁止从本文件、会话记忆或旧版入口重构 Stage、门禁、角色分工和验收规则。

以下情况不得启动完整工作流：安装或使用咨询、解释性能概念、分析已有报告或日志、查看 profiling 数据、精度排障、普通代码检视，以及只出现 `cases.csv`、`msprof`、`profiling`、加速比或性能目标但没有明确优化动作和目标算子的请求。此时按普通请求直接回答；不要加载 Skill、冻结输入、创建运行目录或派发专家。

若运行时没有 Skill 调用能力，先定位本文件的真实插件根目录，再直接完整读取 `<plugin-root>/skills/ops-perf-optimize-workflow/SKILL.md`；读取完成前不得编译、上板或派发任何 Stage。

<!-- cannbot:ops-perf-optimize:end -->

<!-- cannbot-knowledge:start -->
## cannbot-knowledge 使用入口

本项目已安装可检索、可溯源的 CANN 知识库。若安装时配置了 `--platform <id>`，该值会写入
`.cannbot/knowledge.env` 的 `CANNBOT_KNOWLEDGE_PLATFORM`，普通检索省略 `--platform` 时使用它；
任务目标与该值不一致时显式传入目标平台。先按问题是否已经绑定唯一知识卡选择入口：

| 当前问题 | 入口 |
|---|---|
| 用户给出了确定、唯一的知识卡路径，且问题只需要该卡内容 | 直接阅读全文，例如 `/home/l00925865/operator_sim/third_party/cannbot-knowledge/knowledge/common/platforms/concepts/ascend_910_95.md` |
| Ascend C API 的契约、选型、相关知识或本地头文件 | `.agents/skills/ascendc-api-knowledge-query/SKILL.md` |
| CANN Runtime API 推荐、比较或 CUDA Runtime/Driver 替代 | `.agents/skills/runtime-api-knowledge-query/SKILL.md` |
| 未知卡片路径、需要多卡综合，或涉及算子实现、调试、优化、测试与跨技术问题 | `.agents/skills/knowledge-query/SKILL.md` |

直接读卡不需要先运行检索；必须读取完整 Frontmatter 和正文，并核对平台、版本、状态与来源。如果
该卡不足以回答问题，再按新增的证据缺口进入对应 Skill。三个检索入口共用 Query 的索引和平台规则，
不需要重复启动。缺少入口时由项目安装流程补齐，不扫描知识树替代 Skill。

### 使用边界

- 没有明确单卡路径时，在 CANN、昇腾 NPU、算子 DSL、图、模型或 Runtime 的首次领域决策前进入
  对应检索 Skill。目标平台确认、候选召回、选卡全文阅读及按需补证依照所选 Skill 进行；加载入口
  本身不算完成检索。
- 知识仓通过项目配置定位，不把业务工程当成知识仓。实际覆盖由 Query 确认，目录占位、导航数量或
  搜索零命中都不证明技术能力是否存在。
- 消费流程保持只读：必要权限被拒绝、索引缺失或陈旧时停止并报告，不绕过检索、不临时重建索引，
  也不自动下载、编译或修改知识、源码、SDK 与全局环境。
- 已读证据足够回答当前问题时结束；出现新的领域证据缺口时定向重入。需要工程实验或贡献知识时，
  必须属于用户本次授权，不因为检索 Skill 提到验证方法就自动执行。

平台识别、草稿补搜、相关链接、来源路径、输出范围和复杂任务审计的具体规则只在对应 Skill 及其
按需参考中维护。本安装提示仅负责触发、路由和使用边界，不另定义检索步骤；用户明确要求不使用
知识库时遵从该要求。
<!-- cannbot-knowledge:end -->

## 建模迭代交付（用户要求）

持续维护两份覆盖整套当前模型的总文档，并在 README 作为主要入口：

1. `docs/modeling-strategy.md`：全量当前建模方法；每个算子的公式及其物理/统计含义、分核/tiling/流水规则、参数与校准来源、适用范围、未校准部分、预测开销及外推边界。不能仅描述本轮改动或用旧文档链接代替当前方法。
2. `docs/error-analysis.md`：全量当前误差分析。开头 Summary 必须是逐算子的误差情况表，列样本数、数据角色、WAPE/最大误差和校准状态；没有耗时校准/实测的算子标 N/A，不能填0或暗示已验证。后续给逐shape预测/CA、实际ACLNN接口与ops-nn/ops-transformer库kernel身份、正确性/指令/分核检查、误差来源和仿真成本。开发回放和冻结独立验证分开统计。

每轮另写 `results/vX.Y/changes.md`，简短记录方法变化、同集误差变化、采集及预测成本、未完成范围；模型、冻结预测和评估JSON保留版本。历史交付文件不改写，不把用于调参的旧留出继续标为独立验证。可复用的原始trace尽量复用，只新增少量有机制区分价值的shape，并在采集前冻结模型和预算。

区分已证实根因与假设，完整实测、部分trace、超时/失败和外推不可混用。明确区分安装包实际调用的算子与仅作参考的开源源码；“采集方式一致性校验”指同旧shape对比采集链路，不把“桥接”当作算子名称。

## 通用建模约束（用户补充）

- 总建模方法文档开头必须说明通用思想：算法工作、实现策略/tiling、硬件资源成本和依赖调度分层；不同算子类别采用适合其机制的方法。
- 不以950DT某些shape的总时延回归/查表冒充可泛化模型。局部拟合只能是明确标记的对照；校准参数须解释物理含义、来源、单位和可辨识性。
- CA正确性、计时资格、机制预测资格分别报告。缺少指令映射/资源吞吐/发起间隔等证据时，不把局部拟合结果填入通用模型误差栏；跨代可替换参数不等于跨代准确性已验证。

## 当前模型验收

默认入口按 Vector、Cube、CV 融合选用对应实现。总方法开头解释每类的思想、实际步骤和算例。总误差表展示该算子的一个当前预测，列明开发与独立样本数量、独立 WAPE / 最大误差、单次模型查询墙钟。每个算子目标至少十个不同的独立 shape，覆盖工作量大小和分支边界。系统仿真中单次模型查询预算 1 秒，离线 CA 开销单列。未达标的路径保留具体原因。

在总建模方法的每个算子介绍内提供至少一个可复核的小算例，说明 shape 与关键属性、如何得到工作量或阶段时间，并标明计时的证据口径。组件与组合分别举例。

## 系统仿真工具交付

首页以调用方为中心，保留工具用途、安装、后端接口、模型与 CA 误差入口。开发和迭代叙述放在 `development/` 或版本化结果目录。发布前从源码目录外验证安装后的公共 API 与参数资源；分别报告设备估计微秒数和查询墙钟，在线查询优先控制在 100 ms 内。

逐算子的流水对照图优先采用用户的 Matmul 两波两页示例。上栏放建模得到的排程，下栏放真实 CA；使用同一设备零点、横轴比例、资源颜色和工作单元行，分别标出模型整窗和完整 CA 整窗。用图注解释首发刻度、描述符等待和未识别的阶段边界。模型只有独占成本时标清预算口径；缺完整 CA 的路径画诊断并标 N/A。由保留的时间戳与图数据重绘，入口在 docs/pipeline-comparisons.md。
