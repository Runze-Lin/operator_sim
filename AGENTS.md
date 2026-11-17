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
