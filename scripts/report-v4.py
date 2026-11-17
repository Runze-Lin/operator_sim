#!/usr/bin/env python3
"""Detailed Chinese validation report and standalone scientific figures."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
OUT=ROOT/'results/v0.4'
COLORS=['#167d9a','#e09c32','#939ba8']

def main():
 e=json.loads((OUT/'evaluation.json').read_text());w=json.loads((OUT/'workload-coverage.json').read_text());d=json.loads((OUT/'calibration-diagnostics.json').read_text())
 attempts=sorted(json.loads((OUT/'dataset.json').read_text())['attempts'],key=lambda a:a['finished_utc'])
 rows=e['rows'];fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained')
 for col,op in enumerate(('quant','fa')):
  rs=[r for r in rows if r['op']==op];x=np.arange(len(rs));labels=['x'.join(map(str,r['shape'])) for r in rs]
  for i,(key,label,color) in enumerate(zip(('predicted_us','v03_us','constant_us'),('v0.4','v0.3 formula','constant (old data)'),COLORS)):
   axs[0,col].bar(x+(i-1)*.24,[100*abs(r[key]/r['reference_us']-1) for r in rs],.24,label=label,color=color)
   axs[1,col].plot(x,[r[key] for r in rs],'o-',label=label,color=color)
  axs[1,col].plot(x,[r['reference_us'] for r in rs],'ks--',label='CA')
  axs[0,col].set_title(f'{op}: {len(rs)} completed fresh holdouts')
  for ax in axs[:,col]:ax.set_xticks(x,labels,rotation=30,ha='right');ax.grid(axis='y',alpha=.2)
  axs[0,col].set_ylabel('Absolute percentage error (%)');axs[1,col].set_ylabel('Instruction makespan (us)')
 axs[0,0].legend(fontsize=8);axs[1,1].legend(fontsize=8)
 fig.suptitle('Same frozen new holdouts | failed/unrun cases excluded from accuracy, listed in report')
 for ext in ('png','svg'):fig.savefig(OUT/f'holdout-comparison.{ext}',dpi=160)
 plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
 for ax,group,title in zip(axs,('holdout_quant','workload_quant','holdout_all'),('Quant holdouts','Workload quant shapes','All planned holdouts')):
  b=e['benchmark'][group];x=np.arange(2)
  for i,(key,color) in enumerate(zip(('v03','v04'),(COLORS[1],COLORS[0]))):
   ax.bar(x+(i-.5)*.32,[b[m][key]['p95_us'] for m in ('uncached','cached')],.32,label=key,color=color)
  ax.set_xticks(x,['Uncached','Cached']);ax.set_yscale('log');ax.set_ylabel('P95 query time (us, log)');ax.set_title(title);ax.grid(axis='y',alpha=.2)
 axs[0].legend();fig.suptitle('Paired full predict API benchmark after CA; same inputs, alternating version order')
 for ext in ('png','svg'):fig.savefig(OUT/f'query-cost.{ext}',dpi=160)
 plt.close(fig)
 # Explicit quant coverage, not whole-workload accuracy. Context aliases are intentional.
 held={tuple(r['shape']) for r in rows if r['op']=='quant'}
 cells=[];labels=[]
 for phase,batches in [('prefill',(1,4,16)),('decode',(1,4,16,32,128,256))]:
  for b in batches:
   labels.append(f'{phase} B={b}');line=[]
   for ctx in (32768,131072,262144,524288):
    keys={s['quant_key'] for s in w['scenarios'] if s['phase']==phase and s['batch_per_card']==b and s['prompt_tokens']==ctx}
    measured=sum(tuple(w['unique_predictions'][k]['shape']) in held for k in keys)
    line.append((measured,len(keys)))
   cells.append(line)
 fig,ax=plt.subplots(figsize=(9,6),layout='constrained');arr=np.array([[a/b for a,b in row] for row in cells]);im=ax.imshow(arr,vmin=0,vmax=1,cmap='Blues',aspect='auto')
 for i,row in enumerate(cells):
  for j,(a,b) in enumerate(row):ax.text(j,i,f'{a}/{b}',ha='center',va='center')
 ax.set_xticks(range(4),['32k','128k','256k','512k']);ax.set_yticks(range(len(labels)),labels)
 ax.set_title('Quant coverage: direct CA / unique predicted shapes\nContext aliases share shapes; no FA validation implied',fontsize=12)
 fig.colorbar(im,ax=ax,label='Directly measured quant-shape fraction')
 for ext in ('png','svg'):fig.savefig(OUT/f'workload-coverage.{ext}',dpi=160)
 plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
 names=[a['case']['id'].replace('_test_','_') for a in attempts];x=np.arange(len(attempts))
 colors=['#167d9a' if a['status']=='ok' else '#c65c4b' for a in attempts]
 axs[0].bar(x,[a['wall_seconds'] for a in attempts],color=colors);axs[0].set_ylabel('Capture wall time (s)')
 axs[1].bar(x,[a['raw_bytes']/2**30 for a in attempts],color=colors);axs[1].set_ylabel('Raw artifacts (GiB)')
 for ax in axs:ax.set_xticks(x,names,rotation=55,ha='right');ax.grid(axis='y',alpha=.2)
 fig.suptitle('Incremental CA cost | blue=complete, red=failed (included in cost)')
 for ext in ('png','svg'):fig.savefig(OUT/f'ca-cost.{ext}',dpi=160)
 plt.close(fig)
 cost=e['collection_cost'];b=e['benchmark'];q=e['metrics']['quant'];small=e['metrics']['quant_small']
 lines=['# v0.4 验证：quant 小 shape 回退、预测开销与长上下文边界','',
  '## 结论与口径','',
  f"复用 38 个历史输入（其中 quant 13 个），仅补充读取此前未归约的 3 个 quant trace。新留出计划 10 个；本轮成功 {len(rows)} 个，未完成/失败 {len(e['missing'])} 个。预测在采集前冻结，采集后没有调参；v0.3 的代码、参数和旧报告保持原状。",'',
  '计时基准是 **CA 跨核指令窗口 makespan**，不含框架、Host enqueue 或窗口外 dispatch。不是实卡端到端延迟。以下误差只针对成功完成的全新留出点，失败点不参与误差平均，也不算通过。', '',
  '| 组 | 完成点数 | v0.4 WAPE | v0.3 WAPE | 固定耗时 WAPE | v0.4 P90 APE |',
  '| --- | ---: | ---: | ---: | ---: | ---: |']
 for group,label in [('quant_small','quant 小工作量'),('quant_ub','quant 较大工作量'),('quant','quant 全部'),('fa','FA 组件'),('all','全部（受长耗时点权重影响）')]:
  a=e['metrics'][group];m=a['predicted_us']
  if m:lines.append(f"| {label} | {m['count']} | {m['wape']:.2%} | {a['v03_us']['wape']:.2%} | {a['constant_us']['wape']:.2%} | {m['p90_ape']:.2%} |")
 lines+=['','验收口径为每类全部完成且 WAPE≤10%、P90 APE≤20%；完整门禁见 evaluation.json。缺失任何该类计划点，即不宣称该类完整通过。查询开销另检查未缓存P95≤1000μs、缓存P95≤10μs。', '',
  'WAPE=Σ|预测−CA|/ΣCA；P90 为逐点 APE 的第 90 百分位。小工作量定义为每核≤4096 元素，独立于本轮测量结果。固定耗时主基线为历史 38 个输入中各算子的均值；另保留 v0.3 原始均值基线，数值见 evaluation.json。', '',
  'v0.3 原有输入校验上限是维度 4096、FA B=1。为了在同一新集合比较，传入 v0.4 Case 仅绕过旧输入 guard；**v0.3 公式和参数没有扩展校准**。宽度 5120/7168、B>1、长 KV 上的 v0.3 比较值应称为“冻结公式外推”，不是原版已支持能力。', '',
  '![新留出误差与耗时](holdout-comparison.png)','',
  '## 逐点独立验证','',
  '| 新留出 shape | CA μs | v0.4 μs | v0.3 μs | 固定 μs | v0.4 APE | 核数 预测/CA (AIC,AIV) |',
  '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
 for r in rows:lines.append(f"| {r['op']} {r['shape']} | {r['reference_us']:.4f} | {r['predicted_us']:.4f} | {r['v03_us']:.4f} | {r['constant_us']:.4f} | {r['ape']:.2%} | {r['predicted_cores']}/{r['observed_cores']} |")
 lines+=['','数值正确性：每个成功 case 均通过调用程序的全输出 golden 检查及 SMOKE_PASS；quant 检查精确 FP8/scale 字节，FA 为完整 CPU softmax(QK)V 对照（atol=0.002、rtol=0.01）。固定输入模式不是随机精度压力测试。', '',
  '| 指令验证 | 预测的最大单核次数 | CA 最大单核次数 | 相对差 |',
  '| --- | ---: | ---: | ---: |']
 for r in rows:
  p=r['predicted_opcode_count'];a=r['observed_opcode_count'];err=f'{abs(p-a)/a:.4%}' if p is not None and a else '未覆盖'
  lines.append(f"| {r['id']} | {p} | {a} | {err} |")
 lines+=['','quant 拟合计数存在浮点尾差，按最近整数与 CA 比较。RV_* 包含 load/store/control，不能直接当纯算术利用率；FA MMAD 计数仅在 Q tile=64 路径有结构公式。Q tile=128 返回未验证。', '',
  '### 失败、超时与未采集','', '| case | shape | 状态 | profiler/外层超时 | 原因 |','| --- | --- | --- | --- | --- |']
 for m in e['missing']:lines.append(f"| {m['id']} | {m['shape']} | {m['status']} | {str(m.get('profiler_timed_out'))+' / '+str(m.get('outer_timed_out'))} | {m['error'] or '见原始 run.log；不补入推测测量值'} |")
 if not e['missing']:lines.append('| 无 | — | — | — | — |')
 lines+=['','注意：profiler 内部超时可能仍返回码0，并打印 All task success；缺少 SMOKE_PASS/正确性检查时一律判失败。部分 trace 只保留作后续诊断，不当成完整 CA 耗时。FA 设定 profiler 内部限时2分钟、外层进程限时3分钟。', '',
  '## quant 回退为什么改','',
  '旧 trace 显示小 case 的大部分时延在 VF 前和写回等待中。v0.3 用 log2(1+活跃核数) 持续抬高控制耗时下界，容易把小 shape 在多核下的等待差异拟合成确定趋势。本轮将控制项输入截断为 min(log2(1+AIV),4)，让多核小工作量回到共享的统计耗时；计算、DMA、UB 双缓冲递推仍保留。截断值是经验模型假设，不是硬件在 15 核饱和的结论。', '',
  '使用旧 13 个 quant shape 比较 5 个候选：原控制式重拟合、常数控制项、饱和控制项、25%/50% 斜率收缩。逐点留出时重拟合全部 phase 参数，按小工作量 WAPE 选择；这仍属于开发集模型选择，不能当独立验证。', '',
  '| 控制候选 | 旧数据逐点留出：全部 WAPE | 小工作量 WAPE |','| --- | ---: | ---: |']
 for k,v in d['candidate_cv'].items():lines.append(f"| {k} | {v['all']['wape']:.2%} | {v['small']['wape']:.2%} |")
 lines+=['','固定耗时为什么有时更好：shape 总元素增长不等于每核工作量增长，核数同时增加时，计算工作会被摊薄。小算子总时间的窄区间主要由启动、同步与排队决定；添加缺少证据的核数/带宽参数反而放大误差。固定均值具有低方差，但无法描述多 UB 迭代及很长 KV 的增长。应分小工作量和大工作量评价，而不是用一个总 WAPE 决定所有 regime。', '',
  '## 未缓存预测开销','',
  'quant 现在只做一次分核/UB 计划、一次流水递推、一次不可变结果构造；不再先完整调用 v0.2/v0.3 并复制结果。旧版本耗时和基线比较放到离线评估。predict_us 提供可选纯标量结果，但下面主比较始终使用包含 plan/phase 的完整 predict API。', '',
  '| 同一输入集、同一 API | v0.3 未缓存 P50/P95 μs | v0.4 未缓存 P50/P95 μs | v0.3/v0.4 缓存 P95 μs |',
  '| --- | ---: | ---: | ---: |']
 for key,label in [('holdout_quant','6 个新 quant'),('workload_quant','场景去重 quant'),('holdout_all','全部 10 个计划输入')]:
  t=b[key];a=t['uncached']['v03'];v=t['uncached']['v04'];lines.append(f"| {label} | {a['p50_us']:.2f}/{a['p95_us']:.2f} | {v['p50_us']:.2f}/{v['p95_us']:.2f} | {t['cached']['v03']['p95_us']:.2f}/{t['cached']['v04']['p95_us']:.2f} |")
 lines+=['','每版本/每模式 4000 次，交替版本顺序，CA 完成后测量，Case 已构造，缓存/未缓存显式分开；未缓存没有隐式 shape 缓存。相同计划输入包括失败 CA 点，仅用于 Python 查询计时。共享主机上的结果有调度波动；未做整网仿真器集成，不能据此声称系统开销已≤5%。', '',
  f"纯标量 predict_us 在去重 quant 场景上未缓存 P95={b['v04_scalar_uncached_workload_quant']['p95_us']:.2f} μs，单独报告，不用于完整 API 加速比。",'',
  '![查询开销](query-cost.png)','',
  '## 实际场景映射及覆盖','',
  '输入长度按 1024 倍数，分别为 32768/131072/262144/524288；输出 1024。采用你给出的单卡 batch：prefill=1/4/16，decode=1/4/16/32/128/256。暂用 TP=8、每请求 prefill chunk=512/2048；这两项尚未固定，实际调度若按整批 token budget 分配，需重新生成 shape。', '',
  '| 阶段 | tokenwise quant 激活 shape | FA 组件代理 shape (Q,KV,D,B,H) |',
  '| --- | --- | --- |',
  '| prefill | (B×chunk, hidden) | (chunk, prompt, 128, B, heads/8) |',
  '| 普通 decode | (B, hidden) | (1, prompt+已生成量, 128, B, heads/8) |',
  '| 推测 8 / 主模型校验 9 | (9B, hidden) | (9, prompt+已生成量, 128, B, heads/8) |', '',
  'decode 采样生成进度 0 和 1024 两端。1k 输出是多步生成过程，不是单算子 Q=1024；实际校验轮数、有效 token 比例还依赖接受率。head TP 不自动把 tokenwise quant 行数再除以 TP。不同上下文常复用完全相同的 quant shape，已显式去重。', '',
  f"生成 {len(w['scenarios'])} 条场景映射，{len(w['unique_predictions'])} 个去重组件预测；这些是**输入合法性与预测路径检查，不是全部 CA 准确性验证**。完整逐项数据见 [workload-coverage.json](workload-coverage.json)。",'',
  '| 模型 | hidden | TP8 query heads | 真实注意力差异 |','| --- | ---: | ---: | --- |',
  '| DeepSeek V4.1 Flash | 5120 | 8 | head_dim=512、window=128、逐层压缩比与复用配置 |',
  '| DeepSeek V4 Pro | 7168 | 16 | head_dim=512、window=128、压缩比 4/128 等 |',
  '| Kimi K3 | 7168 | 12 | MLA qk_nope=128+rope=64、v=128；另有 KDA D128；路由专家宽度3584 |', '',
  '这些差异意味着单凭 config 不能唯一确定实际 ACLNN kernel、融合、cache layout 和每卡 tensor。当前 D128 普通非因果 FA 是有意保留的代理：不能替代 DeepSeek 的 D512/稀疏压缩路径，不能把 Kimi 的 MLA 与 KDA 都当成普通 FA。专家宽度3584、MoE EP/TP 路由量和通信尚未纳入本轮实测。', '',
  '新增 FA 的 B4/KV256 与 B16/KV4096 用于 batch 与分核/流水组件验证；它们数值上对应 32k/128 与512k/128，但不表示已经实现真实压缩注意力。32k 和 512k+1k 直接探针均只有一个 head；全卡 H=8/12/16 或更高 batch 的代价不能由一次单头验证自动证明。128k、256k 的完整 KV 本轮未新增 CA，其结果仍为外推。', '',
  '![场景 quant 实测覆盖](workload-coverage.png)','',
  '图中相同上下文列覆盖一致是预期行为：tokenwise quant 当前批的 shape 不随历史 KV 长度变化。图中每格是直接实测过的去重 quant shape 数/该格预测的 shape 数，不能理解为整网覆盖率。', '',
  '## 仿真成本与复现','',
  f"本轮 {cost['attempts']} 次尝试、{cost['successful_attempts']} 次成功；CA 调用墙钟 {cost['wall_seconds']/60:.2f} 分钟、子进程 CPU {cost['cpu_seconds']/3600:.2f} 小时、原始产物 {cost['raw_bytes']/2**30:.2f} GiB。失败尝试计入成本。无自动重试；38 个历史输入没有重新仿真。墙钟不含离线 trace 解析/绘图。",'',
  '预算：35 分钟 CA 墙钟、10 CPU 小时、12 GiB；普通 quant 每例≤300秒，FA≤180秒。总 CPU/磁盘门槛在 case 边界检查，可能单例越过门槛；超时流程最多额外等待15秒清理。全输出 CPU golden 保留；host 扩展只提高合法维度/工作预算，FA 单 tensor≤64M元素，乘积工作量≤512M。', '',
  f"冻结时间：`{e['frozen_utc']}`；model SHA256：`{e['model_sha256']}`。源码、manifest、调用程序、原始包身份、trace SHA 和预测在 frozen-predictions.json/dataset.json 中可核对。浮点重放沿用 v0.2 的1e-12相对/绝对容差，允许Python版本求和实现的末位差异；模型和文件哈希严格一致。冻结提交：`73444da`。",'',
  '```bash','python -m unittest discover -s tests -q','python -m operator_model --op quant --shape 9 5120',
  'toolchains/envs/cann-9.2.1/bin/python scripts/workload-v4.py',
  'toolchains/envs/cann-9.2.1/bin/python scripts/evaluate-v4.py',
  'toolchains/envs/cann-9.2.1/bin/python scripts/report-v4.py','```','',
  'CA 重放使用 `scripts/collect-v1.py --version 0.4 --split holdout`；会检查冻结文件和成功缓存，失败不自动重试。机器本地的 raw trace 不提交 GitHub，发布的是摘要、输入、预测与完整校验身份。', '',
  '![新增 CA 成本](ca-cost.png)','',
  'CANN 日志仍有部分配置项不识别、缺少 debug_line/source call stack 的诊断。成功样本均满足 extractor 的 SMOKE_PASS、All task success、解析结束和单 kernel 检查，并用逐核时间窗口交叉核对；没有将缺失 source 映射补造为指令源码对应。原始警告保存在 dataset.json。','',
  '## 本轮边界及下一步','',
  'FA/matmul 时延公式本轮保持 v0.3；宽度、batch、KV 范围扩展必须靠留出证据分区验收。没有 A7/A8 的实测校准，没有实卡利用率验证，也没有把经验 busy 比例转换成 ISA 峰值利用率。', '',
  '成功样本未发现分核或整数指令计数不符；下一步应定位 quant 的残余控制/写回时延，并先降低长 shape 的采集开销，再补最少的新留出点。长 KV 若受 trace 成本限制，应先设计汇总采集或验证重复 tile 的可组合性，并单独验证跨 tile、cache 和多核争用，不能直接宣称线性外推可靠。','']
 (OUT/'report.md').write_text('\n'.join(lines))
 print('Wrote report and 4 PNG/SVG figures')
if __name__=='__main__':main()
