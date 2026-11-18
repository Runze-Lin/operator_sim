"""Render the immutable holdout results and repeated-shape controls."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=root/'results/v0.10/wide'
load=lambda f:json.loads((out/f).read_text());rows=load('rows.json');p=load('paired-controls.json');s=load('summary.json');cost=load('costs.json')
fig,axes=plt.subplots(1,2,figsize=(12,4.9),sharey=True)
fig.subplots_adjust(left=.07,right=.98,top=.85,bottom=.31,wspace=.08)
for ax,op,title in zip(axes,('rmsnorm','mhc_post'),('RMSNorm','mHC Post')):
    v=[r for r in rows if r['role']=='validation' and r['op']==op];i=list(range(len(v)));ax.plot(i,[r['reference_us'] for r in v],'ko-',label='CA measurement');ax.plot(i,[r['previous_us'] for r in v],'x--',color='#777777',label='v0.9');ax.plot(i,[r['predicted_us'] for r in v],'s-',color='#1260a1',label='v0.10 default');ax.plot(i,[r['sequence_us'] for r in v],'d:',color='#cc681d',label='v0.10 with stream history');ax.set_xticks(i,['×'.join(str(x) for x in r['shape']) for r in v],rotation=50,ha='right',fontsize=8);ax.set_title(title);ax.grid(axis='y',alpha=.25);ax.set_ylim(bottom=0)
axes[0].set_ylabel('Critical instruction makespan (µs)');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False);fig.suptitle('Same ten frozen holdout shapes per operator');fig.savefig(out/'shape-error.png',dpi=150);plt.close(fig)
text=['# 950DT 阶段归因与独立验证（v0.10）','', '|算子 / 独立分组|形状数|v0.9 WAPE / max|v0.10 默认 WAPE / max|v0.10 已知前序核覆盖 WAPE / max|旧数据固定中位数 WAPE / max|','|---|---:|---:|---:|---:|---:|'];fmt=lambda st:f"{st['wape']:.2%} / {st['max_ape']:.2%}"
for op in s:
    for group,g in s[op]['validation'].items():text.append('|'+op+' / '+group+'|'+str(g['predicted_us']['n'])+'|'+'|'.join(fmt(g[k]) for k in ('previous_us','predicted_us','sequence_us','fixed_us'))+'|')
text+=['','主结果是默认输入不含上下文的单一预测；第三种输入仅用于说明系统仿真器能跟踪同类前序发射时的价值。开发 23/族，均为旧数据；上表来自冻结后独立的十个新大小/族，其中五个 T≤4 且 D≥2048，五个对照；第一次调用也算进表，不挑选热态。[冻结材料](freeze.json) 含两个版本逐行预测、顺序、时间戳、模型与采集器 SHA256、620 秒 / 3 GiB 每批预算。固定基线取每族全部 23 条旧 CA 时延的中位数。WAPE=Σ|预测−CA|/ΣCA，max 是单 shape 最大绝对百分比。','','![各 shape 对比](shape-error.png)','','## 为什么旧模型高估、哪些事情仍不知道','','历史 RMSNorm `[1,6144]` 的 SCALAR/MTE2 分别预计 1.625/0.718 µs，CA 0.858/0.338；mHC Post `[2,4,4096]` 预计 1.635/0.887，CA 0.984/0.282。两者都是各自批的最后一次。Post 旧批从 `[3,4,256]`、`[6,4,512]`、`[17,4,256]` 到 `[2,4,4096]` 关键 AIV 均有 1578/5 条 SCALAR/MTE2，阶段时间却明显不同；RMSNorm 在旧批长宽/普通宽的同实现中也常为 1093/5。单靠指令数及固定启动截距不能解释这一点。按已覆盖核数建模是可被推翻的上下文代理，基于 950 的 CANNBot 流水机制卡采用互斥暴露时间，未把某个硬件缓存宣称为实锤。','','**新冻结同形对照：同一进程、流、shape、类型、生成器和基础 caller，批首与批尾（重复行不计入独立样本数）。**','','|算子 / shape|CA 首→末 µs|SCALAR / MTE2 首→末 µs|SIMD 首→末 µs|首→末 SCALAR/MTE2 指令条数|','|---|---:|---|---:|---|']
for op in s:
    a=next(r for r in rows if r['op']==op and r['role']=='validation');b=next(r for r in p if r['op']==op);phase=lambda r,k:r['phase_us'][k];text.append(f"|{op} {a['shape']}|{a['reference_us']:.3f}→{b['reference_us']:.3f}|{phase(a,'SCALAR'):.3f}/{phase(a,'MTE2'):.3f}→{phase(b,'SCALAR'):.3f}/{phase(b,'MTE2'):.3f}|{phase(a,'SIMD'):.3f}→{phase(b,'SIMD'):.3f}|{a['counts']['SCALAR']}/{a['counts']['MTE2']}→{b['counts']['SCALAR']}/{b['counts']['MTE2']}|")
text+=['','可确认：同一 shape、分核、指令数量存在两种明显时延，v0.9 把可变的暴露时间计作基础成本。RMSNorm 首末 SCALAR+MTE2 合计少 1.68 µs；Post 本次重复的 SCALAR 反而略长，差异主要是 MTE2、SIMD、arrival 与 other，一概归为标量提速会误导。地址/分配器、缓存与发射状态同时随执行历史变化；单对 trace 不能区分各自的因果贡献，更不能承诺跨流适用。','','**退化与剩余误差。** RMSNorm 五个普通对照的默认 WAPE 从 20.65% 增至 25.73%；最差 `[34,1536]` CA 5.576 µs、默认 3.363，MTE3 暴露 2.325 而默认仅 0.667，单此项已解释约 75% 的绝对差；更改 H 后仍为 3.944 µs。写回合并、并发竞争与地址分布是后续应分辨的假设。Post 默认最差 `[3,4,3072]`（3.679 对 2.238 µs）发生在新批首个 shape 已激活 64 核的情况下；旧先验以核数逐渐扩展序列为主，默认仍偏向扩展态；发射记录推得 H=64 时为 2.269。Post `[4,4,8192]` 默认预测和旧版几乎不变。已知执行历史不代表可观测硬件缓存。10 个独立 shape 只约束所采区域，不能声称整个分布或 950 真机同精度。','','## 逐 shape，新留出数据','','|算子|shape|库接口 / 实际 kernel|关键核 SCALAR / MTE2 指令|活跃 AIV CA / 预测|CA (µs)|v0.9|v0.10 默认|有上下文|默认有符号误差|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    if r['role']!='validation':continue
    op=r['op'];api='ops-nn aclnnRmsNorm' if op=='rmsnorm' else 'ops-transformer aclnnMhcPost';text.append(f"|{op}|{r['shape']}|{api} / {r['kernel']}|{r['counts']['SCALAR']} / {r['counts']['MTE2']}|{r['active_aiv']} / {r['prediction']['predicted_active_aiv']}|{r['reference_us']:.3f}|{r['previous_us']:.3f}|{r['predicted_us']:.3f}|{r['sequence_us']:.3f}|{r['predicted_us']/r['reference_us']-1:+.1%}|")
text+=['','全部 20 个独立样本与 2 个重复对照完成 950DT 单流调用、CPU golden 和逐条 DB 检查；RMSNorm 检查 y/rstd 两输出，Post 检查组合输出。所有独立点活跃 AIV 比对一致，首个/末个 launch 正确性或分核相同并不保证时延相同。`rows.json` 可检索开发及验证全部阶段、CPU 输出摘要和关键核指令；`paired-controls.json` 单列重复样本。','','## 成本与复核','','|算子|CA 批次墙钟|DB MiB / 整批保留 MiB|单次预测 µs（中位 / 最大）|','|---|---:|---:|---:|']
for op,c in cost.items():text.append(f"|{op}|{c['batch_wall_s']:.2f} s（11 次发射）|{c['db_bytes']/2**20:.1f} / {c['raw_bytes']/2**20:.1f}|{c['query_median_us']:.2f} / {c['query_max_us']:.2f}|")
text+=['','每 shape 做 100 次无结果缓存、已加载模型的调用；最大值是本次观测，不是延迟 SLA。两批并行；两批 worker 墙钟求和是约 308.5 s，不是用户实际经过时间。DB 保留在 runs/v10wide，本目录只保留小型结果。`development-trace.json` 与冻结合约记录旧 trace 的来源；`summary.json` 含每族每组的开发回放和验证 WAPE/max、偏置及 MAE。图中的上下文预测在采集前基于顺序和预测 AIV 数计算，不读取新 trace。']
(out/'diagnosis.md').write_text('\n'.join(text)+'\n')
