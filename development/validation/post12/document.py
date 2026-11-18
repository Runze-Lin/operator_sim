"""Create the detailed timing tables and plot from the frozen evaluation."""
import json
from freeze import OUT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    val=json.loads((OUT/'summary.json').read_text())['mhc_post']['validation']; cost=json.loads((OUT/'costs.json').read_text())['mhc_post']
    rows=[r for r in json.loads((OUT/'rows.json').read_text()) if r['role']=='validation'];name={'all':'全部','few_wide':'少 token 宽通道','controls':'普通','fixed_width_ladder':'其中 D=1536 大小阶梯','first':'小宽 / 首次','expand':'小宽 / 扩核','reuse':'小宽 / 热态'}
    doc=['# mHC Post：冻结后独立验证', '', '| 组别 | 新 shape 数 | v0.11 默认 WAPE / 最大 | v0.12 默认 WAPE / 最大 | v0.12 已知 H WAPE / 最大 | 固定耗时 WAPE / 最大 |','|---|---:|---:|---:|---:|---:|']
    fmt=lambda s:f"{s['wape']:.2%} / {s['max_ape']:.2%}"
    for g,st in val.items(): doc.append('| '+name[g]+f" | {st['predicted_us']['n']} | "+' | '.join(fmt(st[k]) for k in ('previous_us','predicted_us','sequence_us','fixed_us'))+' |')
    doc.extend(['','WAPE=Σ|预测−CA|/ΣCA；最大误差为最大的 |预测−CA|/CA。两条默认曲线都只用 shape 和算子属性；显式 H 列仅用采集前已知的同流规划。小宽定义为 T≤4、D≥2048。固定耗时取 43 个开发点中位数。原有 43 个 shape 是开发回放，10 个新 shape、预测和预算在采集前冻结。样本包含有意构造的首次/扩核场景，汇总不代表部署状态占比。', '', '![逐 shape 微秒数，实测与两个冻结版预测](latency.svg)','', '| 新 shape [T,4,D] | H / 状态 | CA µs | v0.11 | v0.12 | 有符号误差 | 已知 H | 实际入口 / AIV | SCALAR / MTE2 / MTE3 / SIMD 指令 | 已审计输出 B |','|---|---|---:|---:|---:|---:|---:|---|---|---:|'])
    for r in rows:
        t,n,d=r['shape'];a=r['reference_us'];p=r['predicted_us'];ctx=r['sequence_prediction']['context']; cnt='/'.join(str(r['counts'][k]) for k in ('SCALAR','MTE2','MTE3','SIMD'))
        doc.append(f"| [{t},4,{d}] | {r['prior_aiv']} / {ctx} | {a:.3f} | {r['previous_us']:.3f} | {p:.3f} | {p/a-1:+.1%} | {r['sequence_us']:.3f} | {r['kernel']} / {r['active_aiv']} | {cnt} | {r['output_audit']['transfer_bytes']:,} |")
    doc.extend(['','## 如何解释误差','','复用历史同形对照 `[2,4,6144]`：首发与批尾分别 4.838、2.905 µs。关键核两次都是 1578 条 SCALAR、5 条 MTE2、5 条 MTE3、581 条 SIMD；MTE2 暴露时间却从 1.202 降到 0.623 µs，SIMD 从 1.384 到 0.670 µs，arrival 从 0.545 到 0.050 µs。这说明仅凭指令数量和 shape 无法区分两次；不能由这些计数证明某个 cache 是根因。若只允许给这两次一个数，即使提前知道两次实测，最大相对误差也至少 (4.838−2.905)/(4.838+2.905)≈25.0%。这条重复控制没有重复加入 43 个校准点。','','按 10 个旧 CA 进程逐个整进程留出，每折重新拟合阶段参数与缺省状态数；用相同折内阶段预测换两种状态选择法，以免把新训练数据的影响误算成选择法收益。43 个历史 shape 的 WAPE 为硬选 28.0%、折衷 22.9%；12 个历史小宽点为 38.3%→33.7%，小宽最大 89.8%→47.0%。两次历史小宽首发平均仍差，首发 WAPE 从 36.4% 到 39.1%；不能承诺每个进程或每个状态都变好。三态 H 已知的历史小宽 WAPE 4.8%。[整进程留出明细](process-crosscheck.json)','','新集中 5 个小宽点有两次首发、两次扩核、一次热态；旧默认因预测复用而系统性低估前两态。新版对首发、扩核上调预测，代价是高估热态。一个热态不足以外推真实 decode 流命中率。普通组 `[31,4,256]` 无需切通道；该组旧进程序列首次/扩核/复用=4/2/2，缺省中位选了扩核。本次实际 H=64、A=31，是热态。此点的偏差主要来自未观测的状态；即使新版本折衷修复切通道分支，该组仍应优先传入 `resident_aiv`。',''])
    large=[r for r in rows if r['shape'][2]==1536 and r['shape'][0]>=64]
    if len(large)==3:
        a,b,c=(r['reference_us'] for r in large); p,q,h=(r['predicted_us'] for r in large)
        doc.append(f'固定 D=1536、H=64 时，T=64/128/256 各有 1/2/4 行每核，CA 为 {a:.3f}/{b:.3f}/{c:.3f} µs，默认预测 {p:.3f}/{q:.3f}/{h:.3f} µs。每多一行的实测斜率分段为 {b-a:.3f}、{(c-b)/2:.3f} µs；模型对应 {q-p:.3f} µs/行。只有三个同宽点、无重复测量，不能据此确认饱和转折点或更大工作量下的全局线性。逐核阶段、指令数和冻结前工作量可在 JSON 复核。')
        last=large[-1]; phases=last['phase_us']; estimate=last['prediction']['stages_us']
        doc.append(f"\nT=256 的误差可以从阶段定位：MTE2 估 {estimate['MTE2']:.3f}、实测暴露 {phases['MTE2']:.3f} µs；MTE3 估 {estimate['MTE3']:.3f}、实测 {phases['MTE3']:.3f} µs；SIMD 估 {estimate['SIMD']:.3f}、实测 {phases['SIMD']:.3f} µs。固定单位字节、每额外行的可见搬运成本偏大；这支持下一轮检查 MTE 与 SIMD 遮盖以及逐行发起间隔，不能由单点认定 DRAM 带宽变化。")
    doc.extend(['','## 实测资格和成本','',f"全部 {len(rows)} 个独立点实测为 CANN 9.2.1 / 950DT 的 `aclnnMhcPost`（ops-transformer 家族）；每条确认 CPU 全输出 `bad=0`、helper `status=ok` 且无无效时序、matched launch_id、1 个任务完成标记，核数和入口逐点对照。展开每条 MTE3 的目标 burst，要求总传输长度、地址并集和跨度均等于唯一 BF16 输出 2TND 字节。没有把零记录或混入其他 launch 的 DB 当作有效计时。两次预热不计入独立 shape。上轮被污染的原批数据库未用于本轮训练。",'',f"四个 CA 进程最多两路并发，包括预热共 {cost['launches']} 次发射，总进程墙钟 {cost['batch_wall_s']:.1f} s，生成数据库 {cost['db_bytes']/2**20:.1f} MiB，计时失败 {cost['failed_timing_attempts']} 次。在线三态/六阶段计算（已加载参数、无 shape 结果缓存）每点重复 100 次，合并中位 {cost['query_median_us']:.2f} µs、最慢 {cost['query_max_us']:.2f} µs；上下文列和设备微秒均不是 Python 墙钟。这些数字不构成任意系统负载的硬实时保证。",'','[逐 shape 开发与验证记录](rows.json) · [模型/预算/两版预测冻结](freeze.json) · [按组指标](summary.json) · [进程和开销](costs.json)'])
    (OUT/'diagnosis.md').write_text('\n'.join(doc)+'\n')
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none','svg.hashsalt':'post12'})
    fig,ax=plt.subplots(figsize=(8,4));ix=list(range(len(rows)));ax.axvspan(-.5,4.5,color='#f5f3ec');
    for col,label,marker,color in [('reference_us','CA','o','#212121'),('previous_us','v0.11 (shape only)','x','#bd6e32'),('predicted_us','v0.12 (shape only)','D','#186f98'),('sequence_us','v0.12 (known H)','+','#368452')]:ax.scatter(ix,[r[col] for r in rows],label=label,marker=marker,c=color,s=36)
    ax.set_xticks(ix,[f"{r['shape'][0]}×{r['shape'][-1]}\n{r['sequence_prediction']['context']}" for r in rows],rotation=35,ha='right');ax.set_ylabel('Critical-core time (µs)');ax.set_xlabel('T × D (N=4), observed dispatch state');ax.spines[['top','right']].set_visible(False);ax.legend(ncol=2,frameon=False);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(OUT/'latency.svg',metadata={'Date':None});plt.close(fig)

if __name__=='__main__': main()
