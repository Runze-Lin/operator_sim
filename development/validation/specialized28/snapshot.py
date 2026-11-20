"""Versioned release snapshot; long CA supervisors maintain separate live reports."""
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.28'
DT,PR='Ascend950DT_9582','Ascend950PR_957b'
def read(name):
    return json.loads((OUT/name).read_bytes())
def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(path):
    return sha256(path.read_bytes()).hexdigest()
def replace_section(path,tag,body,anchor):
    head,tail=f'<!-- {tag} -->',f'<!-- /{tag} -->'
    s=path.read_text()
    if head in s:
        a,b=s.split(head,1);_,c=b.split(tail,1)
        s=a+head+'\n'+body.rstrip()+'\n'+tail+c
    else:
        pos=s.index(anchor)
        s=s[:pos]+head+'\n'+body.rstrip()+'\n'+tail+'\n\n'+s[pos:]
    path.write_text(s)
def cell(r,n='shapes',mx='max'):
    return f'{r[n]} · {r["wape"]:.2f}% / {r[mx]:.2f}%' if r.get(n) else '0 · N/A'
def release():
    mat=read('matmul/evaluation.json');rms=read('rms/evaluation.json');post=read('post/channel-score.json')
    sink=read('fusion/independent.json');cast=read('cast/score.json');wheel=read('wheel-check.json')
    future=read('matmul/long-prefill/freeze.json')
    asof=datetime.now(timezone.utc).isoformat(timespec='minutes')
    prereg=['rms/freeze.json','fusion/register.json','post/registration.json','cast/register.json','matmul/freeze.json',
            'matmul/streamk-freeze.json','matmul/long-prefill/freeze.json','matmul/wide-prefill/freeze.json','matmul/narrow/freeze.json','matmul/multi/freeze.json']
    if (OUT/'fusion/eligibility.freeze.json').exists():
        prereg.append('fusion/eligibility.freeze.json')
    source=list((ROOT/'operator_model').glob('*.py'))+list((ROOT/'configs/chips').glob('Ascend*.json'))+[ROOT/'MANIFEST.in',OUT/'rms/prices.json',OUT/'fusion/prices.json',OUT/'post/parameters.json']+[OUT/f for f in prereg]
    save(OUT/'model.json',dict(version='0.28.0',baseline='v0.27',snapshot_utc=asof,registrations=prereg,
         checksum_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(source)},online='Estimate from inputs and chip config only; history-conditioned APIs use dispatch history from caller. Archived or pending CA is calibration/assessment only.'))
    evals=['matmul/evaluation.json','matmul/public-score.json','rms/evaluation.json','post/channel-score.json','post/cross-chip.json','fusion/independent.json','cast/score.json','wheel-check.json','replay.json']
    evals.extend(name for name in ('matmul/streamk-score.json','matmul/overview-prices.json',
                 'matmul/multi/score.json','fusion/eligibility.json') if (OUT/name).exists())
    save(OUT/'evaluation.json',dict(snapshot_utc=asof,model_sha256=sha(OUT/'model.json'),heldout_frozen_before_acquisition=prereg,
         evidence={n:sha(OUT/n) for n in evals},matmul={'groups':mat['groups'],'pending':mat['pending'],'rejected':mat['rejected']},
         long_prefill={'registered':future['rows'],'progress':read('matmul/long-prefill/state.json')},
         wide_prefill={'registered':read('matmul/wide-prefill/freeze.json')['rows'],'progress':read('matmul/wide-prefill/state.json')},
         rmsnorm={'summary':rms['summary'],'paired_windows':rms['paired'],'terminal':rms['terminal'],'passes':rms['passes'],'pending':rms['pending']},
         cast=cast,post_channel=post['metrics'],sinkhorn=sink['by_sku'],wheel=wheel,
         interpretation='Snapshots of current results. Subsequent durable queue receipts and live scores are linked in changes.md. All 970DT values are projections.'))
    lines=[]
    def add(label,pair,role,state):
        lines.append(f'| {label} | {pair[0]} | {pair[1]} | {role} | {state} |')
    for lane in ['decode','prefill']:
        g=mat['groups'][lane]
        add(f'Matmul（{lane}，普通排程）',[cell(g[t],'n','max_ape') for t in ['dt','pr']],'v0.28，三模型真实矩阵几何，冻结后完整窗','按已返回子集；分开核对 Stream-K')
    if (OUT/'matmul/streamk-score.json').exists():
        add('Matmul（单输出分 K，显式接口）',['1 · 5.65% / 5.65%','2 · 2.33% / 2.64%'],'只计候选登记时尚未开始的完整窗','窄输出已核对；跨宽度待验')
    for label,key in [('RMSNorm（无 H，v0.28 专项）','us'),('RMSNorm（可见 H 与同形重复）','specific_context_us')]:
        add(label,[cell(rms['summary'][c]['all'][key]) for c in (DT,PR)],'新形，同形两种前序合并；首次/重复组另计','分组结果见下文')
    add('mHC Post（通道切分的同形重复）',[f'{v["distinct_shapes"]} · {v["candidate_us"]["wape"]:.2f}% / {v["candidate_us"]["maximum"]:.2f}%' for c in (DT,PR) for v in [post['metrics'][c]['all_repeat']]],'显式条件模型，原冻结加新返回的完整窗','按十形准入检查')
    add('mHC Sinkhorn（寄存器组候选，未知 H）',[cell(sink['by_sku'][c]['all']['models']['point_us'],'geometries','max_ape') for c in (DT,PR)],'冻结后新几何；含首遇与有序窗','样本见独立表')
    add('Cast（165 MiB，写压力模型）',['1 · 5.61% / 5.61%','1 · 0.61% / 0.61%'],'新增独立单形、两端全核，容量子组另见回放','本形达标 / 本形达标')
    # Append rows within the continuous opening table, never insert comments inside it.
    path=ROOT/'docs/error-analysis.md';s=path.read_text()
    cutoff='| Matmul（decode，普通排程）'
    table_end=s.index('\n\n主表按指定协议')
    lower=s.index(cutoff) if cutoff in s[:table_end] else table_end+1
    s=s[:lower]+('\n' if not s[:lower].endswith('\n') else '')+'\n'.join(lines)+s[table_end:]
    s=s.replace('达标 / 容量组超限','旧均匀缓存策略 / 旧容量组超限')
    s=s.replace('| Cast（Pro B8×256Ki） | 1 · 8.38% / 8.38% | 1 · 1.69% / 1.69%','| Cast（Pro B8×256Ki，新版回放） | 1 · 8.38% / 8.38% | 1 · 4.30% / 4.30%')
    s=s.replace('520.357 / 529.287（−1.69%）','506.517 / 529.287（−4.30%，旧版 520.357）').replace('DT/PR 耗时比 0.436 / 0.468','DT/PR 耗时比 0.448 / 0.468')
    s=s.replace('本版逐 shape 的 950DT/PR 预测、CA、库 kernel 与 970DT 外推](../results/v0.27/shape-errors.md)','本版逐 shape 的 950DT/PR 预测、CA、库 kernel 与 970DT 外推](../results/v0.28/shape-errors.md)')
    protocol_heading='## 分协议核验与历史同集对照'
    parts=s.split(protocol_heading+'\n\n',1)
    detail=parts[-1] if len(parts)==2 else s.split('\n\n',1)[1]
    source_table=detail.split('\n\n主表按指定协议')[0]
    names=['DynamicMXQuant','Matmul','FA（新形）','RMSNorm（仅 shape）','Addmm','Lightning Indexer',
           'mHC Pre','mHC Sinkhorn','mHC PreSinkhorn（融合新形）','mHC Post','mHC 拆分链','KDA recurrent',
           'KDA chunk，融合 V1','KDA chunk，FLA 26.9.1 三段 V2','DeepSeek 扩展 HcPost',
           'GMM1–SwiGLU–MXQuant','MoE GMM2','MoE FFN 组合','MoE sigmoid Top-K','Cast BF16→FP32',
           'Kimi KDA beta Sigmoid','Kimi AttnResFwd']
    lookup={r.split('|')[1].strip():r for r in source_table.splitlines() if r.startswith('| ')}
    compact=['# 算子耗时预测与 CA 校准误差','', '| 算子 | 950DT 形数 · WAPE / 最大 | 950PR 形数 · WAPE / 最大 | 主表数据角色 | 当前核验 |','|---|---:|---:|---|---|']
    for name in names:
        row=lookup[name]
        if name=='Matmul':
            g=read('matmul/public-score.json')['groups']['v0.28 application replay']
            vals=[cell(g[c]['public_us'],'n','max_ape') for c in (DT,PR)]
            row=f'| Matmul | {vals[0]} | {vals[1]} | 应用形按当前默认路由回放；新分 K 补充另计 | WAPE 达标；库重选输出块时最大误差超限 |'
        elif name.startswith('RMSNorm'):
            which='us' if rms['passes'] and rms['terminal'] else 'old_us'
            vals=[cell(rms['summary'][c]['all'][which]) for c in (DT,PR)]
            row=f'| RMSNorm | {vals[0]} | {vals[1]} | v0.28 新形，公开无前序；每形首遇和重复各一窗 | 少核首次/复跑须分看，见下表 |'
        elif name=='Cast BF16→FP32':
            stats=[next(r for r in read('replay.json')['summary'] if r['op']=='cast_bf16_fp32' and r['chip']==c) for c in (DT,PR)]
            vals=[cell(r,'shapes','maximum') for r in stats]
            row=f'| {name} | {vals[0]} | {vals[1]} | 已归档完整窗的现版回放；独立 165 MiB 另列 | 新容量对照达标；大形仍有块内定时误差 |'
        compact.append(row.replace('FA（新形）','FA').replace('mHC PreSinkhorn（融合新形）','mHC PreSinkhorn'))
    compact+=['','每格先列不同 shape 数，再列 WAPE（按实测时间加权的绝对误差）与最大单窗绝对相对误差。门槛为 WAPE ≤25%、最大误差 ≤50%。各行注明使用冻结后的独立形状或旧数据回放。Matmul 的不同分支、RMSNorm 和 Post 的前序状态，以及不合格窗口均在下表展开。970DT 只有架构外推。','',protocol_heading,'']
    path.write_text('\n'.join(compact)+'\n'+detail)
    body=f'''## 哪些超限来自算子分支，哪些需要前序输入

以下为 UTC {asof[:16]} 的核验快照。全套历史合格窗按当前安装版重新预测，旧观测统一作为回放；新形只在预测冻结后获得的窗口中计独立误差。[逐形旧版、当前、CA、970DT 外推](../results/v0.28/shape-errors.md) · [新冻结与复核索引](../results/v0.28/evaluation.json)

- **Matmul。** 默认 auto/resource 已比较单输出矩形的分 K 与普通方案。该路径在四个 M 的 DT/PR 回放 WAPE 分别为 3.65% / 2.97%，最大 5.72% / 5.42%。另一形 [144,64,7168] 的实际 host 切 K 之前会重新选择输出块，当前普通分块仍高估约一倍。[当前版本与直接复用 DT](../results/v0.28/matmul/public-report.md) 。Pro [1,64,7168] 的原冻结预测是 1 个 AIC、14 个串行 K 页，DT/PR 分别高估 170.84% 和 176.66%。两端实际 key=4098，用 32/28 个 AIC 各做一次 MMAD，再交由 64/56 个 AIV 合并。普通 prefill [32768,128,512] 的实际 key=131074，只用 AIC，DT/PR 预测 / CA 是 19.153 / 15.562、28.820 / 26.690 µs。两种结果都来自安装包 ops-nn 的 aclnnMatmul；每次对全部 BF16 输出做数值校验。混合分支要求每核存在物理 END，且所有执行引擎最后退休的事件均与聚合时间轴和官方核表对齐，异步引擎可晚于标量 END。原判据与补充核验各有回执。[逐形成本、核数和 MMAD 数](../results/v0.28/matmul/report.md) · [从 32,768 到 2,097,152 行的完整波和带宽假设](examples/matmul.md) · [单输出分 K 的逐核与前瞻对照](../results/v0.28/matmul/streamk-analysis.md) · [全部 28 个 shape 的同一张结果表](../results/v0.28/matmul/README.md) · [三个模型的整批 B8×32Ki 补充](../results/v0.28/matmul/wide-prefill/report.md) · [128Ki、256Ki 双端采集](../results/v0.28/matmul/long-prefill/report.md)
- **RMSNorm 与 DynamicMXQuant。** 两个算子同为 BF16 [9,5120]，DT 却分别用 9 与 45 个 AIV。PR 的 RMSNorm [4,784] 前后两个窗有相同归约时长，但 gamma 的发起差 3933 周期、x 的读响应差 678 周期。首次 / 紧邻重复在无历史时不可辨识；可见历史时按物理分区、归约、统计量和向量写分别排程。十个新形总体达到双门槛。对旧 DT 的十个高误差紧凑窗回放，WAPE 从 44.75% 降到 30.99%，仍超限；给前序时为 6.64%。A≤4 的两态静态下界超过 35%；大形应另查共享带宽和多 UB 轮次。全 hidden Quant 另有原生包络；B4×32Ki 的新原生验证以原队列最终回执为准。[四个相同输入的两个算子](../results/v0.28/vector-matched.md) · [RMS 每个新形、前序与双卡比值](../results/v0.28/rms/report.md) · [旧超限组与每算子实际流的重放](../results/v0.28/rms/stream-replay.md) · [Quant 原生独立任务](../results/v0.27/quant-transfer/README.md)
- **内置 mHC Post 与 Sinkhorn。** Post 通道分核有两笔输入、四路输出；按 token 分核有不同的读写顺序。给出前一同族调用可显著降低误差，在 10 个同形的内置 Post 验证中，每形只能返回一个时间时，把全部 CA 都交给一个知道答案的估计器，让它为每形选最优中位数，DT/PR 的 WAPE 下界仍是 27.64% / 28.81%。再要求任一窗最大偏差 ≤50%，约束下的最优 WAPE 变为 28.46% / 29.43%。这批窗从输入上就无法同时满足用户的两个门槛。按同一流的真实成功派发提供前序后，条件模型在 21/22 窗回放的 WAPE 是 7.71% / 7.53%，最大 30.71% / 31.78%。可直接比较两套输入合同。[形状单值的下界与条件结果](../results/v0.28/post/history-replay.json) 。Sinkhorn 一组寄存器的 VF 包含发起成本；多组时只重复组内工作。专项模型改善宽形的已知 H 回放，但未知前序在旧域的总体 WAPE 有退化。两条候选分别注明状态，不能用可见 H 的数值代表只给 shape 的入口。[Post 按入口与协议](../results/v0.28/post/report.md) · [Sinkhorn 的组数消融](../results/v0.28/fusion/report.md) · [Sinkhorn 新几何](../results/v0.28/fusion/forward.md)
- **Cast、Sigmoid 与 KDA。** 在 165 MiB 的新形上，Cast 的 PR 写压力模型将误差从 +24.45% 降至 −0.61%，DT 为 −5.61%。同形 DT/PR 耗时比误差也从 −24.15% 缩到 −5.03%。Pro 768 MiB 的 PR 总时延误差从 −1.69% 变为 −4.30%；实测支持保留更多脏输出这个工作假设，尚不能直接识别真实的 L2 替换策略。Sigmoid 对相同参数的独立验证未支持全局切换。融合 KDA V1 仍有实际重复 SET，计时资格 N/A；V2 的数值和计时按独立合同给出。[Cast 同集分组与反例](../results/v0.28/cast/score.json) · [KDA 正确性与同步核验](../results/v0.27/kda/report.md)
'''
    replace_section(path,'v028-specialization',body,'## 当前公开模型的双端验证')
    costs=wheel['cost']
    (OUT/'changes.md').write_text(f'''# v0.28 方法和误差变化

- **Matmul** 增加 28 个新几何、两端共 56 项，其中 9 个为 prefill、19 个为 decode 或校验，含三种模型、普通 decode 的 B=1/9/11/16、MTP8 对应 M=9/81/99/144、full prefill 与 chunk。32Ki Flash [32768,128,512] 已双端完成；[262144,32,5120] 每卡仅输入就达 2.5 GiB。B8×128Ki、B8×256Ki 两个满秩几何分别有 1,048,576 与 2,097,152 行。预测先行冻结，每项真实 CA 才进入误差。此快照的首批 32 项已有 {32-mat['pending']-mat['rejected']} 个合格、{mat['rejected']} 个不合格。另为单输出分 K 登记 6 个未见形、重选输出块登记 2 个未见形。[六个窄投影](matmul/narrow/report.md) · [两个重选块的输入和预测](matmul/multi/freeze.json)。另登记 Kimi 和 Pro 的整批 B8×32Ki，复用 RMS 完成后的单槽。单输出切 K 已接入系统后端。同形当前版本回放和登记时尚未开跑的前瞻另计。[默认模型的同集误差](matmul/public-report.md) · [全部 28 个 shape 的实时逐形表](matmul/README.md) · [完整 Pro 和 Kimi B8×32Ki](matmul/wide-prefill/report.md) · [长 prefill](matmul/long-prefill/report.md) · [单输出候选将 DT 一例 / PR 两例前瞻 WAPE 减至 5.65% / 2.33%](matmul/streamk-analysis.md)
- **Cast** 升级为按活动核发起与片外写压计算脏输出留驻。新 Flash [5632,5120] 的 DT 旧/新 27.431/27.431 µs，CA 29.062；PR 68.721/54.881 µs，CA 55.220。两次采集墙钟 685/658 s。大 Pro 的 PR 总窗偏差从 −1.69% 变为 −4.30%。[双端回执和旧样本分组](cast/score.json)
- **RMSNorm** 两芯片的十个新形通过了分别检验首遇与复跑的预登记门槛。无前序、最忙核最多两个 UB 块的情况已接入默认。将 gamma/x 就绪、寄存器组归约、输出与 rstd 写回分别排程，公开条件接口可指定哪些核已参与过。独立同形两态已完成 {rms['paired']} 个可配对芯片窗。[公式与参数](rms/method.md) · [当前的分组 WAPE、最大误差、DT/PR 比值和成本](rms/report.md)
- **mHC Post / Sinkhorn** 分别按通道与 token 路线、寄存器组固定发起与组内工作定价。PR 登记 T233、I16 的新形检查每个 VF 的实际寄存器组数，DT 对应一组的原价。[换组假设、双端冻结和 CA 资格](fusion/eligibility.md)。Post 提供前序条件接口；Sinkhorn 的宽形已知 H 回放改善，同时保留未知 H 在旧域的退化。[Post 独立与回放](post/report.md) · [Sinkhorn 独立](fusion/forward.md)
- **预测成本与适用域** 安装版普通查询 {costs['estimate']['n']} 个不同输入，P95 {costs['estimate']['p95_ms']:.3f} ms，最大 {costs['estimate']['maximum_ms']:.3f} ms。所有 970DT 数字均为外推；缺失的物理退出、未完成的长任务和数值异常均不计入真值。[完整安装版测时](wheel-check.json) · [全量逐形回放](shape-errors.md) · [带时间戳的资格快照](evaluation.json)
''')
if __name__=='__main__':release()
