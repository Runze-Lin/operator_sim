"""Per-route and per-SKU scores. Preserve incomplete/failed launches and old cohorts."""
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate
from development.validation.sku19.campaign import CHIPS, read, write

OUT = ROOT/'results/v0.21'
V12 = 'results/v0.12/evaluation.json'
V19 = 'results/v0.19/evaluation.json'
V20 = 'results/v0.20/evaluation.json'
SIDES = ('dt','pr')
VECTOR = {'cast_bf16_fp32','sigmoid_fp32'}
FUSION = {'attn_res_fwd','mhc_sinkhorn','mhc_pre_sinkhorn'}
DISPATCH = {'addmm','mhc_post'}
CHANGED = VECTOR | FUSION | DISPATCH | {'rmsnorm'}
EXT = {'lightning_indexer','mhc_pre','mhc','kda_recurrent','gmm2','grouped_matmul_swiglu_quant_v2','moe_ffn'}
NAMES = dict(quant='DynamicMXQuant',matmul='Matmul',fa='FA',rmsnorm='RMSNorm',addmm='Addmm',
    lightning_indexer='Lightning Indexer',mhc_pre='mHC Pre',mhc_sinkhorn='mHC Sinkhorn',
    mhc_pre_sinkhorn='mHC PreSinkhorn',mhc_post='mHC Post',mhc='mHC 拆分链',kda_recurrent='KDA recurrent',
    kda_chunk='KDA chunk',grouped_matmul_swiglu_quant_v2='GMM1–SwiGLU–MXQuant',gmm2='MoE GMM2',
    moe_ffn='MoE FFN 组合',moe_gating_top_k='MoE sigmoid Top-K',cast_bf16_fp32='Cast BF16→FP32',
    sigmoid_fp32='Kimi KDA beta Sigmoid',attn_res_fwd='Kimi AttnResFwd')


def score(rows, pred='us'):
    if not rows:
        return None
    errors = [abs(r[pred]-r['ca_us']) for r in rows]
    ratios = [e/r['ca_us'] for e,r in zip(errors,rows)]
    worst = rows[ratios.index(max(ratios))]
    return dict(n=len(rows),shapes=len({tuple(r['shape']) for r in rows}),
                wape=sum(errors)/sum(r['ca_us'] for r in rows),max_ape=max(ratios),
                worst_shape=worst['shape'],worst_pred_us=worst[pred],worst_ca_us=worst['ca_us'])


def receipt(obs):
    return {k:obs[k] for k in ('api','library','kernel','trace','trace_sha256','correctness','active_aic','active_aiv','ca_active','binary_sha256','mmad') if k in obs}


def archive(rows,op,shape,attrs,sku,ca,cohort,src,obs):
    if ca is None:
        return
    chip=CHIPS[sku]
    r=estimate(op,shape,attrs,chip=chip)
    future=estimate(op,shape,attrs,chip=CHIPS['future'])
    if r.latency_us is None or future.latency_us is None:
        raise ValueError((op,shape,attrs,r.reason,future.reason))
    rows.append(dict(op=op,shape=shape,attrs=attrs,chip=chip,us=r.latency_us,ca_us=ca,
                     future_us=future.latency_us,cohort=cohort,receipt=receipt(obs),source=src))


def collect():
    rows=[]; rejected=[]; extdata=read(ROOT/V12)['rows']
    for r in extdata:
        op=r['op']
        if r['role']=='validation':
            cohort='primary' if op in EXT else 'DT v0.12 holdout replay'
        elif r['role']=='development':
            cohort='DT v0.12 development replay'
        else:
            continue
        archive(rows,op,r['shape'],r.get('attrs',{}),'dt',r['reference_us'],cohort,V12,
                dict(kernel=r['kernel'],trace=r['source'],trace_sha256=r.get('trace_or_db_sha256',r.get('db_sha256')),correctness=r['correctness']))
    for r in read(ROOT/V19)['rows']:
        op=r['op']
        if op in VECTOR or op=='kda_chunk':
            continue
        for side in SIDES:
            cohort='prior published pair replay'
            if op not in CHANGED and op!='rmsnorm' and not (op in EXT and side=='dt'):
                cohort='primary'
            # Older reused DT observations keep their explicit origin in detailed output.
            archive(rows,op,r['shape'],r.get('attrs',{}),side,r.get(side+'_ca_us'),cohort,V19,r.get(side+'_observation') or {})
    for version in ('0.17','0.18'):
        filename=f'results/v{version}/evaluation.json'
        for r in read(ROOT/filename)['rows']:
            if r['op']=='rmsnorm':
                for side in SIDES:
                    obs=r[side+'_observation']
                    if obs['ca_us']!=r[side+'_ca_us'] or obs['api']!='aclnnRmsNorm':
                        raise ValueError((version,side,r['shape']))
                    archive(rows,'rmsnorm',r['shape'],r.get('attrs',{}),side,obs['ca_us'],
                            f'v{version} historical RMS replay',filename,obs)
                    rows[-1]['original_role']=r.get(side+'_role',r.get('role'))
                    rows[-1]['launch_context']=obs.get('launch_context')
    h=read(ROOT/V20)['prefill_matmul']
    for side in SIDES:
        archive(rows,'matmul',h['shape'],{},side,h.get(side+'_ca_us'),'primary',V20,h.get(side+'_observation') or {})
    rms=ROOT/'results/v0.20/rms/records.json'
    for r in read(rms):
        side=next(s for s in SIDES if CHIPS[s]==r['chip'])
        archive(rows,'rmsnorm',r['shape'],r.get('attrs',{}),side,r['ca_us'],'v0.20 frozen RMS replay',str(rms.relative_to(ROOT)),r)
    vd=read(OUT/'vector/evaluation.json')
    vraw={(r['op'],tuple(r['shape']),r['chip']):r for r in read(OUT/'vector/holdout.json')}
    for r in vd['rows']:
        independent=r['role']=='post-freeze independent'
        raw=vraw.get((r['op'],tuple(r['shape']),r['chip']),{}) if independent else {}
        rows.append(dict(op=r['op'],shape=r['shape'],chip=r['chip'],us=r['us'],ca_us=r['ca_us'],
                    future_us=r['projected_970_us'],old_us=r['old_us'],attrs={},source=r['source'],
                    receipt=receipt({**raw.get('stage',{}), **{k:raw[k] for k in ('api','correctness') if k in raw}}),
                    cohort='primary' if independent else 'vector development replay'))
    rejected.extend(vd['excluded'])
    fd=read(OUT/'fusion/evaluation.json')
    for r in fd['independent_rows']:
        rows.append(dict(op=r['op'],shape=r['shape'],chip=r['chip'],us=r['pred_us'],ca_us=r['ca_us'],
                         future_us=r['future_us'],old_us=r['old_us'],attrs=r['attrs'],receipt=receipt(r),
                         source='results/v0.21/fusion/evaluation.json',cohort='primary'))
    for r in read(OUT/'fusion/historical-windows.json')['rows']:
        side=next(s for s in SIDES if CHIPS[s]==r['chip'])
        archive(rows,r['op'],r['shape'],r['attrs'],side,r['ca_us'],'fusion distinct historical windows',
                r['source'],r)
    rejected.extend(fd['rejections'])
    def registered():
        for folder in ('dispatch','rms'):
            for j in read(OUT/folder/'freeze.json')['jobs']:
                yield folder,j
    for folder,j in registered():
        y=j['outputs'][0]
        for side in SIDES:
            chip=CHIPS[side]
            path=OUT/folder/'receipts'/(j['id']+'_'+chip+'.json')
            if not path.exists():continue
            data=read(path)
            if not data['qualified']:
                rejected.append({**{k:v for k,v in data.items() if k!='cost'},'source':str(path.relative_to(ROOT))})
                path=OUT/folder/'retries'/path.name
                if not path.is_file():
                    continue
                data=read(path)
                if not data['qualified']:
                    rejected.append({**{k:v for k,v in data.items() if k!='cost'},'source':str(path.relative_to(ROOT))})
                    continue
            obs=data['observation']
            row=dict(op=j['op'],shape=j['shape'],chip=chip,attrs=j['attrs'],us=y[side+'_us'],ca_us=obs['ca_us'],
                     future_us=y['future_us'],old_us=y['old_'+side+'_us'],receipt=receipt(obs),
                     source=str(path.relative_to(ROOT)),cohort='primary')
            if side+'_first_us' in y:row.update(first_us=y[side+'_first_us'],protocol='isolated first launch')
            rows.append(row)
    rows.extend(read(OUT/'coverage/supplement.json'))
    audit=ROOT/'results/rms-dispatch-audit'
    sequence=read(audit/'independent.json')['rows']
    future={(r['protocol'],r['i']):r['previous_unknown_us']
            for r in read(audit/'freeze.json')['rows'] if r['chip']==CHIPS['future']}
    for r in sequence:
        if r['protocol']=='ordered new sequence':
            rows.append(dict(op='rmsnorm',shape=r['shape'],attrs={},chip=r['chip'],
                             us=r['previous_unknown_us'],ca_us=r['ca_us'],future_us=future[r['protocol'],r['i']],
                             cohort='frozen ordered RMS sequence, shape-only forecast',source=r['source'],
                             qualification=r['qualification'],receipt=receipt({**r,'api':'aclnnRmsNorm'})))
    pages=OUT/'matmul-pages-pr/observation.json'
    if pages.exists():
        rec=read(pages)
        if rec['qualified']:
            obs=rec['observation']
            stage,=obs['stages']
            fixed=obs['frozen']
            rows.append(dict(op='matmul',shape=obs['shape'],attrs=obs['attrs'],chip=obs['chip'],
                             us=fixed['pr_us'],ca_us=obs['ca_us'],future_us=fixed['future_us'],
                             cohort='new PR two-wave Matmul, shape matched to historical DT',
                             receipt=receipt({**stage,'api':obs['api'],'correctness':obs['correctness']}),
                             source=str(pages.relative_to(ROOT))))
        else:
            rejected.append(dict(source=str(pages.relative_to(ROOT)),reason=rec.get('reason')))
    return rows,rejected


def analyze():
    rows,rejected=collect()
    groups=defaultdict(list)
    for r in rows:
        groups[r['op'],r['chip'],r['cohort']].append(r)
    summary={}
    for op in NAMES:
        summary[op]={}
        for side in SIDES:
            chip=CHIPS[side]
            metrics={role:score(group) for (o,c,role),group in groups.items() if o==op and c==chip}
            primary=metrics.get('primary')
            bad=[role for role,d in metrics.items() if d['wape']>.20 or d['max_ape']>.40]
            enough = primary is not None and primary['shapes'] >= 10
            if any(role != 'primary' or enough for role in bad):
                state = '超门槛'
            elif not enough:
                state = '待验证（暂超）' if bad else '待验证'
            else:
                state = '达标'
            if op=='kda_chunk' and primary is None:
                state='缺完整 CA'
            summary[op][side]=dict(status=state,primary=primary,cohorts=metrics,outside=bad)
    out=dict(thresholds={'wape':0.2,'max_ape':0.4,'minimum_distinct_shapes':10},
             policy='Each SKU and route separately. Any violating tracked cohort is disclosed. The primary for modified models is a new frozen cohort. Prior releases are replayed through the public auto entry point.',
             summary=summary,rows=rows,rejected=rejected)
    write(OUT/'evaluation.json',out)
    return out


def fmt(m):
    return '0 · N/A' if m is None else f"{m['shapes']} · {m['wape']:.1%} / {m['max_ape']:.1%}"


def report():
    out=analyze()
    lines=['# 算子耗时预测与 CA 校准误差','',
        '| 算子 | 950DT 样本 · WAPE / 最大 | 950PR 样本 · WAPE / 最大 | 主表数据角色 | 验收（DT / PR） |',
        '|---|---:|---:|---|---|']
    for op,name in NAMES.items():
        d=out['summary'][op]
        role='v0.21 冻结独立' if op in CHANGED else '历史冻结合同' if op!='kda_chunk' else '无合格耗时'
        lines.append(f"| {name} | {fmt(d['dt']['primary'])} | {fmt(d['pr']['primary'])} | {role} | {d['dt']['status']} / {d['pr']['status']} |")
    lines+=['',
        '每一格独立核对 WAPE＝Σ|预测−CA|/ΣCA ≤20%，以及 max＝max(|预测−CA|/CA) ≤40%。未满十个不同 shape 时，已完成的结果只表示进度。旧留出一旦用于开发即转作回放；历史回放超限会保留在验收结论。组合算子按该合同中串行 kernel 的设备窗口相加，排除宿主间隙。970DT 始终为配置外推。','',
        '[逐 shape 的三芯片结果与误差](../results/v0.21/shape-errors.md) · [同形 DT/PR 加速比与旧版](../results/v0.21/pairs.md) · [带 kernel、trace 和角色的机器可读结果](../results/v0.21/evaluation.json) · [逐块的模型与 CA 同轴图](pipeline-comparisons.md)','',
        'v0.12 的 development 与 validation 分组保留，曾参与校准的旧样本只按回放计分。对于 Indexer、mHC Pre、mHC 拆分链、KDA Recurrent、GMM 与 MoE FFN，DT 主表采用原始 v0.12 明确标记 validation 的十形，PR 采用 v0.17 冻结十形。Matmul、FA 复用 v0.17 资源模型验证合同，DT 包含历史同形对照。RMSNorm 的历史设备窗口用于启动状态开发；十个冻结新形逐端采集。首轮宿主自设的宽度上限和 64 列对齐限制会在部分形状进入 RMSNorm 前报错，此类尝试未产生有效 kernel 窗口，原预测及失败原因保留。公开默认入口 auto 用于主表。Quant、Matmul、FA 另对 270 个不同 trace 的完整窗口逐一核对；补充归档保留原角色，跨组可能重用同一个窗口。上述审计完成后，另用同一宿主首次采集 PR 的 [1280,2048,512]。模型和 CA 的 416 条 MMAD、52 个输出块全部相符，耗时误差 +6.12%。970DT 外推单列。[PR 矩形尾块逐页证据](../results/v0.21/matmul-pages-pr/report.md) · [三条核心路线的全窗口审计](../results/v0.21/coverage/report.md)','',
        '### 当前超限的数据组','',
        '| 算子 | 芯片 | 数据组 | WAPE / 最大 | 最大误差 shape | 预测 / CA µs |','|---|---|---|---:|---|---:|']
    for op,by in out['summary'].items():
        for side,d in by.items():
            for role in d['outside']:
                m=d['cohorts'][role]
                lines.append(f"| {NAMES[op]} | {side.upper()} | {role} | {m['wape']:.2%} / {m['max_ape']:.2%} | {m['worst_shape']} | {m['worst_pred_us']:.3f} / {m['worst_ca_us']:.3f} |")
    lines.extend(['','### 系统调度提供的额外信息','',
        '下表按归档宿主的实际提交顺序调用 OperatorStream。输入仍为 shape 和调用属性，前序核数由模型推断。单独启动的进程从空流开始；连续序列要求从第 0 个 launch 起完整可重放。这里统计历史开发回放，两种模式共用每个 CA 观测。','',
        '| 算子 | 芯片 | 调用情景 | 窗口 | 仅 shape WAPE / 最大 | 已知顺序 WAPE / 最大 |','|---|---|---|---:|---:|---:|'])
    protocols={'ordered compact stream':'连续同族调用','isolated official launch':'独立首发'}
    for r in read(OUT/'dispatch/stream-replay.json')['summary']:
        a,b=r['unknown'],r['stream']
        lines.append(f"| {NAMES[r['op']]} | {r['chip'].split('_')[0][6:]} | {protocols[r['protocol']]} | {a['n']} | {a['wape']:.2%} / {a['max_ape']:.2%} | {b['wape']:.2%} / {b['max_ape']:.2%} |")
    lines+=['','RMSNorm 的 43 个 DT 旧开发形状也纳入公开默认入口的压力回放。部分已热的短宽形被当成了可能首发，新增 AIV 也可能是同核另一通道。按输出地址恢复的 1,418 个 AIV 窗口支持在已知前序时细分启动和统计量写回，候选的仅 shape 和有顺序误差分别保留，等待独立实测。[RMSNorm 逐块证据和候选验证](../results/rms-dispatch-audit/report.md)','',
        '同一 stream 上十个新形的原采集在导出时达到磁盘预算；每端九个窗口满足全核完整性。已知派发顺序时，RMS 候选在 DT 的 WAPE / 最大误差为 6.60% / 15.64%，PR 为 7.60% / 14.47%；旧模型在同组分别为 17.86% / 47.26% 和 17.32% / 52.16%。仅凭 shape 时，候选在 PR 的 WAPE 从 16.37% 变为 17.84%。候选仍单独报告，主表使用已发布版本。[原始进程状态与逐核恢复](../results/rms-dispatch-audit/scope.md) · [两种输入方式的同集比较](../results/rms-dispatch-audit/report.md)','',
        '每核多轮的 RMSNorm [32768,1280] 含 41,943,040 个 BF16 元素。v0.21 在 DT 按最忙核 512 行、22 轮预测 39.393 µs，实测 44.585 µs；在 PR 按 586 行、25 轮预测 61.616 µs，实测 69.729 µs。这是 v0.20 登记的大 shape；v0.21 用于后续回放，970DT 的 21.436 µs 为外推。[PR 的 25 轮对照图](examples/figures/rms-wide-tiles-pr.png) · [两端全核计时、输出与采集开销](../results/v0.21/wide-rms/receipt.json)','',
        'mHC Post 的 DT 同形 [1,4,128]、同 ELF 在单独首发与真实前序生产链后分别耗时 4.752、2.672 µs。任何仅依赖本次输入的单值预测，在这两次观测各计一次时的 WAPE 至少为 (4.752−2.672)/(4.752+2.672)＝28.03%。该下界只适用于这一配对；完整算子样本集的 WAPE 仍以逐形数据计算。[两端完整配对与 WAPE 可行域](../results/post-candidate/protocol-feasibility.json)','',
        '[按实际派发顺序的逐条预测](../results/v0.21/dispatch/stream-replay.json) · [mHC Post 同形跨状态对照](../results/v0.21/dispatch/post-identifiability.json) · [45 个进程的阶段审计与未启用候选](../results/post-candidate/report.md)','',
        '在大 Cast 上，DT 的最大历史误差发生在一个 96 MiB 工作集，实测没有物理写入 HBM。把输入、输出双缓冲分别计价后，该形状由 13.33 升至 16.59 µs，CA 为 17.04 µs。12 个双端大窗口的历史回放及另行冻结的 6 个新形状单独计分。[Cast 的物理读写计数与候选验证](../results/cast-capacity/report.md)','',
        '数据角色 primary 表示上表该路线的主验证组。未合格的采集保留在分项报告，N/A 不进入 WAPE 分母。KDA chunk 的运行时 SKU、同步错误和校验状态逐次保留。[KDA chunk 算例](examples/kda_chunk.md) · [KDA 库版本与事件核对](../results/chunk-recovery/report.md) · [融合路线的拒收回执](../results/v0.21/fusion/evaluation.json)','','## v0.20 完成时的报告与历史校准明细','']
    doc=ROOT/'docs/error-analysis.md'
    text=doc.read_text();marker='## v0.20 完成时的报告与历史校准明细\n'
    history=text.split(marker,1)[1].lstrip() if marker in text else text.split('\n',1)[1].lstrip()
    doc.write_text('\n'.join(lines)+history)
    table=['# 逐算子、逐芯片的校准与外推','',
           '设备时间均为 µs。DT 和 PR 各行对应自己的完整 CA 窗口；970DT 列是同一输入在假定芯片上的外推。v0.21 新独立组使用采集前封存的预测。源回执、库 kernel、数值结果见同目录 evaluation.json 及其 source。']
    for op,name in NAMES.items():
        table+=['',f'## {name}','',
                '| 数据组 | 芯片 | shape | v0.20 | 当前 | CA | 有符号误差 | 970DT 外推 |','|---|---|---|---:|---:|---:|---:|---:|']
        for r in out['rows']:
            if r['op']!=op:continue
            old=f"{r['old_us']:.3f}" if r.get('old_us') is not None else '—'
            table.append(f"| {r['cohort']} | {r['chip'].split('_')[0][6:]} | {r['shape']} | {old} | {r['us']:.3f} | {r['ca_us']:.3f} | {r['us']/r['ca_us']-1:+.1%} | {r['future_us']:.3f} |")
        if not any(r['op']==op for r in out['rows']):table.append('| 无合格窗口 | DT、PR | — | — | — | N/A | N/A | N/A |')
    (OUT/'shape-errors.md').write_text('\n'.join(table)+'\n')
    return out


if __name__=='__main__':
    report()
