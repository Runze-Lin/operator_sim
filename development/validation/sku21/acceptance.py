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
        if r['role']!='validation':
            continue
        op=r['op'];cohort='primary' if op in EXT else 'DT v0.12 holdout replay'
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
                rejected.append({k:v for k,v in data.items() if k!='cost'});continue
            obs=data['observation']
            row=dict(op=j['op'],shape=j['shape'],chip=chip,attrs=j['attrs'],us=y[side+'_us'],ca_us=obs['ca_us'],
                     future_us=y['future_us'],old_us=y['old_'+side+'_us'],receipt=receipt(obs),
                     source=str(path.relative_to(ROOT)),cohort='primary')
            if side+'_first_us' in y:row.update(first_us=y[side+'_first_us'],protocol='isolated first launch')
            rows.append(row)
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
        '重复使用的 950DT 归档有些来自调参批次。对于 Indexer、mHC Pre、mHC 拆分链、KDA Recurrent、GMM 与 MoE FFN，DT 主表采用原始 v0.12 明确标记 validation 的十形，PR 采用 v0.17 冻结十形。Matmul、FA 复用 v0.17 资源模型验证合同，DT 包含历史同形对照。RMSNorm 的历史设备窗口用于启动状态开发，本轮十个新形在原有大样本之后串行验证。主表全部使用公开默认入口 auto；970DT 外推单列。','',
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
    lines+=['','[按实际派发顺序的逐条预测](../results/v0.21/dispatch/stream-replay.json) · [mHC Post 同形跨状态对照](../results/v0.21/dispatch/post-identifiability.json) · [45 个进程的阶段审计与未启用候选](../results/post-candidate/report.md)','',
             '数据角色 primary 表示上表该路线的主验证组。当前未合格的采集保留在分项报告；N/A 不进入 WAPE 分母。KDA chunk 的运行时 SKU、同步错误和校验状态逐次保留。[KDA chunk 审计](examples/kda_chunk.md) · [融合路线的拒收回执](../results/v0.21/fusion/evaluation.json)','','## v0.20 完成时的报告与历史校准明细','']
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
