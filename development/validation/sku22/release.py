"""Reprice the frozen measurement ledger through the public backend, retaining data roles."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import OperatorStream, __version__, estimate
from development.validation.sku19.campaign import CHIPS, read, write
from development.validation.sku21.acceptance import NAMES, CHANGED, fmt, score

OUT=ROOT/'results/v0.22'
DT,PR,FUT=CHIPS.values()
CAST='cast_bf16_fp32'


def us(op,s,attrs,chip):
    return estimate(op,s,attrs,chip=chip).latency_us


def ledger(archive):
    old=read(archive/'results/v0.21/evaluation.json')
    cast={tuple(r['shape']):r['role'] for r in read(archive/'results/cast-capacity/two-slot-scores.json')['rows']}
    records=[]
    for prior in old['rows']:
        r=prior.copy();op=r['op'];shape=r['shape'];attrs=r.get('attrs',{});chip=r['chip']
        r['v021_us']=r['us'];r['us']=us(op,shape,attrs,chip);r['future_us']=us(op,shape,attrs,FUT)
        if op==CAST and r['cohort']=='primary' and cast[tuple(shape)]!='prospective':
            r['cohort']='Cast replay: capture began before two-slot candidate freeze'
        records.append(r)
    fusion=read(archive/'results/v0.21/fusion/freeze.json')['cases'];dispatch=read(archive/'results/v0.21/dispatch/freeze.json')['jobs']
    excluded=[]
    for p in sorted((OUT/'recovered').glob('*.json')):
        r=read(p)
        if 'qualified' not in r:continue
        if not r['qualified']:
            excluded.append({'source':str(p.relative_to(ROOT)),'reason':r.get('reason')});continue
        op,shape,chip=r['op'],r['shape'],r['chip'];ob=r['observation']
        if op=='mhc_post':
            job=next(j for j in dispatch if j['op']==op and j['shape']==shape);attrs=job['attrs']
            prev=job['outputs'][0][('dt' if chip==DT else 'pr')+'_us']
        else:
            job=next(j for j in fusion if j['op']==op and j['shape']==shape and j['chip']==chip);attrs=job['attrs'];prev=job['latency_us']
        assert not any(x['op']==op and x['shape']==shape and x['chip']==chip and x['cohort']=='primary' for x in records)
        records.append(dict(op=op,shape=shape,attrs=attrs,chip=chip,protocol='isolated first launch',cohort='primary',us=us(op,shape,attrs,chip),
                            v021_us=prev,ca_us=ob['ca_us'],future_us=us(op,shape,attrs,FUT),source=str(p.relative_to(ROOT)),
                            receipt={k:ob[k] for k in ('kernel','api','trace','trace_sha256','binary_sha256','correctness','active_aic','active_aiv')}))
    freeze=read(OUT/'cast-freeze.json')
    for p in sorted((OUT/'cast-receipts').glob('*.json')):
        r=read(p)
        if not r['qualified']:
            excluded.append({'source':str(p.relative_to(ROOT)),'reason':r.get('reason')});continue
        s,c=r['shape'],r['chip'];new=us(CAST,s,{},c);assert isclose(r['us'],new,abs_tol=1.e-12)
        fut=next(j for j in freeze['jobs'] if j['shape']==s and j['chip']==FUT)
        records.append(dict(op=CAST,shape=s,attrs={},chip=c,cohort='primary',us=new,v021_us=r['old_us'],ca_us=r['stage']['ca_us'],future_us=fut['us'],
                            source=str(p.relative_to(ROOT)),receipt={**{k:r['stage'][k] for k in ('kernel','trace','trace_sha256','binary_sha256','active_aic','active_aiv')},'correctness':r['correctness'],'api':r['api']}))
    return records,old['rejected']+excluded


def with_order(archive,records):
    rows=[]
    first_prices={}
    for folder in ('rms','dispatch'):
        for job in read(archive/f'results/v0.21/{folder}/freeze.json')['jobs']:
            for side,c in (('dt',DT),('pr',PR)):
                if side+'_first_us' in job['outputs'][0]:
                    first_prices[job['op'],tuple(job['shape']),c]=job['outputs'][0][side+'_first_us']
    for chip in (DT,PR):
        real=read(archive/'results/rms-dispatch-audit/prefix-recovery/comparison.json')['rows']
        stream=OperatorStream(chip);future=OperatorStream(FUT)
        seq=sorted((r for r in real if r['chip']==chip),key=lambda r:r['i']);assert [r['i'] for r in seq]==list(range(10))
        for r in seq:
            s=r['shape'];value=stream.advance('rmsnorm',s).latency_us;assert isclose(r['candidate_known_us'],value,abs_tol=1.e-10)
            rows.append(dict(op='rmsnorm',chip=chip,shape=s,ordinal=r['i'],cohort='ordered prospective',us=value,ca_us=r['ca_us'],
                             v021_us=r['previous_known_us'],unknown_us=us('rmsnorm',s,{},chip),future_us=future.advance('rmsnorm',s).latency_us,source=r['source']))
        for r in records:
            if r['chip']!=chip or r['op'] not in ('rmsnorm','mhc_post') or r['cohort']!='primary':continue
            s=r['shape'];a=r['attrs'];op=r['op'];known=OperatorStream(chip).advance(op,s,a).latency_us
            rows.append(dict(op=op,chip=chip,shape=s,ordinal=0,cohort='isolated first launch',us=known,ca_us=r['ca_us'],v021_us=first_prices[op,tuple(s),chip],
                             unknown_us=r['us'],future_us=OperatorStream(FUT).advance(op,s,a).latency_us,source=r['source']))
        path=archive/f'results/post-candidate/paired-{chip}.json'; pair=read(path);s=pair['shape'];a={'dtype':'bf16','has_h_res':True};stream=OperatorStream(chip);future=OperatorStream(FUT)
        for r in pair['rows']:
            value=stream.advance('mhc_post',s,a).latency_us;assert isclose(r['current_us'],value)
            rows.append(dict(op='mhc_post',chip=chip,shape=s,ordinal=r['ordinal'],cohort='same stream repeated',us=value,v021_us=r['current_us'],unknown_us=r['current_unknown_us'],
                             ca_us=r['stage']['ca_us'],future_us=future.advance('mhc_post',s,a).latency_us,source=str(path.relative_to(archive))))
    return rows


def publish(archive):
    rows,rejected=ledger(archive); ordered=with_order(archive,rows);groups=defaultdict(list)
    for r in rows:groups[r['op'],r['chip'],r['cohort']].append(r)
    metrics={}
    for op in NAMES:
        metrics[op]={}
        for c in (DT,PR):
            roles={role:score(g) for (o,k,role),g in groups.items() if (o,k)==(op,c)}
            primary=roles.get('primary');bad=[role for role,m in roles.items() if m['wape']>.2 or m['max_ape']>.4]
            status='超限' if bad else (('V2 待建模' if op=='kda_chunk' and c==DT else '缺完整 CA') if not primary else '采集中' if primary['shapes']<10 else '达标')
            metrics[op][c]=dict(roles=roles,status=status,over=bad)
    v2_path=archive/'results/chunk-recovery/fla2691/v2/verified.json'
    diagnostic=read(v2_path)
    assert diagnostic['qualified'] and diagnostic['sku']==DT and all(x[1]==0 for x in diagnostic['complete_golden'].values())
    v2=dict(op='kda_chunk',api='aclnnChunkKdaFwdV2',shape=[1,2,1,1,128,128],chip=DT,role=diagnostic['role'],
            ca_us=diagnostic['serialized_device_us'],elapsed_with_gaps_us=diagnostic['device_elapsed_first_to_last_us'],
            us=None,future_us=None,kernels=[st['kernel'] for st in diagnostic['stages']],source=str(v2_path.relative_to(archive)))
    write(OUT/'evaluation.json',dict(version=__version__,unpriced_measurements=[v2],thresholds=dict(wape=.2,max_ape=.4,distinct_shapes=10),summary=metrics,rows=rows,rejected=rejected,stream=ordered))
    table=['# v0.22 · 实测设备耗时与外推','', '时间单位为 µs。v0.21 与 v0.22 均使用同一条 CA 观测。970DT 按该芯片配置外推。数据组在同一芯片内分别统计，重叠回放不会相加。','']
    for op,name in NAMES.items():
        table+=['',f'## {name}','','| 数据组 | 芯片 | shape | v0.21 | v0.22 | CA | 偏差 | 970DT 外推 |','|---|---|---|---:|---:|---:|---:|---:|']
        for r in sorted((r for r in rows+ordered if r['op']==op),key=lambda r:(r['cohort']!='primary',r['cohort'],r['shape'],r['chip'])):
            table.append(f"| {r['cohort']} | {r['chip'][6:11]} | {r['shape']} | {r['v021_us']:.3f} | {r['us']:.3f} | {r['ca_us']:.3f} | {r['us']/r['ca_us']-1:+.2%} | {r['future_us']:.3f} |")
        if op=='kda_chunk':
            table.append(f"| 公开 V2 直接入口 · 开发测量 | 950DT | {v2['shape']} | N/A | N/A | {v2['ca_us']:.3f} | N/A | N/A |")
            table.append('| 默认小形 V1 | DT / PR | — | N/A | N/A | N/A | N/A | N/A |')
    (OUT/'shape-errors.md').write_text('\n'.join(table)+'\n')
    title=['# 算子耗时预测与 CA 校准误差','', '| 算子 | 950DT 形状数 · WAPE / 最大 | 950PR 形状数 · WAPE / 最大 | 主表数据角色 | 全组验收 DT / PR |','|---|---:|---:|---|---|']
    for op,name in NAMES.items():
        d=metrics[op]; role=('双槽冻结后独立' if op==CAST else 'v0.21 冻结独立' if op in CHANGED else '历史冻结合同') if op!='kda_chunk' else 'V1 未校准；另有 DT V2 实测 1 形'
        title.append(f"| {name} | {fmt(d[DT]['roles'].get('primary'))} | {fmt(d[PR]['roles'].get('primary'))} | {role} | {d[DT]['status']} / {d[PR]['status']} |")
    title+=['','主表通过公开 `estimate` 计算每卡 shape 和调用属性的点估计。每组 WAPE＝Σ|预测−CA|/ΣCA，最大误差＝max |预测/CA−1|。验收要求两者分别 ≤20% 和 ≤40%，每端主验证含至少十种 shape；任一跟踪组超限都保留。Cast 的早期四对转入回放，见下表和逐行数据。通过了数值检查且全核指令窗完整，才纳入 CA 耗时。','',
            '[每个算子及 shape 的 950DT/PR 预测、实测和 970DT 外推](../results/v0.22/shape-errors.md) · [双端同形加速比与直接复用 DT](../results/v0.22/pairs.md) · [原始观测、实际库 kernel、正确性与分核字段](../results/v0.22/evaluation.json) · [同一时间轴上的模型与实测](pipeline-comparisons.md)','',
            '## 同一系统调用知道前序时','', '每个 rank 和每条流分别维护 `OperatorStream`。从空流开始，核数由各芯片自己的分块规则计算。RMSNorm 的有序十形在候选冻结后采集；20 个有效目标包括按原前序补采的两个末点，重复前缀只核验采集条件。表内的首次与连续调用分组各自拥有独立的时间窗口。','',
            '| 算子 | 芯片 | 实际前序 | 窗口数 | 无历史 WAPE / 最大 | 已知前序 v0.21 | 已知前序 v0.22 |','|---|---|---|---:|---:|---:|---:|']
    labels={'same stream repeated':'同一输入连续两次','ordered prospective':'十形连续','isolated first launch':'独立进程首发'}
    for op,c,role in sorted({(r['op'],r['chip'],r['cohort']) for r in ordered}):
        g=[r for r in ordered if (r['op'],r['chip'],r['cohort'])==(op,c,role)]
        values=[score(g,key) for key in ('unknown_us','v021_us','us')];summary=' | '.join(f"{m['wape']:.2%} / {m['max_ape']:.2%}" for m in values)
        title.append(f'| {NAMES[op]} | {c[6:11]} | {labels[role]} | {len(g)} | {summary} |')
    title+=['','只给相同 shape 和属性时，两次 Post 共享同一预测。实测长窗 L、短窗 S 给出必然的最大相对误差下界 (L−S)/(L+S)，DT 为 43.34%、PR 为 41.08%。这个下界约束包含这对样本的任何整体最大误差；整体 WAPE 依赖其他观测。[同值输入、同一 ELF 与两次完整流水](examples/mhc_post.md) · [RMS 两个有序形状的公开调用和完整推导](examples/rmsnorm.md)','',
            '## 仍超限的数据组','','| 算子 | 芯片 | 数据组 | WAPE / 最大 | 最差 shape | 预测 / CA µs |','|---|---|---|---:|---|---:|']
    for op,d in metrics.items():
        for c,x in d.items():
            for role in x['over']:
                m=x['roles'][role];title.append(f"| {NAMES[op]} | {c[6:11]} | {role} | {m['wape']:.2%} / {m['max_ape']:.2%} | {m['worst_shape']} | {m['worst_pred_us']:.3f} / {m['worst_ca_us']:.3f} |")
    title+=['','## 采集、算例与硬件条件','', 'RMSNorm 未提供历史时，旧 DT 短宽组仍高估部分已热调用；已知前序可拆分同核新增通道和新物理核。Cast 的四槽地址可以核对，但大流式窗口的共享带宽、读写响应与发起间隔仍有误差。B8×32k 全 hidden 的低输出量采集另列原生包络，逐指令主表只统计已完成的指令窗。[两项全尺寸任务的预测与采集状态](../results/v0.22/full-prefill.md)。下方附历史各算子的库、误差及硬件条件，当前数字使用上方 v0.22 表。[RMS 写线与归一化临界路径](examples/rmsnorm.md) · [Cast 两侧槽地址和实际重叠](examples/cast_bf16_fp32.md) · [v0.22 的开销与变更](../results/v0.22/changes.md)','']
    title+=['在能重建实际 Pre → Sinkhorn → Post 顺序的 34 条历史链上，跨族切换的 Post 使用公开入口的未知历史估计。PR 的 10 形 WAPE / 最大为 6.93% / 14.29%。如果直接传入前一个 Sinkhorn 实测的 AIV 数作为 Post 的驻留数，WAPE 升至 24.29%，最大为 30.61%。这项对照检验了调用方可以安全传入什么信息。[DT 与 PR 分组、真实 ELF 与端到端三段误差](../results/v0.22/post-stream/report.md)','',
            'Kimi K3 的 FLA 26.9.1 在默认 BF16、64-token 块与 128 维 QK/V 的合同下，按本次提交的各序列长度决定融合 V1 或三段 V2。单例 V1 的 96 个核都有 END，出现 240 次重复置位，数值校验未完成。同一发行包直接调 V2 的 DT 开发小例 [1,2,1,1,128,128] 已通过四组金标和真实结束核验。三段各为 15.281、4.739、3.221 µs，设备窗和 23.241 µs，含两段调度间隙为 28.034 µs。该例尚无事前耗时预测，WAPE 和 970 外推均为 N/A。[实际分派与厂商包指纹](../results/chunk-recovery/fla2691/report.md)','']
    title+=['RMSNorm 有 199 个已去重全窗。同 SKU、同正式 ELF 与相同输入的 14 对重跑都具有相同前序，最大可证的两窗下界仅 1.86%。DT 历史 compact [2,4608] 首发/复用差得更多，但数据库缺少可验证的 ELF 身份。在十条无历史误差超过 40% 的 DT compact 窗口中，提供可重建的前序后 WAPE / 最大为 6.64% / 13.88%；这是定位误差来源的历史子集。[强弱证据分别核验](../results/v0.22/rms-unknown/report.md)','']
    old_doc=(archive/'docs/error-analysis.md').read_text();anchor='## v0.20 完成时的报告与历史校准明细';historical=old_doc[old_doc.index(anchor):]
    (ROOT/'docs/error-analysis.md').write_text('\n'.join(title)+'\n'+historical)
    return rows,ordered

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive',type=Path,default=ROOT);args=parser.parse_args();publish(args.archive.resolve())
