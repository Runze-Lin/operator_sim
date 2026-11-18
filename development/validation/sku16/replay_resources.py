"""Frozen v0.16. Historical CA replay and one-factor chip counterfactuals."""
import json
import sys
from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate,load_chip

OUT = ROOT/'results/v0.16'
OPS = ('quant','matmul','fa')
ARCHIVES = ('v0.13/sku/pairs','v0.13/sku/holdout','v0.14/sku/holdout','v0.15/sku/holdout')


def load(name):
    return json.loads((ROOT/'results'/(name+'.json')).read_text())


def key(row):
    op,shape = row['op'],tuple(row['shape'])
    return op,shape+((1,1) if op=='fa' and len(shape)==3 else ())


def phase_sources():
    training={}
    quant={r['sha256'] for r in load('v0.16/quant-dt-evidence')['samples']}
    for r in load('v0.4/imported-calibration')['records']:
        m=r.get('measurement') or {}
        if m.get('trace_sha256') in quant:
            training[key(r['case'])] = m['reference_us'],m['trace_sha256']
    assert len(training)==len(quant)
    for r in load('v0.16/matmul-dt-provenance')['traces']:
        training['matmul',tuple(r['shape'])] = r['reference_ticks']/1650,r['sha256']
    for r in load('v0.16/fa-dt-evidence')['rows']:
        if r['db'].startswith('runs/v8core/'):
            ca=load('v0.8/core/'+r['db'].split('/')[2])['reference_us']
        else:
            long=load('v0.5/trace-analysis')
            assert long['db_sha256']==r['sha256']
            ca=long['reference_us']
        training['fa',tuple(r['shape'])] = ca,r['sha256']
    return training


def records():
    training=phase_sources()
    dt,pr={},{}
    for r in load('v0.13/evaluation')['rows']:
        if r['op'] in OPS:
            dt[key(r)]=dict(ca_us=r['dt_ca_us'],source='v0.13/evaluation',id=r['id'])
    for file in ARCHIVES:
        for r in load(file):
            assert key(r) not in pr
            for sku,table in [('dt',dt),('pr',pr)]:
                m=r[sku]
                if m is None:
                    continue
                digest=m.get('trace_sha256') or m['db_sha256']
                table[key(r)]=dict(ca_us=m['ca_us'],trace_sha256=digest,source=file,kernel=m['kernel'])
    source,target=load_chip(),load_chip('Ascend950PR_957b')
    for table,chip in ((dt,source),(pr,target)):
        for k,record in sorted(table.items()):
            op,shape=k
            role='other_dt' if table is dt else 'historical_pr'
            if table is dt and k in training:
                role='phase_training_dt'
                # Use the exact training trace when a shape also has a repeat.
                record=dict(ca_us=training[k][0],trace_sha256=training[k][1],
                            source={'quant':'v0.16/quant-dt-evidence','fa':'v0.16/fa-dt-evidence',
                                    'matmul':'v0.16/matmul-dt-provenance'}[op])
            new=estimate(op,shape,chip=chip,timing_mode='resource')
            ref=estimate(op,shape,chip=source)
            assert new.latency_us and ref.latency_us
            row=dict(op=op,shape=shape,chip=chip.name,role=role,**record,resource_us=new.latency_us,
                     dt_reference_us=ref.latency_us,stages_us=new.projection['stages_us'])
            if table is pr:
                old=estimate(op,shape,chip=chip,timing_mode='calibrated')
                assert old.latency_us
                row['pr_calibrated_us']=old.latency_us
            yield row


def metrics(rows,field):
    ca=sum(r['ca_us'] for r in rows)
    worst=max(rows,key=lambda r:abs(r[field]/r['ca_us']-1))
    return dict(wape_pct=100*sum(abs(r[field]-r['ca_us']) for r in rows)/ca,
                max_ape_pct=100*abs(worst[field]/worst['ca_us']-1),worst=worst['shape'])


def summary(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[r['op'],r['role']].append(r)
    for (op,role),group in sorted(groups.items()):
        out=dict(op=op,role=role,count=len(group),resource=metrics(group,'resource_us'),
                 dt_reference=metrics(group,'dt_reference_us'))
        if role=='historical_pr':
            out['pr_calibrated']=metrics(group,'pr_calibrated_us')
        yield out


def experiments():
    base=replace(load_chip(),name='hypothetical-resource-sensitivity')
    shape=(32,8192,64,1,32)
    trials=[('基准',base),('Vector FP32 27 → 6.75 TOPS',replace(base,vector_tops={**base.vector_tops,'32':6.75})),
            ('Cube BF16 432 → 54 TOPS',replace(base,cube_tops={**base.cube_tops,'16':54})),
            ('HBM 4000 → 1000 GB/s',replace(base,hbm_bandwidth_gbs=1000)),
            ('时钟 1.65 → 1.8 GHz',replace(base,frequency_ghz=1.8)),
            ('256-token 发起间隔 ×2',replace(base,details={**base.details,'fa_issue_cycles':2*load('v0.16/fa')['issue_cycles_256']})),
            ('C:V 32:64 → 32:32',replace(base,vector_cores=32)),('C:V 32:64 → 32:16',replace(base,vector_cores=16))]
    for label,chip in trials:
        model=estimate('fa',shape,chip=chip,timing_mode='resource')
        d=model.projection['timing_diagnostics']
        costs=d['wave_types'][0]['first']['demands_cycles']
        yield dict(change=label,op='fa',shape=shape,us=model.latency_us,stages_us=d['stages_us'],
                   aic=model.active_aic,aiv=model.active_aiv,waves=d['target_plan']['waves'],
                   dominant=max(costs,key=costs.get),first_full_tile_demands_cycles=costs,
                   full_tile_ii_us=max(costs.values())/(chip.frequency_ghz*1000))


def render(doc):
    lines=['# 冻结后回放与单变量资源敏感性','',
           '同一个 950DT 目标上，resource 按工作量和资源单价预测，DT 参考列采用工具既有的 950DT 默认模型。WAPE 为误差绝对值之和除以 CA 耗时之和，max 为单 shape 最大绝对百分比误差。每个芯片每个算子同一 shape 只计一次。',
           '','## 950DT 历史 CA','',
           '| 算子 | 数据角色 | shape 数 | 资源 WAPE / max | DT 参考 WAPE / max | 资源模型最差 shape |',
           '|---|---|---:|---:|---:|---|']
    def pct(a):
        return f"{a['wape_pct']:.2f}% / {a['max_ape_pct']:.2f}%"
    stats={(s['op'],s['role']):s for s in doc['summary']}
    for op in OPS:
        for role,title in [('phase_training_dt','精确阶段校准 trace'),('other_dt','其他历史 shape')]:
            a=stats[op,role]
            lines.append(f"| {op} | {title} | {a['count']} | {pct(a['resource'])} | {pct(a['dt_reference'])} | {list(a['resource']['worst'])} |")
    lines+=['','DT 阶段校准行使用提取参数时的原 trace（含 SHA256），重复运行同 shape 时保留这次测量。另一组排除了全部阶段校准 shape，但 tiling 规则、路由选择及参考模型可能已受历史观测影响，因此属于回顾性比较。样本覆盖 v0.13 总评中的三个算子以及 v0.13–v0.15 两端配对 CA 的并集。',
            '','## 950PR 旧 CA','',
            '| 算子 | 历史 shape 数 | 资源 WAPE / max | 直接复用 DT WAPE / max | 既有 PR 校准 WAPE / max |',
            '|---|---:|---:|---:|---:|']
    for op in OPS:
        a=stats[op,'historical_pr']
        lines.append(f"| {op} | {a['count']} | {pct(a['resource'])} | {pct(a['dt_reference'])} | {pct(a['pr_calibrated'])} |")
    lines+=['','这些 PR 测量在本轮前已知，既有 PR 校准也曾使用部分旧 PR 阶段数据。本轮新采的 PR 独立留出另行统计。[逐点 CA、预测、资源阶段与来源](resource-replay.json)',
            '','## 同一 FA shape 的资源敏感性','',
            '固定 BF16 非因果 SBH [32,8192,64,1,32]，即每卡 32 个头。以 DT 参数为基准，每次只改表中一项，其他输入（包括其他精度档位、峰值 TOPS、核数和 GB/s）均保持原值。总耗时、稳态间隔均为资源情景外推。',
            '','| 唯一变动 | AIC/AIV | 波数 | 满 tile 主导项 | 稳态间隔 µs | 总耗时 µs | 相对基准 |',
            '|---|---:|---:|---|---:|---:|---:|']
    names={'vector':'Vec1 + Vec2','cube':'QK + PV','issue_control':'K/V 发起与缓冲节拍','k_v_read':'K/V 搬运'}
    ref=doc['sensitivity'][0]['us']
    for a in doc['sensitivity']:
        lines.append(f"| {a['change']} | {a['aic']}/{a['aiv']} | {a['waves']} | {names[a['dominant']]} | {a['full_tile_ii_us']:.4f} | {a['us']:.3f} | {(a['us']/ref-1)*100:+.2f}% |")
    lines+=['','基准的 128-token tile 由向量服务决定。降低 Cube 算力或 HBM 带宽可分别切换主导项。当只提高时钟而峰值 TOPS 和 GB/s 不变，向量稳态微秒数不变，缩短的主要是按周期计价的开销。32:32 保持整芯片 FP32 峰值，每个 AIV 的吞吐加倍，抵消了每个 AIC 从两个向量工作者降为一个的代价。32:16 再使 AIC 并发数降至 16，产生两波。这些实验用于区分资源的独立作用，均未作目标芯片 CA 实测。','']
    return '\n'.join(lines)


def main():
    freeze=load('v0.16/sku/freeze')
    for op in OPS:
        for path in (f'operator_model/{op}_resource.py',f'results/v0.16/{op}.json'):
            assert sha256((ROOT/path).read_bytes()).hexdigest()==freeze['sha256'][path]
    rows=list(records())
    doc=dict(frozen_utc=freeze['utc'],role_definition='historical replay',rows=rows,
             summary=list(summary(rows)),sensitivity=list(experiments()))
    (OUT/'resource-replay.json').write_text(json.dumps(doc,indent=2,ensure_ascii=False)+'\n')
    (OUT/'resource-replay.md').write_text(render(doc))


if __name__=='__main__':
    main()
