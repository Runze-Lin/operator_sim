"""Publish separate ordered/standalone scores for the frozen dispatch candidate."""
from collections import defaultdict
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.rms_dispatch_audit.extract import OUT
from development.validation.sku21.acceptance import read,write,score,CHIPS


def replay_earlier(params):
    from operator_model import estimate, load_chip
    from development.validation.rms_dispatch_audit.schedule import both

    rows=[]
    for version in ('0.17','0.18'):
        filename=f'results/v{version}/evaluation.json'
        for row in read(ROOT/filename)['rows']:
            if row['op']!='rmsnorm':
                continue
            for side in ('dt','pr'):
                chip=load_chip(CHIPS[side])
                prediction=estimate('rmsnorm',row['shape'],row['attrs'],chip=chip)
                # The second output marginalizes over launch histories; the first
                # needs the actual upstream dispatch sequence and is excluded here.
                _,unknown=both(row['shape'],chip,0,params,prediction)
                obs=row[side+'_observation']
                rows.append(dict(version=version,chip=chip.name,shape=row['shape'],old_us=prediction.latency_us,
                                 us=unknown,ca_us=obs['ca_us'],trace_sha256=obs['trace_sha256'],
                                 source=filename,original_role=row.get(side+'_role',row.get('role')),
                                 launch_context=obs.get('launch_context')))
    summary=[]
    for version in ('0.17','0.18'):
        for side in ('dt','pr'):
            group=[r for r in rows if r['version']==version and r['chip']==CHIPS[side]]
            summary.append(dict(version=version,chip=CHIPS[side],old=score(group,'old_us'),candidate=score(group)))
    write(OUT/'earlier-shape-only.json',dict(role='historical replay after candidate registration; defaults only',
                                            summaries=summary,rows=rows))
    return summary


def publish():
    reg=read(OUT/'freeze.json')
    rows=[];failed=[]
    ordered={}
    for side in ('dt','pr'):
        chip=CHIPS[side]
        path=OUT/('ordered-'+chip+'.json')
        if path.exists():
            batch=read(path)
            if batch['qualified']:
                ordered.update({(chip,r['i']):(r,path,'complete host and parser') for r in batch['rows']})
            else:
                failed.append(dict(chip=chip,protocol='ordered new sequence',reason=batch['reason'],
                                   source=str(path.relative_to(ROOT))))
                recovery=OUT/('verified-ordered-'+chip+'.json')
                if recovery.exists():
                    verified=read(recovery)
                    for row in verified['rows']:
                        if row.get('timing_qualified'):
                            ordered[chip,row['i']]=(row,recovery,'verified per-launch full instruction window')
    for r in reg['rows']:
        if r['chip'] not in reg['measured_skus']:
            continue
        if r['protocol']=='ordered new sequence':
            recovered=ordered.get((r['chip'],r['i']))
            if recovered is None:continue
            outcome,source,quality=recovered
            assert outcome['shape']==r['shape'] and outcome['chip']==r['chip']
            obs=outcome['stage'];checks=outcome['correctness']
        else:
            name='rmsnorm_'+'_'.join(map(str,r['shape']))+'_'+r['chip']+'.json'
            path=ROOT/'results/v0.21/rms/receipts'/name
            if not path.is_file():continue
            result=read(path)
            if not result['qualified']:
                failed.append(dict(chip=r['chip'],shape=r['shape'],protocol=r['protocol'],reason=result['error'],
                                   source=str(path.relative_to(ROOT))))
                retry=ROOT/'results/v0.21/rms/retries'/name
                if not retry.exists():
                    continue
                result=read(retry)
                if not result['qualified']:
                    failed.append(dict(chip=r['chip'],shape=r['shape'],protocol=r['protocol'],
                                       reason=result.get('error',result.get('reason')),
                                       source=str(retry.relative_to(ROOT))))
                    continue
                path=retry
            source=path;quality='complete host and parser'
            obs=result['observation'];checks=obs['correctness']
        rows.append(dict(**r,ca_us=obs['ca_us'],trace=obs['trace'],kernel=obs['kernel'],
                         trace_sha256=obs['trace_sha256'],correctness=checks,
                         source=str(source.relative_to(ROOT)),qualification=quality))
    grouped=defaultdict(list)
    for r in rows:grouped[r['chip'],r['protocol']].append(r)
    keys=('previous_known_us','candidate_known_us','previous_unknown_us','candidate_unknown_us')
    scores=[dict(chip=k[0],protocol=k[1],scores={name:score(group,name) for name in keys})
            for k,group in sorted(grouped.items())]
    write(OUT/'independent.json',dict(rows=rows,summaries=scores,rejections=failed,
        planned_shapes_per_protocol=10,registered_before_ca=reg['utc']))
    dev=read(OUT/'candidate.json')
    txt=['# RMSNorm 的扩核与统计量写回','',
      '在 950DT 复用 53 个完整库窗口、1,418 个活跃 AIV 的指令，从 rstd 的 GM 地址还原每核逻辑行。从同一批次中剔除整个宿主进程，再在两侧剔除相同 shape，测试阶段参数能否迁移到另一组调用。PR 的首发归档另列；开发数据缺少实际连续 PR RMS 的扩核阶段。','',
      '## 工作量如何变成设备时间','',
      '设 T 为行数、D 为通道数、C 为目标 AIV 数，R＝ceil(T/C) 为最忙核行数，A＝ceil(T/R) 为活跃 AIV 数。当前实现的一次首块可以容纳 R 行时，每个 AIV 产生一笔 rstd 和一笔 y。多 UB 块和独立首发沿用已冻结的逐块资源模型。','',
      '1. **落核与启动。** H 表示同流此前已预热的 AIV 高水位。找出本次每个逻辑块对应的硬件分区、物理核和通道，将其分为已热、同核新通道、新物理核和新分区。两个 950DT 分区分别承接 ceil(A/2) 与 floor(A/2) 个连续块；每个分区中一对 AIV 共用物理核。53 个归档上的 1,418 次映射和实际输出地址全部一致。分区并发与物理核布局在 970DT 是继承假设，两代间的启动周期仍需确认。','',
      '2. **何时发起 rstd。** 首发时刻＝首条指令到达周期＋前端准备周期＋max(本核输入字节÷读端口吞吐，归约组数×每组周期)。输入字节为 2D(R+1)，包括 gamma；组数为 R×ceil(D/128)。准备周期由实际首笔 rstd 的发起时间扣除已定价的工作获得。已热核还包含按活跃 AIV 数估计的发起竞争，每多一个 AIV 约 10.35 周期。','',
      '3. **谁在争同一条写线。** 第 k 个块的 rstd 从偏移 4kR 写入，每行占 4 B。对落入同一 128 B 区间的描述符，按预计到达先后递推 Fᵢ＝max(Iᵢ＋L, Fᵢ₋₁＋J)。Iᵢ 是发起周期，Fᵢ 是完成周期，L 是单写响应，J 是已排队写的退休间隔。完整开发档案估计 L＝273、J＝157 cycles，分别来自单写区间和写请求均已发出之后的相邻退休时间。相同状态各核的先后按该次参与写线的核数和观测到的发起离散程度估计。','',
      '4. **写 y 和结束。** rstd 发起后开始归一化，y 可以独立写出。去掉被 rstd 阻塞到最后的 y 写回，以单笔描述符字节量估计响应约 314.25＋5.19×KiB 周期。取全片最晚的 rstd、y 及共享带宽下界，描述符后还留实测的尾部退休周期。按目标主频转换成 µs。不同芯片上的核数、Vector 与 MTE 配置、UB、L2 和片外带宽先进入计算。','',
      '未提供驻留状态时，另外返回热态和可能冷态窗口的调和中心；它在两个端点上有相等的相对误差。以下单独计分，尚未作为默认模型发布。','',
      '## 模型、CA 与误差','',
      '![同核第二个 AIV 加入时，按实际块地址对齐的模型与 CA](../../docs/examples/figures/rms-expanding-sibling.png)','',
      'DT [4,4608] 接在使用 3 个 AIV 的 RMSNorm 后。core16 的第一条 AIV 已在前序调用中预热，第二条此次加入。旧版估计 3.792 µs，候选 2.924 µs，CA 2.633 µs。候选仍高估了新通道到首笔 rstd 的时间。若从一个 AIV 扩到两个，第二个 AIV 在另一个分区，两个宽通道样本的新分区首发偏移超过 2,300 周期。','',
      '| 回放组 | 芯片 | N | 已知派发顺序 旧 → 候选 WAPE / 最大 | 仅 shape 旧 → 候选 WAPE / 最大 |','|---|---|---:|---|---|']
    def pair(a,b):return f"{a['wape']:.2%} / {a['max_ape']:.2%} → {b['wape']:.2%} / {b['max_ape']:.2%}"
    for group in dev['summary']:
        s=group['scores']
        txt.append(f"| {group['cohort']} | {group['chip']} | {s['us']['n']} | {pair(s['old_known_us'],s['known_us'])} | {pair(s['old_us'],s['us'])} |")
    txt+=['','## 新形状的前向验证','',
          '已在原独立 RMS 队列启动前同时冻结两个版本。沿用的十形各自单独发起，另有十个不同的形状在一个真实 stream 上按顺序调用。每端只增加一个十形宿主进程，分别检查每个 y、rstd 和实际 kernel 全核窗口。候选在 DT、PR、970 三档共 60 个冻结预测中的单次最大查询成本为 '+f"{max(r['query_ms'] for r in reg['rows']):.3f} ms。970DT 只有资源外推。",'',
          '| 协议 | 芯片 | 合格 / 登记 | 已知顺序 旧 → 候选 WAPE / 最大 | 仅 shape 旧 → 候选 WAPE / 最大 |','|---|---|---:|---|---|']
    for protocol in ('ordered new sequence','isolated frozen v0.21'):
        for c in reg['measured_skus']:
            group=next((x['scores'] for x in scores if x['chip']==c and x['protocol']==protocol),None)
            txt.append(f'| {protocol} | {c} | '+(f"{group[keys[0]]['n']} / 10 | {pair(group[keys[0]],group[keys[1]])} | {pair(group[keys[2]],group[keys[3]])} |" if group else '0 / 10 | N/A | N/A |'))
    txt+=['','## 更早归档的仅输入预测','',
          '候选冻结后，补入 v0.17 和 v0.18 的 40 个历史端点。v0.17 的 DT 样本含先前多次调用及同形复测，保留其原始数据角色。下表全部省略前序信息。前序未知时，更细的扩核模型仍会高估热态与低估首发，尚不足以替代默认模型。','',
          '| 归档 | 芯片 | 形状数 | 仅 shape 的 v0.21 → 候选 WAPE / 最大 |','|---|---|---:|---|']
    for record in replay_earlier(reg['params']):
        txt.append(f"| v{record['version']} | {record['chip']} | {record['old']['shapes']} | {pair(record['old'],record['candidate'])} |")
    txt+=['','[早期逐形回放与前序来源](earlier-shape-only.json)']
    if (OUT/'scope.md').exists():
        txt+=['','连续调用的训练与预测均在采集前冻结。部分宿主在数值检查及设备执行完成后触及导出文件预算；只有与官方每核 duration、实际入口及 END 指令逐核相符的独立完整窗口计分。原进程终态与逐调用计时依据同时保留。[逐次恢复的证据与范围](scope.md)']
    txt+=['','[时间戳与原始数据库位置](core-windows.json) · [逐进程交叉回放](candidate.json) · [两个协议的事前预测](freeze.json) · [新回执](independent.json)']
    (OUT/'report.md').write_text('\n'.join(txt)+'\n')


if __name__=='__main__':
    publish()
