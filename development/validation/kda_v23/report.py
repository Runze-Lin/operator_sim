"""Build an atomic progress table with trace-backed per-phase counts and paired comparisons."""
from pathlib import Path
import time
import sys

from series import DT,PR,OUT,REG,probe,read,save
from features import write as features


def number(value):
    return 'N/A' if value is None else f'{value:.4f}'


def update():
    plan=read(REG);status=read(OUT/'status.json');cases=[dict(id='reused-ascend950dt-t2-h1',sku=DT,shape=[1,2,1,1,128,128]),*plan['cases']]
    rows=[];by_shape={}
    for case in cases:
        path=OUT/(case['id']+'.json')
        d=read(path) if path.exists() else {}
        valid=d.get('qualified',False)
        summary=dict(case_id=case['id'],sku=case['sku'],T=case['shape'][1],H=case['shape'][3],
                     role='reused development' if case['id'].startswith('reused') else 'new development',
                     state=('complete' if valid else 'ineligible') if d else ('running' if status.get('id')==case['id'] else 'queued'),
                     measured_us=d.get('serialized_device_us'),main_us=d.get('main_stages_us'),wall_s=d.get('cost',{}).get('seconds'))
        if d:
            summary.update(receipt=path.name,receipt_sha256=probe.digest(path),warnings=d.get('sync_errors'),
                           stages=[dict(family=r.get('family'),ca_us=r['ca_us'],aic=r.get('active_aic'),aiv=r.get('active_aiv'),mmads=r.get('mmads')) for r in d.get('stages',[])],
                           numeric=d.get('golden_qualified'),window=d.get('full_windows'))
        if valid:
            features(path);summary['resources']='resources/'+path.name
            by_shape.setdefault((summary['T'],summary['H']),{})[case['sku']]=summary['measured_us']
        rows.append(summary)
    pairs=[dict(T=t,H=h,dt_us=v[DT],pr_us=v[PR],dt_over_pr=v[DT]/v[PR]) for (t,h),v in sorted(by_shape.items()) if DT in v and PR in v]
    data=dict(state=status['state'],role=plan['role'],registration_sha256=probe.digest(REG),rows=rows,paired=pairs)
    save(OUT/'summary.json',data)
    lines=['# FLA Chunk KDA V2 开发采集','',
     '这组样本直接调用 FLA 26.9.1 的公开三阶段入口。BF16，BSND，B=1，K=V=128，C=64，Q/K/V 与 raw gate 的数值和布局沿用已通过的 Kimi 小例。T 是此次调用的 token 数，H 是本地 Q 与 V 的头数。它们都是开发数据，生产 Python 在这里的工作量会选 V1。表中设备耗时是同一条 ACL 流中各实际设备 launch 的独立全核窗口之和，排除 launch 间隔。','',
     '| 芯片 | T | H | 角色与状态 | Prepare / FwdH / Finalize µs | 全调用 µs | CA 墙钟 s | 逐核证据 |',
     '|---|---:|---:|---|---|---:|---:|---|']
    for q in rows:
        phases=[s for s in q.get('stages',[]) if s['family']]
        link=('['+q['case_id']+']('+q['receipt']+')') if 'receipt'in q else '—'
        lines.append(f"| {q['sku'][6:11]} | {q['T']} | {q['H']} | {q['role']} · {q['state']} | "
                     f"{' / '.join(number(s['ca_us']) for s in phases) or 'N/A'} | {number(q['measured_us'])} | "
                     f"{number(q['wall_s'])} | {link} |")
    lines+=['','## 同形双端全窗','','| T | H | DT µs | PR µs | DT / PR |','|---:|---:|---:|---:|---:|']
    for q in pairs:
        lines.append(f"| {q['T']} | {q['H']} | {number(q['dt_us'])} | {number(q['pr_us'])} | {number(q['dt_over_pr'])} |")
    lines+=['','硬件模型分别通过 runtime SoC 与物理 DRAM 配置核对。两端分别使用 HBM3 与 LPDDR5。每条合格记录核查四组逐元素参考、初态保持、发布库与设备 ELF、所有活动核真实 END、分核与合并 trace 的物理指令/同步完成点及官方时长。旧 DT 对照的本轮新增 CA 墙钟为零；表中保留其历史采集开销。','',
            '[原数学协议的开发登记](development-register.json) · [原协议同集结果](summary.json) · [单核资源证据](resources/) · [发布精度对照、独立留出及加速比](mechanisms.md)','',
            '资源表保留逐核 MMAD 的 M/N/K 及 PC、首发与后续发起间隔，并按 DMA 源/目的地址空间拆开搬运量、描述符时延和相邻发起间隔。OUT 是 GM 指令地址空间，实际到达芯片外存的流量还受缓存影响。单个双端比值覆盖一次仿真实现的同形复放；样本间差异不能直接解释成芯片理论峰值比。']
    p=OUT/'report.md.tmp';p.write_text('\n'.join(lines)+'\n');p.replace(OUT/'report.md')
    print(status['state'],len([q for q in rows if q['state']=='complete']),len(rows),flush=True)
    return status['state']


if __name__=='__main__':
    while True:
        state=update()
        if sys.argv[1:]!=['--watch'] or state!='running':break
        time.sleep(40)
