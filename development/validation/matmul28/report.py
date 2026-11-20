"""Publish the registered Matmul comparisons. Only qualified full windows enter errors."""
import fcntl
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/v0.28/matmul'
DT, PR, FUTURE = 'Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'


def read(path):
    return json.loads(path.read_bytes())


def write(path, contents):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(contents, encoding='utf-8')
    temp.replace(path)


def metrics(pairs):
    if not pairs:
        return dict(n=0, wape=None, max_ape=None)
    return dict(n=len(pairs), wape=100*sum(abs(p-t) for p,t in pairs)/sum(t for _,t in pairs),
                max_ape=max(100*abs(p/t-1) for p,t in pairs))


def fmt(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def report():
    reg = read(OUT/'freeze.json')
    rows, groups = [], {}
    for r in reg['rows']:
        slug = '-'.join(map(str,r['shape']))
        refs = {}
        for tag, chip in [('dt', DT), ('pr', PR)]:
            pred = r['predictions'][chip]
            receipt = OUT/(slug+'-'+tag+'.json')
            d = read(receipt) if receipt.exists() else {}
            if d.get('qualified') is False and 'end marker disagrees' in d.get('reason',''):
                audit = OUT/(slug+'-'+tag+'-drain.json')
                if not audit.exists():
                    from drain import recover
                    try:
                        data = recover(r,d,ROOT/'runs/matmul28'/d['case'])
                    except (ValueError,KeyError,AssertionError) as error:
                        data = dict(qualified=False,reason=str(error),ca_us=None)
                    write(audit,json.dumps(data,ensure_ascii=False,indent=2)+'\n')
                result=read(audit)
                if result.get('qualified'):
                    d.update(result)
                    receipt=audit
            refs[tag] = dict(chip=chip, prediction_us=pred['us'], model_aic=pred['active_aic'],
                             model_aiv=pred['active_aiv'], ca_us=d.get('ca_us'), qualified=d.get('qualified'),
                             status='pending' if not d else ('qualified' if d['qualified'] else d.get('reason')),
                             error_pct=d.get('signed_error_pct'), stages=d.get('stages',[]),
                             wall_s=d.get('cost',{}).get('seconds'), receipt=receipt.name if d else None)
        rows.append(dict(shape=r['shape'], lane=r['lane'], node=r['node'], model=r['model'],
                         requests=r['requests'], query_per_request=r['query_per_request'],
                         extrapolated_970dt_us=r['predictions'][FUTURE]['us'], wall_cap_s_per_sku=r['wall_s'],
                         **refs))
    for lane in ['decode','prefill']:
        cases = [r for r in rows if r['lane']==lane]
        group = {}
        for tag in ['dt','pr']:
            group[tag] = metrics([(r[tag]['prediction_us'], r[tag]['ca_us']) for r in cases if r[tag]['qualified']])
        both = [r for r in cases if r['dt']['qualified'] and r['pr']['qualified']]
        ratios = [(r['dt']['prediction_us']/r['pr']['prediction_us'],r['dt']['ca_us']/r['pr']['ca_us']) for r in both]
        group.update(dt_over_pr=metrics(ratios),direct_dt_for_pr=metrics([(r['dt']['prediction_us'], r['pr']['ca_us']) for r in both]),
                     direction_matches=sum((p>1)==(t>1) for p,t in ratios))
        groups[lane]=group
    pending = sum(r[t]['qualified'] is None for r in rows for t in ['dt','pr'])
    rejected = sum(r[t]['qualified'] is False for r in rows for t in ['dt','pr'])
    report_data=dict(registered_utc=reg['utc'],rows=rows,groups=groups,pending=pending,rejected=rejected,
                     max_frozen_prediction_ms=max(p['query_ms'] for r in reg['rows'] for p in r['predictions'].values()))
    write(OUT/'evaluation.json',json.dumps(report_data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# Matmul 的 decode、MTP 与大 prefill 实测','',
           '表内形状均为单 rank 的 [M,N,K]。输入 X[M,K]、权重 W[K,N] 均为连续 BF16 ND，调用安装包 aclnnMatmul，逐元素核验全部 M×N 个输出。N、K 来自三模型在 TP8 下该投影的权重。生产运行的量化权重、转置形式、padding 或融合派发可能改变 kernel。','',
           '普通 decode 每个请求贡献一行；MTP8 校验每请求最多贡献九行。B=1、Q=9 与 B=9、Q=1 都可产生 M=9。同一权重和输入几何只验证一次。prefill 的 B8×512 表示一轮调度 4096 个 token，B8×32768 表示 262144 行一次提交。','',
           f'冻结 16 个新 shape、两端 32 个采集。当前 {32-pending-rejected} 个合格完整窗、{rejected} 个不合格、{pending} 个待回收。冻结时三种配置的最大单次预测开销为 {fmt(report_data["max_frozen_prediction_ms"])} ms。970DT 全部是外推。','',
           '| 场景 | 模型 · 投影 | B×每请求 token | [M,N,K] | DT 预测 / CA (误差) µs | PR 预测 / CA (误差) µs | 970DT 外推 µs |',
           '|---|---|---:|---|---:|---:|---:|']
    for r in rows:
        vals=[]
        for t in ['dt','pr']:
            q=r[t]; ca='待回收' if q['qualified'] is None else ('未合格' if not q['qualified'] else f'{fmt(q["ca_us"])} ({q["error_pct"]:+.1f}%)')
            vals.append(f'{fmt(q["prediction_us"])} / {ca}')
        lines.append(f'| {r["lane"]} | {r["model"]} · {r["node"]} | {r["requests"]}×{r["query_per_request"]} | {r["shape"]} | {vals[0]} | {vals[1]} | {fmt(r["extrapolated_970dt_us"])} |')
    lines+=['','| 场景 | DT 形数 · WAPE / 最大 | PR 形数 · WAPE / 最大 | DT/PR 比值 WAPE / 最大 | 直接用 DT 预测 PR，同形 WAPE |','|---|---:|---:|---:|---:|']
    for lane,g in groups.items():
        def summary(m):
            return f'{m["n"]} · '+(f'{m["wape"]:.1f}% / {m["max_ape"]:.1f}%' if m["n"] else 'N/A')
        lines.append(f'| {lane} | {summary(g["dt"])} | {summary(g["pr"])} | {summary(g["dt_over_pr"])} | {fmt(g["direct_dt_for_pr"]["wape"],1)} |')
    lines+=['','## 已返回的库分支和成本','',
            '| [M,N,K] | 芯片 | 预测 AIC/AIV | 实测 kernel · AIC/AIV · MMAD 数 | 墙钟 (s) | 资格 |','|---|---|---:|---|---:|---|']
    for r in rows:
        for t in ['dt','pr']:
            q=r[t]
            if q['qualified'] is None: continue
            stage=' + '.join(f'{s["kernel"]} · {s["active_aic"]}/{s["active_aiv"]} · {s["mmad"]}' for s in q['stages']) or '—'
            lines.append(f'| {r["shape"]} | {t.upper()} | {q["model_aic"]}/{q["model_aiv"]} | {stage} | {fmt(q["wall_s"],1)} | [{q["status"]}]({q["receipt"]}) |')
    lines+=['','CA 从所有活动核最早发起计至最后退出。多 kernel 派发保留每个完整窗口及成本，只有冻结合同对应的一次独立 Matmul 窗参加聚合。WAPE 以实测时间加权；分母为零个合格样本时记 N/A。两个队列按完成情况更新此表。[冻结的模型输入、预算与三芯片预测](freeze.json) · [逐形机器可读表](evaluation.json) · [单卡模型规格和切分](../../../docs/prefill-shapes.md)','']
    write(OUT/'report.md','\n'.join(lines))


if __name__=='__main__':
    lock = ROOT/'runs/matmul28/report.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        report()
