"""Measure signed latency bias on exactly the windows used for absolute-error acceptance."""
from collections import defaultdict
import json
from math import ceil,isclose
from statistics import median
from time import perf_counter_ns


def summarize(samples):
    pairs = list(samples)
    if not pairs:
        return dict(n=0,wape=None,maximum=None,bias_pct=None,faster_fraction=None,
                    median_signed_pct=None,most_negative_pct=None,most_positive_pct=None)
    total = sum(a for p,a in pairs)
    signed = [100*(p-a)/a for p,a in pairs]
    return dict(n=len(pairs),wape=100*sum(abs(p-a) for p,a in pairs)/total,
                maximum=max(map(abs,signed)),bias_pct=100*sum(p-a for p,a in pairs)/total,
                faster_fraction=sum(p<a for p,a in pairs)/len(pairs),median_signed_pct=median(signed),
                most_negative_pct=min(signed),most_positive_pct=max(signed))


def audit(root,out,archive,multi,acceptance,chips,matmul_supply,default_estimate):
    def read(path): return json.loads(path.read_bytes())
    data=defaultdict(dict)
    def add(label,chip,identity,pred,ca):
        if pred is None:return
        key=tuple(identity) if isinstance(identity,list) else identity
        values=data[label,chip]
        if key in values:
            assert all(isclose(a,b,abs_tol=1e-8) for a,b in zip(values[key],(pred,ca)))
        else:
            values[key]=(pred,ca)

    for r in archive:
        name=r['op']
        if name=='mhc_post':
            for label in ('shape','visible stream','experimental fragment stream'):
                add('mhc_post · '+label,r['chip'],r['trace_sha256'],r[label+'_us'],r['ca_us'])
        elif name!='mhc_sinkhorn':
            add(name,r['chip'],r['trace_sha256'],r['current_us'],r['ca_us'])
    for r in read(root/'results/v0.30/sink/score.json')['rows']:
        add('mhc_sinkhorn',r['sku'],r['trace_sha256'],r['public_us'],r['ca_us'])
    for r in read(root/'results/v0.30/post/score.json')['rows']:
        for label,col in [('shape','stateless_us'),('visible stream','ordered_us'),
                          ('experimental fragment stream','candidate_us')]:
            add('mhc_post · '+label,r['chip'],r['sha256'],r[col],r['ca_us'])
    rms=read(root/'results/v0.28/rms/stream-replay.json')
    for r in rms['archived']:
        label='compact CA' if r['provider']=='npusim compact' else 'msopprof'
        add('rmsnorm · '+label,r['chip'],r['trace'],r['unknown_us'],r['ca_us'])
    for r in rms['pairs']:
        for label,col in [('msopprof','unknown_us'),('visible stream','stream_us')]:
            add('rmsnorm · '+label,r['chip'],r['trace'],r[col],r['ca_us'])
    cast=root/'results/v0.28/cast'
    prices=read(cast/'register.json')['prices']
    for c,r in read(cast/'score.json')['heldout'].items():
        if r['qualified']:
            src=read(root/r['receipt'])
            add('cast_bf16_fp32',c,[src['stage']['trace_sha256']],prices[c]['pressure']['us'],src['ca_us'])
    for c,rows in multi.items():
        for r in rows:
            add('kda_recurrent · dense multi-chain',c,tuple(r['shape']),r['pred'],r['ca_us'])

    report={}
    for label,group in acceptance.items():
        report[label]={}
        for c in chips:
            d=summarize(data[label,c].values())
            expected=group[c]
            assert d['n']==expected['n'],(label,c,d['n'],expected['n'])
            if d['n']:
                assert all(isclose(d[k],expected[k],abs_tol=1e-7) for k in ('wape','maximum')),(label,c)
            report[label][c]=d
    heldout={c:summarize((r['current_us'],r['ca_us']) for r in archive if r['chip']==c and
                          r['role']=='v0.31 prospective, device') for c in chips}
    alternate={}
    changed=[]
    cost_ms=[]
    for c in chips:
        same_windows=[]
        for r in archive:
            if r['op']!='matmul' or r['chip']!=c:continue
            start=perf_counter_ns()
            scenario=matmul_supply(r['shape'],chip=c)
            price=scenario['latency_us']
            if price is None:
                price=default_estimate('matmul',r['shape'],dtype='bf16',chip=c).latency_us
                cost_ms.append((perf_counter_ns()-start)/1e6)
                assert isclose(price,r['current_us'],abs_tol=1e-9)
            else:
                cost_ms.append((perf_counter_ns()-start)/1e6)
                changed.append(dict(shape=r['shape'],chip=c,default_us=r['current_us'],scenario_us=price,
                                    ca_us=r['ca_us'],modeled_mmad=scenario['mmad_total'],trace_sha256=r['trace_sha256']))
            same_windows.append((price,r['ca_us']))
        alternate[c]=summarize(same_windows)
    cost=dict(n=len(cost_ms),p95_ms=sorted(cost_ms)[ceil(.95*len(cost_ms))-1],max_ms=max(cost_ms))
    assert cost['max_ms']<100
    (out/'direction.json').write_text(json.dumps(dict(by_protocol=report,independent_b11_matmul=heldout,
                                                 conditional_matmul_supply=dict(same_windows=alternate,changed=changed,query_cost=cost)),
                                                 ensure_ascii=False,indent=2)+'\n')
    lines=['# 同一批 CA 窗口上，预测偏快还是偏慢', '',
           '每个已定价窗口同时进入绝对误差和方向统计。下表含归档开发回放，B11 新 Matmul 留出单列。'
           '带符号偏差为 100 × Σ(预测耗时 − CA) / ΣCA。负数表示累计预测更快。'
           '“偏快占比”是预测耗时严格小于 CA 的窗口数除以全部定价窗口数，每窗权重相同。'
           '正负残差可能抵消，因此验收仍分别要求每端 WAPE ≤25% 且最大单窗绝对误差 ≤50%。', '',
           '| 算子与合同 | DT 窗数 | DT 有符号偏差 / 偏快占比 | PR 窗数 | PR 有符号偏差 / 偏快占比 |',
           '|---|---:|---:|---:|---:|']
    for label,group in [*sorted(report.items()),('Matmul · 独立 B11 两形',heldout)]:
        cells=[]
        for c in chips:
            d=group[c]
            cells.extend([str(d['n']),f"{d['bias_pct']:+.2f}% / {100*d['faster_fraction']:.1f}%" if d['n'] else 'N/A'])
        lines.append(f"| {label} | {' | '.join(cells)} |")
    lines+=['', '## 长 Matmul 的可选资源排程', '',
            '当目标分块允许 B 在每个 AIC 的 L1 常驻、M 有多波、K 页与缓冲容量满足条件时，'
            '调用 estimate_matmul_supply()，其余输入沿用默认 estimate()。选择只用 shape、dtype、布局与芯片配置。'
            '此表对两种选择使用完全相同的设备窗，包含开发回放；下方 JSON 逐形标出改变的输入。', '',
            '| 芯片 | 改变的窗 / 全部窗 | 默认 WAPE / 最大 / 有符号偏差 | 采用常驻情景后的 WAPE / 最大 / 有符号偏差 |',
            '|---|---:|---:|---:|']
    for c in chips:
        a,b=report['matmul'][c],alternate[c]
        fmt=lambda x:f"{x['wape']:.2f}% / {x['maximum']:.2f}% / {x['bias_pct']:+.2f}%"
        lines.append(f"| {c[6:11]} | {sum(r['chip']==c for r in changed)} / {a['n']} | {fmt(a)} | {fmt(b)} |")
    lines+=['', '在 DT 的 B8×32Ki 窄投影中，常驻排程估 16,384 条 MMAD，与 CA 相符，默认路径为 8,192 条。'
            '常驻模型的 85.246 µs 比 CA 的 85.993 µs 快 0.87%；同形 PR 为 176.124 / 174.953 µs，慢 0.67%。'
            '权重复用和输出跨缓存的读写服务都随目标配置重算。更宽输出的 Kimi 实验仍采用普通波次模型。'
            f"按此策略实际调两个公开入口（适用时只调常驻排程），{cost['n']} 个已载入模型参数的进程内调用 "
            f"P95 为 {cost['p95_ms']:.3f} ms，最大 {cost['max_ms']:.3f} ms。"
            '[常驻条件、流水与双端新形](../v0.30/matmul/report.md)']
    lines+=['','模型选择优先保留能解释 tile、流量与流水的策略。在这些证据和独立绝对误差相近时，'
            '系统仿真可以优先采用略快的可实现排程。若减少的时间需要改变内核、缓存前提或向量归并，'
            '应给出假设，再与当前安装库的校准值分别报告。970DT 的时延和未来实现都只有外推。', '',
            '[逐合同 WAPE 和最大误差](acceptance.md) · [逐合同方向统计与极端残差](direction.json) · '
            '[逐 shape CA 和预测](replay.json) · [全量误差与独立验收](../../docs/error-analysis.md)', '']
    (out/'direction.md').write_text('\n'.join(lines))
    return report
