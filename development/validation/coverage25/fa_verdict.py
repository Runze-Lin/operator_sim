"""Reproduce model-vs-CA point, phase, cross-chip and query-latency evidence."""
import json,time,statistics
from pathlib import Path
from audit import ROOT,OUT,read,DT,PR,FUT,call,summary


def build():
    reg=read(OUT/'fa-freeze.json');old=read(OUT/'audit.json');observed=[]
    for item in reg['jobs']:
        r=read(OUT/(item['id']+'.json'));assert r['qualified']
        sku,shape=r['chip'],r['shape']
        values=item['forecasts'][sku];target=call('fa',shape,{},sku)
        assert target.latency_us==values['candidate'] and (target.active_aic,target.active_aiv)==(r['active_aic'],r['active_aiv'])
        assert target.projection['timing_diagnostics']['mmads_total']==r['stages'][0]['mmad']
        samples=[]
        for _ in range(25):
            before=time.perf_counter_ns();call('fa',shape,{},sku);samples.append((time.perf_counter_ns()-before)/1e6)
        observed.append(dict(shape=shape,sku=sku,old=values['old'],candidate=target.latency_us,ca=r['ca_us'],
            aic=target.active_aic,aiv=target.active_aiv,mmad=r['stages'][0]['mmad'],future_us=item['forecasts'][FUT]['candidate'],
            query_p50_ms=statistics.median(samples),query_max_ms=max(samples),cost_s=r['cost']['seconds'],receipt=item['id']+'.json'))
    pairs=[]
    for i in range(0,len(observed),2):
        dt,pr=observed[i:i+2]
        ratio={k:dt[k]/pr[k] for k in ('old','candidate','ca')};pairs.append(dict(shape=dt['shape'],**ratio))
    ah=100*sum(abs(r['candidate']-r['ca']) for r in pairs)/sum(r['ca'] for r in pairs)
    old_ratio=100*sum(abs(r['old']-r['ca']) for r in pairs)/sum(r['ca'] for r in pairs)
    directions={k:sum((p[k]>1)==(p['ca']>1) for p in pairs) for k in ('old','candidate')}
    return dict(rows=observed,pairs=pairs,ratio_wape_old=old_ratio,ratio_wape_candidate=ah,direction_correct=directions,
                duration_s=sum(r['cost_s'] for r in observed),max_model_ms=max(r['query_max_ms'] for r in observed),
                historical_and_prospective=[s for s in old['summary'] if s['op']=='fa'])


def fmt(m):
    return f"{m['wape']:.2f}% / {m['max_ape']:.2f}%"


def render(x):
    lines=['# Dense BF16 FA 的统一资源排程','',
        '| 数据角色 | SKU | 不同形 / 窗口 | v0.24 WAPE / 最大 | 当前 WAPE / 最大 |',
        '|---|---|---:|---:|---:|']
    for g in x['historical_and_prospective']:
        lines.append(f"| {g['role']} | {g['sku'][6:11]} | {g['current']['shapes']} / {g['current']['n']} | {fmt(g['v024'])} | {fmt(g['current'])} |")
    lines+=['','公共接口 `estimate("fa", [Q,KV,D,B,H], chip=…)` 对每块目标芯片都将 QK、PV、softmax、搬运和发起间隔分别计价。v0.24 在参考 DT 上转用了历史局部定价器，在目标 PR 上使用资源排程，相同芯片参数仅改名都可能换时间。本次复用已经冻结的资源公式，v0.24 显式 calibrated 路径仍可复算。','',
       '## 单个 KV 页怎样计时','',
       '以下以独立输入 [9,1008,64,1,1] 为例。Q 为一次校验的 query 数，KV 为有效上下文，D 为 head_dim，B、H 分别为本 rank 的请求数和本地头数。分核得到一个 AIC、两个 AIV，8 个 KV 页依次为七个 128 和一个 112。每页有 QK、PV 两项 MMAD，实测整次共 16 条。','','每项 MMAD 的服务周期等于固定发起响应加上 2·pad16(Q)·pad16(KV页)·pad16(D) / 每核 Cube ops/cycle；pad16 是上取 16 对齐。K/V 请求按实际有效字节除以本地和共享预算中较小的可用带宽。首 QK 需要 Q/K 和 L0；首 PV 要等 V 及 QK/softmax 生产者。后续页的发起周期 II 取 Cube、Vector、MTE2、MTE1、FIX、前端六项服务需求的最大值。所有服务均使用目标芯片配置，时钟周期除以频率变成微秒。实际库合同为 aclnnFlashAttentionScoreV2、BF16 SBH dense noncausal，无 mask。','',
       '![同一 KV 页的资源估计及 CA 指令](fa-dt-1008-pages.png)','',
       '时间零点为全核第一条设备指令，虚线标记最后核退出。950DT 首 QK 模型/CA 为 7.128 / 7.115 µs，首 PV 为 8.138 / 8.323 µs。末页 PV 发起为 9.948 / 11.028 µs。预期间隔 0.259 µs，实测 8 次 QK/PV 发起的相邻间隔中位数分别是 0.408 / 0.409 µs。首发已经接近，短页流水中的间隔和同步仍低估；从整窗差值无法区分发起端与下游反压的份额。代码段、逐页尺寸、指令 PC 和原始 SHA 见配套数据。[每条实际/预测时标及覆盖范围](fa-dt-1008-pages.json)','',
       '| SKU | Q / KV / D / B / H | AIC/AIV | MMAD | v0.24 / 当前 / CA µs | 970DT 外推 µs | CA 秒 |',
       '|---|---|---:|---:|---:|---:|---:|']
    for r in x['rows']:
        lines.append(f"| {r['sku'][6:11]} | {r['shape']} | {r['aic']}/{r['aiv']} | {r['mmad']} | {r['old']:.3f} / {r['candidate']:.3f} / {r['ca']:.3f} | {r['future_us']:.3f} | {r['cost_s']:.1f} |")
    lines+=['','| shape | DT/PR 旧 | DT/PR 当前 | DT/PR 实测 |','|---|---:|---:|---:|']
    for r in x['pairs']:
        lines.append(f"| {r['shape']} | {r['old']:.4f} | {r['candidate']:.4f} | {r['ca']:.4f} |")
    dr=x['direction_correct']
    lines+=['',f"双端耗时比 WAPE {x['ratio_wape_old']:.2f}% → {x['ratio_wape_candidate']:.2f}%，快慢方向 {dr['old']}/3 → {dr['candidate']}/3。新模型在这三种低占用 shape 上预测 DT/PR 接近 1；两个形状真实 DT 慢约 4%，约束条件超出本地峰值能够区分的范围。历史 950DT [64,320,128,1,1] 则由旧版约 +9.0% 扩大到 +12.43%，已保留在全路线逐窗表。",'',
        f"六次串行 CA 共 {x['duration_s']:.1f} 秒；每个 shape/SKU 的公开入口各重跑 25 次，最大单次 {x['max_model_ms']:.3f} ms。{sum(len(v['cores']) for v in read(OUT/'fa-physical-windows.json'))} 个实际核窗口均有唯一物理 END，与正式 duration 和聚合窗口一致，六端实际设备 ELF 均与既有正式 FA 家族一致。长 KV / 真实模型稀疏合同属于另外的覆盖问题。[逐核结束与 ELF](fa-physical-windows.json) · [事前封存](fa-freeze.json) · [精确计算结果](fa-scores.json) · [全路线总表](report.md)"]
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    x=build();(OUT/'fa-scores.json').write_text(json.dumps(x,indent=2)+'\n');(OUT/'fa-method.md').write_text(render(x))
