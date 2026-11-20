"""Score previously frozen prices after each finished physical device receipt."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from development.validation.matmul31.device.campaign import ca, OUT, REG, CHIPS


def write(path, body):
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(body);tmp.replace(path)


def metrics(rows, field):
    if not rows:return dict(windows=0,wape=None,maximum=None)
    errors=[abs(r[field]-r['ca_us']) for r in rows]
    return dict(windows=len(rows),wape=100*sum(errors)/sum(r['ca_us'] for r in rows),
                maximum=100*max(e/r['ca_us'] for e,r in zip(errors,rows)))


def render():
    reg=ca.read(REG);state=ca.read(OUT/'state.json') if (OUT/'state.json').exists() else {}
    rows=[];ratios=[];cost_s=0;cost_bytes=0
    for job in reg['rows']:
        m,n,k=job['shape'];group=[]
        for sku,tag in CHIPS:
            p=job['predictions'][sku]; plan=p['geometry']; stem=f'{m}-{n}-{k}-{tag}';path=OUT/(stem+'.json')
            r=dict(shape=job['shape'],sku=sku,scenarios=job['scenarios'],old_us=p['old_us'],forecast_us=p['us'],future_970dt_extrapolated_us=job['predictions'][ca.FUT]['us'],
                expected_roster=[plan['active_aic'],plan['active_aiv']],expected_productive_aic=plan['productive_aic'],expected_groups=plan['output_tiles'],
                expected_splits=plan['k_splits'],expected_mmad=plan['mmads_total'],ca_us=None,
                status=state['phase'] if state.get('case')==stem else 'pending')
            if path.exists():
                d=ca.read(path)
                assert d['registration_sha256']==ca.digest(REG) and d['chip']==sku and d['shape']==job['shape']
                cost=d['cost']; cost_s+=cost['seconds'];cost_bytes+=cost['peak_bytes']
                r.update(status='qualified' if d['qualified'] else 'collection not qualified',cost_s=cost['seconds'],peak_bytes=cost['peak_bytes'],receipt=path.name)
                if not d['qualified']:r['reason']=d.get('reason')
                else:
                    stage,=d['stages']; ms=d.get('mechanism_checks');g=d.get('actual_geometry',{})
                    r.update(ca_us=d['ca_us'],old_signed_pct=100*(p['old_us']/d['ca_us']-1),
                        signed_pct=100*(p['us']/d['ca_us']-1),geometry_verified=bool(ms) and all(ms.values()),checks=ms,
                        actual_groups=g.get('groups'),actual_splits=g.get('splits_per_output'),actual_productive_aic=g.get('productive_aic'),
                        actual_mmad=stage['mmad'],actual_roster=[stage['active_aic'],stage['active_aiv']],kernel=stage['kernel'],trace_sha256=stage['trace_sha256'],
                        correctness=d['correctness'],terminal_roster=len(d.get('terminal',stage['windows'])))
            rows.append(r);group.append(r)
        dt,pr=group
        if all(r['ca_us'] for r in group):
            ratio=lambda key:pr[key]/dt[key]
            ratios.append(dict(shape=job['shape'],pr_over_dt=ratio('ca_us'),old=ratio('old_us'),new=ratio('forecast_us'),reuse_dt=1.,signed_pct=100*(ratio('forecast_us')/ratio('ca_us')-1)))
    summary={}
    for sku,_ in CHIPS:
        measured=[r for r in rows if r['sku']==sku and r['ca_us']]
        summary[sku]=dict(old=metrics(measured,'old_us'),new=metrics(measured,'forecast_us'),geometry_pass=sum(r['geometry_verified'] for r in measured))
    result=dict(freeze_sha256=ca.digest(REG),rows=rows,summary=summary,ratios=ratios,total_ca_seconds=cost_s,total_raw_peak_bytes=cost_bytes)
    write(OUT/'score.json',json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    lines=['# 两种 B11 投影的 Matmul 独立设备校验','',
        '| 实际模型、每卡 [M,N,K] | SKU | 旧 / 当前预测 / CA µs | 当前偏差 | 预测 / 实测 MMAD | 分核、K 段与退休 | 970DT 外推 µs |',
        '|---|---|---:|---:|---:|---|---:|']
    for r in rows:
        desc=r['scenarios'][0];t=r['ca_us'];a=r['expected_roster'];p=r['expected_productive_aic']
        evidence='进行中' if r['status'] in ('acquiring','qualifying') else '待采' if r['status']=='pending' else '采集未合格'
        if t:
            evidence=f"[{r['actual_roster'][0]} AIC / {r['actual_roster'][1]} AIV，{r.get('actual_productive_aic','—')} 个 AIC 计算]({r['receipt']})，{'分块全符' if r['geometry_verified'] else '分块待核查'}"
        shape='×'.join(map(str,r['shape'])); brief=f"{desc['model']} `{desc['node']}` · Q{desc['tokens_per_request']} · {shape}"
        lines.append(f"| {brief} | {r['sku'][6:11]} | {r['old_us']:.3f} / {r['forecast_us']:.3f} / {f'{t:.3f}' if t else '—'} | {format(r['signed_pct'],'+.2f')+'%' if t else '—'} | {r['expected_mmad']} / {r.get('actual_mmad','—')} | {evidence} | {r['future_970dt_extrapolated_us']:.3f} |")
    lines+=['','| 芯片 | 已测 / 冻结 shape | 旧 WAPE / 最大 | 当前 WAPE / 最大 | 所有分块核对通过 |','|---|---:|---:|---:|---:|']
    for sku,_ in CHIPS:
        s=summary[sku]; n=s['new']['windows'];fmt=lambda x:f"{x['wape']:.2f}% / {x['maximum']:.2f}%" if n else 'N/A'
        lines.append(f"| {sku[6:11]} | {n} / {len(reg['rows'])} | {fmt(s['old'])} | {fmt(s['new'])} | {s['geometry_pass']} / {n} |")
    if ratios:
        lines+=['','| [M,N,K] | 实测 PR/DT | 旧预测比 | 当前预测比 | 比值误差 |','|---|---:|---:|---:|---:|']
        for r in ratios:lines.append(f"| {r['shape']} | {r['pr_over_dt']:.4f} | {r['old']:.4f} | {r['new']:.4f} | {r['signed_pct']:+.2f}% |")
    lines+=['',f"四个端点按顺序各采一次。截至本次评分，已结束端点共用时 {cost_s/60:.1f} 分钟，累计盘占峰值 {cost_bytes/2**20:.1f} MiB。源投影、旧版和候选预测、两端的切 K 段数及真实 MMAD 数的预测均在首次设备采集前登记。此前知道两个形状的 host workspace 大小。此处验证连续 ND BF16、无偏置的 `aclnnMatmul` 实现。正式安装包为 `ops_nn/mat_mul_v3`。模型部署中若使用其他精度、转换或融合，需要相符的 CA 合同。970DT 全为外推。",'',
        '预测按目标 L0A/L0B 容量重选输出矩形并分配 K 段。首个 shape 是 B11 推测八个 token 后每请求校验 Q9。第二个 shape 是 B11 每请求一行的普通 decode。每个合格端点均核对 gold、安装 ELF、official duration 表和全部物理核结束。机制检查另从每核真实 A/B 装载首址、非重叠输出矩形、MMAD 序列、FP32 部分和地址以及所有 AIV 的最终写回，重建每个矩形对整个 K 轴的覆盖。','',
        '[冻结资源与预算](freeze.json) · [逐形机器数据](score.json) · [后台采集状态](state.json)','']
    write(OUT/'report.md','\n'.join(lines))
    return all('receipt' in r for r in rows)

if __name__=='__main__':
    if '--watch' in sys.argv:
        while not render():time.sleep(20)
    else:render()
