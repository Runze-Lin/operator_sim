"""Deduplicate complete three-stage evidence and compare one frozen prediction on both ends."""
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from evidence import ROOT,OUT,OLD,read,sha
from frozen import predict


def measures(rows,idx,key):
    actual=[r['actual'][idx] for r in rows]
    preds=[r[key][idx] for r in rows]
    err=[abs(y-x)/x for x,y in zip(actual,preds)]
    return dict(n=len(err),wape=100*sum(abs(x-y) for x,y in zip(actual,preds))/sum(actual),maximum=100*max(err)) if err else None


def build():
    reg=read(OUT/'register.json');expected={c['id']:c for c in reg['cases']}
    # The archive and candidate are SHA-checked when the frozen engine loads.
    # Full host/ELF/physical-trace verification was recorded by each capture.
    observed=set();rows=[];rejected=[]
    files=[*sorted(OLD.glob('*ascend*.json')),*sorted(OUT.glob('independent-Ascend*.json'))]
    for f in files:
        rc=read(f);case=rc.get('case');role='independent' if f.parent==OUT else 'archive'
        if not case:continue
        if not rc['qualified']:
            rejected.append(dict(role=role,file=str(f.relative_to(ROOT)),numeric=rc['golden_qualified']));continue
        st=[s for s in rc['stages'] if s.get('family')]
        hashes=tuple(s['aggregate_sha256'] for s in st)
        if hashes in observed:continue
        observed.add(hashes)
        shape=case['shape'];sku=case['sku'];pred=predict(shape,sku)
        if role=='independent':
            assert sha(OUT/'register.json')==rc['registration_sha256']
            assert expected[case['id']]['predictions'][sku]==pred
        ca=[s['ca_us'] for s in st]
        assert abs(sum(ca)-rc['main_stages_us'])<1e-8 and len(st)==3
        assert [s['active_aic'] for s in st]==pred['active_aic']
        rows.append(dict(role=role,shape=shape,sku=sku,file=str(f.relative_to(ROOT)),receipt_sha256=sha(f),stage_sha256=hashes,
                         actual=ca+[sum(ca)],v023=pred['stages_old']+[pred['v023_us']],candidate=pred['stages_us']+[pred['us']],
                         ca_cost_s=rc.get('cost',{}).get('seconds'),raw_peak_bytes=rc.get('cost',{}).get('peak_bytes'),
                         prediction_970_us=predict(shape,'Ascend970DT')['us']))
    summary={};pairs=[]
    for role in ('archive','independent'):
        role_rows=[r for r in rows if r['role']==role]
        for sku in ('Ascend950DT_9582','Ascend950PR_957b'):
            sub=[r for r in role_rows if r['sku']==sku]
            summary.setdefault(role,{})[sku]={name:{v:measures(sub,i,v) for v in ('v023','candidate')} for i,name in enumerate(('prepare','fwdh','finalize','total'))}
        grouped={}
        for r in role_rows:grouped.setdefault(tuple(r['shape']),{})[r['sku'][-4:]]=r
        for s,ends in grouped.items():
            if len(ends)!=2:continue
            dt,pr=ends['9582'],ends['957b'];ratios={k:dt[k][3]/pr[k][3] for k in ('candidate','v023','actual')}
            pairs.append(dict(role=role,shape=s,**ratios,correct_direction=(ratios['candidate']>1)==(ratios['actual']>1),
                              relative_error=100*(ratios['candidate']/ratios['actual']-1)))
    timing=[]
    cfg=read(ROOT/'configs/models/Kimi-K3.json')['text_config']['linear_attn_config']
    local_heads=cfg['num_heads']//8
    assert cfg['num_heads']%8==0 and cfg['head_dim']==128
    shapes=[[1,80,1,1,128,128],[1,145,2,2,128,128]]
    shapes += [[b,seq,local_heads,local_heads,128,128] for b in (1,8) for seq in (32768,131072,262144)]
    for shape in shapes:
        for sku in ('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'):
            costs=[]
            for _ in range(9):
                start=perf_counter_ns();p=predict(shape,sku);costs.append((perf_counter_ns()-start)/1e6)
            timing.append(dict(shape=shape,sku=sku,prediction_us=p['us'],fwdh_us=p['stages_us'][1],
                               eligible_for_new_fwdh='one sequence' if shape[0]*shape[3]<=p['active_aic'][1] else 'fallback: multihead workgroup',
                               query_max_ms=max(costs),query_median_ms=median(costs)))
    return dict(order=['prepare','fwdh','finalize','sum of complete device windows'],register_sha256=sha(OUT/'register.json'),
                summary=summary,rows=rows,pairs=pairs,rejected_prereg_or_archived=rejected,timings=timing,
                independent_pending=[k for k in expected if not (OUT/(k+'.json')).exists()])


def render(x):
    out=['# KDA FwdH 跨块读与状态等待','',
         '| 角色 | SKU | 合格窗口 | FwdH WAPE 旧 → 新 | FwdH 最大误差 旧 → 新 | 三段 WAPE 旧 → 新 | 三段最大误差 旧 → 新 |',
         '|---|---|---:|---:|---:|---:|---:|']
    for role,targets in x['summary'].items():
        for sku,stats in targets.items():
            n=stats['total']['candidate']
            if not n:out.append(f'| {role} | {sku} | 0 | N/A | N/A | N/A | N/A |');continue
            fmt=lambda st,k:' → '.join(f'{stats[st][v][k]:.2f}%' for v in ('v023','candidate'))
            out.append(f'| {role} | {sku} | {n["n"]} | {fmt("fwdh","wape")} | {fmt("fwdh","maximum")} | {fmt("total","wape")} | {fmt("total","maximum")} |')
    out+=['','archive 为已归档的开发样本及过去版本的留出，本轮统一按回放统计。independent 为本次冻结后采集。观察值为三个合格完整设备窗，先按三个原始 trace 的 SHA 组合去重；旧数值失败的采集单列于 JSON。'
          f'当前独立待回执 {len(x["independent_pending"])} 个。','',
          '## 每个样本','',
          '输入均为 BF16、FP32 初态、128 维 key/value，B=1，Hq=Hv=H。四个数值组独立验证，实际 API 是 FLA 26.9.1 的 aclnnChunkKdaFwdV2。单位为 µs，以下三段合计排除宿主和 kernel 间空隙。','',
          '| 角色 | SKU | T / H | Prepare 预测 / CA | FwdH 旧 / 新 / CA | Finalize 预测 / CA | 三段旧 / 新 / CA | 970DT 新合计外推 |',
          '|---|---|---|---:|---:|---:|---:|---:|']
    for r in sorted(x['rows'],key=lambda r:(r['role'],r['shape'][1:3],r['sku'])):
        v,n,a=r['v023'],r['candidate'],r['actual'];_,s,h,*_=r['shape']
        out.append(f'| {r["role"]} | {r["sku"]} | {s} / {h} | {n[0]:.3f} / {a[0]:.3f} | {v[1]:.3f} / {n[1]:.3f} / {a[1]:.3f} | {n[2]:.3f} / {a[2]:.3f} | {v[3]:.3f} / {n[3]:.3f} / {a[3]:.3f} | {r["prediction_970_us"]:.3f} |')
    out+=['','## DT 与 PR 的完整三段加速比','',
          '| 角色 | T / H | 旧 DT/PR | 新 DT/PR | CA DT/PR | 新比值误差 |','|---|---|---:|---:|---:|---:|']
    for r in x['pairs']:
        out.append(f'| {r["role"]} | {r["shape"][1]} / {r["shape"][2]} | {r["v023"]:.4f} | {r["candidate"]:.4f} | {r["actual"]:.4f} | {r["relative_error"]:+.2f}% |')
    for role in ('archive','independent'):
        pairs=[p for p in x['pairs'] if p['role']==role]
        if pairs:
            observed=[p['actual'] for p in pairs]
            errors=[100*sum(abs(p[k]-p['actual']) for p in pairs)/sum(observed) for k in ('v023','candidate')]
            directions=sum(p['correct_direction'] for p in pairs)
            out.extend(['',f'{role} 共 {len(pairs)} 对，DT/PR 比值 WAPE {errors[0]:.2f}% → {errors[1]:.2f}%，新模型判对方向 {directions}/{len(pairs)}。'])
    costs=[r for r in x['rows'] if r['role']=='independent']
    if costs:
        seconds=sum(r['ca_cost_s'] for r in costs)
        peak=max(r['raw_peak_bytes'] for r in costs)/(1024**2)
        out.extend(['',f'独立 CA 串行 {len(costs)} 次，计费进程共 {seconds:.1f} s，单次结果目录采样最大 {peak:.1f} MiB。每个端分别通过四组输出、三段 ELF 和整窗门禁。'])
    timing=max(r['query_max_ms'] for r in x['timings'])
    out+=['',f'常驻 Python 查询中 {len(x["timings"])} 个芯片/形状配置各 9 次，最大 {timing:.3f} ms。记录包含真实 8 条长序列，与一条更长的递推链分开；多头共享一组前端时保留旧阶段估计，另标局限。','',
          '[公式、来源和逐阶段读图](method.md) · [精简模型/实测时标](dt-129-fwdh.json) · [逐窗口及耗时记录](comparison.json)']
    return '\n'.join(out)+'\n'


if __name__=='__main__':
    v=build();(OUT/'comparison.json').write_text(json.dumps(v,indent=2)+'\n');(OUT/'report.md').write_text(render(v))
    print({k:{s:p['total'] for s,p in ss.items()} for k,ss in v['summary'].items()})
