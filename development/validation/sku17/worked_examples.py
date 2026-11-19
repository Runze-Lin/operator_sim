"""Render reproducible per-operator examples from frozen forecasts and qualified CA traces."""
import sys
from campaign import ROOT,OUT,OPS,CORE,DT,PR,FUT,read
from example_traces import pipelines, issue_landmarks
from operator_model import estimate,load_chip
import core_examples
import fusion_examples
import vector_examples
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DOC=ROOT/'docs/examples'
VECTORS={'quant','mhc_post','rmsnorm','mhc_sinkhorn','kda_recurrent'}
PREFERRED={'matmul':[1536,1792,16],'fa':[9,8191,64,1,1],'quant':[144,5120]}
AXES={'quant':'行, 列','matmul':'M,N,K','fa':'Q,KV,D,B,H','rmsnorm':'token, 通道','addmm':'M,N,K',
      'lightning_indexer':'B,Q,KV,查询头数,D','mhc_pre':'T,N,D','mhc_pre_sinkhorn':'T,N,D',
      'mhc_sinkhorn':'T,N','mhc_post':'T,N,D','mhc':'T,N,D','kda_recurrent':'B,T,Hq,Hv,K,V',
      'kda_chunk':'B,T,Hq,Hv,K,V','moe_ffn':'T,K,F（SwiGLU 中间宽度）,E','grouped_matmul_swiglu_quant_v2':'T,K,F（SwiGLU 中间宽度）,E','gmm2':'T,K,F（SwiGLU 中间宽度）,E'}
NAMES={'quant':'DynamicMXQuant','matmul':'MatMulV3','fa':'FlashAttentionScoreV2','addmm':'Addmm',
       'rmsnorm':'RMSNorm','lightning_indexer':'Lightning Indexer','mhc':'mHC 三段链','mhc_pre':'mHC Pre',
       'mhc_pre_sinkhorn':'mHC Pre Sinkhorn','mhc_post':'mHC Post','mhc_sinkhorn':'mHC Sinkhorn',
       'kda_recurrent':'Recurrent KDA','kda_chunk':'Chunk KDA','moe_ffn':'MoE FFN 两段链',
       'gmm2':'GroupedMatmulV5（GMM2）','grouped_matmul_swiglu_quant_v2':'GroupedMatmulSwigluQuantV2（GMM1）'}
LIMITS={'matmul':'末波切分、FIX 写回完成和 L2 驻留决定多波误差。CA 指令窗口可用于识别重叠；仅凭 FIX 的忙时无法识别片外实际写入字节。',
        'fa':'只有工作量大到出现重复 KV 块时，发起间隔才会主导。首 MMAD 就绪、Vector 交接、K/V 的真实 cache 重用与读队列延迟仍可改变总时长。',
        'quant':'局部 MTE 描述符、同一写粒度内多个核的 scale 写和全片共享读写在 UB 槽间交错。端到端带宽含首尾延迟；应同时核对读写描述符的发起间隔和核数。',
        'rmsnorm':'写描述符与 SIMD 部分重叠，尾部行数和先前调用的驻留状态会影响暴露量。',
        'mhc_post':'小 token、宽通道的首次、扩核与热态首发控制不同；未提供 resident_aiv 时输出为多个状态的综合估计。',
        'addmm':'最长队列外的暴露比例来自 DT 关键核；固定比例只近似覆盖跨设备排队和尾波。',
        'kda_recurrent':'时序状态更新串行，关键核 Vector 和标量控制不随整芯片核数成比例加速。',
        'mhc_sinkhorn':'打包行数、循环与尾核决定关键核工作；标量启动和小数据量下的搬运占比较高。'}


def pipeline_plot(op,results):
    fig,axes=plt.subplots(sum(map(len,results.values())),1,squeeze=False,
                          figsize=(9,2.45*sum(map(len,results.values()))))
    text=[];pos=0
    for chip,stages in results.items():
        for stage_i,v in enumerate(stages):
            ax=axes[pos,0];pos+=1;tracks=sorted(v['tracks'].items())
            if not tracks:raise ValueError(('empty trace',op,chip))
            ax.imshow([x['density'] for _,x in tracks],cmap='Blues',vmin=0,vmax=1,aspect='auto',
                      extent=[0,v['span_us'],len(tracks)-.5,-.5],interpolation='nearest')
            ax.set_yticks(range(len(tracks)),[k for k,v in tracks],fontsize=7)
            ax.set_title(f'{chip} | kernel {stage_i+1} | full window {v["span_us"]:.3f} us',loc='left',fontsize=9)
            ax.set_xlabel('Time since first core instruction (us)',fontsize=8)
            busy=sorted(tracks,key=lambda t:-t[1]['union_us'])[:3]
            text.append(f'{chip} 第 {stage_i+1} 段 {v["span_us"]:.4f} µs；所选核上占用最长的队列为 '+ '、'.join(f'{k} {x["union_us"]:.4f} µs' for k,x in busy)+'。')
    fig.tight_layout()
    file=DOC/'figures'/f'{op}.png';file.parent.mkdir(exist_ok=True)
    fig.savefig(file,dpi=112);plt.close(fig)
    return text


def render(op,case,pair):
    s,a=case['shape'],case['attrs']
    if op=='kda_chunk':
        a={**a,'safe_gate':True,'lower_bound':-0.03125,'export_gk':True}
    mode='resource' if op in CORE else 'auto'
    variants={k:estimate(op,s,a,chip=sku,timing_mode=mode) for k,sku in [('dt',DT),('pr',PR),('future',FUT)]}
    w=variants['dt'].work or {};o=variants['future'];dtype=a.get('dtype',w.get('dtype','bf16'))
    explain=(getattr(core_examples,{'kda_chunk':'chunk'}.get(op,op),None)
             or (vector_examples.explain if op in VECTORS else fusion_examples.explain))
    lib='ops-nn' if op in ('matmul','quant','rmsnorm','addmm') else 'ops-transformer'
    lines=[f'# {NAMES[op]} 算例','',f'输入轴顺序为 **{AXES[op]}**。本例比较同一输入契约在 950DT、950PR 与假设的 970DT 上的设备指令窗口。','',
           '```python','from operator_model import estimate','',f'shape = {s!r}',f'attrs = {a!r}',
           f'for chip in ({DT!r}, {PR!r}, {FUT!r}):',
           f'    r = estimate({op!r}, shape, attrs, chip=chip, dtype={dtype!r}, timing_mode={mode!r})',
           '    print(r.latency_us, r.status, r.active_aic, r.active_aiv, r.work)','```','',
           '## 输入到工作量','',f'该路线调用 CANN 9.2.1 的 {lib} 算子族。实际调用接口为 `{OPS[op]}`。',
           f'工作量接口给出的 Cube 操作为 {w.get("cube_ops"):,}，Vector 计数为 {w.get("vec_ops"):,}。逻辑读/写分别为 {w.get("logical_read_bytes")} / {w.get("logical_write_bytes")} B。此处 None 表示缺少计数；片外实际流量须另行观测，逻辑字节不含中间层重复搬运。', '',
           '## 分块、资源和关键路径','',explain(op,s,a),'','## 设备时间及误差','','| 平台 | 模型 (µs) | CA (µs) | 有符号误差 | 预测 AIC / AIV | CA AIC / AIV |','|---|---:|---:|---:|---|---|']
    for k,title in [('dt','950DT'),('pr','950PR'),('future','970DT 外推')]:
        ref=pair.get(k) if pair and k!='future' else None
        v=variants[k];c=ref and ref['ca_us']
        num=lambda val:'N/A' if val is None else f'{val:.5f}'
        cores=lambda val:'N/A' if val is None else str(val)
        err=f'{100*(v.latency_us/c-1):+.2f}%' if c and v.latency_us else 'N/A'
        lines.append(f'| {title} | {num(v.latency_us)} | {num(c)} | {err} | {cores(v.active_aic)} / {cores(v.active_aiv)} | '+(f'{cores(ref["active_aic"])} / {cores(ref["active_aiv"])}' if ref else 'N/A')+' |')
    if pair:
        lines.extend(['',f'DT/PR 加速比取同形耗时之比，模型 {pair["speedup_pred"]:.4f}×，CA {pair["speedup_ca"]:.4f}×。正时延误差为高估。'])
        data={}
        for k,title in [('dt','950DT'),('pr','950PR')]:
            rec=pair[k];checks=rec['correctness'];stages=pipelines(rec)
            if not stages:raise ValueError(('missing trace',op,k,rec['trace']))
            data[title]=stages
            lines+=['',f'**{title} CA**　{rec["backend"]}，角色 {rec["role"]}；核名 `{rec["kernel"]}`。该宿主的数值检查共 {len(checks)} 组；子算子切片共享宿主检查。各设备窗以最早到最晚核指令计，多 kernel 取串行窗之和。']
        lines+=['','## CA 流水与模型差距','',f'![{NAMES[op]} 实测队列](figures/{op}.png)','',
        '横轴为每次 kernel 自身的设备时间，两端各自从零开始。每种核取最晚结束的一个核。每行按该队列有效指令区间的并集着色，100 个等宽时间格，深蓝表示格内占满。不同队列可重叠，行长不能相加。DMA/FIX 行表示描述符的在途窗口，可能包含等待依赖的时间；颜色深度不能直接换算 HBM 总线利用率。','']
        lines+=pipeline_plot(op,data)
        if op in CORE:
            for k,title in [('dt','950DT'),('pr','950PR')]:
                marks=issue_landmarks(pair[k],op)
                if not marks:continue
                lines+=['',f'**{title} 发起与完成**　'+marks[0],'',
                    '| 队列 | 指令及程序计数器 | 临界核次数 | 首次发起距全核起点 (µs) | 描述符中位持续 (µs) | 相邻发起中位间隔 |',
                    '|---|---|---:|---:|---:|---|',*marks[1:]]
                if op in ('matmul','fa'):
                    diag=variants[k].projection['timing_diagnostics']
                    lines+=['',f'模型对本次调用的全核 MMAD 预计 {diag["mmads_total"]} 次。在有多核重叠时，全核次数用于核对分块，时间取设备窗。']
                if op=='fa':
                    diag=variants[k].projection['timing_diagnostics']
                    wv=diag['wave_types'][0];clock=1000*load_chip(pair[k]['chip']).frequency_ghz
                    lines+=['',f'到首次 QK 的模型时间为启动 {diag["stages_us"]["launch"]:.4f} µs 加本波 QK 就绪 {wv["qk_start_cycles"]/clock:.4f} µs。首个满 KV tile 的稳态间隔为 {wv["first"]["ii_cycles"]/clock:.4f} µs；它应与同一 K 或 V 读程序计数器的连续发起间隔比较。']
        delta=pair['pr_us']-pair['pr_ca_us']
        lines+=['',f'PR 的模型与 CA 相差 {delta:+.4f} µs。'+LIMITS.get(op,'本路线使用 DT 的队列重叠或阶段模板；目标芯片的实际发起、核间同步和尾部任务分布可变。')+'上图给出可观测的队列活动；一次端到端差值无法唯一分摊到这些因素。']
    else:
        rejected=[r for r in read(OUT/'rejections.json') if r['case'].startswith(op+'_')]
        lines+=['','## CA 流水与模型差距','',f'本次 PR 尝试：{rejected[0]["reason"] if rejected else "无合格完整窗口"}。实际宿主开启 safe gate、lower_bound=−0.03125 并导出 GK；冻结清单遗漏了这些属性，默认门控不同。上面的结构演算采用宿主实际属性。950DT 的同类同步失败已归档。两端没有通过正确性与完整窗口检查的本例 trace；无法画出合格流水图或计算误差。']
    lines+=['','[总建模方法](../modeling-strategy.md) · [全部 shape 的误差](../error-analysis.md) · [970DT 配置与来源](../../configs/chips/Ascend970DT.json)']
    (DOC/f'{op}.md').write_text('\n'.join(lines)+'\n')


def main(ops):
    pairs=read(OUT/'pairs.json')
    cases=[o for grp in ('core','extension') for j in read(OUT/f'{grp}-freeze.json')['jobs'] for o in j['outputs']]
    for op in ops or OPS:
        available=[p for p in pairs if p['op']==op]
        preferred=next((p for p in available if p['shape']==PREFERRED.get(op)),None)
        pair=preferred or next((p for p in available if p['backend_match']),None) or next(iter(available),None)
        if not pair and op!='kda_chunk':raise RuntimeError(f'No qualified pair yet for {op}')
        render(op,pair or next(c for c in cases if c['op']==op),pair)
        print(op,(pair or {}).get('shape'),flush=True)


if __name__=='__main__':
    main(sys.argv[1:])
