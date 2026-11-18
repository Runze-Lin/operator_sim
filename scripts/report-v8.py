"""Publish the single current predictor using development and frozen validation cases."""
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from operator_model.predictor import predict, VECTOR, CUBE
OUT=ROOT/'results/v0.8'
LEGACY=ROOT/'results/v0.7/evaluation.json'
def read(path):
    return json.loads(path.read_text())

def error(rows):
    if not rows:return None,None
    diffs=[abs(r['predicted_us']-r['reference_us']) for r in rows]
    return sum(diffs)/sum(r['reference_us'] for r in rows),max(d/r['reference_us'] for d,r in zip(diffs,rows))

def percent(x):
    return 'N/A' if x is None else f'{100*x:.2f}%'

def core_rows():
    rows=[]
    for r in read(LEGACY)['rows']:
        if r['op'] not in ('quant','fa'):continue
        historical_fields={'constant_us','previous_v05_us','previous_v06_us','local_v7_proxy_us','mechanism_predicted_us'}
        evidence={key:value for key,value in r.items() if key not in historical_fields}
        rows.append(dict(evidence,attrs={},role='validation' if r['role']=='retained_v6_frozen_holdout' else 'development',
                         active_aic=r['observed_cores'][0],active_aiv=r['observed_cores'][1]))
    for path in sorted((OUT/'core').glob('*.json')):
        r=read(path)
        if r.get('status')!='ok':continue
        m=r['measurement']
        rows.append(dict(id=r['id'],op=r['op'],shape=r['shape'],attrs={},role='validation',source=str(path.relative_to(ROOT)),
                         predicted_us=r['predicted_us'],reference_us=r['reference_us'],kernel=m['kernel'],active_aic=m['active_aic'],
                         active_aiv=m['active_aiv'],correctness=r['correctness'],db_sha256=m['db_sha256']))
    return rows

def main():
    old=read(LEGACY)
    rows=core_rows()
    for group in ('vector','cube','fusion'):
        rows.extend(read(OUT/group/'rows.json'))
    ids=set()
    for r in rows:
        key=(r['op'],tuple(r['shape']),json.dumps(r.get('attrs',{}),sort_keys=True))
        if key in ids:raise ValueError(f'duplicate case {key}')
        ids.add(key)
        op,shape,attrs=r['op'],r['shape'],r.get('attrs',{})
        times=[]
        for _ in range(5):
            start=time.perf_counter_ns()
            current=predict(op,shape,attrs)
            times.append((time.perf_counter_ns()-start)/1e6)
        if not math.isclose(current['predicted_us'],r['predicted_us'],rel_tol=1e-8):
            raise ValueError(f'Frozen prediction differs for {op} {shape}: {current} vs {r["predicted_us"]}')
        r['query_max_ms']=max(times)
    summary=[];detail=[];table=['| 算子 | 方法 | 开发 / 独立 shape | 验证误差 WAPE / 最大 | 查询最慢 ms |','|---|---|---:|---:|---:|']
    for ref in old['operator_summary']:
        op=ref['op'];name=ref['name']
        sample=[r for r in rows if r['op']==op];dev=[r for r in sample if r['role']=='development'];val=[r for r in sample if r['role']=='validation']
        unique=len({tuple(r['shape']) for r in val})
        w,m=error(val);worst=max((r['query_max_ms'] for r in sample),default=None)
        family='Vector' if op in VECTOR else 'Cube' if op in CUBE else 'CV'
        summary.append(dict(op=op,name=name,method=family,development=len(dev),independent=unique,wape=w,max_ape=m,query_max_ms=worst))
        table.append(f'| {name} | {family} | {len(dev)} / {unique} | {percent(w)} / {percent(m)} | '+(f'{worst:.3f}' if worst is not None else 'N/A')+' |')
        api=ref['api'];lib=ref['library']
        detail.extend([f'### {name}',f'\n{lib} · `{api}`。\n'])
        if sample:
            detail+=['| shape | 用途 | 预测 µs | CA µs | 有符号偏差 | 入口 | 活跃 AIC / AIV |','|---|---|---:|---:|---:|---|---|']
            for r in dev+val:
                sig=r['predicted_us']/r['reference_us']-1
                a=r.get('attrs',{})
                setting='，I='+str(a['num_iters']) if 'num_iters' in a else ''
                if 'expert_counts' in a:
                    setting='，各专家行数 '+str(a['expert_counts'])
                detail.append(f"| `{r['shape']}`{setting} | "+('开发' if r['role']=='development' else '验证')+f" | {r['predicted_us']:.3f} | {r['reference_us']:.3f} | {sig:+.1%} | {r['kernel']} | {r.get('active_aic')} / {r.get('active_aiv')} |")
            if val:
                worst_case=max(val,key=lambda r:abs(r['predicted_us']/r['reference_us']-1))
                detail.append(f"\n最大偏差出现在 `{worst_case['shape']}`，实际入口为 `{worst_case['kernel']}`。")
        else:detail.append('普通整块和 safe gate 双头候选均在 CA 主 kernel 触发重复 set_flag。完整 CA 耗时 0 个；另有 10 个冻结 shape 通过逻辑工作量、依赖与 CPU 稀疏前向核对，S=1 至 256，头数 1 至 8。[逐 shape 结构验证](../results/v0.8/cube/chunk-structure-validation.json)。')
        detail.append('')
    if any(r['query_max_ms']>=1000 for r in rows):raise ValueError('1 second model query budget exceeded')
    core_cost=sum(read(p).get('wall_seconds',0) for p in (OUT/'core').glob('*.json'))
    cube_cost=read(OUT/'cube/attempts.json')['sum_ca_wall_s']
    fusion_cost=read(OUT/'fusion/costs.json')['cumulative_worker_wall_seconds']
    vector_cost=read(OUT/'vector/costs.json')['worker_seconds']
    cost={'vector_worker_seconds':vector_cost,'quant_fa_worker_seconds':core_cost,'cube_and_chunk_worker_seconds':cube_cost,'fusion_worker_seconds':fusion_cost}
    def plot():
        os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'build/mpl-cache'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        points=[r for r in summary if r['wape'] is not None]
        fig,ax=plt.subplots(figsize=(10,6))
        ax.barh([r['op'] for r in points],[100*r['wape'] for r in points],color=[{'CV':'#307760','Vector':'#336b99','Cube':'#a35624'}[r['method']] for r in points])
        for index,row in enumerate(points):
            ax.text(100*row['wape'],index,f"  {100*row['wape']:.1f}%",va='center',fontsize=8)
        ax.set_xlim(0,max(100*r['wape'] for r in points)*1.18)
        ax.set_xlabel('Independent validation WAPE (%)');ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        fig.tight_layout()
        fig.savefig(OUT/'validation.png',dpi=140)
        current_dir=ROOT/'results/current'
        (current_dir/'all-operators.png').write_bytes((OUT/'validation.png').read_bytes())
        svg_path=current_dir/'all-operators.svg'
        fig.savefig(svg_path,metadata={'Date':None})
        svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines())+'\n')
        plt.close(fig)
    plot()
    evaluation={'version':'v0.8','rows':rows,'operator_summary':summary,'collection_cost':cost,'query_scope':'one persistent Python process; parameter load on first route use included; no result cache; maximum wall milliseconds over five calls per shape'}
    payload=json.dumps(evaluation,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    (OUT/'evaluation.json').write_text(payload);(ROOT/'results/current/evaluation.json').write_text(payload)
    missing=[r['name'] for r in summary if r['independent']<10]
    message='、'.join(missing) if missing else '无'
    lines=['# 当前整套算子误差分析','',*table,'',f'每行对应默认模型的一种实现。独立列按不同 shape 计数，少于 10 个合格独立 shape 的路径为 {message}。开发样例用于建立或诊断模型。', '',
      'WAPE 是绝对偏差之和除以 CA 时间之和，最大误差是单例最大的绝对相对误差。表内只统计独立集。模型查询计时使用同一 Python 进程，每个 shape 调用统一入口 5 次，每次重新计算结果。最慢列取全部这些调用的最大墙钟毫秒数。硬件为 950DT 的 CA 模型，参考值为跨核指令窗口，多 kernel 串行求各窗口之和。', '',
      '![各算子独立验证误差](../results/v0.8/validation.png)','',

      '向量批采在一次 CA 初始化中依次同步多个 shape，可能保留前例热状态。MoE 组件和组合共享同一次链运行，mHC Pre 与拆分链也共享采集。完整数值检查、核数和数据源列于[逐例 JSON](../results/v0.8/evaluation.json)。', '',
      '## 逐 shape 结果','',*detail,'## 误差来源与采集成本','',
      'quant 的 UB 头尾开销、Matmul 的 tile 分支、FA 的 KV 流水均沿用冻结模型。FA 的 128k 证据仅适用于已测 Q9、D64、单 batch 单 head 路径，256k/512k 为外推。新增路由使用有限的开发 trace 估算阶段服务。任务数、算法分支和资源重叠变化都可能破坏模板假设。具体诊断如下。','',
      '- **RMSNorm 与 mHC Post。** RMSNorm 的 [65,128] 按模型分配给 64 核，实测仅启动 33 核。分配每核两行时，33 核即可覆盖 65 行。Post 的 [32,4,512] 模板同样预计 64 核，实测为 32 核。历史 Post 两个开发点 [1,4,128] 与 [32,4,128] 的入口编号分别为 5 和 6，已按 DB 修正旧版文字。错误的分支和分核会改变临界核的实际工作量。', '',
      '- **Addmm。** [512,512,256] 预测 6.402 µs，CA 为 4.479 µs。最后结束的核上，读、算、写的真实活动区间并集为 2.121 µs，模型给出 4.326 µs。预计的控制时间 2.076 µs 与实测未覆盖时间 2.107 µs 相近。该例的主因是首波内部的阶段重叠被低估。[临界核区间分解](../results/v0.8/cube/worst-addmm.json)', '',
      '- **FA。** WAPE 按参考时间加权，单个 128k 长循环可以显著影响总数。短 Q、不同 D 的单例相对误差另见上表。只有 Q9、B1、H1、D64 的相应长循环有长上下文 CA，不能把该 WAPE 套到任意 batch 或上下文。', '',
      '- **mHC Pre 与 Indexer。** Pre 的 D128、256、512、1024 样例中，同一入口上核内最多的 MMAD 均为 8 条。冻结模板按 N×D 放大工作，T4、D1024 预测 35.01 µs，实际 8.73 µs。指令数证据表明，直接按宽度放大旧的忙时会夸大成本。Indexer 的 L8192 有 64 条 MMAD，对应 8192/128 个段，预测 29.91 µs 与实测 22.71 µs 仍有差距。段数吻合只能验证工作划分，服务时间和重叠还需校准。', '',
      '- **融合与专家计算。** Indexer 模板按 KV 点积段数和 TopK 归并段数伸缩，mHC 混合链按每段库入口分别归属，MoE 需用本 rank 的真实专家 token 数。开发 trace 内同时工作的流水集合沿用到新 shape 是一项假设，算法切换及专家尾块会改变集合。逐例记录可用于核对入口、AIC/AIV 数和各段占用。', '',
      f'quant/FA、Cube/KDA 和融合采集分别累计 {core_cost/60:.2f}、{cube_cost/60:.2f}、{fusion_cost/60:.2f} worker 分钟。并行 worker 的时间可以重叠。Vector 批采累计 {vector_cost/60:.2f} worker 分钟，基础设施故障按已记录的 caller 时间计。','']
    lines.extend(['[Vector 采集成本](../results/v0.8/vector/costs.json) · [Cube 与 KDA 失败记录](../results/v0.8/cube/attempts.json) · [融合采集成本](../results/v0.8/fusion/costs.json) · [版本变化与重点偏差](../results/v0.8/changes.md)',''])
    report='\n'.join(lines)
    (ROOT/'docs/error-analysis.md').write_text(report)
    (OUT/'error-analysis.md').write_text(report.replace('../results/v0.8/','./'))
if __name__=='__main__':main()
