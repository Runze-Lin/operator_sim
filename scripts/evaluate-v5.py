#!/usr/bin/env python3
"""Replay frozen v0.5 predictions, report evidence, and draw static figures."""
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.model_v5 import Predictor,Case
from operator_model.model_v4 import Predictor as V4
from operator_model.evaluation_v5 import evaluate_record

OUT=ROOT/'results/v0.5'
def read(p):return json.loads(p.read_text())
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|',
        *['| '+' | '.join(map(str,r))+' |' for r in rows]])+'\n'
def metrics(rows,key):
    if not rows:return None
    es=[r[key]-r['reference_us'] for r in rows]
    return {'n':len(rows),'wape':sum(map(abs,es))/sum(r['reference_us'] for r in rows),
        'max_ape':max(abs(e)/r['reference_us'] for e,r in zip(es,rows)),
        'mae_us':statistics.mean(map(abs,es)),'bias_us':statistics.mean(es)}

def benchmark(new,old,cases):
    result={}
    # Models are loaded before timing. Alternate pairs; no CANN in query path.
    for cache in (False,True):
        times={k:[] for k in ('v05','v04')}
        for c in cases:new.predict(c);old.predict(c)
        for i in range(2000):
            c=cases[i%len(cases)]
            for key,p in ([('v05',new),('v04',old)] if i%2 else [('v04',old),('v05',new)]):
                t=time.perf_counter_ns();p.predict(c,use_cache=cache);times[key].append((time.perf_counter_ns()-t)/1000)
        result['cached' if cache else 'uncached_full']={k:{'median_us':statistics.median(v),'p95_us':sorted(v)[int(.95*(len(v)-1))]} for k,v in times.items()}
    return result

def main():
    frozen_path=OUT/'frozen-predictions.json';frozen=read(frozen_path)
    if sha256(OUT/'model.json')!=frozen['model_sha256']:raise ValueError('Changed model')
    for f,digest in frozen['artifact_sha256'].items():
        if sha256(ROOT/f)!=digest:raise ValueError('Changed frozen artifact '+f)
    new=Predictor(OUT/'model.json');old=V4(ROOT/'results/v0.4/model.json')
    records=[];rows=[];missing=[]
    for case in frozen['predictions']:
        path=ROOT/'runs/model-v5'/case['id']/'result.json'
        current=new.predict(Case('fa',case['shape']))
        if abs(current['predicted_us']-case['prediction']['predicted_us'])>1e-10:raise ValueError('Prediction replay drift')
        if not path.exists():missing.append(case['id']);continue
        r=read(path);row=evaluate_record(case,r,frozen,sha256(frozen_path));records.append(r)
        if row:rows.append(row)
    # The old 32k holdout is explicitly reclassified as development this iteration.
    trace=read(OUT/'trace-analysis.json');calcase=Case('fa',trace['calibration_shape'])
    calibration={'shape':list(calcase.shape),'reference_us':trace['reference_us'],
        'v05_us':new.predict_us(calcase),'v04_us':old.predict_us(calcase),
        'constant_us':old.predict(calcase)['constant_baseline_us'],'role':'development_from_previous_holdout'}
    history=[]
    previous=read(ROOT/'results/long-fa/evaluation.json')
    for r in previous['historical']:
        c=Case(r['op'],r['shape']);p=new.predict(c)
        history.append({'id':r['id'],'op':r['op'],'shape':r['shape'],'role':r['role'],
            'reference_us':r['reference_us'],'v05_us':p['predicted_us'],'v04_us':r['predicted_us'],
            'constant_us':r['constant_us'],'ape':abs(p['predicted_us']-r['reference_us'])/r['reference_us'],
            'kernel':r['measurement']['kernel'],'kernel_binary_sha256':r['measurement'].get('binary_sha256'),
            'trace_sha256':r['measurement'].get('trace_sha256'),'api':r['api']})
    assert len({r['id'] for r in history})==len(history)==45
    assert all(abs(r['v05_us']-r['v04_us'])<1e-12 for r in history)
    extrap=[]
    for length in (16385,32768,131072,132096,262144,263168,524288,525312):
        case=Case('fa',(9,length,64,1,1));m=next((r for r in rows if r['shape']==list(case.shape)),None)
        extrap.append({'shape':list(case.shape),'v05_us':new.predict_us(case),'v04_us':old.predict_us(case),
            'reference_us':m['reference_us'] if m else trace['reference_us'] if length==32768 else None,
            'role':'independent_measured' if m else 'development' if length==32768 else 'unvalidated_extrapolation'})
    data={'model_sha256':frozen['model_sha256'],'freeze_sha256':sha256(frozen_path),'rows':rows,'missing':missing,
        'failed':[r['case']['id'] for r in records if r['status']!='ok'],
        'calibration':calibration,'historical':history,'extrapolation':extrap,
        'metrics':{key:metrics(rows,key) for key in ('v05_us','v04_us','constant_us')},
        'cost':{'attempts':len(records),'new_calibration_ca':0,'wall_seconds':sum(r['wall_seconds'] for r in records),
            'cpu_seconds':sum(r['cpu_seconds'] for r in records),'cpu_is_lower_bound':any('lower_bound' in r['cpu_accounting'] for r in records),
            'raw_bytes':sum(r['raw_bytes'] for r in records),'extraction_seconds':sum(r['extraction_seconds'] for r in records)},
        'extension_invocations':read(ROOT/'docs/extensions/smoke-950dt.json'),
        'benchmark':benchmark(new,old,[Case('fa',c['shape']) for c in frozen['predictions']])}
    write('dataset.json',{'records':records});write('evaluation.json',data)
    report(data,records,trace);plot(data,trace)
    print(json.dumps({'metrics':data['metrics'],'cost':data['cost'],'missing':missing},indent=2))

def report(d,records,trace):
    rows=d['rows'];cost=d['cost'];cal=d['calibration'];m=d['metrics']
    summary=[]
    for r in rows:
        summary.append(f"- **{r['id']} {r['shape']}**：CA **{r['reference_us']:.6f} µs**；v0.5 **{r['v05_us']:.6f} µs / APE {100*r['v05_ape']:.2f}%**，旧v0.4 **{r['v04_us']:.6f} µs / {100*r['v04_ape']:.2f}%**，固定耗时 **{r['constant_us']:.6f} µs / {100*r['constant_ape']:.2f}%**。")
    if rows:summary.append(f"- 同一新留出集WAPE：v0.5 **{100*m['v05_us']['wape']:.2f}%**，v0.4 **{100*m['v04_us']['wape']:.2f}%**，固定耗时 **{100*m['constant_us']['wape']:.2f}%**。样本仅{len(rows)}个、每个采集一次，没有统计置信区间。")
    summary += [f"- 32k旧数据用于本轮校准：预测由111.985723变为76.183878µs，相对75.98µs的误差由47.39%降至0.27%；**不是本轮独立验证**。",
        f"- 新增FA CA {cost['attempts']}次，新增校准CA 0次；FA采集墙钟 **{cost['wall_seconds']/60:.2f}分钟**，记录CPU **{cost['cpu_seconds']/3600:.2f}小时**，保留数据 **{cost['raw_bytes']/2**20:.2f}MiB**。",
        f"- 另有RMSNorm/Addmm两个极小调用验证，共{d['extension_invocations']['cost']['wall_seconds']:.2f}秒；全部4次CA墙钟合计 **{(cost['wall_seconds']+d['extension_invocations']['cost']['wall_seconds'])/60:.2f}分钟**。两点均通过完整输出检查，但未用于时延拟合。",
        '- 修正仅覆盖Q9/D64/B1/H1普通非因果FA长KV路径；quant、matmul及其他FA保留v0.4。256k/512k仍是未实测外推。',
        '- 新增算子族已分离结构支持与时延校准状态：RMSNorm、Addmm、LightningIndexer、mHC、KDA、MoE融合链的泛化预测耗时均为null，不混入本页FA准确率。']
    if d['missing'] or d['failed']:summary.append(f"- 未完成/失败：{d['missing']+d['failed']}；不计作实测误差。")
    text=['# v0.5 误差分析\n','## Summary / 本轮结论\n','\n'.join(summary)+'\n',
        '配套 [建模策略、公式与方法解释](modeling-strategy.md)。本页比较始终使用同一组成功新留出点；不能把32k校准回放与128k独立验证混在一个准确率里。\n',
        '![新留出误差与成本](holdout-comparison.png)\n',
        '## 1. 实际库算子和身份\n',
        '实际运行CANN9.2.1的950 **ops-transformer库**，接口为 `aclnnFlashAttentionScoreV2`。输入Q=[Q,B,H×D]、K/V=[KV,B,H×D]，BF16/SBH、无mask/dropout、非因果、scale=1/sqrt(D)。它是普通FA代理，不是KDA、MLA或decode专用FIA。quant和matmul由950 ops-nn提供，分别调用aclnnDynamicMxQuant/aclnnMatmul。\n',
        'FA安装对象：`opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/flash_attention_score_apt/FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5.o`，SHA256 `ac006aee669c8696aeab3bb3fe64fc29e5f3c8e2d172bf75f46db7176a2d4580`，与旧msopprof实际导出对象匹配。本轮compact未再次导出ELF，只记录实际DB entry；安装对象SHA不能替代一次新导出的运行时SHA。\n',
        '目标SoC固定Ascend950DT_9582，host二进制SHA `2cf2c77e48233383ef4619ad61f136cc51a1462ae0d5ea63f226e64b4d2bd0e8`。每点的DB SHA、SKU适配器SHA、完整命令和model/freeze SHA见 [dataset.json](dataset.json)。文档中的源码仓是参考实现，不宣称与安装包逐字同源。\n',
        '此前“桥接”仅指**同旧shape的新旧采集方式一致性校验**，不是额外算子。本轮没有重复该CA；沿用已检验的compact DB路径。\n',
        '## 2. 独立留出逐点结果\n',
        table(['ID / shape','CA µs','v0.5 µs / APE','v0.4 µs / APE','固定 µs / APE','偏差 µs'],[
            [r['id']+' '+str(r['shape']),f"{r['reference_us']:.6f}",f"{r['v05_us']:.6f} / {100*r['v05_ape']:.2f}%",f"{r['v04_us']:.6f} / {100*r['v04_ape']:.2f}%",f"{r['constant_us']:.6f} / {100*r['constant_ape']:.2f}%",f"{r['signed_error_us']:+.6f}"] for r in rows]),
        table(['ID','实际kernel entry','预测/实际 AIC,AIV','预测/实际 MMAD','数值检查'],[
            [r['id'],r['kernel'],str(r['predicted_cores'])+' / '+str(r['observed_cores']),str(r['predicted_mmad'])+' / '+str(r['observed_mmad']),'; '.join(r['correctness'])] for r in rows]),
        '正确性是确定性输入的全部attention输出对CPU double golden，atol=0.002、rtol=0.01；并非随机全域数值测试。计时使用完整DB所有执行核的共同指令窗口，不用部分trace、最大单核span或主机墙钟替代。entry与tiling key相同也不能保证所有循环内部细节完全相同。\n',
        '## 3. 校准机制、残差与固定耗时对照\n',
        f"32k开发样本：CA {cal['reference_us']:.6f}、v0.5 {cal['v05_us']:.6f}、v0.4 {cal['v04_us']:.6f}、固定{cal['constant_us']:.6f}µs。数据角色已从上一轮留出改为本轮开发；中段112个tile也只是同一次trace内部观测，不是112个独立样本。\n",
        '**已证实的修正依据**：Q读一次，K/V分别128次；已知QK/PV维度对应的PC不同，可以按流还原时间，不能把混排MMAD每4条直接等同源码循环。单次K/V读取延迟约0.85/0.75µs，稳态同流发起间隔约0.517µs，旧模型加权每轮增量约0.811µs。新模型只把长延迟用于首轮fill，后续按有效II推进。\n',
        '**统计边界**：J来自启动时间对tile索引的回归，包含标量、同步、buffer和下游反压；它不是单独HBM吞吐。F是首轮读完成后的剩余路径，D是最后PV后的排空。来自同一32k样本的参数不能唯一分离所有硬件效应；即使留出误差很小，也不证明可跨Q、D、B/H或代际复用。\n',
        '**新shape误差应怎样解释**：128k检验4倍长循环累积误差与是否换kernel；16k+1检验不同循环长度及尾块处理。模型为尾块计完整256 tile，但MMAD按128分段，可能产生小额保守偏差。若entry、核数或MMAD变化，应先判定路由不匹配；若它们吻合而耗时有偏差，则有效II、首尾启动和运行间变化是待分离因素，不能无证据全归因于访存或噪声。\n',
        '**固定耗时为何失去优势**：小shape启动主导、总工作增大常被分核吸收，均值可比复杂拟合更稳定；这里B=H=1，KV增长增加的是单核循环，常数9.538626µs无法随循环增长。主对照常数冻结自旧数据，本轮没有重估均值。\n',
        '![旧32k数据库中的读流与新旧模型](read-streams.png)\n',
        '图中K/V开始时刻和首轮读窗口是旧32k数据库实测；两种II的线用于解释校准，不是新留出结果。K/V中段分块回归斜率：'+str(trace['block_slopes_us'])+' µs/tile。\n',
        '## 4. 旧算子与旧shape的回归检查\n',
        '45个去重历史记录的预测与v0.4完全相同，未重跑CA。下表包含38个旧开发样本和7个v0.4历史留出；这些数据不混入本轮两个新FA的WAPE。实际kernel、trace SHA、接口见evaluation.json，完整历史流水证据可复用上一轮报告。\n',
        table(['ID','算子 / shape','数据角色','CA µs','v0.5=v0.4 µs','APE'],[
            [r['id'],r['op']+' '+str(r['shape']),r['role'],f"{r['reference_us']:.6f}",f"{r['v05_us']:.6f}",f"{100*r['ape']:.2f}%"] for r in d['historical']]),
        '## 5. 长度外推、场景覆盖和未验证范围\n',
        table(['Q,KV,D,B,H','v0.5 µs','v0.4 µs','CA µs','证据角色'],[
            [r['shape'],f"{r['v05_us']:.6f}",f"{r['v04_us']:.6f}",f"{r['reference_us']:.6f}" if r['reference_us'] is not None else '—',r['role']] for r in d['extrapolation']]),
        '132096/263168/525312表示128k/256k/512k输入加1k输出接近结束的上下文端点。没有完整实测的行明确为外推。B1/H1/D64普通非因果FA不能代表用户的全部单卡batch、TP、prefill chunk或实际模型专用注意力。A7/A8无新证据。\n',
        '新增算子族：见 [结构模型、准确API及校准状态](../../docs/extensions/README.md)。本轮FA的低误差不能转移给这些算子；KDA校验Q=9、mHC融合形态和MoE后半段GMM须单独检查。\n',
        table(['新算子调用 / 参数','实际entry','CA µs','AIC/AIV / MMAD','正确性'],[
            [r['arguments'],r['measurement']['kernel'],f"{r['measurement']['reference_us']:.6f}",
             f"{r['measurement']['active_aic']}/{r['measurement']['active_aiv']} / {r['measurement']['max_core_mmad']}",'; '.join(r['correctness'])]
            for r in d['extension_invocations']['records']]),
        '以上分别是ops-nn的aclnnRmsNorm和aclnnAddmm（BF16、alpha=beta=1、self bias）。两个调用各仅观察到一个kernel，耗时是完整指令窗口；没有泛化预测值，因此不计算APE。详细调用命令、DB SHA、正确性阈值及成本见 [调用证据](../../docs/extensions/smoke-950dt.json)。\n',
        '## 6. 采集和在线预测成本\n',
        table(['ID','状态','墙钟s','CPU秒','保留MiB','SQL归约s'],[
            [r['case']['id'],r['status'],f"{r['wall_seconds']:.2f}",f"{r['cpu_seconds']:.2f}",f"{r['raw_bytes']/2**20:.2f}",f"{r['extraction_seconds']:.3f}"] for r in records]),
        'FA预算为最多两个新shape、总墙钟20分钟；128k最多960秒、尾块最多240秒，CPU/保留空间在case边界检查，另有临时产物2GiB停止门槛。新增算子调用验证另设每点120秒，RMSNorm/Addmm分别消耗40.74/38.80秒、CPU409.68/369.77秒、保留460330/386345字节。没有自动重试或新校准CA。保留字节不是峰值临时空间；CPU若进程超时为下界，当前CPU下界标记='+str(cost['cpu_is_lower_bound'])+'。\n',
        table(['查询口径','v0.5中位/P95 µs','v0.4中位/P95 µs'],[
            [k,f"{v['v05']['median_us']:.3f} / {v['v05']['p95_us']:.3f}",f"{v['v04']['median_us']:.3f} / {v['v04']['p95_us']:.3f}"] for k,v in d['benchmark'].items()]),
        '基准在模型已加载后，两个冻结shape、完整predict API、交替顺序各2000次；与CA采集成本分开。新长FA公式对KV长度为O(1)，缓存容量4096；这不是模型系统端到端性能测量。\n',
        '## 7. 版本与复现\n',
        '模型与采样冻结提交 `24d0590`；所有拟合/预测/调用器/适配器SHA在frozen-predictions.json。评估拒绝shape、SoC、host、适配器、模型或冻结文件身份变化，并要求采集开始晚于冻结。失败、超时和缺完整数值检查不会生成误差值。\n',
        '`python3 scripts/evaluate-v5.py` 可从本地采集摘要重建报告，不运行CA；生成图片需matplotlib。已提交dataset/evaluation可供脱离SDK审阅。历史模型/报告不覆盖；本轮实际库路径与新证据审计见evidence.json。\n']
    (OUT/'error-analysis.md').write_text('\n'.join(text))

def plot(d,trace):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'build/matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    rows=d['rows'];fig,ax=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for i,(key,label,color) in enumerate([('v05_us','v0.5','#338b73'),('v04_us','v0.4','#d8793f'),('constant_us','Fixed mean','#929292')]):
        ax[0].bar([j+(i-1)*.24 for j in range(len(rows))],[100*abs(r[key]-r['reference_us'])/r['reference_us'] for r in rows],width=.24,label=label,color=color)
    ax[0].set_xticks(range(len(rows)),[str(r['shape'][1])+' KV' for r in rows]);ax[0].set(ylabel='Absolute percentage error (%)',title='New frozen holdouts only');ax[0].legend()
    x=[r['shape'][1]/1024 for r in sorted(d['extrapolation'],key=lambda r:r['shape'][1])]
    ex=sorted(d['extrapolation'],key=lambda r:r['shape'][1])
    ax[1].plot(x,[r['v05_us'] for r in ex],'--',color='#338b73',label='v0.5 formula / extrapolation')
    ax[1].plot(x,[r['v04_us'] for r in ex],':',color='#d8793f',label='Frozen v0.4')
    ax[1].scatter([r['shape'][1]/1024 for r in rows],[r['reference_us'] for r in rows],c='black',label='Independent complete CA')
    ax[1].scatter([32],[75.98],marker='s',facecolors='none',edgecolors='#338b73',label='32k development CA')
    ax[1].set(xscale='log',yscale='log',xlabel='KV length (Ki tokens)',ylabel='Instruction makespan (us)',title='Q9 D64 B1 H1; beyond measured points unvalidated');ax[1].legend(fontsize=7)
    ax[1].set_xticks([16,32,128,256,512],['16','32','128','256','512'])
    ax[1].xaxis.set_minor_formatter(NullFormatter())
    for a in ax:a.grid(axis='y',alpha=.2)
    for ext in ('png','svg'):fig.savefig(OUT/f'holdout-comparison.{ext}',dpi=170)
    plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name,color in [('k','#317eb3'),('v','#338b73')]:
        stream=trace['read_streams'][name]
        ax[0].plot(range(128),[r['start_us'] for r in stream],label=name.upper()+' read starts',color=color)
        for i,r in enumerate(stream[:8]):ax[1].plot([r['start_us'],r['end_us']],[i+(0 if name=='k' else .3)]*2,lw=4,color=color)
    c=trace['costs'];ax[0].plot(range(128),[c['prefix_us']+i*.8108798059932323 for i in range(128)],'--',color='#d8793f',label='Old weighted per-tile increment (illustrative)')
    ax[0].set(xlabel='KV tile index',ylabel='Read start (us)',title='Existing 32k DB: separate stream initiation intervals');ax[0].legend(fontsize=7)
    ax[1].set(xlabel='Time from global instruction start (us)',ylabel='Tile (V offset by 0.3)',title='First 8 K/V reads: latency windows overlap')
    for a in ax:a.grid(alpha=.2)
    for ext in ('png','svg'):fig.savefig(OUT/f'read-streams.{ext}',dpi=170)
    plt.close(fig)
    for p in OUT.glob('*.svg'):p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')

if __name__=='__main__':main()
