#!/usr/bin/env python3
"""Rebuild lightweight report artifacts from immutable records, never fit or run CA."""
import json
import os
from pathlib import Path
import re
import sqlite3
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.model_v4 import Case,Predictor
from operator_model.npusim import extract_compact

OUT=ROOT/'results/long-fa'
API={'quant':'aclnnDynamicMxQuant','matmul':'aclnnMatmul','fa':'aclnnFlashAttentionScoreV2'}

def read(path):return json.loads(path.read_text())
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|',
        *['| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in rows]])+'\n'

def metrics(rows):
    if not rows:return None
    errors=[r['predicted_us']-r['reference_us'] for r in rows]
    return {'count':len(rows),'wape':sum(map(abs,errors))/sum(r['reference_us'] for r in rows),
        'max_ape':max(abs(e)/r['reference_us'] for e,r in zip(errors,rows)),
        'signed_bias_us':sum(errors)/len(errors),
        'constant_wape':sum(abs(r['constant_us']-r['reference_us']) for r in rows)/sum(r['reference_us'] for r in rows)}

def compare(predictor,op,shape,m):
    p=predictor.predict(Case(op,shape),use_cache=False)
    if 'max_core_mmad' in m:opcode=m['max_core_mmad']
    else:opcode=max(sum(v for k,v in c['dynamic_opcodes'].items() if k.startswith('RV_')) if op=='quant' else c['dynamic_opcodes'].get('MMAD',0) for c in m['cores'])
    # Reuse complete historical JSON by hash; do not duplicate millions of opcode entries.
    compact={**m,'cores':[{k:v for k,v in c.items() if k not in ('dynamic_opcodes','opcode_duration_us','events')} for c in m['cores']]}
    return {'op':op,'shape':shape,'api':API[op],'reference_us':m['reference_us'],
        'predicted_us':p['predicted_us'],'constant_us':p['constant_baseline_us'],
        'signed_error_us':p['predicted_us']-m['reference_us'],
        'ape':abs(p['predicted_us']-m['reference_us'])/m['reference_us'],
        'plan':p['plan'],'predicted_cores':[p['plan']['active_aic'],p['plan']['active_aiv']],
        'observed_cores':[m['active_aic'],m['active_aiv']],
        'predicted_opcode_count':p['estimated_max_core_selected_opcode_count'],'observed_opcode_count':opcode,
        'measurement':compact}

def steady_diagnostic(rows,predictor):
    target=next((r for r in rows if r['id']=='kv32k-compact-complete'),None)
    if target is None:return None
    p=Path(target['measurement']['db_path'])
    with sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True) as c:
        ticks=[r[0] for r in c.execute("SELECT ExecInstrTickStart FROM ExecutedInstructions WHERE CoreTypeId=1 AND ExecInstrName='MMAD' ORDER BY ExecInstrTickStart")]
        reads=list(c.execute("SELECT ExecInstrTickStart,ExecInstrTickEnd,json_extract(ExtendParams,'$.processed_bytes') FROM ExecutedInstructions WHERE ExecInstrName='MOV_OUT_TO_L1_MULTI_ND2NZ' ORDER BY ExecInstrTickStart"))
    assert len(ticks)==512
    intervals=[(ticks[i+4]-ticks[i])/1650 for i in range(0,len(ticks)-4,4)]
    p=predictor.predict(Case('fa',target['shape']),use_cache=False)
    s=p['stage_services_us'];ii=max(s['read'],s['cube']+s['fix'],s['vector']+s['cube']/2)
    return {'case':target['id'],'db_sha256':target['measurement']['db_sha256'],
        'method':'Consecutive groups of four chronological AIC MMAD starts; proxy for repeated work, not proof of per-source-loop dependency',
        'four_mmad_start_intervals_us':intervals,'median_interval_us':statistics.median(intervals),
        'quarter_mean_us':[statistics.mean(intervals[i:i+32]) for i in range(0,len(intervals),32)],
        'model_stage_services_us':s,'model_uncorrected_ii_us':ii,'model_weighted_ii_us':ii*0.709561101627441,
        'read_payload_counts':{str(size):sum(b==size for a,z,b in reads) for size in sorted({r[2] for r in reads})},
        'median_32kib_dma_span_us':statistics.median((z-a)/1650 for a,z,b in reads if b==32768),
        'limits':'Observed interval distribution is development diagnosis of this holdout, not an independently validated replacement predictor.'}

def main():
    OUT.mkdir(exist_ok=True)
    model=ROOT/'results/v0.4/model.json';predictor=Predictor(model)
    assert sha256(model)=='1b7a116c234d07fed008634e7874b41f73e930426a42833944769367f03c02d0'
    old=[]
    for filename,role in [('imported-calibration.json','reused_development'),('dataset.json','reused_v04_holdout')]:
        for r in read(ROOT/'results/v0.4'/filename)['records']:
            if r['status']!='ok':continue
            if role=='reused_v04_holdout' and r['case']['split']!='holdout':continue
            row=compare(predictor,r['case']['op'],r['case']['shape'],r['measurement'])
            row.update(id=r['case']['id'],role=role,identity=r['identity'],cache_key=r['cache_key'])
            old.append(row)
    assert len({r['cache_key'] for r in old})==len(old), 'Do not count imported calibration twice'
    attempts=[];fresh=[]
    for f in sorted((ROOT/'runs/long-fa').glob('*/result.json')):
        raw=read(f);folder=f.parent;name=folder.name
        # Initial two probes predate the backend field; their recorded command is msopprof.
        if 'backend' not in raw:
            assert 'msopprof' in raw['command']
            raw={**raw,'backend':'default'}
        a={**raw,'name':name,'result_path':str(f.relative_to(ROOT)),'result_sha256':sha256(f)}
        a['cpu_accounting']=raw.get('cpu_accounting','lower_bound_unreaped_descendants' if raw['backend']=='npusim' and raw['outer_timeout'] else 'reaped_child_processes')
        m=None;start=time.perf_counter()
        if raw['status']=='ok':m=raw['measurement']
        elif raw['status']=='correctness_only':
            try:
                if raw['backend']=='npusim':m=extract_compact(folder/'profile',folder/'run.log')
                elif raw.get('native_helper'):m=extract_compact(folder,folder/'run.log',native_returncode=raw['returncode'])
            except (ValueError,OSError) as exc:a['extraction_error']=str(exc)
        a['extraction_wall_seconds']=time.perf_counter()-start
        a['accepted_timing']=m is not None
        if m:
            row=compare(predictor,'fa',raw['shape'],m)
            row.update(id=name,role='independent_long_holdout' if raw['shape'][1]>=32768 else 'repeated_bridge',
                       model_sha256=raw['model_sha256'],host_binary_sha256=raw['binary_sha256'],
                       wall_seconds=raw['wall_seconds'],cpu_seconds=raw['cpu_seconds'],raw_bytes=raw['raw_bytes'])
            fresh.append(row)
        attempts.append(a)
    extrap=[]
    for kv in (32768,33792,131072,132096,262144,263168,524288,525312):
        shape=[9,kv,64,1,1];p=predictor.predict(Case('fa',shape),use_cache=False)
        measured=[r for r in fresh if r['shape']==shape]
        extrap.append({'shape':shape,'predicted_us':p['predicted_us'],
            'measurement_status':'complete' if measured else 'not_completely_measured',
            'reference_us':measured[-1]['reference_us'] if measured else None,
            'max_core_mmad_prediction':p['estimated_max_core_selected_opcode_count'],
            'scope':'ordinary noncausal FA B1/H1/D64 proxy; no network equivalence'})
    diagnosis=steady_diagnostic(fresh,predictor)
    aggregate={'model_sha256':sha256(model),'historical_source_sha256':{n:sha256(ROOT/'results/v0.4'/n) for n in ('dataset.json','imported-calibration.json')},'historical':old,'new_capture':fresh,'long_context':extrap,
        'steady_diagnosis':diagnosis,
        'metrics':{'reused_v04_holdout':{op:metrics([r for r in old if r['role']=='reused_v04_holdout' and r['op']==op]) for op in API},
            'independent_long':metrics([r for r in fresh if r['role']=='independent_long_holdout'])},
        'cost':{'attempts':len(attempts),'wall_seconds':sum(a['wall_seconds'] for a in attempts),
            'recorded_cpu_seconds_lower_bound':sum(a['cpu_seconds'] for a in attempts),'retained_raw_bytes':sum(a['raw_bytes'] for a in attempts)}}
    write('evaluation.json',aggregate);write('capture-attempts.json',attempts)
    # Compact kernel identities; all per-case hashes remain in evaluation.json.
    identities=sorted({(r['op'],r['measurement']['kernel'],r['measurement'].get('binary_sha256') or r['measurement'].get('kernel_binary_sha256') or '未导出') for r in old+fresh})
    ids={key:f'K{i+1:02}' for i,key in enumerate(identities)}
    def kid(r):
        m=r['measurement'];return ids[(r['op'],m['kernel'],m.get('binary_sha256') or m.get('kernel_binary_sha256') or '未导出')]
    def rows(rs):
        return [[r['id'],r['op'],str(r['shape']),kid(r),f"{r['reference_us']:.6f}",f"{r['predicted_us']:.6f}",
                 f"{100*r['ape']:.2f}%",f"{r['signed_error_us']:+.6f}",
                 '/'.join(map(str,r['observed_cores'])),
                 f"{r['predicted_opcode_count']:.2f}/{r['observed_opcode_count']}" if r['predicted_opcode_count'] is not None else f"未预测/{r['observed_opcode_count']}"] for r in rs]
    headers=['ID','算子','shape','kernel','CA µs','预测 µs','APE','偏差 µs','AIC/AIV 实测','指令 预测/实测']
    lines=['# 当前建模策略的误差分析：长 FA 采集迭代\n',
        '配套 [建模策略与完整公式](modeling-strategy.md)。本轮冻结 v0.4 预测，不根据新测量回调参数。所有时间误差以完整指令 makespan 为准；本报告由 `scripts/report-long-fa.py` 重建。\n',
        '## 1. 数据角色和当前结论\n',
        f"复用历史成功记录 **{len(old)}** 条，其中38条为本版开发数据、7条为v0.4历史留出；本轮有 **{len(fresh)}** 条可接受计时采集，其中旧 shape 重复桥接 {sum(r['role']=='repeated_bridge' for r in fresh)} 次、新长 shape 完整实测 {sum(r['role']=='independent_long_holdout' for r in fresh)} 次。重复桥接不是新留出。\n"]
    longrows=[r for r in fresh if r['role']=='independent_long_holdout']
    if longrows:
        for r in longrows:lines.append(f"- **{r['id']}**：CA {r['reference_us']:.6f} µs，冻结模型 {r['predicted_us']:.6f} µs，APE **{100*r['ape']:.2f}%**；正确性通过，核数预测/实测 {r['predicted_cores']}/{r['observed_cores']}，MMAD {r['predicted_opcode_count']}/{r['observed_opcode_count']}。\n")
    else:lines.append('尚无32k及以上完整 CA 实测，不给出该范围准确率。超时数据只用于诊断采集成本。\n')
    lines+=['\n![长上下文与采集成本](long-fa-validation.png)\n',
        '图中圆点为完整实测，虚线为冻结公式外推；未实测区不是CA曲线。成本比较是同一旧 shape 的单次探索结果，没有重复统计显著性。\n',
        '## 2. 实际测的是什么库算子\n',
        mdtable(['模型名','ACLNN API','输入/属性'],[
            ['quant',API['quant'],'[R,C]；BF16→E4M3FN，尾轴block32，E8M0 scale，rint'],
            ['matmul',API['matmul'],'[M,K]×[K,N]→[M,N]，BF16 ND，cubeMathType=0'],
            ['FA',API['fa'],'Q=[Q,B,H×D]，K/V=[KV,B,H×D]，BF16 SBH，非因果、无mask/dropout，scale=1/sqrt(D)']]),
        '调用器为 `smoke/three_ops.cpp`，本轮没有改源码或二进制。FA验证全部输出，CPU double golden，atol=0.002、rtol=0.01；matmul atol=0.01、rtol=0.005；quant输出和scale逐字节精确比较。输入是确定性模式，不能视为全面随机数值测试。\n',
        'CANN 9.2.1；SoC `Ascend950DT_9582`；本轮 host SHA256 `2cf2c77e48233383ef4619ad61f136cc51a1462ae0d5ea63f226e64b4d2bd0e8`。每个历史记录的 package/platform/config/host SHA 保留在 evaluation.json。源码参考为 ops-nn `602e4ad985da00aad28c16432841e9c5069e10de`、ops-transformer `14d022b1a4c3df263bf697534e3c25c7c992ff5d`；不能断言这些源码与安装包二进制逐字对应。\n',
        '缓存口径补充：旧记录中的cache_state提到`lib/config.json`里L2 cache_enable=0，只能证明该文件的值。本轮发现实际执行是camodel全SoC后端，尚未证明它采用该核wrapper配置，故不据此声称所有CA采集关闭了L2；历史字段保留，不把这个未确认假设用于误差归因。\n',
        mdtable(['索引','算子','实际 kernel/entry','kernel 二进制 SHA256'],[[ids[k],*k] for k in identities]),
        'helper DB 的 `entry_kernel_2050000040140220` 与旧 FlashAttentionScore tiling key `2328361008425599520` 对应，但该采集未导出 kernel ELF；“未导出”不补成旧 ELF 哈希，也不把 tiling key 当完整二进制身份。\n',
        '## 3. 本轮完整测量：桥接与新留出分开\n',mdtable(headers,rows(fresh)),
        '新鲜进程、单次采集；本轮没有建立置信区间。core-lib 名称表示一次尝试，msopprof实际仍选择camodel，不能将它作为另一个后端的结果。threads8仅为拒绝采用的调度探索，不并入正式校准。\n',
        '## 4. 当前策略在复用数据上的逐点误差\n',
        '### v0.4 历史独立留出（本轮未重跑）\n',mdtable(headers,rows([r for r in old if r['role']=='reused_v04_holdout'])),
        'quant 6点 WAPE=5.54%，v0.3冻结公式=6.46%，固定均值=17.91%；FA 1点APE=0.79%。这些是上一轮已冻结结果，本轮不能宣传为新增精度改善。\n',
        '### 开发/校准复用集（不是本轮独立准确率）\n',mdtable(headers,rows([r for r in old if r['role']=='reused_development'])),
        '每条记录的预测计划、实际核数、指令计数、原始trace/DB哈希、数值检查都保存在 [evaluation.json](evaluation.json)。matmul本轮没有新增CA，表中如出现偏差不能归因于本轮采集优化。\n',
        '## 5. 误差来源：证据与待验证假设\n',
        '**quant 已证实的旧问题**：v0.2把含参数/排队的MTE2聚合窗当每轮服务，重复计算后续预取；VF命令窗也不等于Vector实际忙碌区间。v0.3按UB槽依赖修正；v0.4对小shape控制路径饱和。本轮复用结果仍可见单点约12.8%误差，未因改采集方式消失。\n',
        '**quant 尚未证实的残差归因**：控制前缀、核间起始偏差与写回排队在小shape占比较高，单次采集波动可能影响数个百分点，但没有逐因素实验或重复统计，不能全部称为随机噪声。三点多UB校准仍不足以识别所有尾块、槽竞争与更长循环。\n',
        '**matmul 边界**：输出tile重平衡是有限启发式，gemv/K-tail/粗tile各只少数旧样本；核数吻合不证明内部tile、访存与流水完全一致。参数中较大的fix截距可能吸收其他固定开销，不能解释为真实硬件FIXP固有延迟。\n',
        '**FA 已证实的指令问题**：外层KV256仍有QK/PV内部128分段，旧的“2×外层次数”会少算；当前Q64结构公式已在旧trace检验。长shape的实际次数与核数见上表。指令次数正确只说明工作量分解通过，不保证每轮关键路径正确。\n',
        '**FA 时延模型的局限**：现有 II=max(R,C+F,V+C/2) 来自小KV的聚合服务拟合，省略了每轮标量、同步、softmax递推和存储竞争的详细依赖。单凭总时间不能认定“Vector bound”或将差额全归到L2/HBM。\n',
        (f"**32k具体诊断（复用本次DB，无新增CA）**：按时间排序每4条MMAD取一次起点，共127个间隔，中位数 **{diagnosis['median_interval_us']:.6f} µs**，四段均值分别为 {', '.join(f'{x:.6f}' for x in diagnosis['quarter_mean_us'])} µs；旧模型加权后每KV tile增量 **{diagnosis['model_weighted_ii_us']:.6f} µs**。实测读payload为1次1152B的Q和256次32768B的K/V搬运；单次32KiB DMA窗口中位数 {diagnosis['median_32kib_dma_span_us']:.6f} µs。AIC MTE2非排除区间并集67.8515 µs，Cube27.6394 µs，最大AIV SIMD28.2861 µs，存在明显重叠。**证实了总工作次数正确、旧模型稳态增量偏大；尚未证明每个MMAD分组等于源码完整循环，或唯一根因是哪一条依赖。** 下一版应分离首轮读延迟、读发起间隔和稳态递推，不能直接将本点调准后称为独立验证。\n\n![32k稳态诊断](fa-steady-diagnostic.png)\n" if diagnosis else ''),
        '**为什么固定耗时曾更好**：旧小shape由启动/控制主导，增加总工作常被分核吸收，时延区间很窄，多参数拟合容易放大噪声；常数方差小。长KV在B=H=1下没有增加活跃核，核内循环增加，常数失去结构解释。固定值仍以旧数据均值比较，不用新留出估计均值。\n',
        '**采集与计时口径**：npusim compact 路径仍采完整指令，仅省去JSON生成和大量文本格式化。SKU适配器仅在soc_version=Ascend950DT且显式目标DT_9582时生效；无适配器的PR_9589结果全部拒绝。原生summary启动/结束窗口与helper指令窗不同，尚不作为替代参考。\n',
        '## 6. 长上下文：已实测与未实测外推\n',
        mdtable(['shape [Q,KV,D,B,H]','预测 µs','完整CA µs','状态','最大核MMAD预测'],[
            [r['shape'],f"{r['predicted_us']:.6f}",f"{r['reference_us']:.6f}" if r['reference_us'] is not None else '—',
             '完整实测' if r['reference_us'] is not None else '未完成实测；仅公式外推',r['max_core_mmad_prediction']] for r in extrap]),
        'Q=9代表草拟8个token后的主模型校验。32/128/256/512k按1024倍数解释，L+1024表示输出接近1k的上下文端点。该表仅B1/H1/D64普通FA代理；用户的其他batch、prefill chunk、TP与MLA/KDA/压缩注意力未因该表而获得验证。核函数在更长shape上是否换路由尚需独立证据。\n',
        '本轮32k成功后，128k–512k完整CA尚未执行：按32k 222.3秒、约20–60秒固定部分粗估，128k单点约12–14分钟，512k约44–54分钟；这是预算外推，不是仿真时间实测。本轮已完成14次采集探索，停止增加高成本长点。鉴于32k已暴露47.4%偏差，表中更长KV公式仅保留可审计预测，不满足准确性验收。下一轮先用已有DB修正服务与发起间隔，再决定是否采用窗口采集或一次128k完整留出；512k目前仍未完成验证。\n',
        '## 7. 所有尝试与采集开销\n',
        mdtable(['尝试','Q/KV','路径','状态','墙钟 s','记录CPU s','保留 MiB'],[
            [a['name'],f"{a['shape'][0]}/{a['shape'][1]}",a['backend'],
             '完整计时' if a['accepted_timing'] else ('超时' if a['outer_timeout'] or a['profiler_timeout'] else a['status']),
             f"{a['wall_seconds']:.2f}",f"{a['cpu_seconds']:.2f}"+(' 下界' if 'lower_bound' in a['cpu_accounting'] else ''),f"{a['raw_bytes']/2**20:.2f}"] for a in attempts]),
        f"本轮{len(attempts)}次探索合计墙钟 **{aggregate['cost']['wall_seconds']/60:.2f}分钟**、记录CPU至少 **{aggregate['cost']['recorded_cpu_seconds_lower_bound']/3600:.2f}小时**、保留原始数据 **{aggregate['cost']['retained_raw_bytes']/2**30:.3f}GiB**。包含失败设置、错误SKU、超时和拒绝采用的配置，不只统计最终成功case。CPU下界来自npusim超时前端未回收后代的getrusage限制；字节数是结束时保留量，不是峰值临时空间。\n",
        '采集预算：旧基线和早期探针是120秒；`kv32k-compact-complete`显式提高为360秒，只用于获得一次完整32k参考。若它完成，不能将预算增加的作用宣传为降开销。1线程和8线程配置没有墙钟收益；未改全局SDK配置。更长上下文继续以成本门槛决定是否完整仿真。\n',
        '原始产物留在本地runs/long-fa，不进入Git。可审计尝试命令、日志/config/host/model哈希和CPU口径见 [capture-attempts.json](capture-attempts.json)。在线模型仍为v0.4，无新增CA校准拟合。\n']
    (OUT/'error-analysis.md').write_text('\n'.join(lines))
    plot(extrap,fresh,attempts,diagnosis)

def plot(extrap,fresh,attempts,diagnosis):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'build/matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(13,4.6),layout='constrained')
    ax[0].plot([r['shape'][1]/1024 for r in extrap],[r['predicted_us'] for r in extrap],'--',label='Frozen v0.4 formula (unvalidated beyond points)')
    measured=[r for r in fresh if r['role']=='independent_long_holdout']
    if measured:ax[0].scatter([r['shape'][1]/1024 for r in measured],[r['reference_us'] for r in measured],s=60,c='#d45b32',label='Complete CA measurement',zorder=4)
    ax[0].axhline(9.538626484510798,color='gray',ls=':',label='Frozen constant baseline')
    ax[0].set(xscale='log',yscale='log',xlabel='KV length (Ki tokens)',ylabel='Instruction makespan (us)',title='FA Q=9, D=64, B=H=1')
    ax[0].legend(fontsize=7);ax[0].grid(alpha=.2)
    names=['bridge-1153-core-lib','bridge-1153-npusim-local-sku','bridge-1153-native-dirs','bridge-1153-threads8']
    selected=[next(a for a in attempts if a['name']==n) for n in names]
    labels=['msopprof full\n(actual camodel)','Compact DB\n32 threads','Native summary\nnot timing-qualified','Compact DB\n8 threads']
    bars=ax[1].bar(range(len(selected)),[a['wall_seconds'] for a in selected],color=['#8093b9','#498a70','#a7a7a7','#a7a7a7'])
    for bar,a in zip(bars,selected):ax[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"{a['wall_seconds']:.1f}s\n{a['raw_bytes']/2**20:.1f} MiB",ha='center',fontsize=8)
    ax[1].set_xticks(range(len(selected)),labels,fontsize=8)
    ax[1].set(ylabel='Collection wall time (s)',title='Same bridge shape [17,1153,64,1,1]',ylim=(0,max(a['wall_seconds'] for a in selected)*1.24))
    fig.suptitle('Complete CA points, formula extrapolation, and capture costs',fontsize=13)
    for ext in ('png','svg'):fig.savefig(OUT/f'long-fa-validation.{ext}',dpi=170)
    plt.close(fig)
    if diagnosis:
        fig,ax=plt.subplots(figsize=(9,3.6),layout='constrained')
        values=diagnosis['four_mmad_start_intervals_us']
        ax.plot(range(1,len(values)+1),values,lw=1,label='Measured: consecutive 4-MMAD group start interval')
        ax.axhline(diagnosis['model_weighted_ii_us'],ls='--',color='#d45b32',label='Frozen model: weighted KV-tile increment')
        ax.axhline(diagnosis['median_interval_us'],ls=':',color='#498a70',label='Measured median (diagnosis only)')
        ax.set(xlabel='Chronological 4-MMAD group interval',ylabel='Time (us)',title='32k FA: correct instruction count, excessive modeled steady increment')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
        for ext in ('png','svg'):fig.savefig(OUT/f'fa-steady-diagnostic.{ext}',dpi=170)
        plt.close(fig)
    for path in OUT.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')

if __name__=='__main__':main()
