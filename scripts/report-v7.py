#!/usr/bin/env python3
"""Render cumulative v7 error evidence without running CA or fitting parameters.

Freeze predictions are authoritative for newly acquired holdouts. Components of
one chain share its acquisition and are not counted as extra CA attempts.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/v0.7'
CURRENT=ROOT/'results/current'
from operator_model.model_v7 import Case, Predictor
from operator_model.mechanism import predict_case as mechanism_predict

OPS={
 'quant':('dynamicMXquant','ops-nn','aclnnDynamicMxQuant'),
 'matmul':('Matmul','ops-nn','aclnnMatmul → MatMulV3'),
 'fa':('FA','ops-transformer','aclnnFlashAttentionScoreV2'),
 'rmsnorm':('RMSNorm','ops-nn','aclnnRmsNorm'),
 'addmm':('Addmm','ops-nn','aclnnAddmm'),
 'lightning_indexer':('Lightning Indexer','ops-transformer','aclnnLightningIndexer'),
 'mhc_pre':('mHC Pre','ops-transformer','aclnnMhcPre'),
 'mhc_sinkhorn':('mHC Sinkhorn','ops-transformer','aclnnMhcSinkhorn'),
 'mhc_pre_sinkhorn':('mHC PreSinkhorn','ops-transformer','aclnnMhcPreSinkhorn'),
 'mhc_post':('mHC Post','ops-transformer','aclnnMhcPost'),
 'mhc':('mHC组合','ops-transformer','aclnnMhcPre + aclnnMhcSinkhorn + aclnnMhcPost'),
 'kda_recurrent':('KDA recurrent','ops-transformer','aclnnRecurrentKda'),
 'kda_chunk':('KDA chunk','ops-transformer','aclnnChunkKdaFwd'),
 'grouped_matmul_swiglu_quant_v2':('GMM1–SwiGLU–MX量化','ops-transformer','aclnnGroupedMatmulSwigluQuantV2'),
 'gmm2':('MoE GMM2','ops-transformer','aclnnGroupedMatmulV5'),
 'moe_ffn':('MoE FFN组合','ops-transformer','aclnnGroupedMatmulSwigluQuantV2 + aclnnGroupedMatmulV5'),
}
STATUS={'qualified_local_mechanism':'A5机制有局部验证；跨代未验证','local_proxy_only_mechanism_unqualified':'局部回归对照已校准；通用机制缺证据','mechanism_no_new_validation':'A5机制只有开发回放','invocation_only':'只有完整调用证据；耗时模型未合格','unmeasured':'无合格完整CA'}
INDEPENDENT={'retained_frozen_holdout','new_frozen_holdout','retained_v6_frozen_holdout','v7_frozen_holdout'}

def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data): Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def fmt(x,n=3): return 'N/A' if x is None else f'{x:.{n}f}'
def pct(x): return 'N/A' if x is None else f'{100*x:.2f}%'
def metric(rows,key='predicted_us'):
    rs=[r for r in rows if r.get(key) is not None and r.get('reference_us') is not None]
    if not rs:return None
    err=[abs(r[key]-r['reference_us']) for r in rs]
    return dict(n=len(rs),wape=sum(err)/sum(r['reference_us'] for r in rs),max_ape=max(e/r['reference_us'] for e,r in zip(err,rs)),mae_us=sum(err)/len(rs))
def mget(m,key):return None if m is None else m.get(key)
def table(headers,rows):
    cell=lambda x:str(x).replace('|','&#124;').replace('\n','<br>')
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+['| '+' | '.join(cell(v) for v in r)+' |' for r in rows])+'\n'
def load_fit():
    spec=importlib.util.spec_from_file_location('fit_v7',ROOT/'scripts/fit-v7.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def frozen_prediction(obs, frozen, predictor):
    c=obs['case']
    if c['split']!='holdout':return predictor.predict(Case(c['op'],tuple(c['shape'])),c['attrs'])
    parent=next((r for r in frozen['predictions'] if r['id']==obs['parent_id']),None)
    if parent is None:raise ValueError('Missing frozen prediction '+obs['parent_id'])
    if c['id']==obs['parent_id']:return deepcopy(parent['prediction'])
    component=next((r for r in parent['prediction'].get('components',[]) if r['op']==c['op']),None)
    if component is None:raise ValueError('Missing frozen component prediction '+c['op'])
    return deepcopy(component)

def build():
    legacy=read(OUT/'legacy-v6-evaluation.json');model=read(OUT/'model.json');frozen=read(OUT/'frozen-predictions.json')
    if frozen['model_sha256']!=sha(OUT/'model.json'):raise ValueError('Model changed since freeze')
    if model['base_v6']!=read(ROOT/'results/v0.6/model.json'):raise ValueError('Legacy base changed; cannot reuse v6 evaluation')
    integrity={p:sha(ROOT/p)==s for p,s in frozen['artifact_sha256'].items()}
    if not all(integrity.values()):raise ValueError('Frozen artifacts changed: '+str([p for p,v in integrity.items() if not v]))
    fit=load_fit();records=fit.records();obs=fit.observations(records);byid={r['case']['id']:r for r in records};p=Predictor(model)
    hardware=read(ROOT/'configs/hardware-950dt-v7.json')
    frozen_mechanisms={r['id']:r['prediction'] for r in frozen.get('mechanism_predictions',[])}
    rows=[deepcopy(r) for r in legacy['rows'] if r['op'] in ('quant','matmul','fa')]
    for r in rows:
        current=p.predict(Case(r['op'],tuple(r['shape'])),r.get('attributes',{}))['predicted_us']
        if not math.isclose(current,r['predicted_us'],rel_tol=1e-10,abs_tol=1e-10):raise ValueError('Legacy delegated prediction changed '+r['id'])
        r['previous_v06_us']=r['predicted_us'];r['evaluation_version']='v0.7';r['component_model_version']='v0.6'
        r['mechanism_predicted_us']=r['predicted_us'];r['local_v7_proxy_us']=None
        r['legacy_role']=r['role']
        if r['role'] in INDEPENDENT:r['role']='retained_v6_frozen_holdout'
        r['reused_measurement']=True;r['independence_note']='Legacy role retained; v7 does not change this family model or select its parameters.'
    for o in obs:
        c=o['case'];r=byid[o['parent_id']];payload=frozen_prediction(o,frozen,p);ref=o['reference_us'];ks=o['kernels']
        pred=payload.get('local_proxy_prediction',payload)
        mechanism=frozen_mechanisms.get(c['id']) if c['split']=='holdout' else mechanism_predict(c['op'],c['shape'],c.get('attrs',{}),hardware)
        if mechanism is None:
            mechanism=payload.get('mechanism_prediction',{'status':'not_frozen_or_not_qualified','predicted_us':None,'candidate_us':None,'missing_evidence':['No qualified frozen hardware-separated mechanism prediction supplied']})
        guess=mechanism.get('predicted_us');local=pred.get('predicted_us')
        if c['split']=='holdout' and r.get('started_utc','')<frozen['frozen_utc']:raise ValueError('Holdout acquired before freeze '+c['id'])
        correctness=r['measurement']['correctness']
        if not correctness or not all('bad=0' in x for x in correctness):raise ValueError('Unqualified correctness '+c['id'])
        rows.append(dict(id=c['id'],parent_id=o['parent_id'],op=c['op'],shape=c['shape'],attributes=c['attrs'],
            role='v7_frozen_holdout' if c['split']=='holdout' else 'development_calibration',
            reference_us=ref,predicted_us=guess,mechanism_predicted_us=guess,mechanism_prediction=mechanism,local_v7_proxy_us=local,previous_v06_us=None,constant_us=pred.get('constant_us'),
            local_proxy_ape=abs(local-ref)/ref if local is not None else None,
            ape=abs(guess-ref)/ref if guess is not None else None,signed_error_us=guess-ref if guess is not None else None,
            api=OPS[c['op']][2],library=OPS[c['op']][1],kernel=' + '.join(k['kernel'] for k in ks),kernels=ks,
            reference_scope=o['scope'],prediction=mechanism,local_proxy_prediction=pred,source=o['record_path'],source_sha256=sha(ROOT/o['record_path']),
            identity=r['identity'],correctness=correctness,correctness_scope='parent caller complete outputs' if c['id']!=o['parent_id'] else 'caller all contract-valid outputs; undefined saved auxiliaries excluded',
            cost=dict(wall_seconds=r.get('wall_seconds',0),cpu_seconds=r.get('cpu_seconds',0),raw_bytes=r.get('raw_bytes',0),extraction_seconds=r.get('extraction_seconds',0),
                      shared_acquisition_id=o['parent_id'],reused=r.get('reused',False),cpu_accounting=r.get('cpu_accounting','historical/reused')),
            observed_cores=[[k.get('active_aic'),k.get('active_aiv')] for k in ks],predicted_cores=None,
            observed_selected_opcode_count=[dict(mmad=k.get('max_core_mmad'),rv=k.get('max_core_rv_count')) for k in ks],predicted_selected_opcode_count=None,opcode_ape=None,
            opcode_comparison_scope='DB ExecutedInstructions records; no new per-op instruction-count model',
            trace_or_db_sha256=[k.get('db_sha256',k.get('trace_sha256')) for k in ks],
            runtime_binary_sha256=[k.get('kernel_binary_sha256',k.get('binary_sha256')) for k in ks],
            frozen_prediction_id=o['parent_id'] if c['split']=='holdout' else None,reused_measurement=r.get('reused',False)))
    summaries=[]
    for op,(name,lib,api) in OPS.items():
        rs=[r for r in rows if r['op']==op];dev=[r for r in rs if r['role'] not in INDEPENDENT];val=[r for r in rs if r['role'] in INDEPENDENT]
        summaries.append(dict(op=op,name=name,library=lib,api=api,complete_unique_shapes=len({tuple(r['shape']) for r in rs}),
            development=metric(dev),independent=metric(val),development_count=len(dev),independent_count=len(val),
            local_development=metric(dev,'local_v7_proxy_us'),local_independent=metric(val,'local_v7_proxy_us'),
            independent_previous_v06=metric(val,'previous_v06_us'),
            independent_constant=metric(val,'constant_us'),development_constant=metric(dev,'constant_us'),
            status='qualified_local_mechanism' if metric(val) else ('local_proxy_only_mechanism_unqualified' if metric(val,'local_v7_proxy_us') or metric(dev,'local_v7_proxy_us') else ('mechanism_no_new_validation' if metric(dev) else ('invocation_only' if rs else 'unmeasured')))))
    failed=deepcopy(legacy['failed_or_unqualified'])
    for r in records:
        if r['status']=='ok':continue
        c=r['case'];entry=dict(id=c['id'],op=c['op'],shape=c['shape'],status=r['status'],reference_us=None,predicted_us=None,
            error=r.get('error') or r.get('stop_reason') or 'inspect run.log',wall_seconds=r.get('wall_seconds'),source=r['record_path'])
        run=(ROOT/r['record_path']).parent
        if (run/'manual-abort.json').exists():
            entry['manual_abort']=read(run/'manual-abort.json');entry['original_collection_error']=entry['error']
            entry['error']=entry['manual_abort'].get('reason') or entry['error']
        log=(run/'run.log').read_text(errors='replace') if (run/'run.log').exists() else ''
        entry['correctness_lines']=[line for line in log.splitlines() if 'correctness ' in line]
        entry['caller_smoke_pass']='SMOKE_PASS' in log
        entry['repeated_set_flag_messages']=log.count('execute_set_flag already has same set_flag')
        entry['partial_databases']=[]
        for db in sorted(run.rglob('*.db')):
            complete=False
            try:
                with sqlite3.connect('file:'+str(db.resolve())+'?mode=ro',uri=True,timeout=1) as conn:
                    info=dict(conn.execute('SELECT * FROM KernelInfo'))
                    complete=bool(info.get('exec_time_ns') is not None and info.get('instrs_exec_num') is not None)
            except sqlite3.Error:pass
            entry['partial_databases'].append(dict(path=str(db.relative_to(ROOT)),bytes=db.stat().st_size,has_completed_kernel_metadata=complete))
        failed.append(entry)
    specs=[read(f) for f in sorted(ROOT.glob('configs/extension-*-v7.json'))]
    planned=[c for spec in specs for c in spec['cases'] if c.get('split') in ('calibration','holdout')]
    retired=[dict(c,plan_container=name) for spec in specs for name,cs in spec.items() if (name.startswith('retired') or name.startswith('fallback')) and isinstance(cs,list) for c in cs]
    retired.extend(dict(c,plan_container='cases',plan_status=c['split']) for spec in specs for c in spec['cases'] if c.get('split') not in ('calibration','holdout'))
    pending=[dict(id=c['id'],op=c['op'],shape=c['shape'],status='not_collected',reference_us=None,predicted_us=None) for c in planned if c['id'] not in byid]
    new=[r for r in records if not r.get('reused')]
    costs=deepcopy(legacy['collection_cost_by_round'])+[dict(round='v0.7',attempts=len(new),successful_attempts=sum(r['status']=='ok' for r in new),
            wall_seconds=sum(r.get('wall_seconds',0) for r in new),cpu_seconds=sum(r.get('cpu_seconds',0) for r in new),
            cpu_is_lower_bound=any('lower_bound' in r.get('cpu_accounting','') or (ROOT/r['record_path']).parent.joinpath('manual-abort.json').exists() for r in new),
            cpu_accounting='recorded CPU; interrupted/unreaped descendants may be missing',raw_bytes=sum(r.get('raw_bytes',0) for r in new),extraction_seconds=sum(r.get('extraction_seconds',0) for r in new))]
    bench=[]
    for op in OPS:
        rs=[r for r in rows if r['op']==op and (r['predicted_us'] is not None or r.get('local_v7_proxy_us') is not None)]
        if not rs:continue
        times=[]
        for i in range(300):
            r=rs[i%len(rs)];case=Case(op,tuple(r['shape']));attrs=r.get('attributes',{})
            start=time.perf_counter_ns();p.predict(case,attrs);times.append((time.perf_counter_ns()-start)/1000)
        bench.append(dict(op=op,n=300,median_us=statistics.median(times),p95_us=sorted(times)[284],scope='preloaded local proxy full prediction; Python process; no CA; base cache may be warm for quant/matmul/FA'))
    mechanism_bench=[]
    for op in list(OPS)[3:]:
        cases=[o['case'] for o in obs if o['case']['op']==op] or [r['case'] for r in records if r['case']['op']==op]
        if not cases:continue
        c=cases[0];times=[];result_prediction=None
        for _ in range(100):
            start=time.perf_counter_ns();result_prediction=mechanism_predict(op,c['shape'],c.get('attrs',{}),hardware);times.append((time.perf_counter_ns()-start)/1000)
        mechanism_bench.append(dict(op=op,n=100,median_us=statistics.median(times),p95_us=sorted(times)[94],
            result_status=result_prediction.get('status'),scope='structure construction and resource-evidence qualification; incomplete hardware profile returns null latency'))
    result=dict(model_version='v0.7',model_path='results/v0.7/model.json',model_sha256=sha(OUT/'model.json'),generated_utc=datetime.now(timezone.utc).isoformat(),
        scope='Cumulative measured operator routes with explicit development/independent roles; component and composed errors are correlated.',
        primary_policy='Qualified hardware-separated mechanism only for extensions; legacy v6 A5 models retained; local v7 regression never fills missing mechanism latency',
        mechanism_engine_path='operator_model/mechanism.py',mechanism_engine_sha256=sha(ROOT/'operator_model/mechanism.py'),
        hardware_profile_path='configs/hardware-950dt-v7.json',hardware_profile_sha256=sha(ROOT/'configs/hardware-950dt-v7.json'),
        local_proxy_model_path='results/v0.7/model.json',frozen_predictions_path='results/v0.7/frozen-predictions.json',
        frozen_utc=frozen['frozen_utc'],artifact_integrity=integrity,rows=rows,operator_summary=summaries,failed_or_unqualified=failed,pending=pending,
        retired_plans=retired,collection_cost_by_round=costs,benchmark=bench,mechanism_benchmark=mechanism_bench,legacy_benchmark=legacy['benchmark'],role_counts=dict(Counter(r['role'] for r in rows)))
    return result

def normalize_svg(path):
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')

def plots(e):
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"build/matplotlib-cache"))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    sums=e['operator_summary'];labels=[r['op'] for r in sums];y=np.arange(len(labels))
    fig,ax=plt.subplots(figsize=(12,9))
    for offset,key,label,col in [(-.3,'development','Existing A5 mechanism: development','#4477aa'),(-.1,'independent','Existing A5 mechanism: frozen','#228833'),(.1,'local_independent','Local regression comparison: frozen','#aa3377'),(.3,'independent_constant','Fixed baseline: same holdout','#cc6677')]:
        vals=[100*mget(r[key],'wape') if r[key] is not None else np.nan for r in sums]
        ax.barh(y+offset,vals,height=.19,label=label,color=col)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('WAPE (%) — missing observations omitted');ax.set_title('Accuracy by operator: A5 models and local baselines')
    ax.grid(axis='x',alpha=.2);ax.legend(fontsize=9);fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(OUT/f'accuracy-by-operator.{ext}',dpi=160)
    plt.close(fig)
    val=[r for r in e['rows'] if r['role'] in INDEPENDENT and (r['predicted_us'] is not None or r.get('local_v7_proxy_us') is not None)]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for op in OPS:
        rs=[r for r in val if r['op']==op]
        if rs:axes[0].scatter([r['reference_us'] for r in rs],[r['predicted_us'] if r['predicted_us'] is not None else r['local_v7_proxy_us'] for r in rs],s=34,label=op+(' (local)' if all(r['predicted_us'] is None for r in rs) else ''))
    refs=[r['reference_us'] for r in val]+[r['predicted_us'] if r['predicted_us'] is not None else r['local_v7_proxy_us'] for r in val]
    if refs:lo=min(refs)*.8;hi=max(refs)*1.2;axes[0].plot([lo,hi],[lo,hi],':',color='black')
    axes[0].set(xscale='log',yscale='log',xlabel='CA reference (us)',ylabel='Prediction (us)',title='Frozen parity (local = regression comparison)')
    axes[0].legend(fontsize=6,loc='upper left');axes[0].grid(alpha=.2)
    cost=e['collection_cost_by_round'];axes[1].barh([r['round'] for r in cost],[r.get('wall_seconds',0)/60 for r in cost],color='#4477aa')
    axes[1].set(xlabel='Acquisition wall minutes (includes failed attempts)',title='Incremental CA cost by round');axes[1].invert_yaxis();fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(OUT/f'validation-and-cost.{ext}',dpi=160)
    plt.close(fig)
    for path in OUT.glob('*.svg'):normalize_svg(path)

def pipeline_digest(kernel):
    """Human report shows one latest-finishing core per type; JSON retains all."""
    bytype={}
    for core in kernel.get('cores',[]):
        typ=core.get('core_type_id')
        if typ not in bytype or core.get('end_ticks',0)>bytype[typ].get('end_ticks',0):bytype[typ]=core
    return '; '.join('type'+str(typ)+'/core'+str(c['core_id'])+': '+json.dumps({p:round(v,4) for p,v in c.get('pipeline_union_us',{}).items()}) for typ,c in sorted(bytype.items())) or 'N/A'

SOURCE_NOTES={
 'rmsnorm':'固定行宽的核内归约和分核波次代理；少量点无法辨识不同 UB 切分、尾行与调度开销。',
 'addmm':'含矩阵乘和 self 融合的 tile 波次代理；小矩阵启动/搬运可能主导，不能直接沿用大 Matmul 的利用率。',
 'lightning_indexer':'实际 BF16 LightningIndexer，包含 scoring 与 TopK；受限形状上的 tile 工作代理不能证明真实稀疏注意力或 FP8 路由覆盖。',
 'mhc_pre':'Pre 包含投影/混合以及相关输出；该路径与 PreSinkhorn 不可相加，输出和 FP32 投影属性影响路由。',
 'mhc_sinkhorn':'FP32 迭代归一化。固定 num_iters 的每轮开销与头部同步未独立辨识。',
 'mhc_pre_sinkhorn':'真实融合 PreSinkhorn 调用，不能以独立 Pre+Sinkhorn 的计时相加代替。',
 'mhc_post':'Vector 混合/归约，工作量相同的核内尾部和核数波次仍可能产生不同耗时。',
 'mhc':'组合调用的净串行 kernel 窗口求和，外部 Attention/MLP 主体不在测量范围；不可作为整网 mHC 层端到端耗时。',
 'kda_recurrent':'同一 state 的 token 更新串行；S 项不能简单除以全部 AIV 核数。当前仅 K=V=128、B/H=1、预激活 gate；S≤8，Q9 不静默拆分。',
 'kda_chunk':'Gate/Prepare/Post-WU/FwdH/Finalize 属于真实 ChunkKdaFwd 的内部阶段。尝试C64/K16 generic与K128 key2 arch35尾块候选；key2不代表内联Gate/融合head-pair/dense FwdH全部启用。S16/Hv1/safeGate=false不满足这些额外条件，CA同步告警与未完整结束的项排除。',
 'grouped_matmul_swiglu_quant_v2':'第一 GMM + SwiGLU + MX 量化融合。专家 token 直方图影响 tile 数和负载，不可用总 M/EP 代替。',
 'gmm2':'实际消费同一caller第一融合阶段产生的量化输出及尺度；该组件计时复用完整链第二个DB，未额外计CA。早期TensorList双重释放的失败尝试仅计成本；r3修复后完整链重新采集。',
 'moe_ffn':'r3实际串联GMM1–SwiGLU–MX量化与消费其输出/尺度的GMM2，两个DB完整后按stream/task/launch匹配，链时延为净kernel窗口求和。组件共用同一采集；不含dispatch/combine/EP通信和host间隔，不宣称整网FFN端到端。旧生命周期失败仍排除。',
}

def report(e):
    s=['# 当前整套算子建模：误差、证据与成本','\n## Summary：逐算子误差情况\n']
    s.append(table(['算子','完整CA shape','开发/独立样本','机制开发WAPE / max','机制独立WAPE / max','局部回归开发WAPE','局部回归独立WAPE / max','固定独立WAPE','状态'],[
        [x['name'],x['complete_unique_shapes'],str(x['development_count'])+'/'+str(x['independent_count']),pct(mget(x['development'],'wape'))+' / '+pct(mget(x['development'],'max_ape')),
         pct(mget(x['independent'],'wape'))+' / '+pct(mget(x['independent'],'max_ape')),pct(mget(x['local_development'],'wape')),
         pct(mget(x['local_independent'],'wape'))+' / '+pct(mget(x['local_independent'],'max_ape')),pct(mget(x['independent_constant'],'wape')),STATUS.get(x['status'],x['status'])] for x in e['operator_summary']]))
    s.append('当前 **v0.7** 包括不改动参数的 v0.6 quant/Matmul/FA（有tiling/流水机制，但仍只在A5校准，未验证跨架构），以及本轮真实库调用的小 shape 局部回归对照。扩展的 a+bW 不作为跨形状/跨硬件的通用机制默认；mechanism 列在物理映射、吞吐/发起间隔等独立资源参数未获证据时为 N/A；当前真实硬件profile不填虚构吞吐。N/A 表示没有合格误差证据，不代表0误差。组合入口和组件共享原始采集；完整shape列逐算子统计，不能跨行相加当作独立CA次数。表中保留旧族仍未用于开发的冻结验证，新扩展仅以本轮冻结后的独立shape统计。\n')
    s.append('![逐算子开发与独立验证误差](../results/v0.7/accuracy-by-operator.png)\n\n![冻结验证预测与CA、历轮采集开销](../results/v0.7/validation-and-cost.png)\n')
    s.append('完整公式、逐参数解释及适用域见 [整体建模方法](modeling-strategy.md)；每例 JSON 证据见 [v0.7 evaluation](../results/v0.7/evaluation.json)，本轮变化见 [changes.md](../results/v0.7/changes.md)。本页由 `scripts/report-v7.py` 生成，不启动 CA、不重新拟合。\n')
    s.append('## 1. 统计口径、冻结与误差归因边界\n')
    s.append(f"模型冻结时间：`{e['frozen_utc']}`；模型SHA256：`{e['model_sha256']}`。新留出的预测直接读取 `frozen-predictions.json`，未根据留出实测重新预测、选模或调参。旧扩展在v0.6只有结构模型或单次调用，旧版耗时必须为N/A；不能把0或Roofline补成旧版精度。\n")
    s.append('APE=|预测−CA|/CA；WAPE=Σ|预测−CA|/ΣCA。开发数据用于校准或机制判断，与独立验证分开；通常每shape仅采一次，未估计重复运行噪声或置信区间。当前测量为950DT/1.65GHz CA完整指令窗口，不是硬件实测。多kernel值为串行单stream各kernel净指令窗口求和，**不含host启动、API初始化、launch间空闲，也不声称跨DB时钟已经对齐**。\n')
    s.append('新扩展每条路由只拟合非负固定项和一个工作量系数。若两例校准工作量相同，模型只能辨识常数；若最优系数为0，也可能只是固定tile/启动占主导。固定基线取校准集均值且与模型一起冻结。低局部误差不等于已获得指令吞吐、跨代利用率或全shape精度。实测指令数用于证据审计，扩展算子的预测指令误差仍N/A。\n')
    s.append(table(['角色','行数'],e['role_counts'].items()))
    s.append('\n### 当前与旧版、固定基线：同一冻结集合\n')
    s.append(table(['算子','独立样本n','机制WAPE','局部回归WAPE','v0.6同集WAPE','固定同集WAPE'],[[x['name'],x['independent_count'],pct(mget(x['independent'],'wape')),pct(mget(x['local_independent'],'wape')),pct(mget(x['independent_previous_v06'],'wape')),pct(mget(x['independent_constant'],'wape'))] for x in e['operator_summary']]))
    s.append('\n## 2. 原有 quant / Matmul / FA：全部逐shape证据\n')
    old=(OUT/'legacy-v6-error-analysis.md').read_text();section=old[old.index('### dynamicMXquant'):old.index('### RMSNorm')]
    s.append('三族参数及预测与v0.6一致；下列完整逐shape表与机制诊断沿用归档内容，其中“v0.5”是历史比较列。原有开发/独立角色不因版本号变化自动重置。\n\n'+section)
    s.append('\n## 3. 扩展算子：真实库实现、逐shape预测/CA与误差来源\n')
    s.append('库组件是本机CANN9.2.1实际ACLNN安装包；开源ops-nn/ops-transformer目录只用于合同与机制核对，不代表重新编译替换了kernel。`entry_kernel_N` 是DB记录的运行入口，必须连同DB SHA、launch序号和caller身份理解，不能单凭入口名判断二进制相同。未导出的运行时ELF SHA仍为null。正确性覆盖接口契约定义为有效的全部请求输出；need_backward=false时文档明确无效的saved辅助输出只保证ABI分配，不把未定义值当作参考数值验证。表中AIC/AIV与指令数为**每kernel**实际活跃核和最大核MMAD/RV记录；helper记录口径与早期成对日志计数不同，不混算误差。\n')
    for op in list(OPS)[3:]:
        name,lib,api=OPS[op];rs=[r for r in e['rows'] if r['op']==op]
        s.append(f'### {name}\n\n实际接口：`{api}`，库组件 **{lib}**。{SOURCE_NOTES[op]}\n')
        if not rs:s.append('尚无合格完整CA，耗时/误差 N/A；失败或未采集情况见后文。\n');continue
        s.append(table(['ID / shape','数据角色','CA µs','机制预测 µs','局部回归 µs','局部回归APE','固定 µs','实际kernel'],[[r['id']+' '+str(r['shape']),r['role'],fmt(r['reference_us']),fmt(r.get('mechanism_predicted_us')),fmt(r.get('local_v7_proxy_us')),pct(r.get('local_proxy_ape')),fmt(r.get('constant_us')),r['kernel']] for r in rs]))
        s.append(table(['ID','AIC/AIV逐kernel','MMAD/RV最大核计数','正确性检查组','CA墙钟s','记录CPU s','DB/日志MiB'],[[r['id'],r['observed_cores'],r['observed_selected_opcode_count'],len(r['correctness']),fmt(r['cost']['wall_seconds']),fmt(r['cost']['cpu_seconds']),fmt(r['cost']['raw_bytes']/2**20)] for r in rs]))
        if op=='rmsnorm':
            diagnostic=[]
            for r in rs:
                if r['shape'] not in ([9,5120],[36,7168]) or len(r['kernels'])!=1:continue
                k=r['kernels'][0];cores=k.get('cores',[])
                if not cores:continue
                core=max(cores,key=lambda c:c['end_ticks']);pu=core.get('pipeline_union_us',{})
                spread=(max(c['start_ticks'] for c in cores)-min(c['start_ticks'] for c in cores))/(1000*k['frequency_ghz'])
                diagnostic.append([r['id'],r['shape'],r['role'],k['kernel'],k.get('active_aiv'),core['rv_count'],fmt(pu.get('MTE3')),fmt(pu.get('MTE2')),fmt(pu.get('SIMD')),fmt(spread),fmt(r['reference_us'])])
            s.append('重点偏差：冻结新留出[36,7168]的CA为6.61394µs，局部回归仅4.48984µs，低估约32.1%。下面是冻结后诊断，未修改该模型或留出预测：\n'+table(['ID','shape','角色','entry','AIV','临界RV','MTE3 union µs','MTE2 union µs','SIMD union µs','核启动spread µs','CA µs'],diagnostic))
            s.append('已证实：由校准[9,5120]到留出[36,7168]，同entry_kernel_1388的活跃AIV9→36，临界RV1183→1631；MTE3 union约0.94727→3.01394µs、MTE2约0.78424→1.30182µs，SIMD仅约0.46545→0.58606µs，核起始时间spread约0.15576→0.38606µs。最大的已观测变化是写出区间扩张，不能归为纯计算量或固定启动差。它提示需要显式刻画多核共享搬运/等待与流水，但仅这两点不能证明某个带宽瓶颈或给出其因果占比。此点仍是v0.7独立留出；下一轮若据其调参，必须转为开发数据并另留新验证。\n')
        if op in ('mhc_post','mhc_sinkhorn'):
            diagnostic=[]
            for r in sorted((r for r in rs if r['role']=='development_calibration'),key=lambda r:r['shape'][0]):
                if len(r['kernels'])!=1 or not r['kernels'][0].get('cores'):continue
                k=r['kernels'][0];core=max(k['cores'],key=lambda c:c.get('end_ticks',0));pu=core.get('pipeline_union_us',{})
                diagnostic.append([r['id'],r['shape'][0],k['kernel'],k.get('active_aiv'),core['core_id'],core.get('rv_count'),fmt(pu.get('SIMD')),fmt(pu.get('SCALAR')),fmt(r['reference_us'])])
            s.append('校准集临界核证据（未用于留出后重新拟合）：\n'+table(['ID','T','entry','活跃AIV','临界核','RV记录','SIMD union µs','SCALAR union µs','kernel窗口 µs'],diagnostic))
            if op=='mhc_post':
                s.append('已证实：T=1与T=32同为entry_kernel_5，临界核RV记录396→208，SIMD union约1.1945→0.1315µs，SCALAR约3.4291→2.2679µs，总窗口4.6461→3.4085µs。因此“每核只处理一个token，所以每核工作固定”的代理假设遗漏了真实指令/分支变化。尚未独立隔离具体哪项tiling或实现分支造成差异，不能笼统归因噪声、频率或核启动。局部常量和激活核数log对照都没有捕获这个执行机制。完整校准诊断见 [机制缺口证据](../results/v0.7/calibration-mechanism-gaps.json)，该诊断没有修改冻结参数。\n')
            else:
                s.append('已证实：T=1与T=64的临界核RV记录均为1903，SIMD union却约3.8261→4.7582µs，总窗口6.2964→7.5976µs。同指令次数不保证同耗时；流水重叠、等待和共享资源竞争仍需独立证据。当前尚未分离其因果贡献，不能把该变化直接写成某一类争用已被证实，也不能把指令duration当成发起间隔。局部常量/激活核数log只是对照，不能替代该机制。\n')
        for r in rs:
            s.append(f"\n<details><summary>{r['id']}：参数、数值检查与逐kernel流水证据</summary>\n\n")
            s.append('调用属性：`'+json.dumps(r['attributes'],ensure_ascii=False)+'`。证据：`'+r['source']+'`。时间口径：`'+r['reference_scope']+'`。\n')
            s.append('机制证据状态：`'+json.dumps({k:r.get('mechanism_prediction',{}).get(k) for k in ('status','predicted_us','candidate_us','missing_evidence')},ensure_ascii=False)+'`。\n')
            s.append('```text\n'+'\n'.join(r['correctness'])+'\n```\n')
            s.append(table(['kernel','blockDim','AIC/AIV','MMAD/RV最大核','DB SHA256','各core-type临界核流水union µs'],[
                [k['kernel'],k.get('block_dim','N/A'),[k.get('active_aic'),k.get('active_aiv')],[k.get('max_core_mmad'),k.get('max_core_rv_count')],k.get('db_sha256','N/A'),
                 pipeline_digest(k)] for k in r['kernels']]))
            s.append('表内每个core-type仅展示最晚结束的临界核（type1为AIC，其余为AIV），完整全部核流水保存在evaluation.json的kernels/cores。流水union时间在不同流水间会重叠，不能求和当作kernel耗时。`max_core`是各计数的最大核，未必同一核；不据此伪造全核指令总数。\n\n</details>\n')
        worst=max((r for r in rs if r.get('local_proxy_ape') is not None),key=lambda r:r['local_proxy_ape'],default=None)
        if worst:s.append(f"本路由局部回归对照的最大已观测APE为 {pct(worst['local_proxy_ape'])}（`{worst['id']}`，{worst['role']}）。这说明受限代理的偏差大小；固定tile、启动、流水填充/尾部、分核变化是待进一步区分的候选来源，**未经对照实验不指定为已证实根因**。数值正确通过不等于该时延误差已经达标。\n")
    s.append('\n## 4. 失败、部分trace和未采集项\n')
    s.append(table(['ID / shape','状态','墙钟s','排除原因'],[[r['id']+' '+str(r['shape']),r['status'],fmt(r.get('wall_seconds')),r.get('error','未采集')] for r in e['failed_or_unqualified']+e['pending']]))
    for r in e['failed_or_unqualified']:
        if 'partial_databases' not in r:continue
        s.append(f"\n<details><summary>{r['id']}：失败与部分采集证据</summary>\n\n")
        s.append(f"caller SMOKE_PASS：{r['caller_smoke_pass']}；重复set_flag告警数：{r['repeated_set_flag_messages']}。这些字段只说明已有证据，不将部分DB窗口当完整算子耗时。\n")
        s.append(table(['保留DB','字节','具有完整KernelInfo'],[[d['path'],d['bytes'],d['has_completed_kernel_metadata']] for d in r['partial_databases']]))
        if r['correctness_lines']:s.append('```text\n'+'\n'.join(r['correctness_lines'])+'\n```\n')
        s.append('\n</details>\n')
    s.append('本轮失败、撤销或被失败门禁阻止的预设计划（不计合格完整实测；实际失败尝试成本已计入）：\n'+table(['ID / shape','计划状态','原因'],[[r['id']+' '+str(r['shape']),r.get('plan_status') or r.get('status') or r.get('plan_container','retired'),r.get('reason') or r.get('note') or ('superseded by '+r['superseded_by'] if r.get('superseded_by') else 'retired/superseded plan')] for r in e.get('retired_plans',[])]))
    s.append('PreSinkhorn两次失败为已定位的caller ABI问题：r1的hRes descriptor rank与接口不符，返回161002；r2把5个必需saved输出传nullptr，返回561103。r3提供全部8个输出descriptor；在need_backward=false时只有3个输出按契约有效，因此只对这3组做数值验证，不验证未定义saved值。两次失败均保留并计成本，不是数值误差或时延模型样本。\n')
    s.append('MoE旧失败存在已定位的caller生命周期根因：aclDestroyTensorList同时销毁成员aclTensor，而原RAII再次销毁造成BlockCache退出挂起；r3转移metadata所有权，1000次无设备元数据自测通过，重新CA才可进入计时。诊断见 [MoE生命周期诊断](../results/v0.7/moe-lifecycle-diagnosis.json)。这与KDA重复set_flag/未结束的未定界同步资格问题不同，不能统称算子数值错误或CA算力慢。\n')
    s.append('失败、超时、仅部分trace、身份不完整和数值检查失败的项不能进入WAPE。旧文件名中的 bridge 仅为历史采集方式一致性校验的任务名，不是一个算子；同shape重复采集只计成本，不增加唯一shape覆盖。\n')
    s.append('\n## 5. 仿真采集开销与预测效率\n')
    s.append(table(['轮次','尝试','CA墙钟分钟','记录CPU小时（部分下界）','保留MiB'],[[r['round'],r.get('attempts','N/A'),fmt(r.get('wall_seconds',0)/60),fmt(r.get('cpu_seconds',r.get('recorded_cpu_seconds_lower_bound',0))/3600),fmt(r.get('raw_bytes',r.get('retained_raw_bytes',0))/2**20)] for r in e['collection_cost_by_round']]))
    s.append('成本按实际collector调用累计，含失败，组件和组合不重复收费；复用旧RMSNorm/Addmm和原51个quant/Matmul/FA的旧trace没有本轮新增CA成本。保留字节不是峰值临时占用。CPU可因多线程高于墙钟；CPU仅为已记录的子进程时间，含lower_bound_unreaped_descendants或手动终止时可能漏计未回收后代，因此部分为下界，累计CPU不宣称精确完整；墙钟按实际collector时段累计。轻量SQL从现有DB提取，不导出巨大逐指令CSV，也不通过不断延长超时伪造效率改善。\n')
    s.append('局部回归对照/原三族模型的完整查询：\n'+table(['算子','预测次数','中位µs','p95 µs'],[[r['op'],r['n'],fmt(r['median_us']),fmt(r['p95_us'])] for r in e['benchmark']]))
    s.append('独立机制引擎建图与资源证据资格检查（当前资源参数不全，输出时延仍为N/A；该项仅衡量软件处理开销）：\n'+table(['算子','次数','中位µs','p95 µs','返回状态'],[[r['op'],r['n'],fmt(r['median_us']),fmt(r['p95_us']),r['result_status']] for r in e.get('mechanism_benchmark',[])]))
    s.append('上述是本机同Python进程、模型预加载的完整predict调用（包括Python对象和域检查），不包含CA、模型加载和DB解析；量化/Matmul/FA的旧路径可能已命中进程内缓存，不能把此列当成其未缓存首次预测。旧v0.6专门区分scalar/full/cache的未缓存基准保留如下，单位µs、不是本轮重测：\n```json\n'+json.dumps(e['legacy_benchmark'],ensure_ascii=False,indent=2)+'\n```\n')
    boundary=old[old.index('## 7. 场景外推与当前整体边界'):old.index('当前模型SHA256')]
    s.append('\n## 6. 场景覆盖与外推边界\n'+boundary.replace('## 7. 场景外推与当前整体边界',''))
    s.append('本轮扩展验证为少量机制代表shape，未完整覆盖用户全部prefill/decode batch、TP/DP/EP、专家不均衡、长序列或不同dtype。KDA recurrent Q9超出当前公开接口单序列≤8限制；KDA chunk的key2尾块候选不等于完整融合快流水，失败采集不能充当已验证路由。A7/A8仍是架构参数假设，不能把A5局部拟合系数按频率/核数简单缩放即宣布跨代准确。\n')
    return '\n'.join(s)

def changes(e):
    newer=[x for x in e['operator_summary'] if x['op'] not in ('quant','matmul','fa')]
    last=e['collection_cost_by_round'][-1]
    return '\n'.join(['# v0.7 迭代变化',
      '\n保持v0.6 quant/Matmul/FA模型和原始trace；新增真实库算子的小shape局部回归对照；通用默认要求算子结构/硬件资源分离，未校准机制预测保持N/A，按路由/属性限定预测域。MoE两阶段和mHC组合按真实串行kernel净窗口描述，不混入host开销。公式及每参数解释维护在全量 modeling-strategy.md。',
      '\n模型先用calibration拟合，再冻结全部holdout预测。旧版新扩展没有可比较时延模型，v0.6同集误差为N/A；原三族同集预测没有变化。新增独立与固定基线比较：\n',
      table(['算子','开发/独立n','机制独立WAPE','局部回归开发WAPE','局部回归独立WAPE / max','固定同集WAPE'],[[x['name'],str(x['development_count'])+'/'+str(x['independent_count']),pct(mget(x['independent'],'wape')),pct(mget(x['local_development'],'wape')),pct(mget(x['local_independent'],'wape'))+' / '+pct(mget(x['local_independent'],'max_ape')),pct(mget(x['independent_constant'],'wape'))] for x in newer]),
      f"\n新增CA {last['attempts']} 次（成功 {last['successful_attempts']}），墙钟 {last['wall_seconds']/60:.2f} 分钟，记录CPU {last['cpu_seconds']/3600:.2f} 小时（部分终止进程为下界），保留 {last['raw_bytes']/2**20:.2f} MiB；DB归约 {last['extraction_seconds']:.2f} 秒。复用记录不再次计费。预测基准逐算子见 evaluation.json，旧族cache和未缓存口径分开。",
      '\n未完成范围：真实整网dispatch形状/路由、全面大shape、独立指令吞吐识别、跨代A7/A8准确性；Q9 recurrent单调用不支持；256k/512k FA未实测部分仍为外推。失败/未采集项保持N/A，不能用本轮小样本误差代替全域准确率。',
      '\n产物：model.json、frozen-predictions.json、evaluation.json、accuracy-by-operator.png/.svg、validation-and-cost.png/.svg；全量 docs/error-analysis.md 覆盖旧与新算子。\n'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--no-publish-current',action='store_true');args=parser.parse_args()
    if not (OUT/'legacy-v6-evaluation.json').exists():raise ValueError('Archive legacy evaluation before modifying current results')
    e=build();plots(e);write(OUT/'evaluation.json',e);text=report(e)
    for name in ('moe-lifecycle-diagnosis.json','shared-opcode-evidence.json'):
        source=ROOT/'runs/model-v7'/name
        if source.exists():(OUT/name).write_bytes(source.read_bytes())
    version_text=text.replace('../results/v0.7/','./').replace('../results/','../').replace('(modeling-strategy.md)','(../../docs/modeling-strategy.md)')
    (OUT/'error-analysis.md').write_text(version_text);(OUT/'changes.md').write_text(changes(e));(ROOT/'docs/error-analysis.md').write_text(text)
    if not args.no_publish_current:
        CURRENT.mkdir(exist_ok=True);write(CURRENT/'evaluation.json',e)
        for ext in ('png','svg'):
            archive=OUT/f'legacy-v6-all-operators.{ext}';old_image=CURRENT/f'all-operators.{ext}'
            if old_image.exists() and not archive.exists():archive.write_bytes(old_image.read_bytes())
            if ext=='svg' and archive.exists():normalize_svg(archive)
            old_image.write_bytes((OUT/f'accuracy-by-operator.{ext}').read_bytes())
    print(json.dumps(dict(rows=len(e['rows']),roles=e['role_counts'],pending=len(e['pending']),version='v0.7'),ensure_ascii=False))
if __name__=='__main__':main()
