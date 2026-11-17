#!/usr/bin/env python3
"""Build cumulative operator error report from committed aggregates, without CA.

The all-shape view is not a new independent test: data roles are explicit and
old development/selection inputs never enter the independent error column.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results/current'
APIS = {'quant': ('dynamicMXquant', 'ops-nn', 'aclnnDynamicMxQuant'),
        'matmul': ('Matmul', 'ops-nn', 'aclnnMatmul → MatMulV3'),
        'fa': ('FA', 'ops-transformer', 'aclnnFlashAttentionScoreV2'),
        'rmsnorm': ('RMSNorm', 'ops-nn', 'aclnnRmsNorm'),
        'addmm': ('Addmm', 'ops-nn', 'aclnnAddmm')}
EXTENSIONS = [
    ('lightning_indexer', 'Lightning Indexer', 'aclnnLightningIndexer', '结构模型；TopK、实际tiling及耗时未验证'),
    ('mhc_pre', 'mHC Pre', 'aclnnMhcPre', '结构模型；投影/混合及分核未校准'),
    ('mhc_sinkhorn', 'mHC Sinkhorn', 'aclnnMhcSinkhorn', '结构模型；FP32迭代归一化未校准'),
    ('mhc_pre_sinkhorn', 'mHC PreSinkhorn', 'aclnnMhcPreSinkhorn', '结构模型；融合路径不能与拆分路径重复计时'),
    ('mhc_post', 'mHC Post', 'aclnnMhcPost', '结构模型；Vector路径未校准'),
    ('mhc', 'mHC组合', '上述Mhc接口组合', '结构模型；不含外部Attention/MLP主体'),
    ('kda_recurrent', 'KDA recurrent', 'aclnnRecurrentKda', '结构模型；S≤8，Q9路由未解决'),
    ('kda_chunk', 'KDA chunk', 'aclnnChunkKdaFwd', '结构模型；内部阶段不等于多次公开launch'),
    ('grouped_matmul_swiglu_quant_v2', 'GMM1–SwiGLU–MX量化', 'aclnnGroupedMatmulSwigluQuantV2', '结构模型；需本rank专家token直方图'),
    ('gmm2', 'MoE GMM2', 'aclnnGroupedMatmulV5（候选）', '仅数学工作及候选API；量化scale ABI尚未串联'),
    ('moe_ffn', 'MoE FFN组合', 'GMM1融合 + GMM2候选', '组合工作量；路由/通信/第二GMM未验证，不是单一kernel'),
]


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def fmt(value, places=3):
    return 'N/A' if value is None else f'{value:.{places}f}'


def pct(value):
    return 'N/A' if value is None else f'{100*value:.2f}%'


def table(headers, rows):
    def cell(v):
        return str(v).replace('|', '&#124;').replace('\n', '<br>')
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|',
                      *['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in rows]]) + '\n'


def metrics(rows, key='predicted_us'):
    rows = [r for r in rows if r.get(key) is not None]
    if not rows:
        return None
    errors = [abs(r[key] - r['reference_us']) for r in rows]
    return {'n': len(rows), 'wape': sum(errors)/sum(r['reference_us'] for r in rows),
            'max_ape': max(e/r['reference_us'] for e, r in zip(errors, rows)),
            'mae_us': sum(errors)/len(rows)}


def observed_count(op, m):
    if op == 'quant':
        cores = m.get('cores', [])
        # Compact captures do not necessarily retain dynamic opcode histograms.
        if not cores or not all('dynamic_opcodes' in c for c in cores):
            return m.get('max_core_rv_count')
        return max(sum(n for name, n in c['dynamic_opcodes'].items() if name.startswith('RV_')) for c in cores)
    if 'max_core_mmad' in m:
        return m['max_core_mmad']
    return max((c.get('dynamic_opcodes', {}).get('MMAD', 0) for c in m.get('cores', [])), default=None)


def normalize(case, measurement, role, source, predictor, old, identity=None, cost=None):
    op = case['op']; shape = case['shape']; ref = measurement['reference_us']
    if not math.isfinite(ref) or ref <= 0:
        raise ValueError('Invalid CA reference: ' + case['id'])
    if measurement.get('reference_scope') != 'cross-core instruction makespan':
        raise ValueError('Incompatible timing scope: ' + case['id'])
    if not measurement.get('correctness') or not all('bad=0' in s for s in measurement['correctness']):
        raise ValueError('Missing full correctness evidence: ' + case['id'])
    p = predictor.predict(predictor.case_type(op, shape))
    q = old.predict(old.case_type(op, shape))
    count = p.get('estimated_max_core_selected_opcode_count')
    seen_count = observed_count(op, measurement)
    return {'id': case['id'], 'op': op, 'shape': shape, 'role': role,
            'source': source, 'source_sha256': digest(source),
            'reference_us': ref, 'predicted_us': p['predicted_us'], 'previous_v05_us': q['predicted_us'],
            'constant_us': q['constant_baseline_us'],
            'ape': abs(p['predicted_us']-ref)/ref, 'signed_error_us': p['predicted_us']-ref,
            'api': APIS[op][2], 'library': APIS[op][1],
            'kernel': measurement['kernel'],
            'runtime_binary_sha256': measurement.get('binary_sha256', measurement.get('kernel_binary_sha256')),
            'trace_or_db_sha256': measurement.get('trace_sha256', measurement.get('db_sha256')),
            'reference_scope': measurement['reference_scope'],
            'observed_cores': [measurement['active_aic'], measurement['active_aiv']],
            'predicted_cores': [p['plan']['active_aic'], p['plan']['active_aiv']],
            'predicted_selected_opcode_count': count, 'observed_selected_opcode_count': seen_count,
            'opcode_ape': abs(count-seen_count)/seen_count if count is not None and seen_count and measurement.get('rv_count_comparable_to_legacy',True) else None,
            'opcode_comparison_scope':measurement.get('rv_count_scope','same legacy CSV event-count convention'),
            'correctness': measurement.get('correctness', []), 'identity': identity or {},
            'cost': cost or {}, 'prediction_coverage': p.get('coverage'),
            'tiling': {k:v for k,v in p['plan'].items() if k in ('tile','branch','waves','iterations','active_aic','active_aiv')}}


def build(version):
    module = importlib.import_module('operator_model.model_v' + version[-1])
    current_path = f'results/{version}/model.json'
    p = module.Predictor(ROOT/current_path); p.case_type = module.Case
    old_module = importlib.import_module('operator_model.model_v5')
    old = old_module.Predictor(ROOT/'results/v0.5/model.json'); old.case_type = old_module.Case
    evaluation = read(f'results/{version}/evaluation.json')
    if evaluation['model_sha256'] != digest(current_path):
        raise ValueError('Evaluation was generated for a different model')
    rows = []
    source = 'results/v0.4/dataset.json'
    for r in read(source)['records']:
        if r['status'] != 'ok':
            continue
        # v0.6 quant selection can use all 19 old quant points. Old development
        # remains conservatively labelled development even for unchanged paths.
        role = 'development_replay'
        if r['case']['id'] == 'f4_test_batch4':
            role = 'retained_frozen_holdout'
        if version == 'v0.5' and r['case']['id'].startswith('q4_'):
            role = 'retained_frozen_holdout'
        if role == 'retained_frozen_holdout':
            archived = next(x for x in read('results/v0.4/evaluation.json')['rows'] if x['id']==r['case']['id'])
            if abs(p.predict(p.case_type(r['case']['op'], r['case']['shape']))['predicted_us']-archived['predicted_us']) > 1e-10:
                raise ValueError('Changed prediction cannot retain an old frozen role')
        rows.append(normalize(r['case'], r['measurement'], role, source, p, old, r['identity']))
    source = 'results/long-fa/evaluation.json'
    for r in read(source)['new_capture']:
        if r['shape'] != [9,32768,64,1,1]:
            continue  # Same-shape capture-method checks never count as new cases.
        rows.append(normalize({'op':'fa','shape':r['shape'],'id':r['id']}, r['measurement'],
                              'development_replay', source, p, old))
    for ver in ('v0.5', 'v0.6'):
        source = f'results/{ver}/dataset.json'
        if ver > version or not (ROOT/source).exists():
            continue
        for r in read(source)['records']:
            if r['status'] != 'ok':
                continue
            role = 'new_frozen_holdout' if ver == version else 'retained_frozen_holdout'
            frozen = read(f'results/{ver}/frozen-predictions.json')
            expected = next(x for x in frozen['predictions'] if x['id']==r['case']['id'])
            if r['case']['split'] != 'holdout' or r['case']['shape'] != expected['shape']:
                raise ValueError('Shape or split does not match frozen predictions')
            if r['identity']['model_sha256'] != frozen['model_sha256']:
                raise ValueError('Collected model identity does not match freeze')
            if abs(p.predict(p.case_type(r['case']['op'], r['case']['shape']))['predicted_us']-expected['prediction']['predicted_us']) > 1e-10:
                raise ValueError('Current prediction differs from independently frozen value')
            rows.append(normalize(r['case'], r['measurement'], role, source, p, old, r.get('identity'),
                                  {k:r[k] for k in ('wall_seconds','cpu_seconds','raw_bytes') if k in r}))
    seen = [(r['op'], tuple(r['shape'])) for r in rows]
    if len(set(seen)) != len(seen):
        raise ValueError('Duplicate completed op/shape; explicit de-duplication required')
    invocations = read('docs/extensions/smoke-950dt.json')['records']
    for r in invocations:
        if r['status'] != 'complete_single_kernel':
            raise ValueError('Invocation evidence is incomplete')
        m = r['measurement']; op = r['op']
        shape = [int(x) for x in r['arguments'][1:] if str(x).isdigit()]
        rows.append({'id':'invocation_'+op, 'op':op, 'shape':shape, 'role':'invocation_only',
            'source':'docs/extensions/smoke-950dt.json','source_sha256':digest('docs/extensions/smoke-950dt.json'),
            'reference_us':m['reference_us'],'predicted_us':None,'previous_v05_us':None,'constant_us':None,
            'ape':None,'signed_error_us':None,'api':APIS[op][2],'library':APIS[op][1],
            'kernel':m['kernel'],'runtime_binary_sha256':None,'trace_or_db_sha256':m['db_sha256'],
            'reference_scope':m['reference_scope'],'observed_cores':[m['active_aic'],m['active_aiv']],
            'predicted_cores':None,'predicted_selected_opcode_count':None,'observed_selected_opcode_count':m['max_core_mmad'],
            'opcode_ape':None,'correctness':r['correctness'],'identity':{k:r[k] for k in ('soc_version','cann_version','host_binary_sha256','host_source_sha256','sku_adapter_sha256')},
            'cost':{k:r[k] for k in ('wall_seconds','cpu_seconds','raw_bytes')},'attributes':{'dtype':'bf16',**({'self_shape':[16],'alpha':1,'beta':1,'cube_math_type':0} if op=='addmm' else {'epsilon':1e-6})}})
    summaries = []
    for op, (name, library, api) in APIS.items():
        group = [r for r in rows if r['op']==op]
        dev = [r for r in group if r['role']=='development_replay']
        independent = [r for r in group if 'frozen_holdout' in r['role']]
        summaries.append({'op':op,'name':name,'library':library,'api':api,'complete_unique_shapes':len(group),
            'development':metrics(dev),'independent':metrics(independent),
            'independent_previous_v05':metrics(independent,'previous_v05_us'),
            'independent_constant':metrics(independent,'constant_us'),
            'status':'calibrated_provisional' if op in ('quant','matmul','fa') else 'invocation_only_uncalibrated'})
    for op,name,api,status in EXTENSIONS:
        summaries.append({'op':op,'name':name,'library':'ops-transformer','api':api,'complete_unique_shapes':0,
            'development':None,'independent':None,'independent_previous_v05':None,'independent_constant':None,
            'status':status})
    failures = []
    for ver in ('v0.4','v0.5','v0.6'):
        file = f'results/{ver}/dataset.json'
        if ver > version or not (ROOT/file).exists():continue
        for r in read(file)['records']:
            if r['status']=='ok':continue
            # Imported failed records are not automatically merged across rounds.
            if ver=='v0.4' and not r['case']['id'].startswith(('q4_','f4_')):continue
            failures.append({'id':r['case']['id'],'op':r['case']['op'],'shape':r['case']['shape'],
                'status':r['status'],'reference_us':None,'error':r.get('error',r.get('stop_reason','See archived capture log')),
                'wall_seconds':r.get('wall_seconds'),'source':file})
    for r in read('results/long-fa/capture-attempts.json'):
        if r['accepted_timing']:continue
        failures.append({'id':r['name'],'op':'fa','shape':r['shape'],'status':r['status'],
            'reference_us':None,'error':r.get('error','Capture or identity qualification did not pass'),
            'wall_seconds':r.get('wall_seconds'),'source':'results/long-fa/capture-attempts.json'})
    costs=[]
    for ver in ('v0.1','v0.2','v0.3','v0.4','long-fa','v0.5','v0.6'):
        if ver.startswith('v') and ver>version:continue
        file=f'results/{ver}/evaluation.json'
        if not (ROOT/file).exists():continue
        d=read(file);c=d.get('cost',d.get('collection_cost',{}))
        costs.append({'round':ver,'source':file,**c})
    costs.append({'round':'extension_invocations','source':'docs/extensions/smoke-950dt.json',**read('docs/extensions/smoke-950dt.json')['cost']})
    benchmark=read(f'results/{version}/evaluation.json').get('benchmark',{})
    data={'model_version':version,'model_sha256':digest(current_path),'model_path':current_path,
          'scope':'Cumulative unique completed op/shape with explicit role; not pooled independent validation',
          'rows':rows,'operator_summary':summaries,'failed_or_unqualified':failures,'collection_cost_by_round':costs,
          'benchmark':benchmark,'role_counts':dict(Counter(r['role'] for r in rows))}
    return data


def render(d):
    version=d['model_version'];rows=d['rows'];ss=d['operator_summary']
    summary=[]
    for s in ss:
        a=s['development'];b=s['independent'];c=s['independent_constant']
        status=s['status']
        if status=='calibrated_provisional':status='已有局部校准；域外仍需验证'
        if status=='invocation_only_uncalibrated':status='仅1个库调用验证；无耗时模型'
        summary.append([s['name'],s['complete_unique_shapes'],a['n'] if a else 0,pct(a['wape']) if a else 'N/A',
                        b['n'] if b else 0,pct(b['wape']) if b else 'N/A',pct(b['max_ape']) if b else 'N/A',
                        pct(c['wape']) if c else 'N/A',status])
    texts=['# 当前整套算子建模：误差、证据与成本\n','## Summary：逐算子误差情况\n',
        table(['算子','完整CA唯一shape','开发回放n','开发WAPE','冻结验证n','冻结验证WAPE','验证最大APE','同集固定基线WAPE','当前状态'],summary),
        f'当前模型 **{version}**。此表覆盖全部已接入算子族，而非只统计最新一轮。N/A表示没有可计算的误差，**不表示0误差**。完整CA数包含仅调用验证的RMSNorm/Addmm；组合入口共享组成算子，不累计为额外独立实测。\n',
        '开发回放用于说明模型对已见数据的拟合情况；冻结验证仅含预测冻结后采集、且未被当前版本用作参数拟合或模型选择的shape。保守地将旧开发池保留为开发；原来是留出的shape一旦被本轮用作开发，立即改变角色。不同算子不合成一个WAPE，避免长FA把其他算子的误差掩盖。\n',
        '总方法、公式与适用域见 [整体建模方法](modeling-strategy.md)；机器可读逐点证据见 [当前数据快照](../results/current/evaluation.json)。本页由 `python3 scripts/report-current.py` 从已提交聚合结果生成，不启动CA。每轮方法/误差变化单独存放，历史报告保持不变。\n',
        '![全部已完成shape：开发与冻结验证分开](../results/current/all-operators.png)\n',
        '## 1. 统一口径、版本与数据角色\n',
        '- 参考平台为CANN9.2.1 / Ascend950DT_9582。预测目标是实际执行核共同时间轴上的完整指令窗口，不是主机CA运行墙钟，也不是整个ACLNN调用的端到端时间。多kernel调用必须另核对跨launch时钟。\n'
        '- APE=|预测−实测|/实测；WAPE=Σ|预测−实测|/Σ实测；同时给出最大APE。小样本、通常每点一次，没有跨运行统计置信区间。\n'
        '- quant是BF16→FP8 E4M3动态MX量化；matmul是BF16稠密ND；FA是BF16/SBH、无mask/dropout的普通非因果代理。它们不代表所有量化格式、布局、真实模型注意力或vLLM dispatch。\n'
        '- compact DB未重新导出运行时ELF时，binary SHA明确为null。安装包对象SHA不能代替每次运行的二进制身份；entry名也未必全域唯一。\n'
        '- 数值检查通过与耗时误差达标是不同证据；未校准结构模型不以FLOPs/峰值冒充预测耗时或利用率。\n',
        table(['当前数据角色','点数'],d['role_counts'].items()),
        '### 历史独立验证保留记录\n',
        historical_table(),
        '上表是当时冻结模型的独立结果，仅保留历史事实。不能与当前不同样本集直接相减宣称收益；当前新旧版本对比见下文同集表。Matmul本轮没有新增独立CA，不能把开发回放当作新验证。\n',
        '## 2. 当前模型与旧版、固定基线的同集对比\n',
        table(['算子','当前独立n','当前WAPE','v0.5同集WAPE','固定同集WAPE','数据范围'],[
            [s['name'],s['independent']['n'],pct(s['independent']['wape']),pct(s['independent_previous_v05']['wape']),pct(s['independent_constant']['wape']),
             '当前仍未参与开发的冻结shape；见逐点角色'] for s in ss if s['independent']]),
        '固定耗时沿用v0.4/v0.5旧校准池的每算子均值，本轮不重算。小shape启动/标量开销占比较大且分核可吸收工作增长时，常数可能胜过复杂模型；长FA的单核循环随KV增长，常数会系统性低估。不能先看到留出结果再重新选常数或修正参数。\n',
        '## 3. 全部已完成唯一shape：耗时、误差和实际库实现\n']
    for op,(name,library,api) in APIS.items():
        group=[r for r in rows if r['op']==op]
        texts += [f'### {name}\n',f'实际接口：`{api}`；安装库组件：**{library}**。shape约定：'+
                  {'quant':'[R,C]','matmul':'[M,N,K]','fa':'[Q,KV,D,B,H]','rmsnorm':'[R,C]，gamma[C]','addmm':'[M,N,K]，self=[N]、alpha=beta=1'}[op]+'。\n',
            table(['ID / shape','角色','CA µs','当前预测 µs','APE','v0.5 µs','固定 µs','实际kernel entry'],[
                [r['id']+' '+str(r['shape']),r['role'],fmt(r['reference_us']),fmt(r['predicted_us']),pct(r['ape']),fmt(r['previous_v05_us']),fmt(r['constant_us']),r['kernel']] for r in group]),
            table(['ID','预测/实际 AIC,AIV','预测/实际所选指令数','指令APE','运行时二进制SHA256','trace/DB SHA256'],[
                [r['id'],str(r['predicted_cores'])+' / '+str(r['observed_cores']),
                 fmt(r['predicted_selected_opcode_count'],2)+' / '+fmt(r['observed_selected_opcode_count'],0),pct(r['opcode_ape']),
                 r['runtime_binary_sha256'] or '未导出',r['trace_or_db_sha256']] for r in group]),
            diagnoses(op,d),
            worst_cases(group),
            ('quant指令表：旧CSV日志事件预测与新DB执行记录不是同一计数口径，新点指令APE为N/A；原始数值分别保留。\n' if op=='quant' else '')+
            '数值检查：'+('; '.join(sorted({x for r in group for x in r['correctness']})))+'。\n']
    texts += ['## 4. 已接入但尚无时延误差的算子\n',
        table(['算子','950库接口/组合','状态与主要误差缺口'],[[name,api,status] for _,name,api,status in EXTENSIONS]),
        '这些入口返回结构工作量及证据支持的shape约束，预测耗时和利用率为null。具体数学公式、dtype/layout/分组约束及候选二进制清单见 [扩展算子目录](extensions/README.md)。RMSNorm与Addmm虽各有一次小case完整实测，仍缺跨shape校准及冻结验证；其误差同样是N/A。\n',
        '## 5. 失败、部分trace与重复采集\n',
        table(['ID / shape','状态','墙钟s','排除原因'],[
            [r['id']+' '+str(r['shape']),r['status'],fmt(r['wall_seconds'],2),r['error']] for r in d['failed_or_unqualified']]),
        '失败或身份不匹配的记录没有CA误差值。相同旧shape用于采集方式一致性校验的重复运行不增加唯一shape样本数；其成本仍记录在对应轮次。早期重试和重复样本明细保留在历史dataset/evaluation，当前表列出v0.4以后相关失败与采集资格问题。\n',
        '## 6. 仿真采集成本与在线预测效率\n',
        table(['轮次','尝试次数','墙钟分钟','CPU小时','保留MiB','口径'],[
            [c['round'],c.get('attempts','N/A'),fmt(c.get('wall_seconds',0)/60,2),
             fmt(c.get('cpu_seconds',c.get('recorded_cpu_seconds_lower_bound',0))/3600,2),
             fmt(c.get('raw_bytes',c.get('retained_raw_bytes',0))/2**20,2),
             'CPU仅记录下界' if 'recorded_cpu_seconds_lower_bound' in c or c.get('cpu_is_lower_bound') else '历史采集聚合，包含该轮失败/重试'] for c in d['collection_cost_by_round']]),
        '本轮quant三个原始采集记录因正确性日志字段不同而被旧提取器拒绝；CA进程和逐字节检查均已成功。离线qualifier校验quant_elements/scale_elements/bad=0、日志和DB哈希后另存qualified-result，原记录不覆盖，没有新增CA。恢复审计见v0.6 dataset的recovery字段。\n',
        '每轮列自己的新增采集成本，复用历史数据不再次计费。保留字节数不是运行中峰值临时空间；线程并发会使CPU时间远大于墙钟。轻量SQL归约节省trace转换与存储，不能消除CA指令执行成本。\n',
        f'当前版本查询基准（模型预加载、未启动CANN；单位和样本数以该轮evaluation为准）：\n\n```json\n{json.dumps(d["benchmark"],ensure_ascii=False,indent=2)}\n```\n',
        '长FA公式查询对KV循环数为O(1)；quant完整诊断路径按UB tile数展开，当前多UB标量路径仅保留两轮状态、临时空间O(1)，scalar-only优化及边界详见总方法。缓存命中与未缓存查询分别报告，不以缓存成绩掩盖首次预测开销。\n',
        '## 7. 场景外推与当前整体边界\n',
        extrapolation_table(d),
        '用户场景包括prefill单卡batch=1/4/16、decode=1/4/16/32/128/256，32k/128k/256k/512k输入、1k输出、推测8个时Q=9。已建立场景映射，但表中的少量代表shape远未覆盖全部batch/TP/EP、prefill chunk、mask和量化组合。配置能给候选shape，运行时分发、专家负载及融合路径还需实际调用证据。\n',
        '256k/512k没有完整CA的项仍是外推；A7/A8、独立ISA峰值利用率和整网系统误差尚未验证。整体误差表中的N/A和这些外推边界应随版本持续保留，直到相应证据补齐。\n',
        f'当前模型SHA256：`{d["model_sha256"]}`。各逐点source SHA、数值检查、身份与分核信息见 [evaluation.json](../results/current/evaluation.json)，当前方法见 [modeling-strategy.md](modeling-strategy.md)。\n']
    (ROOT/'docs/error-analysis.md').write_text('\n'.join(texts))


def diagnoses(op,d):
    if op=='quant':
        prefix='当前quant仍采用v0.4饱和小shape控制回退。' if d['model_version']=='v0.5' else '当前quant的修正与参数见总方法；19个旧quant点均归入开发诊断，不再称独立留出。'
        return prefix+'本轮冻结验证WAPE基本持平、最大APE下降，不能宣称全域精度改善；full诊断接口也没有整体提速。新compact quant的RV数是ExecutedInstructions记录，而历史CSV把成对日志事件都计入；跨口径计数APE标N/A，不把观测值乘2。已证实：旧trace可区分UB搬入、VF、双缓冲复用和写回；所选RV_*计数包括load/store/control，并非纯算术指令。剩余误差来自控制成本、分核策略、UB边界与流水近似的组合；在逐点trace证据不足时，不能把所有残差都归因于访存。\n'
    if op=='matmul':
        return '当前仍是已校准资源服务时间加分支残差；分核tile是源码启发式，并非完整库tiling移植。M/N/K尾块、GEMV路径、粗tile与启动成本可改变误差；无新独立CA。本表指令数是最大核MMAD，不代表指令形状、利用率或总FLOPs全都正确。\n'
    if op=='fa':
        return '长Q9/D64/B1/H1路径从旧32k DB分离首轮读延迟与稳态II；32k因此属于开发。128k/16k+1是先冻结后采集的两点，未重新拟合。短KV、多head/batch、D128仍走旧模型；其更高误差不能被长FA较低WAPE掩盖。尾块按完整256 tile计时是保守近似；尚不能把小幅残差唯一归因为尾块、噪声或同步。\n'
    if op=='rmsnorm':
        return '该小case实测0 AIC/1 AIV；entry 0x1388=5000与NORMAL源码候选key一致，但不能推断其他shape都走同一路径。尚无耗时模型，后续需覆盖归约长度、UB/分核切换及多核尾波。\n'
    return '该bias小case观察到1 AIC/0 AIV、1条MMAD、一个kernel；entry_kernel_2不是唯一二进制身份。完整矩阵self、不同alpha/beta/dtype可能采用其他融合或多kernel路径，不能将普通Matmul时延直接当作Addmm总时延。\n'


def worst_cases(group):
    eligible = [r for r in group if r['ape'] is not None]
    if not eligible:
        return '尚无逐shape预测值，因此没有残差排序；单点实测不能估计泛化误差。\n'
    worst = sorted(eligible, key=lambda r:r['ape'], reverse=True)[:3]
    text = '当前该算子最大残差点（包含开发回放，不能作为独立准确率）：\n\n'
    text += table(['shape / 角色','有符号偏差 µs','APE','预测分支 / tile','核数吻合','指令计数证据'],[
        [str(r['shape'])+' / '+r['role'],f"{r['signed_error_us']:+.4f}",pct(r['ape']),
         str(r['tiling'].get('branch'))+' / '+str(r['tiling'].get('tile')),
         '是' if r['predicted_cores']==r['observed_cores'] else '否',
         'N/A' if r['opcode_ape'] is None else 'APE '+pct(r['opcode_ape'])] for r in worst])
    return text+'正偏差为高估、负偏差为低估。核数/指令吻合只能排除部分工作划分错误，不能证明启动、等待和流水重叠时序正确。\n'


def historical_table():
    result=[]
    for ver in ('v0.2','v0.3'):
        data=read(f'results/{ver}/evaluation.json')['metrics']['holdout']
        for op in ('quant','matmul','fa'):
            metric=data.get(op,{}).get('predicted_us')
            if metric:result.append([ver,APIS[op][0],metric['count'],pct(metric['wape']),pct(metric['max_ape'])])
    for op in ('quant','fa'):
        m=read('results/v0.4/evaluation.json')['metrics'][op]['predicted_us']
        result.append(['v0.4',APIS[op][0],m['count'],pct(m['wape']),pct(m['max_ape'])])
    m=read('results/v0.5/evaluation.json')['metrics']['v05_us']
    result.append(['v0.5','FA新长循环',m['n'],pct(m['wape']),pct(m['max_ape'])])
    return table(['冻结版本','算子','当时独立n','WAPE','最大APE'],result)


def extrapolation_table(d):
    # Do not label new extrapolations as CA measurements based on formula availability.
    module=importlib.import_module('operator_model.model_v'+d['model_version'][-1])
    predictor=module.Predictor(ROOT/d['model_path']);rs=[]
    for length in (32768,131072,132096,262144,263168,524288,525312):
        shape=[9,length,64,1,1]
        found=next((r for r in d['rows'] if r['op']=='fa' and r['shape']==shape),None)
        rs.append([shape,fmt(predictor.predict(module.Case('fa',shape))['predicted_us']),
                   fmt(found['reference_us']) if found else '未实测',found['role'] if found else 'unvalidated_extrapolation'])
    return table(['普通FA [Q,KV,D,B,H]','当前预测 µs','CA µs','数据角色'],rs)


def plot(d):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'build/matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    colors={'development_replay':'#929292','retained_frozen_holdout':'#367ca8','new_frozen_holdout':'#288b70'}
    for ax,op in zip(axes,('quant','matmul','fa')):
        group=[r for r in d['rows'] if r['op']==op]
        for role,color in colors.items():
            rs=[r for r in group if r['role']==role]
            if rs:ax.scatter([r['reference_us'] for r in rs],[r['predicted_us'] for r in rs],color=color,label=role.replace('_',' '),s=35)
        lo=min(min(r['reference_us'],r['predicted_us']) for r in group)*.9
        hi=max(max(r['reference_us'],r['predicted_us']) for r in group)*1.1
        ax.plot([lo,hi],[lo,hi],':',color='black',label='Exact agreement')
        if op=='fa':ax.set(xscale='log',yscale='log')
        ax.set(title=APIS[op][0],xlabel='Complete CA makespan (us)',ylabel='Current prediction (us)')
        ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.suptitle(d['model_version']+' cumulative coverage; development is not independent validation')
    for ext in ('png','svg'):fig.savefig(OUT/f'all-operators.{ext}',dpi=170)
    plt.close(fig)
    for p in OUT.glob('*.svg'):p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',choices=['v0.5','v0.6']);parser.add_argument('--no-plots',action='store_true');args=parser.parse_args()
    version=args.version or ('v0.6' if (ROOT/'results/v0.6/evaluation.json').exists() else 'v0.5')
    OUT.mkdir(exist_ok=True)
    data=build(version);write('evaluation.json',data);render(data)
    if not args.no_plots:plot(data)
    print(json.dumps({'version':version,'rows':len(data['rows']),'roles':data['role_counts']},indent=2))

if __name__=='__main__':main()
