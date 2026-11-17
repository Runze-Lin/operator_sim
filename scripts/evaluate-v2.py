#!/usr/bin/env python3
"""Expanded held-out evaluation, including coverage, utilization and failures."""
from datetime import datetime
import json
import math
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.model_v2 import Case,Predictor
from operator_model.model import Case as OldCase,Predictor as OldPredictor

utilities=runpy.run_path(str(ROOT/'scripts/evaluate-v1.py'))
metrics,benchmark=utilities['metrics'],utilities['benchmark']


def replay_equal(actual, expected):
    """Allow last-bit float changes across Python sum implementations only."""
    if isinstance(actual,dict) and isinstance(expected,dict):
        return actual.keys()==expected.keys() and all(replay_equal(actual[k],expected[k]) for k in actual)
    if isinstance(actual,(list,tuple)) and isinstance(expected,(list,tuple)):
        return len(actual)==len(expected) and all(replay_equal(a,b) for a,b in zip(actual,expected))
    if type(actual) is float and type(expected) is float:
        return math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12)
    return type(actual) is type(expected) and actual==expected


def validate_dataset(data, frozen, model):
    """Reject mixing runs, changing inputs or silently replacing training data."""
    if data['manifest_sha256'] != frozen['manifest_sha256']:
        raise ValueError('Dataset manifest differs from frozen manifest')
    records=data['records']
    if len({r['case']['id'] for r in records}) != len(records):
        raise ValueError('Duplicate selected cases')
    training=[r for r in records if r['case']['split']=='calibration' and r['status']=='ok']
    if {r['cache_key'] for r in training} != set(model['training_keys']):
        raise ValueError('Training records changed after freezing')
    identity_keys=('soc_version','cann_version','package_manifest_sha256','platform_sha256','sim_config_sha256')
    if len({tuple(r['identity'][k] for k in identity_keys) for r in records}) != 1:
        raise ValueError('Incompatible CA environments')
    if len({r['identity']['host_binary_sha256'] for r in records if 'imported_from' not in r}) != 1:
        raise ValueError('Host caller changed within the new round')
    expected={p['id']:p for p in frozen['predictions']}
    for r in records:
        p=expected.get(r['case']['id'])
        if p is None or any(r['case'][k]!=p[k] for k in ('op','shape','split')):
            raise ValueError('Case differs from frozen input')


def main():
    out=ROOT/'results/v0.2';spec=json.loads((ROOT/'configs/sampling-v2.json').read_text())
    frozen=json.loads((out/'frozen-predictions.json').read_text())
    for path,key in [(out/'model.json','model_sha256'),(ROOT/'configs/sampling-v2.json','manifest_sha256'),
                     (ROOT/'operator_model/model_v2.py','implementation_sha256'),(ROOT/'operator_model/model.py','shared_implementation_sha256')]:
        if sha256(path)!=frozen[key]:raise SystemExit('Frozen artifact changed: '+str(path))
    data=json.loads((out/'dataset.json').read_text());records={r['case']['id']:r for r in data['records']}
    if len(records)!=len(data['records']):raise SystemExit('Duplicate selected cases')
    model=Predictor(out/'model.json');old=OldPredictor(ROOT/'results/v0.1/model.json')
    validate_dataset(data,frozen,model.model)
    rows=[];missing=[]
    for item in frozen['predictions']:
        r=records.get(item['id'])
        if r is None or r['status']!='ok':
            missing.append({'id':item['id'],'split':item['split'],'status':r['status'] if r else 'missing'})
            continue
        p=item['prediction'];c=Case(item['op'],item['shape']);m=r['measurement']
        replay=json.loads(json.dumps(model.predict(c)))
        if not replay_equal(replay,p):raise SystemExit('Prediction replay mismatch')
        if item['split']=='holdout' and datetime.fromisoformat(r['finished_utc'])<=datetime.fromisoformat(frozen['frozen_utc']):raise SystemExit('Holdout leakage')
        if item['op']=='quant':observed=max(sum(v for k,v in core['dynamic_opcodes'].items() if k.startswith('RV_')) for core in m['cores'])
        else:observed=max(core['dynamic_opcodes'].get('MMAD',0) for core in m['cores'])
        row={k:item[k] for k in ('id','op','shape','split')}
        row.update({k:p[k] for k in ('predicted_us','coverage','constant_baseline_us','empirical_roofline_us','uncorrected_schedule_us')})
        row.update({'reference_us':m['reference_us'],'ape':abs(p['predicted_us']/m['reference_us']-1),
                    'branch':p['plan']['branch'],'calibration_regime_count':p['calibration_regime_count'],
                    'predicted_cores':[p['plan']['active_aic'],p['plan']['active_aiv']],
                    'observed_cores':[m['active_aic'],m['active_aiv']],
                    'kernel_seen_in_regime':m['kernel'] in p['expected_kernel_candidates'],
                    'predicted_selected_opcode_count':p['estimated_max_core_selected_opcode_count'],'observed_selected_opcode_count':observed,
                    'kernel':m['kernel'],'matrix_instruction_shapes':m['cores'][0]['matrix_instruction_shapes'],
                    'mmad_calls_all_cores':sum(core['dynamic_opcodes'].get('MMAD',0) for core in m['cores']),
                    'mmad_calls_with_shape_detail':sum(s['count'] for core in m['cores'] for s in core['matrix_instruction_shapes']),
                    'max_core_cube_busy_fraction':max(core['busy_fraction_by_pipe'].get('CUBE',0) for core in m['cores']),
                    'max_core_vector_family_busy_fraction':max(core['busy_fraction_by_pipe'].get('VECTOR_FAMILY',0) for core in m['cores']),
                    'aic_coverage':m['active_aic']/32,'aiv_coverage':m['active_aiv']/64})
        row['mapping_matches']=row['predicted_cores']==row['observed_cores']
        try:
            shape=c.shape[:3] if c.op=='fa' and c.shape[3:]==(1,1) else c.shape
            row['v01_prediction_us']=old.predict(OldCase(c.op,shape))['predicted_us']
        except ValueError:row['v01_prediction_us']=None
        rows.append(row)
    aggregate={}
    for split in ('calibration','holdout'):
        aggregate[split]={}
        for op in ('all','quant','matmul','fa'):
            rs=[r for r in rows if r['split']==split and (op=='all' or r['op']==op)]
            aggregate[split][op]={field:metrics(rs,field) for field in ('predicted_us','constant_baseline_us','empirical_roofline_us','uncorrected_schedule_us')}
            comparable=[r for r in rs if r['v01_prediction_us'] is not None]
            aggregate[split][op]['v01_comparable_only']=metrics(comparable,'v01_prediction_us')
    speed=benchmark(out/'model.json',[Case(r['op'],r['shape']) for r in rows],Predictor,Case,'operator_model.model_v2')
    attempts=data.get('attempts',[])
    cost={'attempts':len(attempts),'successful_attempts':sum(r['status']=='ok' for r in attempts),
          'reused_v01_shapes':sum('imported_from' in r for r in records.values()),
          'wall_seconds':sum(r['wall_seconds'] for r in attempts),
          'cpu_seconds':sum(r['cpu_seconds'] for r in attempts),'raw_bytes':sum(r['raw_bytes'] for r in attempts)}
    initial=json.loads((ROOT/'results/v0.1/evaluation.json').read_text())['collection_cost']
    combined={k:cost[k]+initial[k] for k in ('attempts','wall_seconds','cpu_seconds','raw_bytes')}
    gates={}
    for op in ('all','quant','matmul','fa'):
        target=sum(c['split']=='holdout' and (op=='all' or c['op']==op) for c in spec['cases'])
        m=aggregate['holdout'][op]['predicted_us']
        gates[op]=bool(m and m['count']==target and m['wape']<=spec['acceptance']['holdout_wape'] and m['p90_ape']<=spec['acceptance']['holdout_p90_ape'])
    gates['speed']=speed['uncached']['p95_us']<=1000 and speed['cached']['p95_us']<=10
    held=[r for r in rows if r['split']=='holdout']
    diagnostics={'heldout_mapping_matches':sum(r['mapping_matches'] for r in held),
                 'heldout_mapping_count':len(held),'selected_opcode_wape':{},
                 'partial_matrix_detail_cases':[r['id'] for r in rows if r['mmad_calls_with_shape_detail']<r['mmad_calls_all_cores']]}
    for op in ('quant','matmul','fa'):
        rs=[r for r in held if r['op']==op]
        total=sum(r['observed_selected_opcode_count'] for r in rs)
        diagnostics['selected_opcode_wape'][op]=(sum(abs(r['predicted_selected_opcode_count']-r['observed_selected_opcode_count']) for r in rs)/total if total else None)
    result={'model_sha256':frozen['model_sha256'],'frozen_utc':frozen['frozen_utc'],'rows':rows,'metrics':aggregate,
            'gates':gates,'benchmark':speed,'missing':missing,'collection_cost':cost,'combined_cost':combined,
            'diagnostics':diagnostics,
            'failed_attempts':[r for r in attempts if r['status']!='ok'],
            'kernel_counts':{op:len({r['kernel'] for r in rows if r['op']==op}) for op in ('quant','matmul','fa')}}
    (out/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# v0.2 分支与边界扩展结果','',
           '本轮复用 v0.1 的 15 个不同输入作为校准；本轮留出点全部是新 shape，预测在采集前冻结。不能将 v0.1 旧留出点在本轮继续算作独立测试。','',
           '| 算子 | 新留出数 | WAPE | P90 APE | 固定耗时 WAPE | 经验 Roofline WAPE | 达标 |',
           '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for op in ('quant','matmul','fa','all'):
        a=aggregate['holdout'][op];m=a['predicted_us']
        if m:lines.append(f"| {op} | {m['count']} | {m['wape']:.2%} | {m['p90_ape']:.2%} | {a['constant_baseline_us']['wape']:.2%} | {a['empirical_roofline_us']['wape']:.2%} | {gates[op]} |")
    lines+=['','达标目标为 WAPE≤10%、P90 APE≤20%。均匀权重；经验 Roofline 非理论峰值、非用户原始实现。','',
           '| 新留出 shape | CA μs | 预测 μs | APE | 预测/实际 AIC,AIV | 校准同机制点数 | 覆盖 |',
           '| --- | ---: | ---: | ---: | --- | ---: | --- |']
    for r in rows:
        if r['split']=='holdout':lines.append(f"| {r['op']} {r['shape']} | {r['reference_us']:.4f} | {r['predicted_us']:.4f} | {r['ape']:.2%} | {r['predicted_cores']} / {r['observed_cores']} | {r['calibration_regime_count']} | {r['coverage']} |")
    lines+=['','## 分支、分核与指令证据','',
            'kernel 标识相同不表示 tile 相同；保留实际 MMAD 的 m/n/k 和次数。自定义的 `branch` 是模型中的机制分组，不冒充已解码的 CANN tilingKey。','',
            '| Case | 分组 | 核数匹配 | 最大核选定指令数：预测/实际 | Cube busy | Vector family busy |',
            '| --- | --- | --- | --- | ---: | ---: |']
    for r in rows:
        if r['id'].startswith('import_'):continue
        lines.append(f"| {r['id']} | {r['branch']} | {r['mapping_matches']} | {r['predicted_selected_opcode_count']:.1f}/{r['observed_selected_opcode_count']} | {r['max_core_cube_busy_fraction']:.2%} | {r['max_core_vector_family_busy_fraction']:.2%} |")
    lines+=['','busy 为非同步指令区间并集/跨核指令时间窗，Vector family 包含 RegBase load/store/control；这是占用观察，不是峰值算力利用率。稀疏训练分组保持 provisional；不把低时延误差当作准确分核/指令模型的证明。','',
            f"新留出核数匹配 {diagnostics['heldout_mapping_matches']}/{diagnostics['heldout_mapping_count']}；选定指令次数 WAPE："+', '.join(f"{op}={value:.2%}" for op,value in diagnostics['selected_opcode_wape'].items() if value is not None)+'。','',
            '部分 MMAD trace 事件没有 m/n/k 字段；指令总数使用 CSV call_count，维度列表仅代表带完整 detail 的事件，不能用它推导完整执行 FLOPs。存在该情况的 case：'+', '.join(diagnostics['partial_matrix_detail_cases'])+'。','',
            '## 开销与超时','',
            f"新增采集尝试 {cost['attempts']} 次，成功 {cost['successful_attempts']} 次，复用旧 shape {cost['reused_v01_shapes']} 个；本轮墙钟 {cost['wall_seconds']/60:.2f} 分钟、CPU {cost['cpu_seconds']/3600:.2f} 小时、生成原始数据 {cost['raw_bytes']/2**30:.2f} GiB。",'',
            f"两轮合计墙钟 {combined['wall_seconds']/60:.2f} 分钟、CPU {combined['cpu_seconds']/3600:.2f} 小时。CPU 是累计计算资源，不是用户等待时间。",'',
            f"在线未缓存 P50/P95 {speed['uncached']['p50_us']:.2f}/{speed['uncached']['p95_us']:.2f} μs；缓存 P50/P95 {speed['cached']['p50_us']:.2f}/{speed['cached']['p95_us']:.2f} μs；批量 {speed['batch_queries_per_second']:,.0f} 次/秒。",'',
            '失败尝试计入开销，不作为设备耗时样本；慢例重试保留独立目录。缺失项：'+json.dumps(missing,ensure_ascii=False),'']
    for r in result['failed_attempts']:lines.append(f"- {r['case']['id']} attempt={r['attempt']}：{r['status']}，墙钟 {r['wall_seconds']:.1f}s；{r.get('error') or '见原始日志'}。")
    lines+=['','详细归因与未完成项见 [误差分析](../../docs/error-analysis-v2.md)。','',
            '## 结论边界','',
            '- 本轮覆盖 BF16 普通 MatMul、尾轴 block32 FP8 quant、BF16 标准 FA；没有覆盖所有 dtype、layout、mask、量化矩阵、稀疏/MLA/KDA 和所有算法分支。',
            '- 分组只有 1–2 个校准点时不能识别全部成本参数；使用共享阶段成本与收缩残差，并显式标记 provisional。',
            '- CA 配置、时钟与版本仍固定为 950DT/CANN9.2.1。没有将系数直接迁移到 A7/A8。',
            '- 本轮新留出结果不用于修改已冻结模型；若未达标，误差保留，下一轮需要新留出点。','']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps({'gates':gates,'cost':cost,'combined_cost':combined,'benchmark':speed},indent=2))


if __name__=='__main__':main()
