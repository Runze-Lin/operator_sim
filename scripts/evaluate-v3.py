#!/usr/bin/env python3
"""Independent frozen evaluation, with timing, structural and budget checks."""
from datetime import datetime,timedelta
import json
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.model_v3 import Case,Predictor
util=runpy.run_path(str(ROOT/'scripts/evaluate-v2.py'))
metrics,benchmark,replay_equal=util['metrics'],util['benchmark'],util['replay_equal']


def validate(data,frozen,model):
    if data['manifest_sha256']!=frozen['manifest_sha256']:raise ValueError('Manifest changed')
    records=data['records']
    if len({r['case']['id'] for r in records})!=len(records):raise ValueError('Duplicate selected cases')
    if {r['cache_key'] for r in records if r['case']['split']=='calibration'}!=set(model['training_keys']):raise ValueError('Training set changed')
    env_keys=('soc_version','cann_version','package_manifest_sha256','platform_sha256','sim_config_sha256')
    if len({tuple(r['identity'][k] for k in env_keys) for r in records})!=1:raise ValueError('Environment changed')
    expected={x['id']:x for x in frozen['predictions']}
    for r in records:
        p=expected.get(r['case']['id'])
        if p is None or any(r['case'][k]!=p[k] for k in ('op','shape','split')):raise ValueError('Input changed')
        if 'imported_from' not in r and r['identity']['host_binary_sha256']!=frozen['artifact_sha256']['build/smoke/three_ops']:
            raise ValueError('Host binary changed')
        if r['case']['split']=='holdout':
            started=datetime.fromisoformat(r['finished_utc'])-timedelta(seconds=r['wall_seconds'])
            if started<=datetime.fromisoformat(frozen['frozen_utc']):raise ValueError('Holdout predates freeze')


def main():
    out=ROOT/'results/v0.3';spec=json.loads((ROOT/'configs/sampling-v3.json').read_text())
    frozen=json.loads((out/'frozen-predictions.json').read_text())
    if sha256(out/'model.json')!=frozen['model_sha256'] or sha256(ROOT/'configs/sampling-v3.json')!=frozen['manifest_sha256']:raise SystemExit('Frozen model or manifest changed')
    for path,expected in frozen['artifact_sha256'].items():
        if path=='build/smoke/three_ops':continue  # Compare recorded binary identity; offline checkout need not install CANN.
        if sha256(ROOT/path)!=expected:raise SystemExit('Frozen artifact changed: '+path)
    data=json.loads((out/'dataset.json').read_text());model=Predictor(out/'model.json')
    validate(data,frozen,model.model)
    selected={r['case']['id']:r for r in data['records']};rows=[];missing=[]
    for item in frozen['predictions']:
        r=selected.get(item['id'])
        if r is None or r['status']!='ok':
            missing.append({'id':item['id'],'split':item['split'],'status':r['status'] if r else 'missing'});continue
        p=item['prediction'];c=Case(item['op'],item['shape']);m=r['measurement']
        if not replay_equal(json.loads(json.dumps(model.predict(c))),p):raise SystemExit('Prediction replay mismatch')
        observed=max(sum(v for k,v in x['dynamic_opcodes'].items() if k.startswith('RV_')) if c.op=='quant' else x['dynamic_opcodes'].get('MMAD',0) for x in m['cores'])
        row={k:item[k] for k in ('id','op','shape','split')}
        row.update({k:p[k] for k in ('predicted_us','previous_v02_us','constant_baseline_us','previous_constant_baseline_us','coverage')})
        row.update(reference_us=m['reference_us'],ape=abs(p['predicted_us']/m['reference_us']-1),
                   predicted_cores=[p['plan']['active_aic'],p['plan']['active_aiv']],observed_cores=[m['active_aic'],m['active_aiv']],
                   predicted_selected_opcode_count=p['estimated_max_core_selected_opcode_count'],observed_selected_opcode_count=observed,
                   phase_schedule=p.get('phase_schedule'),kernel=m['kernel'],kernel_seen_in_candidates=m['kernel'] in p['expected_kernel_candidates'],
                   ca_wall_seconds=r['wall_seconds'],ca_cpu_seconds=r['cpu_seconds'],raw_bytes=r['raw_bytes'])
        row['mapping_matches']=row['predicted_cores']==row['observed_cores'];rows.append(row)
    fields=('predicted_us','previous_v02_us','constant_baseline_us','previous_constant_baseline_us')
    aggregate={split:{op:{field:metrics([r for r in rows if r['split']==split and (op=='all' or r['op']==op)],field) for field in fields} for op in ('all','quant','fa')} for split in ('calibration','holdout')}
    held=[r for r in rows if r['split']=='holdout'];diagnostics={}
    for op in ('quant','fa'):
        rs=[r for r in held if r['op']==op];total=sum(r['observed_selected_opcode_count'] for r in rs)
        diagnostics[op]={'count':len(rs),'mapping_matches':sum(r['mapping_matches'] for r in rs),
          'opcode_count_wape':sum(abs(r['predicted_selected_opcode_count']-r['observed_selected_opcode_count']) for r in rs)/total if total and all(r['predicted_selected_opcode_count'] is not None for r in rs) else None,
          'opcode_counts_exact':bool(rs) and all(r['predicted_selected_opcode_count']==r['observed_selected_opcode_count'] for r in rs)}
    speed=benchmark(out/'model.json',[Case(r['op'],r['shape']) for r in rows],Predictor,Case,'operator_model.model_v3')
    attempts=data['attempts'];cost={'attempts':len(attempts),'successful_attempts':sum(r['status']=='ok' for r in attempts),
        'wall_seconds':sum(r['wall_seconds'] for r in attempts),'cpu_seconds':sum(r['cpu_seconds'] for r in attempts),'raw_bytes':sum(r['raw_bytes'] for r in attempts),'reused_shapes':32}
    old=json.loads((ROOT/'results/v0.2/evaluation.json').read_text())['combined_cost']
    combined={k:cost[k]+old[k] for k in ('attempts','wall_seconds','cpu_seconds','raw_bytes')}
    gates={}
    for op in ('all','quant','fa'):
        m=aggregate['holdout'][op]['predicted_us'];target=sum(op=='all' or c['op']==op for c in spec['cases'])
        gates[op]=bool(m and m['count']==target and m['wape']<=spec['acceptance']['holdout_wape'] and m['p90_ape']<=spec['acceptance']['holdout_p90_ape'])
    gates['fa_counts']=diagnostics['fa']['count']==3 and diagnostics['fa']['opcode_counts_exact']
    gates['mapping']=len(held)==6 and all(r['mapping_matches'] for r in held)
    gates['speed']=speed['uncached']['p95_us']<=spec['acceptance']['uncached_p95_us'] and speed['cached']['p95_us']<=spec['acceptance']['cached_p95_us']
    result={'model_sha256':frozen['model_sha256'],'frozen_utc':frozen['frozen_utc'],'rows':rows,'metrics':aggregate,'diagnostics':diagnostics,
            'gates':gates,'benchmark':speed,'collection_cost':cost,'combined_cost':combined,'missing':missing,'failed_attempts':[r for r in attempts if r['status']!='ok']}
    (out/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# v0.3：quant UB 与 FA 指令修复验证','',
        '复用此前 32 个输入作为开发/校准数据；以下 6 个留出 shape 全新，预测在采集前冻结。计时为 CA 跨核指令窗口，不包含框架/Host 开销。', '',
        '| 算子 | 新留出数 | v0.3 WAPE | v0.2 WAPE（同一新集） | 更新均值基线 WAPE | v0.3 P90 APE | 时延达标 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for op in ('quant','fa','all'):
        a=aggregate['holdout'][op];m=a['predicted_us']
        if m:lines.append(f"| {op} | {m['count']} | {m['wape']:.2%} | {a['previous_v02_us']['wape']:.2%} | {a['constant_baseline_us']['wape']:.2%} | {m['p90_ape']:.2%} | {gates[op]} |")
    lines+=['','目标：WAPE≤10%、P90 APE≤20%；更新均值基线仅使用旧 32 个校准输入。FA 时延完全保留 v0.2，改善目标是指令次数。matmul 本轮未重新采样。','',
            '| 新 shape | CA μs | v0.3 μs | v0.2 μs | APE | 指令数：预测/CA | 核数匹配 |',
            '| --- | ---: | ---: | ---: | ---: | --- | --- |']
    for r in held:lines.append(f"| {r['op']} {r['shape']} | {r['reference_us']:.4f} | {r['predicted_us']:.4f} | {r['previous_v02_us']:.4f} | {r['ape']:.2%} | {r['predicted_selected_opcode_count']:.3f}/{r['observed_selected_opcode_count']} | {r['mapping_matches']} |")
    lines+=['','## 指令与分核','']
    for op,d in diagnostics.items():
        value=d['opcode_count_wape']
        lines.append(f"- {op}：核数匹配 {d['mapping_matches']}/{d['count']}；指令次数 WAPE {value:.4%}。" if value is not None else f'- {op}：指令数据不完整。')
    lines+=['','FA 结构计数仅覆盖 Q tile=64、D=64/128 的 BF16 普通路径；Q tile=128 路径返回未验证，不套用公式。RV_* 含 load/store/control，不是纯算术指令。','',
        '## 开销','',f"本轮 {cost['attempts']} 次尝试，成功 {cost['successful_attempts']}；新增 CA 采集墙钟 {cost['wall_seconds']/60:.2f} 分钟、CPU {cost['cpu_seconds']/3600:.2f} 小时、原始数据 {cost['raw_bytes']/2**30:.2f} GiB。复用 32 个旧输入，无新增校准 CA。",'',
        f"在线未缓存 P95 {speed['uncached']['p95_us']:.2f} μs；缓存 P95 {speed['cached']['p95_us']:.2f} μs；批量 {speed['batch_queries_per_second']:,.0f} 次/秒。",'',
        '缺失/失败：'+json.dumps(missing,ensure_ascii=False)+'；失败尝试 '+str(len(result['failed_attempts']))+'。','',
        '## 范围','',
        '每类只有 3 个新留出点；未验证实卡、完整工作负载或跨代外推。quant 后续 DMA 间隔/写回成本仍只来自两个旧多 UB shape；控制路径是小工作量统计下界模型，不是纯 Scalar 指令周期求和。', '',
        '方法、trace 证据和固定耗时基线分析见 [模型说明](../../docs/model-v3.md)。','']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps({'gates':gates,'diagnostics':diagnostics,'cost':cost,'benchmark':speed},indent=2))
if __name__=='__main__':main()
