#!/usr/bin/env python3
"""Evaluate immutable predictions; report failures, baselines, speed and CA cost."""
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tracemalloc

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.model import Case, Predictor


def quantile(values,q):
    values=sorted(values)
    pos=(len(values)-1)*q
    lo=int(pos);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(pos-lo)


def metrics(rows,field):
    if not rows:return None
    refs=[r['reference_us'] for r in rows]
    errors=[r[field]-r['reference_us'] for r in rows]
    ape=[abs(e)/t for e,t in zip(errors,refs)]
    return {'count':len(rows),'wape':sum(abs(e) for e in errors)/sum(refs),
            'p90_ape':quantile(ape,.9),'max_ape':max(ape),
            'mae_us':statistics.mean(abs(e) for e in errors),
            'signed_bias_us':statistics.mean(errors)}


def benchmark(model_path,cases,predictor_class=Predictor,case_class=Case,cold_module='operator_model.model'):
    start=time.perf_counter_ns();predictor=predictor_class(model_path)
    load_us=(time.perf_counter_ns()-start)/1000
    # Warm interpreter and use shape diversity. Uncached means full plan,
    # cost evaluation and immutable result construction, not a different API.
    for case in cases:predictor.predict(case)
    samples={}
    for mode,use_cache in [('uncached',False),('cached',True)]:
        times=[]
        for i in range(10000):
            case=cases[i%len(cases)]
            begin=time.perf_counter_ns();predictor.predict(case,use_cache=use_cache)
            times.append((time.perf_counter_ns()-begin)/1000)
        samples[mode]={'p50_us':statistics.median(times),'p95_us':quantile(times,.95)}
    batch=cases*1000
    begin=time.perf_counter_ns();predictor.predict_many(batch)
    batch_ns=time.perf_counter_ns()-begin
    tracemalloc.start()
    cache=predictor_class(model_path,cache_size=4096)
    for i in range(4096):cache.predict(case_class('quant',(i+1,256)))
    current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    # Pure Python cold process, including interpreter/import/JSON load/one prediction.
    cold=[]
    for _ in range(5):
        begin=time.perf_counter()
        subprocess.run([sys.executable,'-c',
            f"from {cold_module} import Predictor,Case; Predictor({str(model_path)!r}).predict(Case('quant',(4,256)))"],
            cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        cold.append((time.perf_counter()-begin)*1000)
    return {'python':sys.version,'model_load_us':load_us,**samples,
            'batch_queries':len(batch),'batch_queries_per_second':len(batch)*1e9/batch_ns,
            'cache_entries':len(cache.cache),'cache_memory_bytes':current,'cache_peak_bytes':peak,
            'cold_process_median_ms':statistics.median(cold),
            'measurement_scope':'single-process library API; no CANN/trace reads; no real system-simulator integration yet'}


def main():
    out=ROOT/'results/v0.1'
    frozen=json.loads((out/'frozen-predictions.json').read_text())
    if sha256(out/'model.json')!=frozen['model_sha256'] or sha256(ROOT/'operator_model/model.py')!=frozen['implementation_sha256']:
        raise SystemExit('Frozen model or implementation changed')
    if sha256(ROOT/'configs/sampling-v1.json')!=frozen['manifest_sha256']:
        raise SystemExit('Frozen sampling manifest changed')
    data=json.loads((out/'dataset.json').read_text())
    if len({r['case']['id'] for r in data['records']})!=len(data['records']):
        raise SystemExit('Duplicate case IDs: choose a single versioned experiment')
    records={r['case']['id']:r for r in data['records']}
    spec=json.loads((ROOT/'configs/sampling-v1.json').read_text())
    predictor=Predictor(out/'model.json')
    for r in data['records']:
        if json.dumps(r['identity'],sort_keys=True)!=json.dumps(predictor.model['identity'],sort_keys=True):
            raise SystemExit('Measurement identity differs from calibrated model')
    if {r['cache_key'] for r in data['records'] if r['case']['split']=='calibration'}!=set(predictor.model['training_keys']):
        raise SystemExit('Training records changed after freezing')
    rows=[];incomplete=[]
    for item in frozen['predictions']:
        r=records.get(item['id'])
        if r is None or r['status']!='ok':
            incomplete.append({'id':item['id'],'status':r['status'] if r else 'missing'})
            continue
        p=item['prediction'];measurement=r['measurement'];case=Case(item['op'],tuple(item['shape']))
        if dict(predictor.predict(case)) != dict(p):
            # JSON lists versus immutable tuples: compare canonical serialization.
            if json.dumps(predictor.predict(case),sort_keys=True)!=json.dumps(p,sort_keys=True):
                raise SystemExit('Predictions do not reproduce')
        predicted_counts=predictor.instruction_counts(case)
        compute_names=['MMAD'] if item['op']!='quant' else [k for k in predicted_counts if k.startswith('RV_')]
        expected_count=sum(predicted_counts.get(k,0) for k in compute_names)
        observed_count=max(sum(c['dynamic_opcodes'].get(k,0) for k in compute_names) for c in measurement['cores'])
        row={k:item[k] for k in ('id','op','shape','split')}
        row.update({k:p[k] for k in ('predicted_us','constant_baseline_us','empirical_roofline_us','uncorrected_schedule_us','coverage')})
        row.update({'reference_us':measurement['reference_us'],
                    'error_us':p['predicted_us']-measurement['reference_us'],
                    'ape':abs(p['predicted_us']-measurement['reference_us'])/measurement['reference_us'],
                    'predicted_cores':[p['plan']['active_aic'],p['plan']['active_aiv']],
                    'observed_cores':[measurement['active_aic'],measurement['active_aiv']],
                    'kernel_seen_in_calibration':measurement['kernel'] in predictor.model['operators'][item['op']]['kernels'],
                    'instruction_family':'RV_* including load/store/control' if item['op']=='quant' else 'MMAD',
                    'predicted_max_core_selected_opcode_count':expected_count,
                    'observed_max_core_selected_opcode_count':observed_count,
                    'ca_wall_seconds':r['wall_seconds'],'ca_cpu_seconds':r['cpu_seconds'],'raw_bytes':r['raw_bytes']})
        row['mapping_matches']=row['predicted_cores']==row['observed_cores']
        rows.append(row)
    aggregates={}
    for split in ('calibration','holdout'):
        aggregates[split]={}
        for op in ('all','quant','matmul','fa'):
            selected=[r for r in rows if r['split']==split and (op=='all' or r['op']==op)]
            aggregates[split][op]={name:metrics(selected,name) for name in ('predicted_us','constant_baseline_us','empirical_roofline_us','uncorrected_schedule_us')}
    speed=benchmark(out/'model.json',[Case(r['op'],tuple(r['shape'])) for r in rows if r['split']!='repeat'])
    acceptance=spec['acceptance']
    gates={}
    for op in ('all','quant','matmul','fa'):
        m=aggregates['holdout'][op]['predicted_us']
        gates[op]=bool(m and m['count']==(6 if op=='all' else 2) and m['wape']<=acceptance['holdout_wape'] and m['p90_ape']<=acceptance['holdout_p90_ape'])
    gates['speed']=speed['uncached']['p95_us']<=acceptance['uncached_p95_us'] and speed['cached']['p95_us']<=acceptance['cached_p95_us']
    repeats=[]
    for r in rows:
        if r['split']=='repeat':
            first=next(x for x in rows if x['id']==r['id'].replace('_repeat_','_cal_'))
            repeats.append({'op':r['op'],'first_us':first['reference_us'],'repeat_us':r['reference_us'],
                            'relative_pair_difference':abs(r['reference_us']-first['reference_us'])/first['reference_us']})
    result={'evaluated_utc':datetime.now(timezone.utc).isoformat(),'model_sha256':frozen['model_sha256'],
            'freeze_utc':frozen['frozen_utc'],'incomplete':incomplete,'rows':rows,'metrics':aggregates,
            'acceptance':acceptance,'gates':gates,'benchmark':speed,'repeat_pairs':repeats,
            'collection_cost':{'attempts':len(records),'successful':sum(r['status']=='ok' for r in records.values()),
                'wall_seconds':sum(r['wall_seconds'] for r in records.values()),
                'cpu_seconds':sum(r['cpu_seconds'] for r in records.values()),
                'raw_bytes':sum(r['raw_bytes'] for r in records.values())},
            'scope':'uniform-weight small synthetic holdouts; no full-model, true hardware or cross-generation validation'}
    for r in data['records']:
        if r['case']['split']=='holdout' and datetime.fromisoformat(r['finished_utc'])<=datetime.fromisoformat(frozen['frozen_utc']):
            raise SystemExit('Holdout measurement predates frozen predictions')
    (out/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# v0.1 CA 校准与留出误差报告','',
        '预测在采集留出集前冻结；下表均对比跨核指令时间窗，不含 host enqueue 或指令窗外的设备调度。',
        '每个算子只有两个留出 shape，权重均为 1；指标不代表完整模型工作负载。','',
        '| 算子 | 留出 WAPE | P90 APE | 固定耗时基线 WAPE | 经验 Roofline WAPE | 目标达成 |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for op in ('quant','matmul','fa','all'):
        a=aggregates['holdout'][op]
        if not a['predicted_us']:continue
        lines.append(f"| {op} | {a['predicted_us']['wape']:.2%} | {a['predicted_us']['p90_ape']:.2%} | {a['constant_baseline_us']['wape']:.2%} | {a['empirical_roofline_us']['wape']:.2%} | {gates[op]} |")
    lines+=['','目标：WAPE ≤10%、P90 APE ≤20%。经验 Roofline 的有效速率来自校准集，不是理论峰值，也不是用户原系统的 Roofline 实现。','',
        '| Case | Shape | Split | CA μs | 预测 μs | APE | 预测/实际 AIC,AIV |',
        '| --- | --- | --- | ---: | ---: | ---: | --- |']
    for r in rows:
        lines.append(f"| {r['id']} | {r['shape']} | {r['split']} | {r['reference_us']:.4f} | {r['predicted_us']:.4f} | {r['ape']:.2%} | {r['predicted_cores']} / {r['observed_cores']} |")
    cost=result['collection_cost']
    lines+=['','## 开销','',f"共 {cost['attempts']} 次采集，成功 {cost['successful']} 次；累计墙钟 {cost['wall_seconds']/60:.2f} 分钟、CPU {cost['cpu_seconds']/3600:.2f} 小时，原始文件 {cost['raw_bytes']/2**30:.3f} GiB。",'',
        f"预测接口：未缓存 P50/P95 {speed['uncached']['p50_us']:.2f}/{speed['uncached']['p95_us']:.2f} μs；缓存命中 P50/P95 {speed['cached']['p50_us']:.2f}/{speed['cached']['p95_us']:.2f} μs；批量 {speed['batch_queries_per_second']:,.0f} 次/秒。",'',
        f"模型加载 {speed['model_load_us']:.1f} μs；独立 Python 进程冷启动中位数 {speed['cold_process_median_ms']:.2f} ms；4096 条缓存约 {speed['cache_memory_bytes']/2**20:.2f} MiB。",'',
        '在线模型不加载 CANN、不读取 trace、不启动 CA。上述为库级微基准，尚未测量接入用户整网仿真后的总额外开销。','',
        '## 复测波动','']
    for pair in repeats:lines.append(f"- {pair['op']}: {pair['first_us']:.4f} → {pair['repeat_us']:.4f} μs，配对差异 {pair['relative_pair_difference']:.2%}。")
    lines+=['','每个基准仅复测一次，不能据此给出统计置信区间。','',
        '## 限制与诊断','',
        '- 正确性、采集失败与超时不当作延迟样本；未完成项：'+json.dumps(incomplete,ensure_ascii=False),
        '- `evaluation.json` 逐点记录分核偏差、kernel 是否在校准集出现、选定指令族计数偏差（quant 为 RV_*，含 load/store/control；matmul/FA 为 MMAD）；这些是诊断线索，不把相关性写成确定根因。',
        '- 校准成本包含 CA 中的延迟/等待影响；busy 区间并集不是执行吞吐峰值。v0.1 不输出未经独立校验的 ideal/CA 效率。',
        '- 本轮不根据留出结果重新调参。未达标的分支应新增校准点并使用新的留出集验收下一版。','']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps({'gates':gates,'cost':cost,'benchmark':speed},indent=2))


if __name__=='__main__':main()
