"""Score the frozen model on qualified windows. All other states remain visible."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import time
from tempfile import NamedTemporaryFile

from extract import OUT, ROOT, read, sha
from model import predict
from series import DT,PR

FUTURE='Ascend970DT'


def atomic(path,txt):
    with NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False) as f:
        f.write(txt)
        temp=Path(f.name)
    temp.replace(path)


def summary(rs):
    if not rs:return None
    w=sum(abs(r['prediction_us']-r['ca_us']) for r in rs)/sum(r['ca_us'] for r in rs)
    worst=max(rs,key=lambda r:abs(r['error_pct']))
    return dict(n=len(rs),wape_pct=100*w,max_abs_pct=abs(worst['error_pct']),worst=worst['shape'])


def verified_row(file,pred,role):
    receipt=read(file);assert receipt['qualified'] and receipt['all_shards_end']
    model_us=pred['device_us']; ca=receipt['ca_us']
    assert pred['active_aiv']==receipt['stage']['active_aiv']
    return dict(shape=receipt['shape'],sku=receipt['chip'],role=role,prediction_us=model_us,ca_us=ca,
                error_pct=100*(model_us/ca-1),active_aiv=pred['active_aiv'],vf_commands=pred['vf_commands'],
                raw_receipt=str(file.relative_to(ROOT)),trace_sha256=receipt['stage']['trace_sha256'],
                raw_bytes=receipt['cost']['raw_bytes'],ca_wall_s=receipt['cost']['seconds'])


def build():
    regfile=OUT/'holdout-register.json';reg=read(regfile);regsha=sha(regfile)
    if sha(Path(__file__).with_name('model.py'))!=reg['hashes']['development/validation/hc_post_timing23/model.py']:
        raise ValueError('Model changed since freeze')
    if sha(OUT/'parameters.json')!=reg['hashes']['results/v0.23/hc-timing/parameters.json']:
        raise ValueError('Parameters changed since freeze')
    rows=[]
    for f in sorted((OUT/'primitives').glob('*.json')):
        j=read(f);t,_,d=j['shape'][1:];rows.append(verified_row(ROOT/j['source'],predict(t,d,j['sku']),'development'))
    pending=[];failed=[];future={}
    for c in reg['cases']:
        _,t,_,d=c['shape'];key=f'{t}x{d}';future[key]=c['predictions'][FUTURE]['device_us']
        file=OUT/(c['id']+'.json')
        if not file.exists():pending.append(c['id']);continue
        receipt=read(file)
        if not receipt['qualified']:failed.append(dict(id=c['id'],reason=receipt.get('reason')));continue
        if receipt['registration_sha256']!=regsha:raise ValueError(c['id'])
        pred=c['predictions'][c['chip']]
        if pred!=predict(t,d,c['chip']):raise ValueError(f'Frozen prediction drift: {key}')
        rows.append(verified_row(file,pred,'independent'))
    sreg=OUT/'supplement-register.json';sup_pending=[];sup_failed=[]
    if sreg.exists():
        extra=read(sreg);assert extra['primary_sha256']==regsha
        for c in extra['cases']:
            _,t,_,d=c['shape'];future[f'{t}x{d}']=c['predictions'][FUTURE]['device_us']
            f=OUT/(c['id']+'.json')
            if not f.exists():sup_pending.append(c['id']);continue
            rr=read(f)
            if not rr['qualified']:sup_failed.append(dict(id=c['id'],reason=rr.get('reason')));continue
            assert rr['registration_sha256']==sha(sreg)
            pred=c['predictions'][c['chip']];assert pred==predict(t,d,c['chip'])
            rows.append(verified_row(f,pred,'supplement'))
    assert len({r['trace_sha256'] for r in rows})==len(rows)
    groups={role:{sku:summary([r for r in rows if r['role']==role and r['sku']==sku]) for sku in (DT,PR)} for role in ('development','independent','supplement')}
    pooled={sku:summary([r for r in rows if r['role']!='development' and r['sku']==sku]) for sku in (DT,PR)}
    pairs=[]
    for role in groups:
        shapes=defaultdict(dict)
        for r in rows:
            if r['role']==role:shapes[tuple(r['shape'])][r['sku']]=r
        for shape,rs in shapes.items():
            if len(rs)!=2:continue
            dt,pr=rs[DT],rs[PR];cm=dt['ca_us']/pr['ca_us'];pm=dt['prediction_us']/pr['prediction_us']
            pairs.append(dict(role=role,shape=shape,actual_dt_over_pr=cm,pred_dt_over_pr=pm,ratio_error_pct=100*(pm/cm-1),same_direction=(cm>1)==(pm>1)))
    out=dict(register_sha256=regsha,rows=rows,summary=groups,pairs=pairs,pending=pending,failed=failed,
             supplement_pending=sup_pending,supplement_failed=sup_failed,accepted_independent_summary=pooled,
             extrapolation_970_us=future,queue_state=read(OUT/'holdout-state.json'),
             supplement_state=read(OUT/'supplement-state.json') if (OUT/'supplement-state.json').exists() else {'status':'queued'})
    scored_pairs=[r for r in pairs if r['role']!='development']
    if scored_pairs:
        out['independent_speedup']=dict(n=len(scored_pairs),correct_direction=sum(r['same_direction'] for r in scored_pairs),
                                         mean_abs_ratio_error_pct=sum(abs(r['ratio_error_pct']) for r in scored_pairs)/len(scored_pairs),
                                         max_abs_ratio_error_pct=max(abs(r['ratio_error_pct']) for r in scored_pairs))
    accepted_files=[ROOT/r['raw_receipt'] for r in rows]
    failed_files=[OUT/(f['id']+'.json') for f in failed+sup_failed]
    costs=[read(path)['cost'] for path in accepted_files+failed_files]
    out['offline_cost']=dict(completed_attempts=len(costs),msopprof_wall_s=sum(c['seconds'] for c in costs),
                             failed_wall_s=sum(read(path)['cost']['seconds'] for path in failed_files),
                             max_attempt_s=max(c['seconds'] for c in costs),
                             raw_bytes=sum(c['raw_bytes'] for c in costs),
                             definition='Total msopprof+host wall clock over completed attempts including failed numerics, excluding post-run qualification. Development includes the six reused traces.')
    return out


def render(v):
    lines=['# vLLM HcPost 分段模型误差','',
        '| 数据角色 | 芯片 | 合格 shape | WAPE | 最大绝对误差 | 最坏 [T,D] |',
        '|---|---|---:|---:|---:|---|']
    expanded={**v['summary'],'all_usable':v['accepted_independent_summary']}
    for role,values in expanded.items():
        for chip,group in values.items():
            label={'development':'开发回放','independent':'首批冻结独立','supplement':'另登记 64-lane 边界','all_usable':'全部合格独立形状'}[role]
            if group:
                _,t,_,d=group['worst'];lines.append(f'| {label} | {chip} | {group["n"]} | {group["wape_pct"]:.2f}% | {group["max_abs_pct"]:.2f}% | [{t},{d}] |')
            else:lines.append(f'| {label} | {chip} | 0 | N/A | N/A | — |')
    lines+=['',f'独立登记共 20 个双端窗口。已取得 {20-len(v["pending"])-len(v["failed"])} 个合格回执，{len(v["pending"])} 个待回执，{len(v["failed"])} 个未合格。'
           'WAPE 按全部合格窗口的绝对误差之和除以 CA 时间之和。±误差按预测除以实测再减 1。','']
    lines+=['另登记 T=9、D=4160 的双端形状。选形时已知原首批有数值失败，冻结设备模型、输入构造和统计方法沿用原合同。'
            f'追加有 {len(v["supplement_pending"])} 个待回执，{len(v["supplement_failed"])} 个不合格。','']
    if v['failed']:
        for f in v['failed']:lines+=['- '+f['id']+'，'+str(f['reason'])]
        lines+=['']
    if (OUT/'failed-tail-mask.json').exists():
        lines+=['D4112 两端主机各比较 115,136 个输出，均有 333 个 BF16 值不相等。逐流掩码的源码重放预期为 0/112/111/110 个差异，合计 333。原始日志仅记录总数，逐流值是推演。该形状保留在登记失败清单，耗时剔出统计。[失败分析](failed-tail-mask.json)','']
    lines+=['## 逐 shape 对照','','激活与输出为 BF16，四个 post 与十六个 comb 系数为 FP32。每行的实际接口是 vLLM Ascend 定制库的 `aclnnHcPost`，设备函数为 `HcPost_d2fe6614722b27afa6f8d964c215447e`。设备时间覆盖全核最早指令至最后完成。CA 墙钟为宿主采集成本，单位秒。','',
            '| 数据角色 | [T,D] | SKU | AIV | 预测 µs | CA µs | 误差 | 970DT 外推 µs | CA 墙钟 s |',
            '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(v['rows'],key=lambda r:(r['role'],r['shape'][1],r['shape'][-1],r['sku'])):
        _,t,_,d=r['shape'];f=predict(t,d,FUTURE)['device_us']
        lines.append(f'| {r["role"]} | [{t},{d}] | {r["sku"]} | {r["active_aiv"]} | {r["prediction_us"]:.3f} | {r["ca_us"]:.3f} | {r["error_pct"]:+.2f}% | {f:.3f} | {r["ca_wall_s"]:.1f} |')
    lines+=['','## 双芯片加速比','','以 DT 时间除以 PR 时间，值大于 1 表示 PR 较快。双端都有回执的形状进入下表；接近 1 的样本对启动抖动敏感。','',
            '| 数据角色 | [T,D] | 预测 DT/PR | CA DT/PR | 比值误差 |','|---|---|---:|---:|---:|']
    for r in v['pairs']:
        _,t,_,d=r['shape'];lines.append(f'| {r["role"]} | [{t},{d}] | {r["pred_dt_over_pr"]:.4f} | {r["actual_dt_over_pr"]:.4f} | {r["ratio_error_pct"]:+.2f}% |')
    if v.get('independent_speedup'):
        ratio=v['independent_speedup']
        lines+=['',f'{ratio["n"]} 个独立双端形状的 DT/PR 比值平均绝对相对误差 {ratio["mean_abs_ratio_error_pct"]:.2f}%，最大 {ratio["max_abs_ratio_error_pct"]:.2f}%；'
               f'其中 {ratio["correct_direction"]} 个预测对了两芯片的快慢方向。单端耗时达到误差要求不能自动保证两个相近耗时的排序。']
    anchors=OUT/'independent-stage-errors.json'
    if anchors.exists():
        all_anchors=read(anchors)
        lines+=['','## 最大独立误差落在哪个阶段','','以下对比每端已合格的最坏形状，数值为模型相对 CA 的时间戳偏差（周期），正值表示模型偏晚。两者都从最早设备指令计时。每列都是累计时标；不能把各列相加。',
                '','| SKU | [T,D] | CA 最后核 | 模型/CA 每核行数 | 首行系数发起 | 首算术 VF 发起 | 首写发起 | 末写发起 | 全窗结束 |',
                '|---|---|---|---:|---:|---:|---:|---:|---:|']
        for chip in (DT,PR):
            pool=[r for r in v['rows'] if r['role']!='development' and r['sku']==chip]
            if not pool:continue
            r=max(pool,key=lambda x:abs(x['error_pct']))
            a=all_anchors.get(Path(r['raw_receipt']).name)
            if a is None:continue
            ticks=a['ticks'];_,t,_,d=a['shape']
            vals=' | '.join(f'{ticks[k]["delta"]:+.0f}' for k in ('row0_coeff_issue','row0_first_vf_issue','first_write_issue','last_write_issue','full'))
            lines.append(f'| {chip} | [{t},{d}] | {a["core"]} | {a["model_critical_rows"]}/{a["actual_rows"]} | {vals} |')
        lines+=['','每个冻结合格样本都可逐个核查阶段时标与最后核的 shard 散列。[物理核阶段残差](independent-stage-errors.json)']
    costs=v['offline_cost']
    lines+=['','## 离线与在线成本','',
            f'已完成 {costs["completed_attempts"]} 次的 msopprof/宿主时间合计 {costs["msopprof_wall_s"]/60:.1f} 分钟，其中数值失败 {costs["failed_wall_s"]/60:.1f} 分钟；最慢一次 {costs["max_attempt_s"]:.1f} 秒，原始数据合计 {costs["raw_bytes"]/2**30:.2f} GiB。'
            '这个总计含已复用的六份开发 trace；后置逐核核验和绘图另计。冻结采集保持单并发。在线预测只读芯片与资源参数。[在线前向及成本](query-cost.json)']
    lines+=['','实际主频读取自冻结的模型配置，两个 CA SKU 都为 1.65 GHz。PR 的公开 FP32 Vector 峰值有取整，两端有很小的名义每核速率差。当资源预测的时间差只有几十周期，现存窗口不足以把速度方向评为跨 SKU 校准。',
            '','[完整方法、参数与适用范围](method.md) · [五分片 CA 对照图](dt-16448-timeline.png) · [逐样本回执与 SHA](comparison.json) · [原始独立登记](holdout-register.json)']
    return '\n'.join(lines)+'\n'


def publish():
    v=build()
    atomic(OUT/'comparison.json',json.dumps(v,indent=2)+'\n')
    atomic(OUT/'comparison.md',render(v))
    return v


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();prev=''
    while True:
        signature=tuple((p.name,p.stat().st_mtime_ns) for pattern in ('holdout-Ascend*.json','supplement-Ascend*.json','*-state.json') for p in sorted(OUT.glob(pattern)))
        if signature!=prev:
            v=publish();prev=signature
        if not args.watch or v['supplement_state']['status'] in ('complete','stopped'):break
        time.sleep(9)
