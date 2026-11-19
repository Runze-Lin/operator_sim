"""Score the frozen model on qualified windows. All other states remain visible."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import time

from extract import OUT, ROOT, read, sha
from model import predict
from series import DT,PR

FUTURE='Ascend970DT'


def atomic(path,txt):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(txt)
    tmp.replace(path)


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
    assert len({r['trace_sha256'] for r in rows})==len(rows)
    groups={role:{sku:summary([r for r in rows if r['role']==role and r['sku']==sku]) for sku in (DT,PR)} for role in ('development','independent')}
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
             extrapolation_970_us=future,queue_state=read(OUT/'holdout-state.json'))
    return out


def render(v):
    lines=['# vLLM HcPost 分段模型误差','',
        '| 数据角色 | 芯片 | 合格 shape | WAPE | 最大绝对误差 | 最坏 [T,D] |',
        '|---|---|---:|---:|---:|---|']
    for role in v['summary']:
        for chip,group in v['summary'][role].items():
            label='开发回放' if role=='development' else '冻结独立'
            if group:
                _,t,_,d=group['worst'];lines.append(f'| {label} | {chip} | {group["n"]} | {group["wape_pct"]:.2f}% | {group["max_abs_pct"]:.2f}% | [{t},{d}] |')
            else:lines.append(f'| {label} | {chip} | 0 | N/A | N/A | — |')
    lines+=['',f'独立登记共 20 个双端窗口。已取得 {20-len(v["pending"])-len(v["failed"])} 个合格回执，{len(v["pending"])} 个待回执，{len(v["failed"])} 个未合格。'
           'WAPE 按全部合格窗口的绝对误差之和除以 CA 时间之和。±误差按预测除以实测再减 1。','']
    if v['failed']:
        for f in v['failed']:lines+=['- '+f['id']+'，'+str(f['reason'])]
        lines+=['']
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
        signature=' '.join(sorted(p.name for p in OUT.glob('holdout-Ascend*.json')))+str((OUT/'holdout-state.json').stat().st_mtime_ns)
        if signature!=prev:
            v=publish();prev=signature
        if not args.watch or v['queue_state']['status'] in ('complete','stopped'):break
        time.sleep(9)
