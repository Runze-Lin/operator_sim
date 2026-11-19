"""Match old exposed-time queue accounting to both new 5120-wide ordinary/repeat launches."""
from collections import Counter,defaultdict
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import read,write
from development.validation.post26.freeze import OUT,OLD
from development.validation.post24.collect import events
from operator_model import load_chip
from operator_model.post_resource import resource_post

def measure(r):
    st=r['stage'];core=max(st['windows'],key=lambda k:st['windows'][k][1])
    path=(ROOT/st['trace']).parent/core/'trace.json';changes=defaultdict(Counter);first=None
    for ev in events(path):
        a=round(ev['ts']*1650);b=round((ev['ts']+ev['dur'])*1650);first=a if first is None else min(first,a)
        d=json.loads(ev['args']['detail'])
        if not d.get('is_duration_excluded'):
            changes[a][d['queue_type']]+=1;changes[b][d['queue_type']]-=1
    changes[first];changes[st['last_tick']];prior=first;busy=Counter();exposed=Counter()
    for t,v in sorted(changes.items()):
        key=next((k for k in ('SIMD','MTE3','MTE2','SCALAR') if busy[k]>0),'other')
        exposed[key]+=t-prior;busy.update(v);prior=t
    ca={k:x/1650 for k,x in exposed.items()};ca['arrival']=(first-st['first_tick'])/1650
    assert abs(sum(ca.values())-st['ca_us'])<1e-9
    us,_,_,diag=resource_post(r['shape'],load_chip(r['chip']),{'resident_aiv':r['prior_aiv']})
    diff={k:v-ca.get(k,0.) for k,v in diag['stages_us'].items()}
    assert abs(sum(diff.values())-(us-st['ca_us']))<1e-10
    return dict(sku=r['chip'],shape=r['shape'],state=r['state'],role=r['role'],prior_aiv=r['prior_aiv'],active_aiv=st['active_aiv'],core=core,
                public_us=us,ca_us=st['ca_us'],opt_in_us=(r['us'] if r['role']=='repeat' else us),
                public_model_stages=diag['stages_us'],ca_exposed_stages=ca,public_minus_ca=diff,trace=st['trace'],sha256=st['trace_sha256'])
def main():
    data=[]
    for p in sorted((OLD/'receipts').glob('width-*.json')):
        for r in read(p)['rows']:
            if r['shape']==[65,4,5120] and r['timing_qualified']:data.append(measure(r))
    write(OUT/'wide-stages.json',dict(rule='Same disjoint time allocation as public v0.12 stage calibration; when multiple queues overlap assign SIMD > MTE3 > MTE2 > SCALAR > other. Arrival is global first dispatch to terminal AIV first instruction. Not a direct intrinsic hardware bandwidth.',rows=data))
    md=['# 宽 token 的普通热态高估来自哪些阶段','', '| SKU / 前序 | CA µs | 默认 µs | 可选 µs | arrival 差 | SCALAR 差 | SIMD 差 | MTE2 差 | MTE3 差 | other 差 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in data:
        md.append(f"| {r['sku']} / {r['role']} | {r['ca_us']:.3f} | {r['public_us']:.3f} | {r['opt_in_us']:.3f} | "+' | '.join(f"{r['public_minus_ca'][k]:+.3f}" for k in ('arrival','SCALAR','SIMD','MTE2','MTE3','other'))+' |')
    md+=['','阶段差为默认模型减去 CA 上相同互斥计价口径的时间，单位 µs。普通热态从 [121,4,1216] 切换到 [65,4,5120]，下一调用立即重复 [65,4,5120]；两者都已有足够的 AIV 驻留。绝对时间很近，说明在该宽度上仅凭“同形”无法解释默认模型的高估。CANN 实际每个 token 有两片，模型把字节数转阶段价格的方法与流水重叠一同影响高估。','',
        'CA 的低优先级队列在更高优先级队列同时活动时让出计价周期。互斥贡献差可以定位偏差并避免重复计费，但不能把它等同于单条指令本身的响应变化。精确重复的可选模型使用不同的依赖里程碑，表内的阶段差专指默认模型。[全部阶段、终核和原始 trace](wide-stages.json)']
    (OUT/'wide-stages.md').write_text('\n'.join(md).rstrip()+'\n')
if __name__=='__main__':main()
