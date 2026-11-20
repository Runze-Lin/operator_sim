"""One geometry per DT/PR CA window, multiple B×Q labels where shapes coincide."""
from collections import defaultdict
from pathlib import Path
from statistics import median
import json
from freeze import ROOT, OUT, DT, PR, FUT
TARGET=OUT/'decode';REG=TARGET/'freeze.json'

def build(score):
    prices=json.loads(REG.read_bytes());pending={tuple(x['shape']) for x in prices['jobs']}
    evidence=defaultdict(list)
    for r in score['rows']:evidence[tuple(r['shape']),r['chip']].append(r)
    rejects={}
    for folder in ['v0.29/matmul/decode','v0.32/matmul/decode']:
        for p in (ROOT/'results'/folder).glob('*.json'):
            if not p.stem.endswith(('-dt','-pr')):continue
            x=json.loads(p.read_bytes())
            if x.get('qualified') is False and x.get('shape'):rejects[tuple(x['shape']),x['chip']]=[x.get('reason'),str(p.relative_to(ROOT))]
    rows=[];state=json.loads((TARGET/'state.json').read_bytes()) if (TARGET/'state.json').exists() else {}
    lines=['# 三个模型的 decode：B1、9、11、16 与主模型 Q9','','M 是**这个 rank 交给一次 Matmul 的行数**。普通 decode 每个请求一行 Q=1；MTP 一次提出八个草稿时主模型校验 Q=9。B=1 开 MTP 和 B=9 普通 decode 得到同一个 M=9，共用设备证据；B=9 的九个请求也各开 MTP，得到 M=81。B=11 的两种情况是 M=11/99，B=16 是 M=16/144。公式仅指本卡无 padding、无额外请求时的一次投影，非注意力 batch 轴推断。','','以下形状依据三模型配置与权重推导的真实 TP8 逻辑尺寸；实测合同为安装包 aclnnMatmul、连续无偏 BF16 ND。图部署若重新量化、转置或融合，按新合同核验。每个已有完整设备窗按 trace 指纹去重；同一形状多样本取中位数。每个模型列一个主要投影和一个替代投影，**同一模型内两个真实投影并非重复样本**。','','| 模型 · 投影 | B×Q → M | 每卡 [M,N,K] | 950DT 预测 / CA µs | 950PR 预测 / CA µs | 970DT 外推 µs |','|---|---|---|---:|---:|---:|']
    for j in prices['application_rows']:
        s=tuple(j['shape']);cells=[];endpoints={}
        for sku,tag in ((DT,'dt'),(PR,'pr')):
            measured=evidence[s,sku];m=median(r['ca_us'] for r in measured) if measured else None
            p=j['predictions'][sku];evidence_link=measured[0]['source'] if measured else None
            stem='-'.join(map(str,s))+'-'+tag;label='已核验' if measured else ('新形采集中' if state.get('case')==stem else '已排队') if s in pending else '无合格窗'
            check=OUT/'pro-b9-mtp/host-check.json'
            if not measured and s==(81,64,7168) and check.exists():
                label='未测 · 主机满载未预约';evidence_link=str(check.relative_to(ROOT))
            if not measured and (s,sku) in rejects:
                label='采集未合格';evidence_link=rejects[s,sku][1]
            ca=(f'[{m:.3f}](../../../{evidence_link})' if measured else f'[— {label}](../../../{evidence_link})' if evidence_link else f'— {label}')
            endpoints[sku]=dict(predicted_us=p['us'],ca_us=m,source=evidence_link,qualified_windows=len(measured),status=label)
            cells.append(f"{p['us']:.3f} / {ca}")
        sc='、'.join(f"{v['batch']}×{v['Q']}" for v in j['scenarios'])+' → '+str(s[0])
        lines.append(f"| {j['model']} · {j['node']} | {sc} | {list(s)} | {cells[0]} | {cells[1]} | {j['predictions'][FUT]['us']:.3f} |")
        rows.append(dict(model=j['model'],node=j['node'],shape=s,scenarios=j['scenarios'],endpoints=endpoints,extrapolated_970dt_us=j['predictions'][FUT]['us']))
    ids={tuple(r['shape']) for r in rows};counts={k:sum(bool(evidence[s,k]) for s in ids) for k in (DT,PR)};both=sum(all(evidence[s,k] for k in (DT,PR)) for s in ids)
    lines[5:5]=[f"当前 {len(ids)} 种唯一几何，DT {counts[DT]}、PR {counts[PR]} 种已有合格时间，{both} 种双端配对。空白项只表示这份合同尚无合格设备时间。",'']
    idle=OUT/'pro-b9-mtp/host-check.json'
    if idle.exists():
        obs=json.loads(idle.read_text())['observation'];quiet='、'.join(str(v['idle80']) for v in obs)
        lines+=['',f"Pro 的同一 indexer.weights_proj [81,64,7168]（B9×Q9）目前只有 host workspace 和本模型预报。已有设备登记和全部相关原始调用均未发现相同采集。三次 2 秒宿主采样只见 {quiet} 个具有至少 80% 空闲的物理核；256 核主机上的旧 Kimi 槽已有锁主和等待者。因此此端点没有另排长队，表里也不填 CA。[资源核对](pro-b9-mtp/host-check.json)。"]
    lines+=['','本轮只新增三个从未采集也未预约的几何：Flash dsa.wq_a 的 [81,1280,5120]、Pro dsa.wq_a 的 [16,1536,7168]、Kimi mla.kv_b_proj 的 [9,3072,512]。三形共六个目标端点，已在采前登记唯一 ELF、数值、分核和 MMAD 预测；在现有 Kimi 大形双端释放同一把锁后逐一执行。前序等待不计为已采或独立准确性。','', '[新增六端冻结](decode/freeze.json) · [实际排队和 CA 状态](decode/state.json) · [逐几何机器数据](decode.json) · [prefill 双端实测和冻结盲测](report.md)']
    from score import save
    save('decode.md','\n'.join(lines)+'\n');save('decode.json',json.dumps(dict(utc=score['utc'],unique_shapes=len(ids),measured_counts=counts,both=both,rows=rows,state=state),indent=2,ensure_ascii=False)+'\n')
