"""Score only groups which had not started at repeat-policy registration."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post24.freeze import sha
from development.validation.post24.report import stats,fmt
from development.validation.post25.calibrate import OUT
from development.validation.sku19.campaign import read,write,now
def main():
    reg=read(OUT/'repeat-policy-freeze.json')
    assert all(sha(ROOT/f)==h for f,h in reg['hashes'].items())
    a,b=reg['measured'];sources=sorted((OUT/'receipts').glob('*.json'));allrows=[];merged={};groups_ok=[]
    future={(r['group'],r['i']):r['policy_us'] for r in reg['rows'] if r['chip'] in reg['extrapolated']}
    for f in sources:
        obj=read(f)
        if obj['group'] not in reg['included_groups']:continue
        assert obj['cost']['started_utc']>reg['utc']
        rows=[r for r in reg['rows'] if r['chip']==obj['chip'] and r['group']==obj['group']]
        groups_ok.append(f.name)
        for p,r in zip(rows,obj['rows']):
            assert p['i']==r['i'] and p['shape']==r['shape']
            if not r['timing_qualified']:continue
            x=dict(**p,ca_us=r['stage']['ca_us'],extrapolated_970_us=future[r['group'],r['i']]);allrows.append(x);merged[r['chip'],r['group'],r['i']]=x
    ratio=[];cols=('policy_us','public_us','full_candidate_us')
    for r in allrows:
        p=merged.get((b,r['group'],r['i']))
        if r['chip']==a and p:
            ratio.append(dict(role=r['role'],shape=r['shape'],ca_us=r['ca_us']/p['ca_us'],**{k:r[k]/p[k] for k in cols}))
    summary={}
    for sku in [a,b]:
        val=[r for r in allrows if r['chip']==sku]
        summary[sku]={group:{k:stats(xs,k) for k in cols} for group,xs in [('all',val),('entry',[r for r in val if r['role']=='entry']),('repeat',[r for r in val if r['role']=='repeat'])]}
    ratios={group:{k:stats(rr,k) for k in cols} for group,rr in [('all',ratio),('entry',[r for r in ratio if r['role']=='entry']),('repeat',[r for r in ratio if r['role']=='repeat'])]}
    write(OUT/'repeat-policy.json',dict(utc=now(),expected=20,qualified=len(allrows),receipts=groups_ok,summary=summary,ratios=ratios,ratio_rows=ratio,rows=allrows))
    md=['# 仅在上次“同一内置 Post 合同”时切换的部署策略','', '| 芯片 / 调用状态 | 已合格窗口 | 仅重复替换 WAPE / 最大 | 当前公开 WAPE / 最大 | 全状态逐行候选 WAPE / 最大 |','|---|---:|---:|---:|---:|']
    for sku,sd in summary.items():
        for label,s in sd.items():
            if s['policy_us']:md.append(f"| {sku} / {label} | {s['policy_us']['n']} | "+' | '.join(fmt(s[c]) for c in cols)+' |')
    md+=['',f"预登记后合格 {len(allrows)}/20 个全核窗口，来自 {len(groups_ok)}/4 个批次。开发期曾查看 boundary 的三个形状并启动 steady 组，两组全部排除。这一版调用策略只用余下 5 个尚未启动的新形，每端各 5 种，去重和重复调用分别评分。已知普通复用即使有相同的 AIV 驻留数，也继续走发布版。仅有 shape 和芯片的未知前序请求同样走发布版。",'',
     '流实例连续保存最后一次成功派发的完整逻辑合同，包含 rank/stream、目标芯片配置、内置 ops-transformer aclnnMhcPost 身份、BF16、含残差、contiguous ND 和 (T,N,D)。实际输出地址与参数值不进入决策。不同算子、不同实现、不同引擎或 stream 重置后都失去“上一条合同完全相同”的证据。仅仅查询价格不会更新上一实际派发；可能已部分送达设备的失败使前序失效。即便合同相等也必须已知同流的 AIV 为可复用状态，且当前为 token 分核路线。作为独立候选，未修改公开模型。', '',
     '## 双端加速比','', '| 调用状态 | 同形同调用对 | 仅重复替换 WAPE / 最大 | 当前公开 WAPE / 最大 | 全状态逐行候选 WAPE / 最大 |','|---|---:|---:|---:|---:|']
    for label,s in ratios.items():
        if s['policy_us']:md.append(f"| {label} | {s['policy_us']['n']} | "+' | '.join(fmt(s[c]) for c in cols)+' |')
    md+=['','[预注册时间、价格与核对哈希](repeat-policy-freeze.json) · [逐形、逐状态与 970DT 外推](repeat-policy.json) · [影子策略实现](../../../development/validation/post25/repeat_policy.py)','']
    (OUT/'repeat-policy.md').write_text('\n'.join(md))
if __name__=='__main__':main()
