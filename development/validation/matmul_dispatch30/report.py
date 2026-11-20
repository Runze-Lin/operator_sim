"""Join device receipts and separate host-only dispatches without counting B*Q aliases twice."""
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.matmul28 import campaign as ca
from development.validation.matmul28.overview import receipt
from development.validation.matmul29 import host_audit

OUT=ROOT/'results/v0.30/matmul-dispatch'
OLD=ROOT/'results/v0.29/matmul'
CHIPS=(ca.DT,ca.PR)
def val(v): return '—' if v is None else f'{v:.3f}'


def render():
    archived={}
    for item in ca.read(ROOT/'results/v0.28/matmul/overview.json')['rows']:
        for chip,tag in ((ca.DT,'dt'),(ca.PR,'pr')):
            r=receipt(item,tag)
            if r.get('ca_us') is not None:
                archived[tuple(item['shape']),chip]=dict(ca_us=r['ca_us'],source='results/v0.28/matmul/'+r['receipt'])
    for pair in ca.read(OLD/'decode/score.json')['unique_rows'].values():
        for chip,r in pair.items():
            if r.get('qualified'):
                archived.setdefault((tuple(r['shape']),chip),dict(ca_us=r['ca_us'],source='results/v0.29/matmul/decode/score.json'))
    for file in [OLD/'decode/retry/receipt.json',*sorted((OUT/'device').glob('*-*.json'))]:
        if not file.is_file() or file.name.endswith('-initial.json'):continue
        r=ca.read(file)
        if r.get('qualified'):
            archived[tuple(r['shape']),r['chip']]=dict(ca_us=r['ca_us'],source=str(file.relative_to(ROOT)))
    host={}
    registration=ca.read(OUT/'freeze.json')
    digest=host_audit.sha(OUT/'freeze.json')
    for chip in CHIPS:
        evidence=ca.read(OUT/(chip+'-workspace.json'))
        assert evidence['qualified_for_host_choice'] and evidence['register_sha256']==digest
        host[chip]={tuple(r['shape']):r for r in evidence['rows']}
    rows=defaultdict(dict)
    for r in registration['rows']:
        if r['scenarios']:
            rows[tuple(r['shape'])][r['soc']]=r
    new=ca.read(OUT/'device/freeze.json')
    for case in new['rows']:
        shape=tuple(case['shape'])
        for chip in CHIPS:
            p=case['predictions'][chip]
            rows[shape][chip]=dict(public_us=p['us'],placement_us=p['placement_us'],supply_us=p['supply_us'],
                                   future_public_us=case['predictions'][ca.FUT]['us'],
                                   scenarios=[dict(model=new['model'],node=new['node'],**s) for s in case['scenarios']])
    joined=[]
    lines=['# 同一投影从 decode 到整批 prefill','',
           '每个输入是单卡一次提交的连续 BF16 ND 矩阵。普通 decode 每请求贡献 1 行，MTP8 主模型校验贡献 9 行。八条未切 chunk 的 prefill 各贡献完整 S 行。表内展示独立库调用；服务侧的量化权重、融合和 chunk 调度会改变实际派发。','',
           '| 模型 · 投影 | 阶段 B×Q | [M,N,K] | DT 预测 / CA µs | PR 预测 / CA µs | 970DT 外推 µs | 库 host 选择 DT / PR |',
           '|---|---|---|---:|---:|---:|---|']
    for shape,targets in sorted(rows.items(),key=lambda x:(x[1][ca.DT]['scenarios'][0]['phase'],x[1][ca.DT]['scenarios'][0]['model'],x[0])):
        s=targets[ca.DT]['scenarios']; scen='、'.join(f"{x['batch']}×{x['tokens_per_request']}" for x in s)
        cells=[];branches=[];devices={}
        for chip in CHIPS:
            seen=archived.get((shape,chip))
            hostrow=host[chip].get(shape)
            branch=None
            if hostrow:
                observed=hostrow['workspace_observed']
                branch={512:'纯 Cube',512+host_audit.SOCS[chip]*262144:'切 K + 归并'}.get(observed,'未识别')
            branches.append(branch or '—')
            measurement=val(seen['ca_us']) if seen else '—'
            if seen:measurement=f"[{measurement}](../../../{seen['source']})"
            cells.append(f"{val(targets[chip]['public_us'])} / {measurement}")
            devices[chip]=dict(public_us=targets[chip]['public_us'],placement_us=targets[chip]['placement_us'],
                               ca=seen,host_workspace_bytes=hostrow['workspace_observed'] if hostrow else None)
        name=s[0]['model']+' · '+s[0]['node']
        future=targets[ca.DT]['future_public_us']
        joined.append(dict(shape=shape,scenarios=s,targets=devices,extrapolated_970_us=future))
        lines.append(f"| {name} | {s[0]['phase']} {scen} | {list(shape)} | {' | '.join(cells)} | {val(future)} | {' / '.join(branches)} |")
    direct=sum(all(v['ca'] for v in r['targets'].values()) for r in joined)
    lines[2:2]=[f"本表涵盖 **{len(rows)} 个场景几何**。其中 {direct} 个已有双端完整设备时间。新增 host 对照每端查询 {len(host[ca.DT])} 个几何，含 3 个旧阳性/阴性控制和 28 个其他几何。各端命中分别为 "+'、'.join(f"{sum(bool(x['agreement']) for x in host[c].values())}/{len(host[c])}" for c in CHIPS)+'。同一 [M,N,K] 的 B1×9 与 B9×1 共用一次测量。','']
    lines+=['','DT 和 PR 实际调用 CANN 9.2.1 的 aclnnMatmulGetWorkspaceSize，逐形使用真实维度和足额分配的张量。表中 host 选择根据已做过设备复核的两类 workspace 签名解释。512 B 指示普通模板，512+262144×AIC数 B 指示带部分和工作区的模板。签名只能检验这两类分支；确切 tile、MMAD 发起数及设备时间需要完整设备窗。两个 SKU 的 host 整批墙钟分别为 36.3、42.6 秒。','',
            '新增 Flash indexer.wk 设备合同固定 N=128、K=512。七个 decode 几何共享同一权重，另测 B8×64Ki，共 16 个端点，只开一个 CA 槽。64Ki 是 32Ki 和 128Ki 两个已有 DT 大形之间的检查点，用于判断 L2 脏输出开始溢出后斜率是否可迁移。未知时间在表内留空，每个正式回执保留全输出数值、发行 ELF、逐核 END 与退休证据。续收回执后运行 report.py 即可更新。','',
            '[两卡 host 预测与输入来源](freeze.json) · [DT 原始 host 回执](Ascend950DT_9582-workspace.json) · [PR 原始 host 回执](Ascend950PR_957b-workspace.json) · [新设备留出与芯片比值](device/report.md) · [设备采前冻结与预算](device/freeze.json) · [设备作业状态](device/state.json) · [全表 JSON](coverage.json) · [已有 33 形档案](../../v0.29/matmul/workloads.md)','']
    (OUT/'coverage.json').write_text(json.dumps(dict(utc=ca.now(),rows=joined,distinct=len(rows),paired_device=direct),indent=2)+'\n')
    (OUT/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':render()
