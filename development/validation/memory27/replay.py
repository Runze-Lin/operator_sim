"""Reprice qualified historical windows on this release. Development replay only."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model import estimate

OUT = ROOT / 'results/v0.27'
AUDIT = ROOT / 'results/v0.25/coverage/audit.json'
ALIASES = {'fla_kda_v2':'kda_chunk','vllm_hc_post':'mhc_post'}


def metrics(rows):
    scored = [r for r in rows if r['latency_us'] is not None]
    if not scored:
        return None
    worst = max(scored,key=lambda r: abs(r['latency_us']/r['ca_us']-1))
    return dict(windows=len(scored),shapes=len({tuple(r['shape']) for r in scored}),
                wape=100*sum(abs(r['latency_us']-r['ca_us']) for r in scored)/sum(r['ca_us'] for r in scored),
                maximum=100*abs(worst['latency_us']/worst['ca_us']-1),
                worst=dict(shape=worst['shape'],attrs=worst['attrs'],ca_us=worst['ca_us'],
                           latency_us=worst['latency_us'],trace=worst['trace_sha256']))


def main():
    groups = defaultdict(list)
    rows = []
    for ref in json.loads(AUDIT.read_bytes())['rows']:
        route = ref['op']
        sku = ref['chip']
        query = (ALIASES.get(route,route),ref['shape'],ref['attrs'])
        pred = estimate(*query,chip=sku)
        future = estimate(*query,chip='Ascend970DT')
        row = {k:ref[k] for k in ('op','chip','shape','attrs','ca_us','kernel','trace_sha256','slices')}
        row.update(latency_us=pred.latency_us, previous_us=ref['current_us'],
                   estimate_status=pred.status, future_970dt_extrapolated_us=future.latency_us)
        rows.append(row)
        for group in {'所有合格窗',*ref['slices']}:
            groups[route,sku,group].append(row)
    summary = [dict(op=op,sku=sku,group=group,result=metrics(values))
               for (op,sku,group),values in sorted(groups.items())]
    output = dict(role='Historical development replay under current public API. New paired validation and any partial or native wrapper windows are reported separately.',
                  registration_sha256=sha256(AUDIT.read_bytes()).hexdigest(),summary=summary,rows=rows)
    (OUT/'replay.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 全部旧指令窗按本版公开入口回放','',
           '同一组已去重的库调用与原始指令哈希。各时延单位为 µs，970DT 全是外推。RMSNorm 和内置 Post 的前序敏感协议在[当前误差总表](../../docs/error-analysis.md)单列。',
           '这些数据均已可用于开发。冻结后取得的独立样本另行判断。','',
           '| 路线 | SKU | shape / 窗 | WAPE / 最大 | 最大误差所在 shape | 预测 / CA µs |',
           '|---|---|---:|---:|---|---:|']
    for r in summary:
        if r['group']!='所有合格窗':
            continue
        s=r['result']
        if s is None:
            lines.append(f"| {r['op']} | {r['sku']} | 0 | N/A | — | — |")
            continue
        w=s['worst']
        lines.append(f"| {r['op']} | {r['sku']} | {s['shapes']} / {s['windows']} | {s['wape']:.2f}% / {s['maximum']:.2f}% | {w['shape']} | {w['latency_us']:.3f} / {w['ca_us']:.3f} |")
    lines.extend(['','[逐 shape 的 DT/PR 本版、旧版、CA、970DT 外推、实现身份与原始指令哈希](replay.json)'])
    (OUT/'replay.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
