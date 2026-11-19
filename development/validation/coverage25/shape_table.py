"""Human-readable per-window lookup; fingerprints and original holdout roles stay in audit.json."""
from collections import defaultdict
from pathlib import Path
import json
from audit import OUT,read


def render():
    data=read(OUT/'audit.json')
    grouped=defaultdict(list)
    for r in data['rows']:grouped[r['op']].append(r)
    table=['# 全路线逐窗输入与误差','',
           '时间单位 µs。每行是一份唯一物理 trace 或有序的多个 kernel 窗。v0.25 实测为冻结之后新采集，其余均为本次开发回放；历史留出标签和逐核证据可用 SHA 查 JSON。归档 SQL、采集宿主和完整性门禁表可从每个窗口的来源追溯。970DT 均为外推。[全量收据和原始角色](audit.json) · [统计口径](report.md)']
    for op,rr in sorted(grouped.items()):
        table+=['',f'## {op}','','| SKU | shape | 调用属性 | 当前角色 | v0.24 | 当前 | CA | 偏差 | 970DT 外推 | 首段 SHA 前缀 |',
                '|---|---|---|---|---:|---:|---:|---:|---:|---|']
        for r in rr:
            attrs=json.dumps(r['attrs'],ensure_ascii=False,separators=(',',':'))
            err=100*(r['current_us']/r['ca_us']-1)
            table.append(f"| {r['chip'][6:11]} | {r['shape']} | `{attrs}` | {'新留出' if r['role'].startswith('prospective') else '回放'} | {r['v024_us']:.3f} | {r['current_us']:.3f} | {r['ca_us']:.3f} | {err:+.2f}% | {r['future_970dt_us']:.3f} | {r['trace_sha256'][0][:12]} |")
    return '\n'.join(table)+'\n'


if __name__=='__main__':
    (OUT/'shape-errors.md').write_text(render())
