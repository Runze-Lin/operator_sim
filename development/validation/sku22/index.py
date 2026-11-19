"""Index source fingerprints, same-shape SKU comparisons and the validation costs."""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.sku21.acceptance import score, NAMES
from development.validation.sku19.campaign import CHIPS,read,write
OUT=ROOT/'results/v0.22'


def main():
    data=read(OUT/'evaluation.json');rows=data['rows'];by=defaultdict(dict)
    for r in rows+data['stream']:
        if r['cohort'] not in ('primary','ordered prospective','isolated first launch','same stream repeated'):continue
        key=(r['op'],tuple(r['shape']),json.dumps(r.get('attrs',{}),sort_keys=True),r['cohort'],r.get('ordinal'))
        if r['chip'] in by[key]:raise ValueError(key)
        by[key][r['chip']]=r
    pairs=[]
    for key,g in sorted(by.items()):
        if {CHIPS['pr'],CHIPS['dt']}!=set(g):continue
        d,p=g[CHIPS['dt']],g[CHIPS['pr']];assert abs(d['future_us']-p['future_us'])<1.e-7
        pairs.append(dict(op=key[0],shape=key[1],attrs=json.loads(key[2]),cohort=key[3],ordinal=key[4],
                          dt_ca=d['ca_us'],pr_ca=p['ca_us'],dt_us=d['us'],pr_us=p['us'],dt_old=d['v021_us'],pr_old=p['v021_us'],
                          measured_ratio=d['ca_us']/p['ca_us'],predicted_ratio=d['us']/p['us'],old_ratio=d['v021_us']/p['v021_us'],
                          direct_dt_as_pr_us=d['us'],future_us=d['future_us']))
    write(OUT/'pairs.json',dict(ratio='DT device window / PR device window',direct_reuse_ratio=1,rows=pairs))
    lines=['# 双端同一调用的加速比与误差','', '芯片间比值＝DT 耗时 / PR 耗时。直接复用 DT 作为 PR 的绝对耗时预测对应比值 1。时间单位为 µs。只配对相同输入与可确认的同一数据组；不同历史样本不强行配对。970DT 只有配置预测。','', '| 算子 / 前序 | shape | DT v0.21 / v0.22 / CA | PR v0.21 / v0.22 / CA | 比值 v0.21 / v0.22 / CA | 970DT 外推 |','|---|---:|---:|---:|---:|---:|']
    for r in pairs:
        lines.append(f"| {NAMES[r['op']]} · {r['cohort']} | {list(r['shape'])} | {r['dt_old']:.3f} / {r['dt_us']:.3f} / {r['dt_ca']:.3f} | {r['pr_old']:.3f} / {r['pr_us']:.3f} / {r['pr_ca']:.3f} | {r['old_ratio']:.3f} / {r['predicted_ratio']:.3f} / {r['measured_ratio']:.3f} | {r['future_us']:.3f} |")
    (OUT/'pairs.md').write_text('\n'.join(lines)+'\n')
    wheel=read(OUT/'wheel-check.json');sources=set((ROOT/'operator_model').glob('*.py'))
    for line in (ROOT/'MANIFEST.in').read_text().splitlines():
        if line.startswith('include '):sources.add(ROOT/line.split(maxsplit=1)[1])
    sources.update(ROOT/p for p in ['MANIFEST.in','pyproject.toml','setup.py']);digest=lambda p:sha256(p.read_bytes()).hexdigest()
    regs=['results/cast-capacity/two-slot-freeze.json','results/rms-dispatch-audit/freeze.json','results/v0.22/cast-freeze.json'];write(OUT/'model.json',dict(version='0.22.0',sources={str(p.relative_to(ROOT)):digest(p) for p in sorted(sources)},forecast_registrations={p:digest(ROOT/p) for p in regs},wheel_sha256=wheel['wheel_sha256']))
    changes=['# v0.22 · 单核写回与双缓冲','','| 路线和输入条件 | 芯片 | 完整新 shape | v0.21 WAPE / 最大 | v0.22 WAPE / 最大 |','|---|---|---:|---:|---:|']
    for op,which in [('rmsnorm','ordered prospective'),('cast_bf16_fp32','primary')]:
        for c in (CHIPS['dt'],CHIPS['pr']):
            g=[r for r in rows+data['stream'] if (r['op'],r['cohort'],r['chip'])==(op,which,c)]
            old,new=score(g,'v021_us'),score(g);changes.append(f"| {NAMES[op]} · {'连续派发' if op=='rmsnorm' else '无状态'} | {c[6:11]} | {new['shapes']} | {old['wape']:.2%} / {old['max_ape']:.2%} | {new['wape']:.2%} / {new['max_ape']:.2%} |")
    cast=[read(p) for p in (OUT/'cast-receipts').glob('*.json')];verified=[read(p) for p in (OUT/'recovered').glob('mhc*.json')];w=wheel['overall'];state=wheel['ordered']
    changes+=['','RMSNorm 已知前序且每核一轮时，单独计算同核新增 AIV、首次动用物理核和写同一 128 B 区段时的队列。仅给 shape 时保留 v0.21 未知历史规则，DT 部分旧组仍超限。Cast 把输入与输出的 UB 槽分别计价，完整块 DMA 的响应时间和平均发起间隔分别约束首块和后续块。两条路线均按目标频率、并行核数和访存端口重新定价；970DT 的派发分区和 DMA 响应系数继承 950，缺少 970 实测。','',
        f"Cast 沿用候选冻结后的 6 对独立样本，补登 4 对新尺寸；本快照新增 {len(cast)}/8 个端点回执，累计 CA 墙钟 {sum(r['cost']['seconds'] for r in cast)/60:.1f} 分钟。8 个 mHC 已完成计算但报告导出触顶的端点重新核验证据，其中 {sum(bool(r.get('qualified')) and r.get('new_ca_seconds')==0 for r in verified)} 个可复用原始采集。RMS 的有序十形沿用 20 个合格窗口（含两次按原前序恢复的目标），本次接入没有为这批结果重复启动 CA。",'',
        f"隔离安装 wheel 后，{w['n']:,} 个不同调用的 Python 墙钟中位 {w['median_ms']:.3f} ms、P95 {w['p95_ms']:.3f} ms、最大 {w['max_ms']:.3f} ms；另有 {state['n']} 次有序调用最大 {state['max_ms']:.3f} ms。模块导入 {wheel['import_ms']:.2f} ms 另计。[按算子查询开销](wheel-check.json)",'',
        'mHC Post 的跨族顺序、相同 Post 的复用以及孤立首发有不同的实测协议。同输入的无历史单点在 DT / PR 有超过 40% 的已证实最大误差下界。FLA 的 Chunk KDA V2 直接入口得到 1 个完整 DT 开发窗口（三段之和 23.241 µs），尚无独立时间预测；V1 未得到合格完整调用。[V2 逐阶段完整性](../chunk-recovery/fla2691/v2/verified.json)[逐算子总误差](../../docs/error-analysis.md) · [新旧同形双端与 970DT](pairs.md) · [跨族前序实测](post-stream/report.md) · [新图及模型推导](../../docs/pipeline-comparisons.md)']
    (OUT/'changes.md').write_text('\n'.join(changes)+'\n')

if __name__=='__main__':main()
