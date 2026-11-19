"""Compare the two registered stream policies on the same qualified target launches."""
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.mixed_history25.campaign import OUT, write


def score(rows, key):
    err = [abs(r[key]-r['ca_us']) for r in rows]
    return {'n':len(rows), 'wape':sum(err)/sum(r['ca_us'] for r in rows),
            'max':max(e/r['ca_us'] for e,r in zip(err,rows))}


def main():
    plan = json.loads((OUT/'independent-freeze.json').read_text())
    future = {(r['group'],r['i']):r for r in plan['rows'] if r['chip']=='Ascend970DT'}
    observed = []
    rejected = []
    cost = []
    accepted=set()
    files=sorted(OUT.glob('independent-*-windows.json'))+sorted(OUT.glob('recovery-*-windows.json'))
    for p in files:
        receipt = json.loads(p.read_text())
        cost.append(receipt['cost'])
        for i,r in enumerate(receipt['rows']):
            if not r['target']:
                continue
            key=r['chip'],r['group'],r['i']
            if key in accepted:
                continue
            # The test concerns a physical AIC-only predecessor, with complete evidence
            # of every successfully executed RMS launch contributing to its high-water.
            prefix = receipt['rows'][:i+1]
            if not all(s['qualified'] for s in prefix) or not prefix[-2]['observed_aic'] or prefix[-2]['observed_aiv']:
                rejected.append({'chip':r['chip'],'group':r['group'],'i':i,'source':p.name,
                                 'reasons':[s.get('reason','ineligible engine mask') for s in prefix if not s['qualified']]})
                continue
            accepted.add(key)
            H=r['previous_aiv']
            state = 'first' if H==0 else 'expand' if H<r['active_aiv'] else 'reuse'
            observed.append(dict(chip=r['chip'],group=r['group'],i=i,shape=r['shape'],state=state,previous_aiv=H,source=p.name,
                                 public_us=r['public_us'],preserve_us=r['preserve_us'],ca_us=r['ca_us'],
                                 future_us=future[r['group'],i]['preserve_us'],trace_sha256=r['trace_sha256']))
    buckets=defaultdict(list)
    for r in observed:
        for state in ['all',r['state']]:
            buckets[r['chip'],state].append(r)
    metrics=[dict(chip=c,state=s,old=score(rows,'public_us'),model=score(rows,'preserve_us'))
             for (c,s),rows in sorted(buckets.items())]
    lines=['# RMSNorm 跨纯 AIC 调用的冻结验证', '',
           '| 芯片 | RMS 前序 | 完整新形 | v0.24 WAPE / 最大 | 保留 AIV 历史 WAPE / 最大 |',
           '|---|---|---:|---:|---:|']
    for m in metrics:
        def pct(k):
            return f"{m[k]['wape']:.2%} / {m[k]['max']:.2%}"
        lines.append(f"| {m['chip']} | {m['state']} | {m['model']['n']} | {pct('old')} | {pct('model')} |")
    lines+=['','设备时间从全核首条指令至最后一核退休。每次目标 RMS 之前的 Addmm 都有完整全输出检查、安装包 ELF、纯 AIC 执行与 MMAD 的证据。同一进程中的前序也必须完整。三个独立进程分别覆盖扩核、充分复用及多行。每行仅在双端均有合格窗口时计入加速比。970 来自预先登记的独立芯片配置。','',
            '| RMS shape | 前序 AIV，DT / PR | DT 旧 / 新 / CA µs | PR 旧 / 新 / CA µs | DT/PR 旧 / 新 / CA | 970DT 外推 µs |',
            '|---|---|---:|---:|---:|---:|']
    def triple(row):
        return ' / '.join(f'{row[k]:.3f}' for k in ('public_us','preserve_us','ca_us')) if row else '待合格窗口'
    paired=[]
    for case in [r for r in plan['rows'] if r['chip']==plan['chips'][0] and r['target']]:
        rows=[next((r for r in observed if r['chip']==c and r['shape']==case['shape']),{}) for c in plan['chips']]
        dt,pr=rows
        ratio={k:dt[k]/pr[k] for k in ('public_us','preserve_us','ca_us')} if all(rows) else {}
        if ratio:paired.append(ratio)
        h=' / '.join(str(r.get('previous_aiv','—')) for r in rows)
        lines.append(f"| {case['shape']} | {h} | {triple(dt)} | {triple(pr)} | {triple(ratio)} | {future[case['group'],case['i']]['preserve_us']:.3f} |")
    acceleration = {k:score(paired,k) for k in ('public_us','preserve_us')} if paired else None
    if paired:
        lines+=['',f"上表 {len(paired)} 对 DT/PR 比值的旧 / 新 WAPE 为 {acceleration['public_us']['wape']:.2%} / {acceleration['preserve_us']['wape']:.2%}。"]
    n_primary=sum(p.name.startswith('independent-') for p in files)
    lines+=['',f"已核验 {n_primary}/6 个原始进程及 {len(files)-n_primary} 个按原顺序复采的进程。"
           f"冻结的 20 个目标中有 {len(observed)} 个合格，不重复计分。"
           f"共排除 {len(rejected)} 次缺少完整前序或本窗的观测，原始原因保留在 JSON。"
           f"已核验进程总墙钟 {sum(c['wall_s'] for c in cost)/60:.2f} 分钟。"]
    lines+=['','[冻结输入与各次调用预测](independent-freeze.json) · [950DT 逐 tile 对照](dt-rms-tile.png) · [950PR 逐 tile 对照](pr-rms-tile.png) · [逐窗证据及比值](comparison.json)']
    occupancy_file=OUT/'occupancy-comparison.json'
    if occupancy_file.is_file():
        occ=json.loads(occupancy_file.read_text())
        if occ['metrics']:
            lines+=['','### 首次采集只核实前序用了多少 AIV 时','',
                    '以下另报原始六个进程。被评分的目标 RMS 以及紧邻的 Addmm 都有完整退出；DT 长组未计时的热身调用按每核入口、全核地址覆盖、输出金标、官方活动核名单及聚合/分片一致性复核。'
                    '其热身时延继续留空。该口径不用于补造原始提前退出的时间。','',
                    '| 芯片 | 合格目标窗 | v0.24 WAPE / 最大 | 本版 WAPE / 最大 |','|---|---:|---:|---:|']
            for c,m in occ['metrics'].items():
                b,a=m['public_us'],m['preserve_us']
                lines.append(f"| {c} | {a['n']} | {b['wape']:.2%} / {b['max']:.2%} | {a['wape']:.2%} / {a['max']:.2%} |")
            lines+=['','[独立核覆盖、资格和每个目标窗](occupancy-comparison.json) · [未计时的热身输出地址](longer-Ascend950DT_9582-warmup-mask.json)']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'comparison.json',dict(rows=observed,metrics=metrics,ratio=acceleration,rejected=rejected,cost=cost))


if __name__=='__main__':
    main()
