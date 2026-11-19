"""Recover actual AIV & sibling-domain membership from ordered device windows, not from elapsed time."""
from collections import defaultdict
from pathlib import Path
import statistics,sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import read,write
OUT=ROOT/'results/v0.24/post'
def main():
    data=[];indices=sorted((OUT/'physical').glob('*.json'));official=read(OUT/'evaluation.json')['rows']
    strict={(r['chip'],r['group'],r['i']):r for r in official}
    for path in indices:
        rows=read(path)['rows'];resident=set();last_kernel=None
        for r in rows:
            if not r['qualified']:
                resident=None;continue
            if resident is None:continue
            cur=r['per_core_physical_ticks'];origin=min(a for a,_ in cur.values());final=max(b for _,b in cur.values());critical=[k for k,(_,b) in cur.items() if b==final]
            groups=defaultdict(list);parent=lambda x:x.split('.')[0];parents={parent(n) for n in resident}
            kinds={k:('issued' if k in resident else 'new sibling' if parent(k) in parents else 'new physical core') for k in cur}
            for k,(a,b) in cur.items():groups[kinds[k]].append([(a-origin)/1650,(b-origin)/1650])
            times={k:dict(n=len(v),p50_start_us=statistics.median(x for x,y in v),max_start_us=max(x for x,y in v),p50_finish_us=statistics.median(y for x,y in v),max_finish_us=max(y for x,y in v)) for k,v in groups.items()}
            q=strict.get((r['chip'],r['group'],r['ordinal']),{})
            resolved=q.get('kernel') or Path(r['trace']).parents[2].name
            data.append(dict(chip=r['chip'],group=r['group'],i=r['ordinal'],shape=r['shape'],role=r['role'],state=r['state'],actual_prior=len(resident),active=len(cur),current_kernel=resolved,
                             previous_kernel=last_kernel,critical=critical,critical_cohort=[kinds[k] for k in critical],cohorts=times,ca_physical_us=r['ca_us'],pred_known_us=r['candidate_us'],strict_us=q.get('ca_us')))
            resident.update(cur);last_kernel=resolved
    archive=read(ROOT/'results/post-candidate/phase-evidence.json')
    old=[x for x in archive if x['state']=='expand']
    old_summary={}
    for key,pred in [('one prior AIV',lambda h:h==1),('two or more prior AIVs',lambda h:h>=2)]:
        xs=[x for x in old if pred(x['prior_aiv'])]
        old_summary[key]=dict(n=len(xs),terminal_arrival_us=[x['phase_us']['arrival'] for x in xs],median_us=statistics.median(x['phase_us']['arrival'] for x in xs))
    write(OUT/'topology.json',dict(physical_scope='Membership in previously issued lanes/physical core. Prior actual activity alone does not establish data or code cache residency. Per-core first and physical last instruction.',archive_expand=old_summary,rows=data))
    md=['# Post 扩核与隔形回访','', '按同一进程各个已经派发过的真实 AIV 名单重建历史。若某个新 AIV 的同物理核兄弟此前已执行，将其标为“新兄弟”，整个物理核均未参与过则标为“新物理核”。该分类来自设备核标识，不预设工作集留在 L2、UB 或指令缓存中。每次设备时间独立从最早物理指令开始。', '', '| SKU | shape / 状态 | 上次家族内已用核数 → 本次核数 | 最近一次/当前 ELF 函数后缀 | 最后完成的核与类别 | 已用 / 新兄弟 / 新物理核的最晚加入 µs | 预测 / 物理窗 µs |','|---|---|---:|---|---|---|---:|']
    for r in data:
        if r['state']!='expand' and r['role']!='revisit' and r['current_kernel']==r['previous_kernel']:continue
        suffix=lambda n:n.rsplit('_',1)[-1] if n else '—'
        a=' / '.join(f"{r['cohorts'][k]['max_start_us']:.3f}" if k in r['cohorts'] else '—' for k in ('issued','new sibling','new physical core'))
        md.append(f"| {r['chip'][6:11]} | {r['shape']} / {r['role'] if r['role']=='revisit' else r['state']} | {r['actual_prior']} → {r['active']} | {suffix(r['previous_kernel'])} / {suffix(r['current_kernel'])} | {', '.join(k+' ('+g+')' for k,g in zip(r['critical'],r['critical_cohort']))} | {a} | {r['pred_known_us']:.3f} / {r['ca_physical_us']:.3f} |")
    md+=['',f"旧训练集中扩核前只用过 1 个 AIV 时，2 个末核的加入偏移中位值是 {old_summary['one prior AIV']['median_us']:.3f} µs；其余 10 个扩核窗中位值为 {old_summary['two or more prior AIVs']['median_us']:.3f} µs。旧扩核证据全部来自 DT。新样本可直接检验先前有一个 AIV 与已经覆盖一对 AIV 这两种历史，但这两个时延分组还不能唯一确定延迟提升来自哪一级硬件。", '', '[每次调用的实际物理核分组及队列起止](topology.json)']
    (OUT/'topology.md').write_text('\n'.join(md)+'\n')
if __name__=='__main__':main()
