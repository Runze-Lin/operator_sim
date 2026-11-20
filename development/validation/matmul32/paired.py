"""Complement the timing report with same-shape transfer tests, including direct DT reuse."""
from statistics import median
import json
from freeze import OUT,DT,PR
from score import save

def render():
    doc=json.loads((OUT/'score.json').read_text())
    paired=[]
    for v in doc['ratios']:
        rows=[r for r in doc['rows'] if r['shape']==list(v['shape'])]
        dt=[r for r in rows if r['chip']==DT];pr=[r for r in rows if r['chip']==PR]
        if not dt or not pr:continue
        role='B 常驻开发回放' if dt[0]['role']=='mechanism development' else '冻结留出' if 'freeze' in dt[0]['role'] or 'acquisition' in dt[0]['role'] else '其他合格算例'
        paired.append(dict(shape=v['shape'],role=role,true_dt=median(r['ca_us'] for r in dt),true_pr=median(r['ca_us'] for r in pr),
                           old_pr=pr[0]['old_us'],current_pr=pr[0]['new_us'],reuse_dt=dt[0]['new_us'],pr_over_dt_measured=v['ca'],pr_over_dt_old=v['old'],pr_over_dt_predicted=v['new']))
    summary={}
    for name,case in [('全体同形双端',paired),('B 常驻开发回放',[r for r in paired if r['role']=='B 常驻开发回放']),('冻结留出',[r for r in paired if r['role']=='冻结留出'])]:
        if not case:
            summary[name]={'pairs':0};continue
        ca=sum(r['true_pr'] for r in case);ratio=sum(r['pr_over_dt_measured'] for r in case)
        summary[name]=dict(pairs=len(case),pr_wape={key:100*sum(abs(r[key]-r['true_pr']) for r in case)/ca for key in ('old_pr','current_pr','reuse_dt')},
                           ratio_wape={key:100*sum(abs((1 if key=='reuse_dt' else r[key])-r['pr_over_dt_measured']) for r in case)/ratio for key in ('pr_over_dt_old','pr_over_dt_predicted','reuse_dt')})
    save('paired.json',json.dumps(dict(input_timestamp=doc['utc'],summary=summary,rows=paired),indent=2,ensure_ascii=False)+'\n')
    lines=['# 同形跨芯片：直接用 DT 预测作基线','','每个形状先取芯片内所有完整 CA 窗的中位数，再配对。直接复用 DT 指 **两端都返回当前模型的 DT 预测时间**，因此该基线预测的 PR/DT 恒为 1；没有把 DT 的实测时间作为预测输入。比值 WAPE 的分母为所有实测比值之和。','', '| 数据角色 | 成对形数 | PR 时间 WAPE：v0.31 / v0.32 / 直接 DT | PR/DT 比值 WAPE：v0.31 / v0.32 / 直接 DT |','|---|---:|---:|---:|']
    for name,v in summary.items():
        fmt=lambda vs:' / '.join(f'{r:.2f}%' for r in vs.values())
        lines.append(f"| {name} | {v['pairs']} | "+(f"{fmt(v['pr_wape'])} | {fmt(v['ratio_wape'])}" if v['pairs'] else '— | —')+' |')
    lines+=['','[所有成对几何及三种预测](paired.json) · [逐窗全表与冻结条件](report.md)'];save('paired.md','\n'.join(lines)+'\n')
if __name__=='__main__':render()
