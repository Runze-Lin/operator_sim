"""No model edits. Diagnose signed errors and layout from independent measured timelines."""
import sqlite3,statistics
from campaign import *
def report():
 rows=load(OUT/'holdout.json'); result=[]
 for r in rows:
  op=r['op'];d=RUNS/r['id'];pr=r['pr'];cores=pr['core_windows'];last=max(cores,key=lambda c:c['end_us']);ph=pr['phase_landmarks']; dt=r['dt']
  rec=dict(op=op,shape=r['shape'],id=r['id'],pr_ca_us=pr['ca_us'],frozen_error_percent=100*(r['pr_us']/pr['ca_us']-1),kernel=r['kernel'],latest_finishing_core=last['core'],window_us=pr['ca_us'],active_predicted=[r['pr_aic'],r['pr_aiv']],active_actual=[r['active_aic'],r['active_aiv']],core_end_spread_us=last['end_us']-min(c['end_us']for c in cores),critical_landmarks=ph.get(last['core'],{}),mmad_events=pr['selected_timeline_counts']['MMAD'])
  if (d/'landmarks.db').is_file():
   with sqlite3.connect((d/'landmarks.db').resolve().as_uri()+'?mode=ro',uri=True) as db:
    by_core=dict(db.execute("select core,count(*) from landmarks where opcode='MMAD' group by core")); assert sum(by_core.values())==rec['mmad_events']
    rec.update(landmarks_sha256=sha(d/'landmarks.db'),db_bytes=(d/'landmarks.db').stat().st_size,mmad_events_by_core=by_core)
    if len(by_core)==1:
     entries=list(db.execute("select ts_us,duration_us from landmarks where opcode='MMAD' order by ts_us"))
     rec.update(first_to_last_mmad_end_us=entries[-1][0]+entries[-1][1]-entries[0][0],median_adjacent_mmad_issue_gap_us=statistics.median(b[0]-a[0] for a,b in zip(entries,entries[1:])))
  if dt:
   rec.update(dt_ca_us=dt['ca_us'],dt_pred_error_us=r['dt_us']-dt['ca_us'],transfer_delta_error_us=(r['pr_us']-r['dt_us'])-(pr['ca_us']-dt['ca_us']))
  result.append(rec)
 write(OUT/'posthoc.json',dict(role='independent outcomes examined only after estimator freeze; features below not used as inputs',rows=result))
 lines=['# 冻结后的独立样例诊断','','记录实际活跃核、最后退出的核以及每个算例的 MMAD 指令事件数。末核可为 AIV，其没有 MMAD 不代表 FA 无 Cube 工作。发起间隔统计混合 QK/PV 与块边界，不作为独立的读延迟。','','| 算子 / 单卡 shape | PR CA µs | 有符号误差 | 预测 → 实际活跃 (C,V) | MMAD 事件 | 首末退出核时间差 µs |','|---|---:|---:|---|---:|---:|']
 for r in result:
  lines.append(f"| {r['op']} `{r['shape']}` | {r['pr_ca_us']:.3f} | {r['frozen_error_percent']:+.2f}% | {r['active_predicted']} → {r['active_actual']} | {r['mmad_events']} | {r['core_end_spread_us']:.3f} |")
 lines+=['','原始 msOpProf trace 的 SHA-256、实际库 kernel、全部核心起止窗和数值正确性校验见 [holdout.json](holdout.json)。长 FA 的补充 SQLite 仅保存指定 opcode 的核心、时间戳与持续时间；完整计时仍来自全核官方 trace。逐点的阶段分解及 DT/PR 配对误差见 [posthoc.json](posthoc.json)。']
 (OUT/'posthoc.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':report()
