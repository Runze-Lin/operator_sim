"""No refitting: errors against same PR CA on the predeclared independent shapes."""
from campaign import *
from statistics import median

def compare():
 rows=load(OUT/'holdout.json'); assert len(rows)==30
 summary={}
 for op in ['quant','matmul','fa']:
  group=[r for r in rows if r['op']==op]; assert len(group)==10 and len({tuple(r['shape']) for r in group})==10
  methods={}
  for key in ['pr_us','legacy_resource_us','dt_us','prior_pr_us']:
   errs=[r[key]-r['pr_ca_us'] for r in group]; pct=[abs(e)/r['pr_ca_us'] for r,e in zip(group,errs)]
   methods[key]=dict(wape=sum(map(abs,errs))/sum(r['pr_ca_us'] for r in group),max_ape=max(pct),median_ape=median(pct),mean_signed_us=sum(errs)/len(errs),worst_id=group[pct.index(max(pct))]['id'])
  info=dict(methods=methods,n=10,max_resource_query_ms=max(r['pr_query_ms']for r in group),dt_pair_count=sum(r['dt_ca_us'] is not None for r in group),cores_matched=sum((r['active_aic'],r['active_aiv'])==(r['pr_aic'],r['pr_aiv']) for r in group))
  summary[op]=info
 write(OUT/'comparison.json',summary)
 c=load(OUT/'campaign-cost.json')
 text=['# 950PR 独立验证：只使用资源的时延外推','','30 个正式 shape 在冻结后首次做 PR CA 采集；预留 6 个替补只用于采集失败。历史 DT CA 不参与选择 PR 赢家。','',
 '| 算子 | shape数 / 双端 CA 数 | v0.16 仅资源 | v0.15 匿名 PR | 直接复用 DT | v0.15 PR 专属校准 | AIC、AIV 均命中 |','|---|---:|---:|---:|---:|---:|---:|']
 for op,s in summary.items():
  metrics=['%.2f%% / %.2f%%'%(100*s['methods'][k]['wape'],100*s['methods'][k]['max_ape']) for k in ('pr_us','legacy_resource_us','dt_us','prior_pr_us')]
  text.append('| %s | 10 / %s | %s | %s/10 |'%(op,s['dt_pair_count'],' | '.join(metrics),s['cores_matched']))
 text+=['','各列为 WAPE / 最大逐点绝对相对误差。v0.16 resource 与 v0.16 auto(PR) 的估计逐点相等；旧 PR 专属列用隔离安装的 v0.15 生成，并与 v0.16 显式 calibrated(PR) 逐点核对。匿名 PR 基线保留原芯片资源，只改 chip name。',f"\n实际离线采集：{c['attempted']} 次（合格 {c['qualified']}），{c['wall_seconds']:.2f} 秒墙钟，已结算 ≥{c['simulator_cpu_seconds']/3600:.2f} CPU 小时（{c['unknown_cpu_rows']} 个被终止进程 CPU 未结算），{c['raw_bytes']/2**30:.2f} GiB。全部使用官方 SDK msopprof 和完整输出 golden，未重测 DT。",'','[逐点：含输入合同/核启动/真值/全部预测](holdout.json) · [冻结清单与 SHA-256](freeze.json) · [失败与成本](attempts.json) · [指令与末核窗口](posthoc.md)']
 (OUT/'comparison.md').write_text('\n'.join(text)+'\n')
if __name__=='__main__':compare()
