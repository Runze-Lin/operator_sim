"""One current 22-route coverage matrix. Separate kernel contracts, protocols and roles."""
from collections import defaultdict
from pathlib import Path
import datetime,json,math,os,sys,time,fcntl
R=Path(__file__).resolve().parents[3];O=R/'results/v0.32/large';B=R/'results/v0.29/large'
DT,PR,F='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'
def get(p):return json.loads(p.read_bytes())
def rel(p):return str(p.relative_to(R))
def write(p,s):
 tmp=p.with_suffix('.'+str(os.getpid())+'.tmp');tmp.write_text(s);tmp.replace(p)
def fmt(x):return 'N/A' if x is None else f'{x:.3f}'
def render():
 base=get(B/'coverage.json'); rows=[]; old='v0.29 large register'
 def add(op,sh,c,p,m,scope,role,mode,state,src,**k):rows.append(dict(op=op,shape=sh,chip=c,forecast_us=p,measured_us=m,contract=scope,role=role,protocol=mode,status=state,source=src,**k))
 for r in base['rows']:
  src=R/r['source'] if (R/r['source']).is_file() else B/r['source']
  add(r['op'],r['shape'],r['sku'],r['model_us'] if r['model_us'] is not None else r['conditional_us'],r['ca_us'],r['role'],old,r['protocol'],r['qualification'],rel(src) if src.is_file() else rel(B/'coverage.json'))
 for r in get(R/'results/v0.31/recurrent/score.json')['rows']:
  add('kda_recurrent',[r['b'],1,12,12,128,128],r['chip'],r['predicted_us'],r['measured_ca_us'],'dense BF16 S1 FP32 gate and state','v0.31 prospective','full',r['status'],r['receipt'] or 'results/v0.31/recurrent/score.json')
 for c,p in get(O/'recurrent-freeze.json')['prices'].items():
  rc=O/('recurrent16-'+('dt' if c==DT else 'pr')+'.json');v=get(rc)['verdict'] if c!=F and rc.is_file() else {}
  add('kda_recurrent',p['shape'],c,p['latency_us'],v.get('ca_us') if v.get('qualified') else None,'dense BF16 S1 FP32 gate and state','v0.32 prospective','full','qualified' if v.get('qualified') else v.get('reason') or ('future scenario' if c==F else 'see recurrent-state.json'),rel(rc if v else O/'recurrent-freeze.json'),rounds=p['chains_per_busiest_aiv'],active_aiv=p['active_aiv'])
 sink=get(O/'sink-score.json')['rows']
 for r in sink:
  t,c=r['shape'][0],r['chip'];src=(O/('sink1024-'+c+'.json')) if t==1024 else O/'archive.json'
  add('mhc_sinkhorn',r['shape'],c,r['known_us'],r['measured_us'],f'FP32 N4 I={r["iterations"]}; '+('isolated cold' if t==1024 else 'ordered wide stream'),r['role'],'full','qualified' if r['qualified'] else r['reason'] or ('future scenario' if c==F else 'see state.json'),rel(src if src.is_file() else O/'freeze.json'),resident_aiv=r['H'],unknown_us=r['unknown_us'],passes=r['passes'])
 for r in get(R/'results/v0.31/post/score.json')['rows']:
  src=R/'results/v0.31/post'/(r.get('receipt') or 'score.json')
  add(r['op'],r['shape'],r['chip'],r['public_us'],r['ca_us'],'N4 D5120 1024 tokens','v0.31 prospective','full',r['stage'],rel(src) if src.is_file() else 'results/v0.31/post/score.json')
 pairs={o:{tuple(s) for s in group} for o,group in base['complete_same_shape_pairs_by_op'].items()};group=defaultdict(set)
 for r in rows:
  if r['role']!=old and r['measured_us'] is not None and r['chip'] in (DT,PR):group[(r['op'],tuple(r['shape']),r['contract'],r['protocol'])].add(r['chip'])
 for (op,s,_,__),a in group.items():
  if a=={DT,PR}:pairs.setdefault(op,set()).add(s)
 lines=['# 大几何覆盖：22 条算子路线','',f'按同形、同合同、双端完整计时统计，**{sum(bool(v) for v in pairs.values())}/{len(pairs)}** 条路线至少有一个合格大形；不是 22 条都已验证。两种较长的 Sinkhorn [513,4] 与 [4097,4] 都是历史开发回放，迭代数分别 6、4。没有把它们算成生产配置 20 次迭代的独立校验。','','| 算子 | 登记形状示例（小端 → 大端） | 双端完整的大形代表：DT / PR µs | 数据 |','|---|---|---|---|']
 for op,ss in sorted(pairs.items()):
  rs=[r for r in rows if r['op']==op];s=sorted({tuple(r['shape']) for r in rs},key=lambda x:(math.prod(x),x));best=max(ss,key=math.prod) if ss else None;show=lambda x:'×'.join(map(str,x))
  hit={r['chip']:r['measured_us'] for r in rs if tuple(r['shape'])==best and r['measured_us'] is not None}
  label=show(best)+': '+fmt(hit.get(DT))+' / '+fmt(hit.get(PR)) if best else 'N/A'
  src=next((r['source'] for r in rs if r['source'].endswith('freeze.json')),rs[0]['source'])
  if op=='kda_recurrent':src=rel(O/'recurrent-freeze.json')
  if op=='mhc_sinkhorn':src=rel(O/'sink-score.json')
  lines.append(f'| {op} | '+ ' → '.join(show(x) for x in ([s[0]] if s[0]==s[-1] else [s[0],s[-1]]))+f' | {label} | [明细]({os.path.relpath(R/src,O)}) |')
 lines+=['','### 每条路线的 DT / PR 状态','','表中为已登记大形的原始状态汇总。完整数按具有合格指令窗的不同 shape 去重，wrapper 另列；非终态仍需查看正在执行该次采集的日志，不能把“已登记”当作已经运行。一个已有完整窗的算子仍可能有其他未验证的大形。','','| 算子 | DT | PR |','|---|---|---|']
 for op in sorted(pairs):
  cells=[]
  for c in (DT,PR):
   x=[r for r in rows if r['op']==op and r['chip']==c]; qual={tuple(r['shape']) for r in x if r['measured_us'] is not None and r['protocol']!='wrapper'};partial={tuple(r['shape']) for r in x if r['measured_us'] is not None and r['protocol']=='wrapper'}
   un=[r['status'] for r in x if r['measured_us'] is None]
   denied=any(z in ('unqualified','failed','unavailable') or 'fail' in z or 'incomplete' in z for z in un)
   cell=f'完整 {len(qual)} 形' if qual else '完整 N/A'
   if partial:cell+=f'；wrapper {len(partial)} 形'
   if denied:cell+='；有未合格路线'
   if any(z in ('acquiring','running') for z in un):cell+='；有采集中'
   if any(z in ('queued','pending') or 'see ' in z or 'awaiting' in z for z in un):cell+='；有登记待回执'
   cells.append(cell)
  lines.append(f'| {op} | '+' | '.join(cells)+' |')
 lines+=['','[原有 22 条路线的逐形口径](../../v0.29/large/report.md)。DT、PR 必须分别有同几何、同合同的合格完整窗口；个别 wrapper 包络单列，不列为完整指令窗。量化和 RMS 已有的 [2048,5120] 双端窗是 2048-token chunk 的控制，不等于 8192/262144 行整批验证。TopK [4096,896,16]、RMS [8192,5120] 的原任务仍由共享槽调度。未合格、等待与 970DT 外推均无 CA 数值。','','## FP32 standalone MhcSinkhorn','','| 形状 / 迭代数 / 角色 | DT 已知 H / 未知 H / CA µs | PR 已知 H / 未知 H / CA µs | 970DT 已知 H / 未知 H 外推 µs |','|---|---:|---:|---:|']
 for t in (513,4097,1024):
  g={r['chip']:r for r in sink if r['shape'][0]==t}; cols=[' / '.join(fmt(g[c][k]) for k in (['known_us','unknown_us','measured_us'] if c!=F else ['known_us','unknown_us'])) for c in (DT,PR,F)]
  lines.append(f'| [{t},4] / {g[DT]["iterations"]} / '+('冻结新形' if t==1024 else '旧形回放')+' | '+' | '.join(cols)+' |')
 lines+=['','[1024,4] 对应每卡 B8、每请求本次调度 128 token 的 chunk，不等于在设备上以一次调用处理整条 32Ki 输入。每端首发 H=0、20 次交替归一化，检查全部 16,384 个 FP32 输出。历史 H=57/52 的有限迭代控制分别报告。已知 H 的值使用相应分支；无 H 按可达状态均值，不把均值叫作实际运行历史。真实接口 aclnnMhcSinkhorn，CANN 9.2.1 安装版 ops-transformer 二进制。保留[模型参数和旧版基线](freeze.json)、[原始/汇总核验](archive.json)、[同集误差和墙钟](sink-score.json)。','','## Kimi decode：独立状态链','','| 单卡 B / 单卡形状 | DT 预测 / CA µs | PR 预测 / CA µs | 970DT 外推 µs |','|---|---:|---:|---:|']
 for b in (6,16,128):
  d={r['chip']:r for r in rows if r['op']=='kda_recurrent' and r['shape'][0]==b and r['role']!=old}
  lines.append(f'| {b} / {d[DT]["shape"]} | '+' | '.join(fmt(d[c]['forecast_us'])+(' / '+fmt(d[c]['measured_us']) if c!=F else '') for c in (DT,PR,F))+' |')
 lines+=['','B16 TP8 使用 Kimi 96/8=12 个本地头、K=V=128。192 条链使 DT 每核 3 轮、PR 最忙核 4 轮。B6/B128 不重复；B16 在新 Sinkhorn 两端结束后复用同一把设备锁，每端最多 1 小时、8 GiB。完整时间要通过全输出金标、发布库 .text、全核官方时长与 END、所有链的 beta 计数及状态写体积。970DT 始终无 CA。','', '[逐形完整 JSON](coverage.json) · [Sinkhorn 当前状态](state.json) · [B16 当前状态](recurrent-state.json) · [B16 冻结与预算](recurrent-freeze.json)','']
 lines.extend(['### 验收口径与开销','',
 '已重验历史循环的四个 950 窗，T=513/I6 每核都是 14 次 VF、总写 32,832 B；T=4097/I4 每核都是 10 次 VF、总写 262,208 B。这符合每核运行 2I+2 个 helper，寄存器的多次遍历发生在 helper 内。T=4097 同已知 H 时，旧模型 DT/PR 高估 26.37% / 37.53%，新模型偏差 −6.79% / −0.17%；H 不传时新模型偏差 −9.58% / −3.26%。T=513 H 不传的偏差仍有 +28.12% / +32.08%，不能据此宣称全部达标。四个窗来自同两次有序四连发；每个芯片整批用 248/252 秒，不能把该墙钟归到某一个 shape。',
 '', '新增 DT/PR 用官方 msopprof simulator --dump=false，只收集判定整窗所需的合并/分核指令和 kernel 对象，保留全部输出金标；不重复抓取逐指令文本 dump。历史结果逐端复核输出字节、真实内存控制器、每核 END 与官方表、全输出、trace 哈希与安装 .text。当前 CPU 利用率与部分日志不能换算成可计分的 CA 时间。',
 '', '[主机进程/锁的实际工作与等候情况](host-audit.json) · [3 秒 CPU 样本](host-sample.json)。主机有 256 个可用 CPU；采样时平均约 25.5 个 CPU 等效空闲，Sinkhorn CA 实际用了 14.5 个 CPU；高负载下旧实验耗时不是这轮的 ETA。', ''])
 lines+=['[每条线路最大的旧双端参照与大形目标](scale-ladder.md) · [资格、同集误差与主机开销](costs-and-evidence.md)','']
 # Numeric statements above describe immutable archive receipts, not the new endpoint.
 j=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_snapshot=base['snapshot_utc'],complete_same_shape_pairs_by_op={k:sorted(v) for k,v in pairs.items()},rows=rows)
 write(O/'coverage.json',json.dumps(j,indent=2,allow_nan=False)+'\n');write(O/'report.md','\n'.join(lines))
def watch():
 # Watch receipts, not a fixed delay between recomputations. No CANN launches.
 dirs=[O,R/'results/v0.31/post',R/'results/v0.31/recurrent',B]
 fixed=[B/'coverage.json',O/'sink-score.json',R/'results/v0.31/post/score.json',R/'results/v0.31/recurrent/score.json',O/'recurrent16-dt.json',O/'recurrent16-pr.json']
 evidence=lambda:(tuple(p.stat().st_mtime_ns if p.exists() else None for p in fixed))
 stop=time.monotonic()+60*3600
 with (R/'runs/coverage32/inventory.lock').open('a') as lk:
  fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB);prev=None
  while time.monotonic()<stop:
   new=evidence()
   if new!=prev:
    import subprocess
    subprocess.run([sys.executable,__file__],check=True);prev=new
   if (O/'recurrent16-pr.json').exists():return
   time.sleep(45)
if __name__=='__main__':(watch if '--watch' in sys.argv else render)()
