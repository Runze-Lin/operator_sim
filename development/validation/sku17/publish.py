"""Publish total release errors from preregistered forecasts and qualified measures."""
from collections import Counter
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[3]
V=ROOT/'results/v0.17'
SKU=V/'sku'
CORE=('quant','matmul','fa')
NN={'quant','matmul','rmsnorm','addmm'}
DT,PR='Ascend950DT_9582','Ascend950PR_957b'
sys.path.insert(0,str(Path(__file__).resolve().parent))
from campaign import OPS, estimate


def load(path):
    return json.loads(path.read_text())


def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def key(row):
    return row['op'],tuple(row['shape']),json.dumps(row['attrs'],sort_keys=True)


def num(value,n=3):
    return 'N/A' if value is None else f'{value:.{n}f}'


def percent(value):
    return 'N/A' if value is None else f'{100*value:.2f}%'


def signed(pred,ca):
    return 'N/A' if pred is None or not ca else f'{100*(pred/ca-1):+.2f}%'


def metric(rows,pred,ref):
    a=[(r[pred],r[ref],r['id']) for r in rows if r.get(pred) is not None and r.get(ref)]
    if not a:
        return dict(n=0,wape=None,max_ape=None,worst=None)
    mx=max(a,key=lambda x:abs(x[0]/x[1]-1))
    return dict(n=len(a),wape=sum(abs(u-v) for u,v,_ in a)/sum(v for _,v,_ in a),
                max_ape=abs(mx[0]/mx[1]-1),worst=mx[2])


def pretty(m):
    return percent(m['wape'])+' / '+percent(m['max_ape']) if m['n'] else 'N/A'


def render_table(head,values):
    return ['| '+' | '.join(head)+' |','|'+'|'.join('---' for _ in head)+'|',
            *['| '+' | '.join(map(str,row))+' |' for row in values]]


def trace_summary(m):
    if not m:
        return None
    obj={k:m.get(k) for k in ('chip','ca_us','kernel','api','correctness','active_aic','active_aiv','backend','role','trace','trace_sha256',
                               'source_hash_match','launch_context','scope','run_log','runner_sha256','input_contract_log','active_aic_per_launch','active_aiv_per_launch')}
    obj['stages']=[{k:s.get(k) for k in ('kernel','ca_us','active_aic','active_aiv','total_events','mmad','binary_sha256')}
                   for s in m.get('stages') or []]
    return obj


def rows():
    pairs={key(p):p for p in load(SKU/'pairs.json')}
    records={(key(r),r['chip']):r for r in load(SKU/'records.json')}
    errors={(r['case'],r['chip']):r['reason'] for r in load(SKU/'rejections.json')}
    result=[]
    for group in ('core','extension'):
        frozen=load(SKU/(group+'-freeze.json'))
        for job in frozen['jobs']:
            for f in job['outputs']:
                k=key(f);p=pairs.get(k,{})
                dt=p.get('dt') or records.get((k,DT))
                pr=p.get('pr') or records.get((k,PR))
                d,pred=f.get('dt_us'),f.get('pr_us')
                act_d=dt.get('ca_us') if dt else None; act_p=pr.get('ca_us') if pr else None
                speed=d/pred if d and pred else None
                measured=act_d/act_p if act_d and act_p else None
                if measured:
                    if dt.get('role')=='new paired DT CA' and dt['backend']==pr['backend']:
                        tier='direct'
                    elif dt['backend']==pr['backend']:
                        tier='archived_same'
                    else:
                        tier='mixed'
                    assert abs(measured-p['speedup_ca'])<1e-12
                else:tier='unpaired'
                record={k:v for k,v in f.items() if k!='dt'}
                record.update(id=job['id'],group=group,api=OPS[f['op']],library='ops-nn' if f['op'] in NN else 'ops-transformer',
                              dt_observation=trace_summary(dt),pr_observation=trace_summary(pr),dt_ca_us=act_d,pr_ca_us=act_p,
                              dt_role=dt['role'] if dt else None,pr_role=pr['role'] if pr else None,future_role='unvalidated prospective extrapolation',
                              pair_tier=tier,speedup_ca=measured,speedup_pred=speed,
                              speedup_error=speed/measured-1 if speed and measured else None,
                              dt_rejection=errors.get((job['id'],DT)),pr_rejection=errors.get((job['id'],PR)))
                for side,obs in [('dt',dt),('pr',pr)]:
                    failure=record[side+'_rejection']
                    if not obs: outcome='failed capture' if failure else 'unmeasured'
                    elif obs['role']=='reused DT qualified archive': outcome='qualified archive; new control failed' if failure else 'qualified archive'
                    else: outcome='qualified recovery' if failure else 'qualified new capture'
                    record[side+'_outcome']=outcome
                result.append(record)
    assert len(result)==len({key(r) for r in result})
    assert {key(r) for r in result if r['speedup_ca'] is not None}==set(pairs), 'Incomplete or orphan pair'
    return result


def add_diagnostics(rows):
    for r in rows:
        if r['op'] not in CORE:
            continue
        run=estimate(r['op'],r['shape'],r['attrs'],chip=PR,timing_mode='resource')
        assert abs(run.latency_us/r['pr_us']-1)<1e-12
        r['pr_model_stages_us']=run.projection['stages_us']
        r['pr_predicted_mmads']=run.projection['timing_diagnostics'].get('mmads_total')
        r['default_dt_reuse_us']=estimate(r['op'],r['shape'],r['attrs'],chip=DT).latency_us


def summarize(group):
    paired=[r for r in group if r.get('speedup_ca') and r.get('speedup_pred')]

    def speed_stats(g):
        if not g:
            return dict(n=0,observed=None,predicted=None,wape=None,max_ape=None)
        s=metric(g,'speedup_pred','speedup_ca')
        return dict(n=len(g),observed=sum(r['dt_ca_us'] for r in g)/sum(r['pr_ca_us'] for r in g),
                    predicted=sum(r['dt_us'] for r in g)/sum(r['pr_us'] for r in g),wape=s['wape'],max_ape=s['max_ape'])
    fut=[r['future_us'] for r in group if r.get('future_us') is not None]
    return dict(n=len(group),dt=metric(group,'dt_us','dt_ca_us'),pr=metric(group,'pr_us','pr_ca_us'),
                old_dt=metric(group,'old_dt_us','dt_ca_us'),old_pr=metric(group,'old_pr_us','pr_ca_us'),
                reuse_dt=metric(group,'dt_us','pr_ca_us'),speed=speed_stats(paired),
                direct=speed_stats([r for r in paired if r['pair_tier']=='direct']),
                tiers=dict(Counter(r['pair_tier'] for r in group)),future_range=[min(fut),max(fut)] if fut else [])


def costs():
    rows=load(SKU/'all-costs.json')
    if not rows:
        return {}
    unique={(r['group'],r.get('attempt',r['case']),r['chip']) for r in rows}
    if len(unique)!=len(rows):
        raise ValueError('Duplicate collection charge')
    stamps=lambda r,k:datetime.fromisoformat(r[k]).timestamp()
    return dict(attempted=len(rows),qualified=sum(r['eligible'] for r in rows),rejected=sum(not r['eligible'] for r in rows),
                wall_seconds=max(stamps(r,'finished_utc') for r in rows)-min(stamps(r,'started_utc') for r in rows),
                sum_seconds=sum(r['seconds'] for r in rows),known_cpu_hours=sum((r['cpu_seconds'] or 0) for r in rows)/3600,
                cpu_missing=sum(r['cpu_seconds'] is None for r in rows),raw_gib=sum(r['raw_bytes'] for r in rows)/2**30,
                by_chip=dict(Counter(r['chip'] for r in rows)))


def draw(summary,names):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/operator-sim-v17-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ops=[op for op in names if summary[op]['pr']['n']]
    if not ops:return
    y=range(len(ops));fig,(a,b)=plt.subplots(1,2,figsize=(12,6.6),layout='constrained',sharey=True)
    for axis in (a,b):
        axis.set_yticks(list(y),[op.replace('grouped_matmul_swiglu_quant_v2','GMM1 + SwiGLU + MX') for op in ops])
        axis.grid(axis='x',alpha=.2)
    for role,label,marker,color in [('pr','PR','o','#16756f'),('dt','DT','x','#88682a'),('reuse_dt','DT reused on PR','+','#525774')]:
        xs=[None if summary[op][role]['wape'] is None else 100*summary[op][role]['wape'] for op in ops]
        a.scatter(xs,list(y),marker=marker,label=label,c=color,s=30)
    for role,label,marker,color in [('observed','CA','o','#16756f'),('predicted','Model','x','#88682a')]:
        b.scatter([summary[op]['speed'][role] for op in ops],list(y),marker=marker,label=label,c=color,s=34)
    a.invert_yaxis();a.set_xlabel('Latency WAPE (%)');b.set_xlabel('Sum of DT times / sum of PR times');b.axvline(1,lw=.8,c='#555555')
    a.legend();b.legend();fig.savefig(V/'comparison.png',dpi=160);plt.close(fig)


def chips():
    data=[load(ROOT/'configs/chips'/(s+'.json')) for s in (DT,PR,'Ascend970DT')]
    lines=render_table(('配置项','950DT_9582','950PR_957b','970DT 情景'),[
        ('频率 GHz',*[c['frequency_ghz'] for c in data]),('Cube / Vector 核',*[f"{c['cube_cores']} / {c['vector_cores']}" for c in data]),
        ('Cube 32/16/8/4 位 TOP/s',*[' / '.join(str(c['cube_tops'][b]) if c['cube_tops'][b] else '未知' for b in ('32','16','8','4')) for c in data]),
        ('Vector 32/16/8/4 位 TOP/s',*[' / '.join(str(c['vector_tops'][b]) if c['vector_tops'][b] else '未知' for b in ('32','16','8','4')) for c in data]),
        ('外存容量 GB',*[f"{c['hbm_capacity_gb']:.2f}" for c in data]),('整片外存带宽 GB/s',*[c['hbm_bandwidth_gbs'] for c in data]),
        ('L2 容量、整片峰值','128 MiB，带宽未知','112 MiB，带宽未知','200 MB，30000 GB/s'),
        ('每核 UB / L1','248 / 512 KiB','248 / 512 KiB','继承 DT 的 248 / 512 KiB'),
        ('本次 CA 外存子模型','HBM3、2 通道','LPDDR5、1 通道','无目标 CA')])
    return lines+['','PR 比 DT 少 12.5% 的 AIC/AIV、名义片外带宽低 65%，单个 BF16 Cube 核的名义吞吐相等。970 的资源与绑定比例来自情景配置；其 Vector 峰值含两路发射，2 operation/MAC 只计一次。PR 的 1400 GB/s 是按容量匹配的档位，CA 日志中的通道仅指其模拟子模型。各 SKU 缺少实际 MTE 峰值、缓存命中或首读延迟时继承阶段假设。严格 FP32 Cube 在三者均未登记。[DT](../configs/chips/Ascend950DT_9582.json) · [PR](../configs/chips/Ascend950PR_957b.json) · [970DT](../configs/chips/Ascend970DT.json) · [模拟器外存日志](../results/v0.13/sku/memory-models.json)']


def obs_core(obs):
    if not obs:return 'N/A'
    def peak(v):
        if isinstance(v,list):return max(v,default=0)
        return v if isinstance(v,int) else '?'
    return f"{peak(obs['active_aic'])}/{peak(obs['active_aiv'])}"


def kernel_label(obs,legend):
    if not obs:return '—'
    name=obs['kernel'] or '历史未存 kernel 名'
    if name not in legend:legend[name]=f'K{len(legend)+1}'
    return legend[name]


def detailed(rows,groups,names):
    lines=['## 本轮逐例与库身份','',
           '时延单位均为 µs，误差有符号。实测倍速与模型倍速均为 950DT/950PR；大于 1 表示 PR 更快。AIC/AIV 栏列出 PR 模型→CA；串行组件各自峰值取最大。D 表示两端新采同宿主，A 表示复用的 DT 是同后端，M 表示历史 DT 和 PR 计时后端或采集条件有差异，缺任一端为 —。ACLNN 与实际 kernel 名取自已安装 CANN 包的运行记录；ops-nn/ops-transformer 表示其算子族的源码归属。开源对照与同名 ELF 的逐行对应关系没有由这些窗口证明。每个 K 编号仅在其所属小节内有效。[完整合同、正确性与指令/二进制来源](../results/v0.17/sku/records.json) · [逐点预测](../results/v0.17/evaluation.json) · [冻结留出上 Quant 首块读写及 Matmul 单/多波的阶段核查](../results/v0.17/heldout-phases.md) · [大 Matmul 实测 MMAD 尺寸直方图](../results/v0.17/sku/campaign-mmad.json)','']
    for op,name in names.items():
        legend={}; table=[]
        for r in groups[op]:
            ca=r['pr_ca_us'];dt=r['dt_ca_us'];pr=r['pr_observation']
            kl=kernel_label(r['dt_observation'],legend)+' / '+kernel_label(pr,legend)
            attrs=' '.join(f'{k}={v}' for k,v in r['attrs'].items() if k in ('num_iters','self_shape','expert_counts','top_k'))
            compact='×'.join(map(str,r['shape']))+(' '+attrs if attrs else '')
            core=f"{r.get('pr_aic')}/{r.get('pr_aiv')}→{obs_core(pr)}" if r.get('pr_us') else 'N/A'
            tier={'direct':'D','archived_same':'A','mixed':'M','unpaired':'—'}[r['pair_tier']]
            table.append((compact,kl,f"{num(r['dt_us'])} / {num(dt)}",signed(r['dt_us'],dt),
                          f"{num(r['pr_us'])} / {num(ca)}",signed(r['pr_us'],ca),num(r['future_us']),
                          num(r['speedup_pred'])+' / '+num(r['speedup_ca']),core,tier))
        lines += [f'### {name}','',f"{'ops-nn' if op in NN else 'ops-transformer'} · {OPS[op]}。",'']
        lines+=render_table(('本卡 shape 与可变属性','DT / PR kernel','DT 预测 / CA','DT 误差','PR 预测 / CA','PR 误差','970 外推','倍速预测 / CA','PR 核数','配对'),table)
        for n,symbol in legend.items():lines+=['',f'{symbol} · {n}']
        qual=[r for r in groups[op] if r['pr_ca_us'] and r['pr_us']]
        if qual:
            w=max(qual,key=lambda r:abs(r['pr_us']/r['pr_ca_us']-1))
            stages=w['pr_observation']['stages']
            stage_time='、'.join(f"{s['ca_us']:.3f}" for s in stages)
            mm=sum(s['mmad'] or 0 for s in stages)
            lines+=['',f"PR 最大误差 {w['shape']} 的设备预测 {w['pr_us']:.3f}、CA {w['pr_ca_us']:.3f} µs（{signed(w['pr_us'],w['pr_ca_us'])}）；CA 的串行 kernel 窗口依次 {stage_time} µs，记录 {mm} 条 MMAD。{interpret(op)}"]
            ok=sum(obs_core(r['pr_observation'])==f"{r['pr_aic']}/{r['pr_aiv']}" for r in qual)
            lines+=['',f"PR 芯片上，预测活动 AIC/AIV 同时与 CA 对齐 {ok}/{len(qual)} 例。"]
            if op in CORE:
                stage_text='、'.join(f"{stage} {time:.3f}" for stage,time in w['pr_model_stages_us'].items())
                lines+=['',f"此例 PR 模型的阶段分解（µs）为 {stage_text}。"]
            if op in ('matmul','fa'):
                equal=sum(r['pr_predicted_mmads']==sum(st['mmad'] for st in r['pr_observation']['stages']) for r in qual)
                lines+=['',f"在所有有效核上加总 MMAD，模型与 PR trace 总数相同 {equal}/{len(qual)} 例；最大时延误差的这个 shape 预测 {w['pr_predicted_mmads']}、实测 {mm} 条。"]
            if op=='matmul' and (SKU/'campaign-mmad.json').exists():
                checks=load(SKU/'campaign-mmad.json')
                matches=sum(r['histogram_matches'] for r in checks)
                lines+=['',f"在本轮大 shape 上逐条读取 MMAD 的实际 (m,n,k) 并按形状计数。已有 {len({tuple(r['shape']) for r in checks})} 个不同 shape、{len(checks)} 个 DT/PR 窗口；其中 {matches}/{len(checks)} 个窗口的尺寸直方图与冻结模型完全相同。"]
        elif op=='kda_chunk':
            lines+=['','本轮主机对 ChunkKdaFwd 启用了 safe gate 与 lower bound，而预注册合同未给这两个属性（raw_unsafe 路线）。因此该运行不能与预注册合同精确配对；运行本身也报重复 set_flag，同步失败，无法给安全门控路线或 raw 路线提供合格窗口。两种合同的耗时误差和加速比均为 N/A。[实际主机参数](../scripts/cube-v8-kda-safe.cpp)']
        lines+=['']
    return lines


def interpret(op):
    return {
        'quant':'短 scale 片段的多核请求等待与大请求 payload 对时长有不同影响。DT 阶段校准采用最忙核对应的 32 B 区段最大生产者数，最晚完成核可能取尾片。见冻结后旧 trace 的逐请求对照。',
        'matmul':'新尾波分块和跨波重复发起可用 MMAD/FIX 时间线核查。剩余的 A/B 复用、写回暴露比例和全片有效带宽仍是近似。',
        'fa':'KV 小块中首发、同步及排空权重大，长循环对稳态间隔更敏感；单条完整窗口不能分离不同端口反压。',
        'rmsnorm':'两笔写的重叠取决于关键核每轮行数，未知前序核驻留用 DT 先验。启动状态与 SRAM 或跨核排队不能由 shape 唯一反演。',
        'mhc_post':'DT 对首次/扩核/复用分开估时。独立调用和同序列历史样本可有不同前序状态，跨芯片搬运按暴露比例缩放，无法单从总时长判定微观根因。',
        'kda_recurrent':'只校准单状态链，token 递推串行。旧 DT 样本数小，控制/带宽的分项来自重叠后的临界核暴露时间。',
        'mhc_sinkhorn':'归一化迭代之间串行，小矩阵跨 token 向量打包；独立调用的发射成本和历史多次调用状态可不同。',
        'lightning_indexer':'此路线以 32 个 Q 行为一波、每 128 个 KV 位置估一次点积块、每 512 个位置估一次选择工作。若真实 TopK 合并的跨核工作由序列长度或并发触发，宏块计数不能逐条还原这部分开销。限定于 dense G8/D128、top512。',
        'mhc_pre':'预测按 ceil(T/16)×ceil(4D/256) 估 Cube 并行任务、按 T 估 Vector 任务；该校准只能约束临界核时间。active AIC/AIV 的逐例对照有助于区分并行工作映射问题。',
        'mhc_pre_sinkhorn':'融合后单 kernel 包含输入归约与迭代工作，两类队列在同一设备窗口内重叠。按同族 DT trace 的临界活动区间外推；即使总算术量相同，子块交接和线程分配仍影响尾部。',
        'mhc':'这是 Pre、Sinkhorn、Post 串行的完整调用链。表中列出的三段实测窗口与模型阶段分别相加，单个阶段高低误差可能抵消；组合与由它抽取的 Pre 共用一次捕获。',
        'grouped_matmul_swiglu_quant_v2':'令 eᵢ 是每个专家实际行数，输出 2F 列。模型以 Σceil(eᵢ/16)×ceil(2F/128) 估 Cube 任务，随专家尾片、门控和重新量化的 Vector 工作定价。活动核不同意味着这一任务到核映射存在偏差。',
        'gmm2':'第二个 kernel 读取分专家的 FP8 激活并进行下投影，输出 K 列。按 Σceil(eᵢ/16)×ceil(K/128) 估输出任务；实际分组描述符、写回或核分配的代价未独立测定。',
        'moe_ffn':'此项是先 GroupedMatmulSwigluQuantV2 再 GroupedMatmulV5 的设备窗口和。两个组件可单独在上表核对，链式调用可能受中间数据驻留影响。组件内的偏差可抵消，组合误差不能视作新的一份独立采集。',
        'addmm':'目标沿用参考分块的启动与队列重叠比例，偏置/矩阵输入的读流量不同；有限目标样本未辨识目标页级调度。'
    }.get(op,'模型迁移 DT 的临界阶段/重叠模板，目标分核与具体核间交接仍是假设。串行组合扣除 host 间隙，组件窗口来自同一条调用链；仅凭完整窗口不能唯一拆出缓存、发射或等待的占比。')


def persistence(payload):
    path=ROOT/'results/current/evaluation.json'
    current=load(path)
    old=[r for r in current.get('paired_sku',[]) if r.get('evaluation_release')!='v0.17']
    assert len(current['rows'])==356 and len(old)==85
    current.update(version='v0.17',previous_version='v0.16',qualification=payload['qualification'],
                   pr_summary={op:m['pr'] for op,m in payload['summary'].items()},
                   release_summary=payload['summary'],release_rows=payload['rows'],
                   paired_sku=old+[{**r,'evaluation_release':'v0.17'} for r in payload['rows'] if r['speedup_ca']],
                   collection_cost=payload['cost'],target_query_wall_ms=payload['wheel'],
                   reference_query_wall_scope='rows and operator_summary retain 356 historical DT calibrated predictions. release_summary and release_rows use the v0.17 frozen routes.')
    save(path,current)


def replay_prior(rows, python):
    """Replay extension contracts against an installed prior release in isolation."""
    import subprocess
    unmatched=[r for r in rows if 'old_dt_us' not in r]
    program="""import json,sys
from operator_model import estimate,__version__,__file__
assert __version__=='0.16.0' and 'site-packages' in __file__
rows=json.load(sys.stdin)
print(json.dumps([[estimate(r['op'],r['shape'],r['attrs'],chip=c).latency_us
                  for c in ('Ascend950DT_9582','Ascend950PR_957b')] for r in rows]))"""
    answer=json.loads(subprocess.check_output([python,'-I','-c',program],
        input=json.dumps([{k:r[k] for k in ('op','shape','attrs')} for r in unmatched]),text=True,cwd='/tmp'))
    assert len(answer)==len(unmatched)
    for r,(old_dt,old_pr) in zip(unmatched,answer):
        r.update(old_dt_us=old_dt,old_pr_us=old_pr)
    for r in rows:
        dt,pr=r['old_dt_us'],r['old_pr_us']
        r.update(speedup_old=dt/pr if dt and pr else None,speedup_reuse_dt=1. if r.get('speedup_pred') else None)


def legacy_dt(archive):
    lines=['## 950DT 参考校准路线的全量历史','',
           '以下 356 条是当前 950DT 校准路线保留的历史记录。开发与原冻结留出按采集时的版本标记。部分旧 trace 已用于后续阶段诊断；旧留出的准确性只对应其原冻结模型。上面的显式资源路径在 DT 的误差单独统计。身份、正确性、分核、指令比对和离线成本保存在逐点 JSON。[完整历史](../results/v0.12/evaluation.json)','']
    for info in archive['operator_summary']:
        op=info['op']
        g=[r for r in archive['rows'] if r['op']==op]
        if not g:continue
        legend={};table=[]
        for r in g:
            symbol=kernel_label(r,legend)
            table.append((r.get('id',r.get('case_id',Path(r['source']).name)),'×'.join(map(str,r['shape'])),symbol,'原留出' if r['role']=='validation' else '开发',
                          num(r['reference_us']),num(r['predicted_us']),signed(r['predicted_us'],r['reference_us'])))
        lines += [f"### {info['name']} · DT 历史",'']+render_table(('记录','shape','kernel','角色','DT CA µs','DT 校准 µs','误差'),table)
        for k,v in legend.items():lines += ['',f'{v} · {k}']
        lines+=['']
    return lines


def summary_table(groups,names,s,archive,wheel):
    table=[]
    for h in archive['operator_summary']:
        op=h['op'];m=s[op];nnew=sum((r['dt_observation'] or {}).get('role')=='new paired DT CA' for r in groups[op])
        future=m['future_range'];orig=f"{percent(h['wape'])} / {percent(h['max_ape'])}" if h['independent'] else 'N/A'
        table.append((names[op],f"{h['development']}/{h['independent']}",orig,f"{nnew}/{m['dt']['n']-nnew}",
                      pretty(m['dt']),str(m['pr']['n']),pretty(m['pr']),
                      num(m['speed']['observed'])+' / '+num(m['speed']['predicted']),
                      '–'.join(num(x,2) for x in future) if future else 'N/A',
                      num(wheel['per_op'][op]['max_ms']),'仅结构，无合格时延' if op=='kda_chunk' else 'DT 资源定价，PR 冻结验证' if op in CORE else 'DT 阶段校准，PR 外推验证'))
    return render_table(('算子','历史 DT 开发/独立 N','历史 DT 独立 WAPE/最大','本轮 DT 新/复用 N','本轮 DT WAPE/最大',
                         'PR 新独立 N','PR WAPE/最大','DT/PR 倍速 CA/预测','970 外推范围 µs','查询最大 ms','计时资格 / 校准状态'),table)


def baselines(s,names):
    lines=['## 在相同合同上比较时延与加速比','',
           '只对两端都合格的同一组 shape 比较加速比。表内时延误差针对同一侧 CA；“复用 DT”指把本轮该 shape 的 DT 模型输出直接填作 PR 预测，它给出的 DT/PR 倍速恒为 1。新旧模型和基线均未读取对应 PR 留出标签。新倍速 WAPE 以逐点 DT/PR 比值作观测；“直接配对”额外要求两端本轮运行同一宿主、相同命令参数和采集后端。旧版对照使用已安装的 v0.16 wheel，核心三类预测在采集前冻结。','']
    table=[]
    for op,name in names.items():
        m=s[op];b=m['direct'];t=m['tiers']
        table.append((name,f"{t.get('direct',0)}/{t.get('archived_same',0)}/{t.get('mixed',0)}",
                      pretty(m['old_pr']),pretty(m['reuse_dt']),pretty(m['pr']),pretty(m['speed']),
                      pretty(m['old_speed']),pretty(m['reuse_speed']),
                      f"{num(b['observed'])} / {num(b['predicted'])}" if b['n'] else 'N/A',pretty(b)))
    actual=lines+render_table(('算子','直接/同后端历史/混合 N','旧 PR WAPE/最大','复用 DT 到 PR','当前 PR','当前倍速 WAPE/最大','旧倍速','恒 1 倍速','直接配对 CA/预测倍速','直接配对倍速 WAPE/最大'),table)+['']

    actual+=['若系统调用默认 `auto` 取得 DT 已校准时延，再直接作为 PR 的预测，三个核心算子的对照如下。其余算子的 DT 比较路线均为 `auto`。','']
    actual+=render_table(('算子','PR N','直接复用 DT 的资源模型 WAPE/最大','直接复用 DT 的 auto 模型 WAPE/最大','当前 PR WAPE/最大'),
                         [(names[o],s[o]['pr']['n'],pretty(s[o]['reuse_dt']),pretty(s[o]['dt_auto_reused_on_pr']),pretty(s[o]['pr'])) for o in CORE])+['']
    return actual


def run_cost(c,wheel):
    w=wheel['overall']
    return ['## 在线查询与离线采集成本','',
            f"安装 wheel 后从仓库外的常驻解释器逐 shape、逐芯片调用公共 API，{w['n']} 次；中位 {w['median_ms']:.3f} ms，P95 {w['p95_ms']:.3f} ms，最大 {w['max_ms']:.3f} ms。首次调用和参数加载计入，导包另需 {wheel['import_ms']:.3f} ms。三种配置的结果与冻结预测逐例一致。设备窗口以 µs 表示，查询墙钟以 ms 表示。[安装后验证](../results/v0.17/wheel-smoke.json)",'',
            f"全轮 CA 新运行 {c['attempted']} 个宿主/芯片组合，合格 {c['qualified']}，失败或停止 {c['rejected']}。去重后原始数据 {c['raw_gib']:.2f} GiB；最早启动至最后结束为 {c['wall_seconds']/60:.2f} min，逐宿主历时相加 {c['sum_seconds']/60:.2f} min。可确认 CPU {c['known_cpu_hours']:.2f} 核时，另有 {c['cpu_missing']} 条无 CPU 计数。一个 MoE 或 mHC 宿主可产出组合与组件的多个窗口，离线成本仅计一次。[每次启动、退出及临时数据量](../results/v0.17/sku/all-costs.json)",'']


def ambiguous(controls,rejections,rows):
    lines=['## 采集条件和不合格记录','',
           'PR 使用 CANN 9.2.1 的 msopprof 官方仿真，检查运行时 SKU、输出 golden 与库 kernel、完整事件窗口和活动核。历史 DT 含官方 profiler 与 npusim compact；不同后端、早先串行或批量宿主的同 shape 时间可能不同。新同条件 DT 可用时优先作为配对端；相同归档合同仍保留供估算这种不确定性。组合同时校验 component 顺序，总时间取设备阶段和。历史条目的稀疏指令计数仅用于比对其自身能对齐的 opcode。关于高估或低估的解释，除明确有阶段证据者外都是待验证因素。','']
    if controls:
        lines+=render_table(('算子与 shape','原 DT 后端、运行上下文','原 DT µs','新 DT µs','变化'),[(r['op']+' '+'×'.join(map(str,r['shape'])),r['archive_backend']+'，'+str(r['archive_context']),num(r['dt_archived_us']),num(r['dt_msopprof_us']),signed(r['dt_msopprof_us'],r['dt_archived_us'])) for r in controls])+['']
        lines+=['同 shape 下历史与新 DT 的差异包含后端、宿主和运行上下文的共同影响。详见[九条宿主路线的控制实验](../results/v0.17/sku/dt-control-comparison.json)。','']
    if rejections:
        state={}
        for row in rows:
            for side,chip in [('dt',DT),('pr',PR)]:
                state[row['id'],chip]=row[side+'_outcome']
        lines+=render_table(('宿主','芯片','本次停止或拒收原因','最终证据','本次墙钟 s','原始 MiB'),
                            [(r['case'],r['chip'],r['reason'],state[r['case'],r['chip']],num(r['seconds'],1),num(r['raw_bytes']/2**20,1)) for r in rejections])+['']
    return lines


def main():
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prior-python',default='/tmp/operator-sim-v16-deploy/bin/python')
    ap.add_argument('--preview',action='store_true',help='Validate report generation without publishing any output')
    args=ap.parse_args()
    if not args.preview and not (SKU/'campaign-completion.json').exists():
        raise SystemExit('unfinished campaign: campaign-completion.json')
    rs=rows();replay_prior(rs,args.prior_python);add_diagnostics(rs)
    archive=load(ROOT/'results/v0.12/evaluation.json')
    names={s['op']:s['name'] for s in archive['operator_summary']}
    groups={op:[r for r in rs if r['op']==op] for op in names}
    s={op:summarize(g) for op,g in groups.items()}
    for op,m in s.items():
        m['old_speed']=metric(groups[op],'speedup_old','speedup_ca')
        m['reuse_speed']=metric(groups[op],'speedup_reuse_dt','speedup_ca')
        m['dt_auto_reused_on_pr']=metric(groups[op],'default_dt_reuse_us','pr_ca_us')
        m['dt_by_role']={role:metric([r for r in groups[op] if r['dt_role']==role],'dt_us','dt_ca_us')
                         for role in ('new paired DT CA','reused DT qualified archive')}
    wheel=load(V/'wheel-smoke.json');assert wheel['matches_freeze_within_relative_1e_12'] and wheel['overall']['n']==len(rs)*3
    c=costs(); controls=load(SKU/'dt-control-comparison.json');reject=load(SKU/'rejections.json')
    if not args.preview:
        # A recovered attempt can have both an earlier rejection and a later eligible capture.
        ledger=load(SKU/'all-costs.json')
        pr_eligible={x['case'] for x in ledger if x['chip']==PR and x['eligible']}
        assert pr_eligible=={r['id'] for r in rs if r['pr_ca_us'] is not None}
        assert sum(m['n'] for m in s.values())==len(rs)
    s_dt=sum(m['dt']['n'] for m in s.values());s_pr=sum(m['pr']['n'] for m in s.values())
    hard=[o for o in names if s[o]['pr']['n'] and s[o]['pr']['wape'] is not None]
    hard=sorted(hard,key=lambda o:s[o]['pr']['wape'],reverse=True)[:3]
    errors='、'.join(f"{names[o]} {percent(s[o]['pr']['wape'])}" for o in hard)
    tiers=Counter(r['pair_tier'] for r in rs)
    detail=f"PR WAPE 最高的三个算子是 {errors}。两端同宿主新测配对 {tiers['direct']}，复用同后端历史 DT {tiers['archived_same']}，复用异后端或后端未明 DT {tiers['mixed']}，尚缺配对 {tiers['unpaired']} 个。"
    core_diag='；'.join(f"{names[o]} 的 PR WAPE {percent(s[o]['old_pr']['wape'])}→{percent(s[o]['pr']['wape'])}" for o in CORE)
    reuse=[]
    for op in CORE:
        m=s[op];now=m['pr']['wape'];ref=m['reuse_dt']['wape']
        if now is None or ref is None:continue
        action='降低' if now<ref else '增加'
        delta=100*abs(now-ref)
        reuse.append(f"{names[op]} {action} {delta:.2f} 个百分点")
    versus='与直接复用 DT 资源模型相比，PR WAPE 的变化为 '+'、'.join(reuse)+'。'
    text=['# 算子耗时预测与 CA 校准误差','']+summary_table(groups,names,s,archive,wheel)+['',
          'WAPE 为 Σ|预测−CA|/ΣCA，最大误差为该组最大的 |预测/CA−1|。倍速栏用同一集合的 ΣDT/ΣPR，大于 1 表示 PR 用时更短。970 的全列均为同一合同在假想芯片上的预测，没有目标芯片观测。N/A 表示缺少合格测时或模型资格。','',
          f"本轮冻结 {len(rs)} 个算子合同，获得 {s_dt} 条可比 DT、{s_pr} 条独立 PR 时延。DT 列含新增与复用，历史 DT 开发/独立只用于说明 950DT 校准路线的既有覆盖。Quant、Matmul、FA 本轮三芯片统一用显式 resource 模式；默认 DT auto 仍走左侧历史校准路线，其他算子用 auto。扩展算子的 DT 归档 shape 在 PR 首次验证，DT 侧的复用不可当作本轮独立验证。每个 shape 的取值和证据等级见后表。",'',
          detail,'',core_diag+'。'+versus+'三组之间的工作量和分支不同，不能用这些数字推断全部调用频率加权的系统误差。','','![时延 WAPE 与配对加速比](../results/v0.17/comparison.png)','',
          '## 芯片资源及模拟条件','']+chips()+['']+baselines(s,names)+ambiguous(controls,reject,rs)+run_cost(c,wheel)+detailed(rs,groups,names)+legacy_dt(archive)
    qualification='Core three use explicit resource on DT/PR/970. Other operators use auto; PR is independent. New paired DT measured after freeze; reused/archived DT may be development evidence. Mixed backends/launch contexts give approximate ratios. 970 is unvalidated projection only.'
    payload=dict(version='v0.17',previous_version='v0.16',qualification=qualification,summary=s,rows=rs,wheel=wheel,cost=c,
                 same_shape_dt_controls=controls,rejected=reject,
                 route_exceptions={'kda_chunk':'Frozen attributes omit safe_gate/lower_bound (raw_unsafe). Host scripts/cube-v8-kda-safe.cpp uses safe gating and lower bound. Repeated set_flag independently disqualifies the capture; neither contract has an eligible timing.'},
                 reference_calibration_release='v0.12',reference_calibration='results/v0.12/evaluation.json',
                 protocols={group:dict(freeze_file=f'results/v0.17/sku/{group}-freeze.json',utc=load(SKU/f'{group}-freeze.json')['utc'],
                                       sha256=sha256((SKU/f'{group}-freeze.json').read_bytes()).hexdigest()) for group in ('core','extension')})
    changes=['# v0.17 建模及验证变化','',
             'Quant 按每轮请求的 payload、scale 生产者冲突、MTE3 端口排队和整片带宽分别定价。Matmul 区分首波/尾波、关键核重复发起和写回排空。FA 按 AIC/AIV 绑定比例分配向量资源，给出一对一与可借空闲核的情景。[Quant 证据](quant-method.md) · [Matmul 证据](matmul-method.md) · [FA 情景](fa-scenarios.md) · [新留出逐阶段核查](heldout-phases.md)','',
             f"冻结 {len(rs)} 个算子合同，本轮具备 {s_pr} 个 PR、{s_dt} 个 DT 合格窗口。下表三类核心算子按本轮同一组冻结 shape 比较；误差单位为 WAPE，倍速方向 DT/PR。",'']
    changes+=render_table(('算子','PR N','v0.16 PR','v0.17 PR','直接复用 DT','倍速 CA/预测','DT N','v0.16 DT','v0.17 DT'),
                         [(names[o],s[o]['pr']['n'],percent(s[o]['old_pr']['wape']),percent(s[o]['pr']['wape']),percent(s[o]['reuse_dt']['wape']),
                           num(s[o]['speed']['observed'])+' / '+num(s[o]['speed']['predicted']),s[o]['dt']['n'],percent(s[o]['old_dt']['wape']),percent(s[o]['dt']['wape'])) for o in CORE])+['',
             f"单次查询最大 {wheel['overall']['max_ms']:.3f} ms。新增 CA 宿主 {c['attempted']}，合格 {c['qualified']}，失败或停止 {c['rejected']}，墙钟 {c['wall_seconds']/60:.1f} min、数据量 {c['raw_gib']:.2f} GiB。KDA chunk 仍缺少完整合格计时。扩展算子复用 DT 归档的异后端倍速置信度低于新配对；所有 970DT 结果均为外推。[全量逐算子结论和逐 shape 明细](../../docs/error-analysis.md) · [冻结评估及成本](evaluation.json)"]
    if args.preview:
        print(f'preview validated: {len(rs)} frozen contracts, {s_dt}/{s_pr} qualified DT/PR; {len(text)} report lines');return
    draw(s,names)
    (ROOT/'docs/error-analysis.md').write_text('\n'.join(text).rstrip()+'\n')
    (V/'changes.md').write_text('\n'.join(changes)+'\n')
    save(V/'evaluation.json',payload);persistence(payload)
    print(f'published {len(rs)} rows, {s_dt} DT and {s_pr} PR observations')


if __name__=='__main__':
    main()
