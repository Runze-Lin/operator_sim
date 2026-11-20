"""Render all sibling-phase replay plus only registered and strictly qualified target windows."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json,math,sys,time,statistics

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/v0.34/attn'
DT,PR,F='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'
def get(p):return json.loads(p.read_bytes())
def sig(p):return sha256(p.read_bytes()).hexdigest()
def metrics(rows):
    if not rows:return dict(n=0,wape=None,maximum=None)
    a,p=zip(*rows)
    return dict(n=len(rows),wape=100*sum(abs(y-x) for x,y in rows)/sum(a),maximum=100*max(abs(y/x-1) for x,y in rows))
def run(prefix=None):
    sys.path.insert(0,str(prefix or ROOT))
    from operator_model import estimate
    from operator_model.attention_residual import attention_residual_work
    from operator_model.attn_res_resident import resident_inference_price,DISPATCH,DRAIN
    from operator_model.chips import load_chip
    phases=get(OUT/'phase-calibration.json');reg=get(OUT/'freeze.json')
    attrs=reg['jobs'][0]['attrs'];rows=[];scope={}
    for r in phases['rows']:
        sh=r['shape'];chip=load_chip(r['chip']);w=attention_residual_work(*sh,chip=chip,addend=True,optimize_prefill=True)
        q=resident_inference_price(sh,chip,w)['per_core_cycles']
        p=dict(dispatch=DISPATCH,weights=q['weight_setup'],prefix=q['prefix'],score_weighted=q['score_softmax_mix'],norm=q['output_norm'],write=q['store'],drain=DRAIN)
        raw=r['cycles'];ca=sum(raw[k] for k in p);pred=sum(p.values())
        rows.append(dict(shape=sh,chip=r['chip'],trace_sha256=r['trace_sha256'],role='DT-phase-development' if r['development'] else 'archived-sibling-phase-replay',
                         raw_common_cycles=ca,predicted_common_cycles=pred,signed_pct=100*(pred/ca-1),predicted_cycles=p,observed_cycles=raw))
    for k in ['DT-phase-development','archived-sibling-phase-replay']:
        for c in (DT,PR):
            subset=[r for r in rows if r['role']==k and r['chip']==c]
            if subset:scope[k+' '+c]=metrics([(r['raw_common_cycles'],r['predicted_common_cycles']) for r in subset])
    direct=[];wall=[];old=ROOT/'results/v0.29/large/vector/freeze.json'
    for j in reg['jobs']:
        t,h,slots,n=j['shape']
        for chip,tag in ((DT,'dt'),(PR,'pr'),(F,'extrap')):
            r=estimate('attn_res_fwd',j['shape'],attrs,chip=chip)
            if not math.isclose(r.latency_us,j['prices'][chip]['us'],abs_tol=1e-8):raise ValueError('public price changed')
            path=OUT/(f'{t}-{n}-{tag}.json') if j['new'] else (ROOT/j['existing'][0 if chip==DT else 1])
            ca=None;role='970DT extrapolation' if chip==F else 'registered; no qualifying receipt'
            if chip!=F and path.exists():
                receipt=get(path);v=receipt.get('verdict',receipt)
                if v.get('qualified') is True:
                    valid=receipt['freeze_sha256']==sig(OUT/'freeze.json' if j['new'] else old) and receipt['shape']==j['shape'] and receipt['chip']==chip and receipt['cost']['started_utc']>reg['utc'] and v['instruction_us']>0 and len(v['trace_sha256'])==64 and '40021' in v['kernel']
                    if not valid:raise ValueError('qualified receipt conflicts with frozen contract '+str(path))
                    ca=v['instruction_us'];role='independent true prefill'
                else:role='terminal without qualified instruction window'
            direct.append(dict(shape=j['shape'],chip=chip,predicted_us=r.latency_us,ca_us=ca,role=role,status=r.status,
                               signed_pct=100*(r.latency_us/ca-1) if ca else None,receipt=str(path.relative_to(ROOT)) if chip!=F and path.exists() else None))
    full=[]
    for context in (32768,131072,262144):
        t=8*context;s=[t,7168,8,1];p={}
        for c in (DT,PR,F):
            r=estimate('attn_res_fwd',s,attrs,chip=c);d=r.projection['timing_diagnostics']
            p[c]=dict(us=r.latency_us,plan=r.projection['target_plan'],read_demand=d['dma_read_bytes'],write_demand=d['dma_write_bytes'])
        full.append(dict(batch=8,context=context,shape=s,prices=p,physical_role='unmeasured whole-batch workload projection'))
    # Different large sizes for each cold cache entry, using preloaded module and a warm interpreter.
    for i in range(60):
        chip=(DT,PR,F)[i%3];start=time.perf_counter_ns();estimate('attn_res_fwd',[262144+i,7168,8,1],attrs,chip=chip)
        wall.append((time.perf_counter_ns()-start)/1e6)
    independent={c:metrics([(r['ca_us'],r['predicted_us']) for r in direct if r['chip']==c and r['ca_us']]) for c in (DT,PR)}
    info=dict(sibling=scope,sibling_rows=rows,independent=independent,direct=direct,whole_batch=full,query_ms=dict(n=len(wall),median=statistics.median(wall),maximum=max(wall)),
              evidence_role='Only qualified preregistered nonmaterialized no-bank-write captures qualify for independent latency; 40020 phase transfer remains exploratory.')
    tmp=OUT/'score.tmp';tmp.write_text(json.dumps(info,indent=2,allow_nan=False)+'\n');tmp.replace(OUT/'score.json')
    display=lambda v:'N/A' if v is None else f'{v:.3f}'
    lines=['# Kimi AttnResFwd 真实前序混合','',
           '## 计时与适用范围','',
           '| 数据口径 | 芯片 | 窗 | WAPE / 最大 |','|---|---|---:|---:|']
    for k,s in scope.items():lines.append(f"| {k.split()[0]} | {k.split()[-1]} | {s['n']} | {s['wape']:.2f}% / {s['maximum']:.2f}% |")
    for c,s in independent.items():lines.append(f"| 新合同冻结后独立整窗 | {c} | {s['n']} | {display(s['wape'])} / {display(s['maximum'])} |")
    lines+=['','前三行只比较 materialized=True 且会写 bank 的 40020 归档中两个变体共有的阶段。共同阶段去掉了旧形物化写与另一条 gamma 读门控的间隔，不等同于已测新算子总时间。各阶段训练来源为 11 个 DT 已观测窗；10 个其它 DT 窗与 21 个 PR 窗只作旧合同回放。DT 到 PR 的阶段价格按每核峰值/频率与端口换算。','',
       '| 输入输出契约 | 设备行为 | 计时口径 |','|---|---|---|',
       '| 归档 materialize=True、write_idx>=0 | 独立写 materialized、指定 bank、prefixOut、Y | 已有旧完整 CA；上表前三行仅提取共有阶段 |',
       '| 真实 Kimi prefill，materialize=False、write_idx=-1 | materialized 与 Y 别名，仅写 prefixOut、Y，bank 保持原值 | 下表按新合同、数据角色和芯片报告端到端时间 |','',
       '| 真实调用 [T,H,S,N] | 芯片 | 模型 µs | 合格 CA µs | 误差 % | 证据 |','|---|---|---:|---:|---:|---|']
    for x in direct:lines.append(f"| {x['shape']} | {x['chip']} | {x['predicted_us']:.3f} | {display(x['ca_us'])} | {display(x['signed_pct'])} | {x['role']} |")
    lines+=['','### B8 整输入的单卡外推','',
        '| 每请求上下文 | 输入行 | DT / PR / 970DT 单次设备 µs |','|---:|---:|---:|']
    for r in full:lines.append(f"| {r['context']} | {r['shape'][0]} | "+' / '.join(f"{r['prices'][c]['us']:.1f}" for c in (DT,PR,F))+' |')
    lines+=['',f"这些场景假设单卡一次调度全部 token，valid=1、UB 可驻留。单解释器内 60 个不同大形、预加载后未缓存的 API 计时中位 {statistics.median(wall):.3f} ms，最大 {max(wall):.3f} ms。所有 970DT 时间均外推。","",
        '真实 Kimi text_config 有 93 层、残差块跨度 12。按框架派发枚举，上述真实 add/mix/norm、write_idx=-1、materialize=False 合同共有170个层/位置，在 950 UB 253952B 下 valid 1–3 合计 66 个位置可驻留。valid 4–8 的104个位置进入 reload 路线，公开入口仍可给出分核、UB 分支与重读量，耗时缺直接证据。模型不覆盖解码图中的所有别名和多样标志。','',
        '新采集调用固定 vLLM-Ascend 自定义库的 ACLNN AttnResFwd，发布源 c09d3e20。每个 SKU 最小跨核块边界 T=65,N=1；第二项 T=121,N=2 在 DT/PR 上分别2与3个忙核迭代。前者每端最多2小时、20GiB，后者3小时、28GiB，在同一既有串行锁逐个放行。512 形保持原有队列。成功需同时过实际 SKU、安装 ELF、全部输出与未改写 bank、供应商全核时长和唯一 END，分核/读写量另对照。没有新的128Ki或256Ki同合同设备窗口。','',
        '[冻结](freeze.json) · [阶段标定及每条原始 SHA](phase-calibration.json) · [逐形与开销](score.json) · [实时槽状态](state.json) · [安装 wheel 逐项核验](wheel-check.json) · [公式与算例](../../../docs/examples/attn_res_fwd.md)','']
    tmp=OUT/'report.tmp';tmp.write_text('\n'.join(lines));tmp.replace(OUT/'report.md')
if __name__=='__main__':
    prefix = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT
    run(prefix)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from figure import render
    render(prefix)
