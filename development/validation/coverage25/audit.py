"""Replay trace-addressed device observations; never promote a selected historical row to a new holdout."""
from collections import defaultdict, Counter
from functools import lru_cache
from hashlib import sha256
import json,re,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'results/v0.25/coverage';sys.path.insert(0,str(ROOT))
from operator_model import estimate, load_chip
DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'
CORE={'quant','matmul','fa'}
EXCLUDED={'rmsnorm','mhc_post','kda_chunk'}
rows={};unknown=[];rejected=[];sources={}


@lru_cache(None)
def read(p):
    p=Path(p);f=p if p.is_absolute() else ROOT/p;data=f.read_bytes();sources[str(f.relative_to(ROOT))]=sha256(data).hexdigest()
    return json.loads(data)


def hashes(value):
    if isinstance(value,list):
        arr=[x for s in value for x in hashes(s)];return arr
    return [value] if isinstance(value,str) and re.fullmatch(r'[a-f0-9]{64}',value) else []


def add(op,chip,shape,attrs,duration,kernel,ids,origin,old_role,**extras):
    if op in EXCLUDED or chip not in (DT,PR) or not op or not duration:return
    shape=tuple(shape)
    if op=='fa' and len(shape)==3:shape+=1,1
    ids=hashes(ids)
    record=dict(op=op,chip=chip,shape=list(shape),attrs=attrs or {},ca_us=duration,kernel=kernel,trace_sha256=ids)
    if not ids:
        unknown.append((record,str(origin),old_role));return
    key=(op,chip,tuple(ids))
    r=rows.setdefault(key,{**record,'origins':{},'contexts':[]})
    if r['shape']!=list(shape) or abs(r['ca_us']-duration)>.003:raise ValueError((key,r['shape'],shape,r['ca_us'],duration))
    if attrs:
        if r['attrs']!=attrs and r['attrs']:
            # Different exports spell defaults explicitly. Check the public prediction
            # against both before using the more explicit contract.
            a=call(op,shape,attrs,chip).latency_us;b=call(op,shape,r['attrs'],chip).latency_us
            if a!=b:raise ValueError(('same measured trace, conflicting runtime attrs',r,attrs))
        r['attrs']=attrs
    r['origins'].setdefault(str(origin),[])
    if old_role not in r['origins'][str(origin)]:r['origins'][str(origin)].append(old_role)
    for name,val in extras.items():
        if val is not None and (name not in r or name=='contexts'):
            if name=='contexts':
                if val not in r[name]:r[name].append(val)
            else:r[name]=val


def call(op,shape,attrs,chip,mode='auto'):
    if op=='fla_kda_v2':op='kda_chunk';attrs={'implementation':'fla_26_9_1_v2'}
    if op=='vllm_hc_post':op='mhc_post';attrs={'implementation':'vllm_hc_post'}
    return estimate(op,shape,attrs,chip=chip,timing_mode=mode)


def populate():
    for v in range(7,13):
        f=f'results/v0.{v}/evaluation.json'
        for r in read(f)['rows']:
            if r['op'] in CORE:continue
            add(r['op'],DT,r['shape'],r.get('attrs'),r.get('reference_us'),r['kernel'],
                r.get('trace_or_db_sha256') or r.get('trace_sha256') or r.get('db_sha256'),f,r['role'],raw_source=r.get('source'),api=r.get('api'),
                correctness=r.get('correctness'),window=r.get('reference_scope'))
    f='results/v0.21/coverage/audit.json'
    for r in read(f)['rows']:
        for o in r['origins']:
            add(r['op'],r['chip'],r['shape'],r['attrs'],r['ca_us'],r['kernel'],r['trace_sha256'],o['receipt'],o['role'],
                api=r['api'],raw_source=r.get('trace'),correctness=r.get('correctness'),window=r['window'])
    for r in read('results/v0.22/evaluation.json')['rows']:
        m=r['receipt']
        add(r['op'],r['chip'],r['shape'],r['attrs'],r['ca_us'],m.get('kernel'),m.get('trace_sha256'),r['source'],r['cohort'],
            correctness=m.get('correctness'),contexts='v0.22 '+r['cohort'])
    for r in read('results/v0.25/coverage/early-db-receipts.json')['accepted']:
        add(r['op'],r['chip'],r['shape'],r['attrs'],r['ca_us'],r['kernel'],r['trace_sha256'],r['source'],
            'early DB complete; original register roles retained in joined release records',raw_source=r['raw'],correctness=r['correctness'])
    for prefix,op in [('attn-res','attn_res_fwd'),('cast','cast_bf16_fp32'),('router','moe_gating_top_k')]:
        for file in sorted((ROOT/'results/v0.19/workloads').glob(prefix+'*.json')):
            group=read(file)
            if not isinstance(group,list):continue
            for r in group:
                if not isinstance(r,dict) or 'qualified' not in r or 'shape' not in r:continue
                if not r['qualified']:
                    rejected.append(dict(receipt=str(file.relative_to(ROOT)),shape=r['shape'],chip=r.get('chip') or r.get('sku'),reason=r.get('reason')));continue
                z=r.get('kernel')
                if not isinstance(z,dict) or 'ca_us' not in z:continue
                attrs={'addend':True,'materialize':True,'write_idx':r['shape'][-1]} if op=='attn_res_fwd' else {}
                add(op,r.get('chip') or r.get('sku'),r['shape'],attrs,z['ca_us'],z['kernel'],z['trace_sha256'],file.relative_to(ROOT),file.stem,
                    correctness=r.get('correctness'),raw_source=z.get('trace'))
    for name in ('holdout','development'):
        f=f'results/v0.20/newop/{name}.json'
        for r in read(f):
            if not r['qualified']:continue
            z=r['kernel'];add('sigmoid_fp32',r['sku'],r['shape'],{},z['ca_us'],z['kernel'],z['trace_sha256'],f,'v0.20 '+name,
                correctness=r['correctness'],raw_source=z.get('trace'))
    f='results/v0.21/fusion/historical-windows.json'
    for r in read(f)['rows']:
        add(r['op'],r['chip'],r['shape'],r['attrs'],r['ca_us'],r['kernel'],r['trace_sha256'],r['source'],'historical full fusion trace',correctness=r['correctness'])
    for f in sorted((ROOT/'results/v0.20/large').glob('cast_*Ascend*.json')):
        r=read(f)
        if r.get('qualified') and r.get('stage'):
            z=r['stage'];add('cast_bf16_fp32',r['chip'],r['shape'],{},r['ca_us'],z['kernel'],z['trace_sha256'],f.relative_to(ROOT),'v0.20 large tensor full window',
                correctness=r['correctness'],raw_source=z.get('trace'))
    for file in (ROOT/'results/v0.23/prefill').glob('*.json'):
        r=read(file);m=read(r['receipt'])
        add(r['op'],r['chip'],r['shape'],{},r['ca_us'],r['kernel'],r['trace_sha256'],r['receipt'],'archived full prefill',contexts='B8 prefill projection/beta',model_name=r['model_name'],correctness=m.get('correctness') if isinstance(m,dict) else None)
    cutoff=read(OUT/'continuing-runs-cutoff.json')
    for pin in cutoff['pro']:
        f=ROOT/pin['receipt']
        r=read(f)
        assert sources[pin['receipt']]==pin['sha256']
        if r.get('qualified') and r.get('stage'):
            z=r['stage']
            add('cast_bf16_fp32',r['chip'],r['shape'],{},r['ca_us'],z['kernel'],z['trace_sha256'],f.relative_to(ROOT),'v0.23 prospective Pro prefill',contexts='B8 prefill projection/beta',correctness=r['correctness'],raw_source=z.get('trace'))
    f='results/v0.21/matmul-pages-pr/observation.json';outer=read(f)
    if outer['qualified']:
        z=outer['observation'];add('matmul',z['chip'],z['shape'],{},z['ca_us'],z['kernel'],z['trace_sha256'],f,'v0.21 prospective PR matching DT geometry',correctness=z['correctness'],raw_source=z['trace'][0])
    f='results/v0.23/hc-timing/comparison.json'
    for r in read(f)['rows']:
        z=read(r['raw_receipt'])
        if z.get('qualified'):
            add('vllm_hc_post',r['sku'],r['shape'],{'implementation':'vllm_hc_post'},r['ca_us'],z['stage']['kernel'],r['trace_sha256'],r['raw_receipt'],r['role'],api='aclnnHcPost',correctness=z.get('checks'),raw_source=z['stage'].get('trace'))
    for pin in cutoff['kda']:
        z=pin['receipt'];src=read(z)
        assert src['qualified'] and sources[z]==pin['sha256']
        inp=src['case'];stages=[s for s in src['stages'] if s.get('family')]
        assert len(stages)==3 and abs(sum(s['ca_us'] for s in stages)-src['main_stages_us'])<1.e-8
        add('fla_kda_v2',inp['sku'],inp['shape'],{'implementation':'fla_26_9_1_v2'},src['main_stages_us'],
            [s['kernel'] for s in stages],[s['aggregate_sha256'] for s in stages],z,pin['role'],
            api='aclnnChunkKdaFwdV2',window='sum of three complete kernel windows',correctness=src.get('output_groups'))
    for f in sorted(OUT.glob('fa-Ascend*.json')):
        r=read(f)
        if not r.get('qualified'):
            rejected.append(dict(receipt=str(f.relative_to(ROOT)),reason=r.get('reason')));continue
        add('fa',r['chip'],r['shape'],{},r['ca_us'],r['kernel'],r['trace_sha256'],f.relative_to(ROOT),'v0.25 prospective',correctness=r['correctness'],api=r['api'],raw_source=r['trace'][0],contexts='v0.25 prospective')
    # Link a legacy summary lacking a hash only when a fully attributed raw
    # window has matching shape, SKU and duration. Those summaries never add votes.
    for record in read('results/v0.22/evaluation.json')['rejected']:
        if record['op'] not in EXCLUDED:
            rejected.append(dict(receipt=record.get('source','results/v0.22/evaluation.json'),op=record['op'],shape=record['shape'],
                                 chip=record['chip'],attempt=record.get('attempt') or record.get('id'),reason=record.get('reason') or record.get('error')))
    for record in read('results/v0.25/coverage/early-db-receipts.json')['other']:
        rejected.append(dict(receipt=record['source'],**{k:record[k] for k in ('op','shape','reason')}))
    for f in ('results/v0.19/prefill/quant_262144_5120.json','results/v0.21/coverage/local-exclusions.json',
              'results/v0.21/large/full-hidden-terminal.json'):
        rec=read(f)
        rejected.append(dict(receipt=f,shape=rec['shape'],op=rec.get('op','cast_bf16_fp32'),
                             reason=rec.get('reason') or rec.get('stop_reason') or rec.get('qualification_error')))
    for item in read('results/v0.23/hc-timing/comparison.json')['failed']:
        rejected.append(dict(receipt='results/v0.23/hc-timing/comparison.json',op='vllm_hc_post',reported_failure=item))
    unresolved=[]
    for r,src,role in unknown:
        matches=[d for d in rows.values() if d['op']==r['op'] and d['chip']==r['chip'] and d['shape']==r['shape'] and abs(r['ca_us']-d['ca_us'])<.00001]
        if len(matches)==1:
            d=matches[0];d['origins'].setdefault(src,[])
            if role not in d['origins'][src]:d['origins'][src].append(role)
        else:unresolved.append(dict(**r,source=src,role=role,reason=f'{len(matches)} unique fingerprint/window matches'))
    return unresolved


def slices(r):
    op,sku,s=r['op'],r['chip'],r['shape']
    if op=='fa':
        q,kv,d,b,h=s
        yield 'KV '+('≤1024' if kv<=1024 else '1025–32767' if kv<32768 else '≥32Ki')
        if q<=17 and 512<=kv<=1024:yield '校验 Q≤17，KV 512–1024'
        if d==128:yield 'D=128'
        if h*b>1:yield '多个注意力头/请求'
    if op in ('cast_bf16_fp32','sigmoid_fp32'):
        issued_bytes=s[0]*s[1]*(6 if op=='cast_bf16_fp32' else 8)
        l2=load_chip(sku).details['l2_bytes']
        yield ('输入+输出 ≤ L2' if issued_bytes<=l2 else 'L2 < 输入+输出 ≤ 2L2' if issued_bytes<=2*l2 else '输入+输出 > 2L2')
        if 'B8 prefill projection/beta' in r['contexts']:
            yield 'B8 全序列'
            yield f'B8，每请求 {s[0]//8//1024}Ki'
        if s[0]>=262144:yield '≥262144 行'
    if op=='quant':
        if s[0]*s[1]>=2**20:yield '至少 2²⁰ 个元素'
        if s[1]>=5120:yield '宽度至少 5120'
    if op in ('matmul','addmm'):
        if s[2]>=512:yield '至少两次 256-K 页'
        if s[0]*s[1]>=2**20:yield '输出至少 2²⁰ 个元素'
    if op in ('mhc_sinkhorn','mhc_pre_sinkhorn','mhc_pre','attn_res_fwd','moe_gating_top_k'):
        yield 'T=1' if s[0]==1 else 'T=2–8' if s[0]<=8 else 'T=9–64' if s[0]<=64 else 'T>64'
        if op=='attn_res_fwd' and s[1]>=5120:yield 'hidden≥5120'
        if op in ('mhc_sinkhorn','mhc_pre_sinkhorn') and s[0]>64:
            yield f"T>64、{r['attrs'].get('num_iters',20)} 次迭代"
    if op in ('moe_ffn','grouped_matmul_swiglu_quant_v2','gmm2'):
        yield '单 expert' if s[-1]==1 else '多个 expert'
        counts=r['attrs'].get('expert_counts',[])
        if counts and min(counts)==1 and max(counts)>1:yield '至少一个 expert 恰有一个 token，其余含多行'
    if op=='lightning_indexer':
        yield 'KV<4096' if s[2]<4096 else 'KV≥4096'
    if op in ('fla_kda_v2','kda_recurrent'):
        yield 'T>64' if s[1]>64 else 'T≤64'
    if op=='vllm_hc_post':yield 'D>4096' if s[-1]>4096 else 'D≤4096'


def summary(items,field):
    if not items:return {}
    worst=max(items,key=lambda x:abs(x[field]/x['ca_us']-1))
    return dict(n=len(items),shapes=len({tuple(r['shape']) for r in items}),wape=100*sum(abs(r[field]-r['ca_us']) for r in items)/sum(r['ca_us'] for r in items),
                max_ape=100*abs(worst[field]-worst['ca_us'])/worst['ca_us'],worst_shape=worst['shape'],pred_us=worst[field],ca_us=worst['ca_us'],
                worst_trace=worst['trace_sha256'])


def build():
    unresolved=populate()
    primary=defaultdict(list);groups=defaultdict(list);observations=[]
    for r in sorted(rows.values(),key=lambda z:(z['op'],z['chip'],z['shape'],z['ca_us'])):
        op,sku,s,attrs=r['op'],r['chip'],r['shape'],r['attrs']
        verdict=call(op,s,attrs,sku)
        if verdict.latency_us is None:raise ValueError(f'Unpriced contract {r} {verdict.reason}')
        r['current_us']=verdict.latency_us
        r['v024_us']=call(op,s,attrs,sku,'calibrated').latency_us if op=='fa' and sku==DT else r['current_us']
        r['future_970dt_us']=call(op,s,attrs,FUT).latency_us
        r['active_aic']=verdict.active_aic;r['active_aiv']=verdict.active_aiv
        r['logical_io_bytes']=verdict.work['logical_read_bytes']+verdict.work['logical_write_bytes']
        role='prospective v0.25' if 'v0.25 prospective' in r['contexts'] else 'development replay v0.25'
        r['role']=role;r['slices']=list(slices(r))
        labels=' '.join(t for v in r['origins'].values() for t in v).lower()
        r['previously_evaluated_as_holdout']=any(k in labels for k in ('holdout','frozen','independent','prospective','primary','reserve','留出'))
        primary[op,sku,role].append(r)
        for slice_name in r['slices']:groups[op,sku,slice_name,role].append(r)
        observations.append(r)
    def collect(cat):
        return [dict(op=k[0],sku=k[1],name=(k[2] if len(k)==4 else '所有合格窗'),role=k[-1],v024=summary(v,'v024_us'),current=summary(v,'current_us')) for k,v in sorted(cat.items())]
    return dict(continuing_job_cutoff_utc=read(OUT/'continuing-runs-cutoff.json')['utc'],source_sha256=sources,gates=dict(wape_pct=20,maximum_pct=40),
                scope='Per-route unique ordered aggregate instruction/SQLite SHA windows; composite and individual routes retain their own metrics. Official cold controls, KDA wrappers and numerically invalid/budget-limited captures excluded.',
                summary=collect(primary),subgroups=collect(groups),rows=observations,unresolved=unresolved,excluded_attempts=rejected)


def table(points):
    lines=['| 算子 | SKU | 数据组 | 不同形 / 窗 | WAPE | 最大 | 最差 shape | 预测 / CA µs |',
           '|---|---|---|---:|---:|---:|---|---:|']
    for p in points:
        m=p['current'];lines.append(f"| {p['op']} | {p['sku'][6:11]} | {p['name']} | {m['shapes']} / {m['n']} | {m['wape']:.2f}% | {m['max_ape']:.2f}% | {m['worst_shape']} | {m['pred_us']:.3f} / {m['ca_us']:.3f} |")
    return lines


def large_endpoints(data):
    groups=defaultdict(dict)
    for r in data['rows']:
        group=groups[r['op']]
        if r['chip'] not in group or r['logical_io_bytes']>group[r['chip']]['logical_io_bytes']:
            group[r['chip']]=r
    lines=['| 算子 | 950DT 最大逻辑读写输入、MiB | 950PR 最大逻辑读写输入、MiB |',
           '|---|---|---|']
    for op,ends in sorted(groups.items()):
        def fmt(sku):
            r=ends.get(sku)
            return 'N/A' if not r else f"{r['shape']} · {r['logical_io_bytes']/2**20:,.3f}"
        lines.append(f'| {op} | {fmt(DT)} | {fmt(PR)} |')
    return lines


def render(data):
    hist=[s for s in data['summary'] if s['role']=='development replay v0.25']
    new=[s for s in data['summary'] if s['role']=='prospective v0.25']
    bad=[p for p in data['subgroups'] if p['current']['wape']>20 or p['current']['max_ape']>40]
    top=max(hist,key=lambda p:p['current']['max_ape'])
    lines=['# 全路线合格设备窗复算','',
           *table(hist),'',
           f"上表 {sum(x['current']['n'] for x in hist)} 个历史调用窗全部按本轮开发回放；当时曾作为留出的记录在 JSON 中保留原标签。每个算子内部按 SKU、原始 trace/数据库的有序 SHA 去重；组合算子与其单段各算一次各自的合同，不能将全表视作同一算子的独立试验数。WAPE 为 Σ|预测−CA| / ΣCA，最大为 max|预测/CA−1|。历史整体单窗最大值 {top['current']['max_ape']:.2f}%（{top['op']}）。当前确有子组超限，见下。",'',
           f"持续队列纳入此版的快照截止 {data['continuing_job_cutoff_utc']}。快照逐端锁定 KDA 与 Pro 原始回执，后续后台作业另有实时报告。[截止时每个已合格端的 SHA](continuing-runs-cutoff.json)",'',
           '## 有物理或工作负载含义的超限分组','',
           *table(bad),'','分组标准在全量 JSON 逐行给出。单端只有一个点时 WAPE 等于该点误差。原主表十形、所有已知形及 B8 真实 prefill 是不同分母。','',
           '## 新冻结 FA 检验','',*table(new),'',
           'FA 旧版 DT 默认使用历史局部调度；新默认与 PR、970DT 使用同一目标资源排程。所有比较在相同 CA 原窗上进行。PR 作为数值不变的对照。[新 FA 上模型、下实测及逐形对照](fa-method.md) · [所有算子的逐形误差表](shape-errors.md) · [旧版、新版、双端和具体收据](audit.json)','',
           '## 每条路线已经测到的数据量','',
           '按各算子的公共逻辑读字节与写字节之和，从合格 CA 窗口选择最大输入。多次读取、缓存行、workspace 或物理 HBM 报告有其他口径，以下 MiB 只表示模型接口可见的输入输出量。相同 shape 上 MoE 的 expert 路由等全部属性见逐形表。','',
           *large_endpoints(data),'','## 证据边界','',
           '- mHC Post、RMSNorm 和融合 KDA V1 的端点由同版各自的有序状态、完整窗或同步审计覆盖。此处仅单列可用的 vLLM HcPost 以及 FLA V2 三个设备窗求和。',
           '- vLLM HcPost 的形状输入使用公开四维 [B,S,4,D]，与 CANN aclnnMhcPost 分开。KDA V2 的三段之和排除了 launch 间隙，标签和三个 SHA 全部保留。',
           '- Flash/Kimi 的真实 hidden 宽单次 B8 全核运行还需查看原生队列；中途写入、wrapper 包络与超时记录均不进入指令终点 WAPE。现有 32k/128k FA 的长 KV 为单头稠密校验。模型的稀疏、压缩、KDA 模板和分块 prefill 不能借用这两个长 KV 点作为独立验证。',
           '- AttnResFwd 仅给出原扩展、有效读写槽和全输出通过的观测。[2,7168,8,3] 等超时尝试缺少完整输出。宿主 stride/地址边界分析表明静态分配内可容纳该调用，信号出现在停机后，尚无证据把失败归因于数据搬运越界。',
           '- 970DT 的逐行预测为芯片参数外推。签名相同的物理窗若在多份开发或留出汇总出现，JSON 保留各个原始角色。独立前序有别的测量保持不同的 trace；SHA 缺失仅允许补连一个同算子、同芯片、同 shape、同 CA 的已知窗。',
           '',f"未能唯一补连的旧汇总 {len(data['unresolved'])} 行；有失败判据的原始脉络见 JSON。"]
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    data=build()
    (OUT/'audit.json').write_text(json.dumps(data,indent=2)+'\n')
    (OUT/'report.md').write_text(render(data))
