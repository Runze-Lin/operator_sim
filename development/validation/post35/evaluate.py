"""Same-window comparisons and preregistered external receipts only."""
from collections import defaultdict
from statistics import median
import math,time
from registration import D,P,F,ROOT,OUT,read,save,sha,rel,schedule


def stats(rows,key):
    if not rows:return dict(n=0,shapes=0,wape=None,maximum=None,bias=None)
    total=sum(r['ca_us'] for r in rows)
    return dict(n=len(rows),shapes=len({tuple(r['shape']) for r in rows}),wape=100*sum(abs(r[key]-r['ca_us']) for r in rows)/total,
                maximum=100*max(abs(r[key]/r['ca_us']-1) for r in rows),bias=100*sum(r[key]-r['ca_us'] for r in rows)/total)

def replay():
    ref=read(ROOT/'results/v0.33/post/replay.json');out=[];times=[];by={}
    oracle=read(ROOT/'results/v0.32/post/replay.json')['same_contract_oracle']
    for x in ref['rows']:
        t=time.perf_counter_ns();v=schedule(x['shape'],x['chip']);ms=(time.perf_counter_ns()-t)/1e6;times.append(ms)
        out.append(dict(chip=x['chip'],shape=x['shape'],trace=x['trace'],sha256=x['sha256'],kernel=x['kernel'],ca_us=x['ca_us'],
              public_us=x['v032_us'],v033_us=x['candidate_us'],us=v['us'],stages=v.get('prices'),states=v['states'],
              hardware_conditional_modes=v['model'],route=v['route'],query_ms=ms,validation_only_prior=x['validation_only_actual_prior'],
              future970_us=schedule(x['shape'],F)['us'],role='historical replay'))
    assert len({r['sha256'] for r in out})==len(out)==239
    for c in (D,P):
        rs=[r for r in out if r['chip']==c];lookup={(tuple(r['shape']),r['ca_us']):r for r in rs}
        same=[lookup[tuple(g['shape']),y] for g in oracle[c]['shapes'] for y in g['measured_us']]
        groups={'all':rs,'post_channel':[x for x in rs if x['route']=='post_channel'],'post_token':[x for x in rs if x['route']=='post_token'],'strict_same_contract':same}
        by[c]={g:{k:stats(x,k) for k in ('public_us','v033_us','us')} for g,x in groups.items()}
    return dict(replay=out,by_chip=by,oracle={s:{k:v for k,v in q.items() if k!='shapes'} for s,q in oracle.items()},
                query_ms=dict(median=median(times),p95=sorted(times)[int(len(times)*.95)],maximum=max(times)),
                benchmark='One preloaded Python process, shape+hardware; includes complete candidate computation and small setup; no CA calls.')
def observed(j,reg,own_key,cost_key='cost'):
    f=ROOT/j['receipt'];pre=j.get('already_started',False);p=j['forecast']['us']
    row=dict(chip=j['sku'],shape=j['shape'],ordinal=j['ordinal'],group=j['group'],us=p,role='awaiting full instruction window',ca_us=None,error_pct=None,forecast_970dt_us=j['future970_us'],receipt=None,collector_wall_s=None)
    if pre:row['role']='already started before freeze; eventual replay'
    if not f.is_file():return row
    d=read(f)
    registered=j.get('owner_sha256',sha(ROOT/j['owner']))
    if registered!=sha(ROOT/j['owner']) or d.get(j['owner_field'])!=registered:
        raise ValueError('owner registration mismatch '+str(f))
    if d.get('chip',d.get('sku'))!=j['sku']:raise ValueError('SKU mismatch '+str(f))
    cost=d.get('cost') or {};started=cost.get('started_utc',cost.get('begin'))
    row.update(receipt=rel(f),collector_wall_s=cost.get('seconds'),receipt_sha256=sha(f),started_utc=started)
    s=None
    if j['group']=='isolated_1024':
        if d.get('qualified') and d['shape']==j['shape']:s,=d['stages']
    else:
        hits=[x for x in d.get('rows',[]) if x.get('i')==j['ordinal'] and x.get('shape')==j['shape'] and x.get('timing_qualified')]
        if hits:
            x,=hits;s=x['stage']
    if not s:
        row['role']='terminal; no eligible full window';return row
    ca=s['ca_us'];assert ca>0 and len(s['trace_sha256'])==64 and s['kernel'].startswith('MhcPost_') and s['active_aiv']>0 and not s['active_aic']
    if cost.get('exit_code')!=0 or cost.get('stop_reason'):raise ValueError('qualified receipt conflicts with budget')
    row.update(ca_us=ca,role='frozen independent' if not pre and started and started>reg['utc'] else 'historical replay',
               error_pct=100*(p/ca-1),source_trace=s['trace'],trace_sha256=s['trace_sha256'],kernel=s['kernel'],active_aiv=s['active_aiv'])
    return row
def forward():
    reg=read(OUT/'freeze.json');g=read(OUT/'fresh-freeze.json')
    rows=[observed(j,reg,'owner_sha256') for j in reg['jobs']]
    for chip,tag in ((D,'dt'),(P,'pr')):
        for p in g['rows'][chip]:
            i=p['i'];j=dict(sku=chip,shape=p['shape'],ordinal=i,group='fresh',forecast={'us':p['candidate_us']},future970_us=g['rows'][F][i]['candidate_us'],
                           owner=rel(OUT/'fresh-freeze.json'),owner_field='freeze_sha256',receipt=rel(OUT/('fresh-'+tag+'.json')))
            row=observed(j,g,'owner_sha256');row.update(public_us=p['public_us'],sequence_role=p['role']);rows.append(row)
    return dict(rows=rows,independent={c:stats([r for r in rows if r['chip']==c and r['role']=='frozen independent'],'us') for c in (D,P)})
def fmt(v):return 'N/A' if v is None else f'{v:.2f}%'
def main():
    e=replay();save(OUT/'replay.json',e);fwd=forward();save(OUT/'forward.json',fwd)
    b=e['by_chip'];lines=['# 内置 mHC Post 的首笔写与多轮发起间隔','',
        '| 芯片与数据角色 | 形 / 窗 | v0.34 公开 WAPE / 最大 | v0.33 候选 WAPE / 最大 | 本轮候选 WAPE / 最大 | 本轮有符号总偏差 |',
        '|---|---:|---:|---:|---:|---:|']
    for c in (D,P):
        for g,nm in [('all','全量历史'),('post_token','token 分核'),('post_channel','通道分核'),('strict_same_contract','严格同形冷热')]:
            s=b[c][g];m=s['us'];parts=[' / '.join(fmt(v[k]) for k in ('wape','maximum')) for v in (s['public_us'],s['v033_us'],s['us'])]
            lines.append(f"| {c} · {nm} | {m['shapes']} / {m['n']} | "+' | '.join(parts)+f" | {m['bias']:+.2f}% |")
        h=fwd['independent'][c]
        lines.append(f"| {c} · 冻结后独立 | {h['shapes']} / {h['n']} | 见逐窗 | N/A | {fmt(h['wape'])} / {fmt(h['maximum'])} | {fmt(h['bias'])} |")
    lines.extend(['','所有行是安装的 CANN 9.2.1 ops-transformer 内置 aclnnMhcPost、连续四路 BF16、含 hRes 的完整系统指令窗。vLLM 的 HcPost 有不同 ABI，单独统计。历史的 239 窗曾用于诊断、估价或选择方案，当前统一按回放。两端全量最大值小于 50%，WAPE 仍高于 25%；通道分核受未知历史影响最大。','',
        '## 关键核究竟在等哪一步','',
        '| 形状 [T,4,D] / 950 | 相邻状态 | 末核实际行 | 首写发起距全片起点，周期 | 后续写发起间隔，周期 | 最后写响应 / 末段，周期 | 设备 µs |','|---|---|---:|---:|---|---:|---:|'])
    for r in read(OUT/'physical.json')['rows']:
        if r['shape'][::2] not in ([65,320],[65,5120],[449,512]):continue
        v=r['actual_phases_ticks'];lines.append(f"| {r['shape']} / {'PR' if r['chip']==P else 'DT'} | {r['roles']['prior']} | {r['rows']} | {v['first_write']} | "+' / '.join(str(n) for n in r['write_intervals'])+f" | {v['last_write_response']} / {v['exit']} | {r['ca_us']:.3f} |")
    lines.extend(['','[65,4,5120] 的末核每行分成两个 2560 通道片。四个旧完整窗中三段相邻写间隔均等于 1377 周期，而首次写发起受真实前序影响，DT 普通换形比立即重复晚 340 周期，PR 晚 279 周期。槽周转可在之前描述符退休前开始。对 [449,4,512]，DT 扩核和热态首写相差 2234 周期，随后七段间隔加总只差 77 周期。PR 扩核时最后退出的物理核实际处理八行，热重复最后核处理九行；以最长行数估临界核仍有潜在误差。另一个两轮样例 [65,4,320] 的 DT 首发→第二笔写比热重复多 533 周期，说明窄单片早期轮次也会受前序影响。','',
        '## 候选怎样计算','',
        '先由目标 AIV、UB 与源码分核规则算每核行数、当前片宽以及输出字节。单片每行发起四笔读（系数 64、16 B，主数据 2D、8D B）与一笔 8D B 写。分别计第一行输入响应、后续行发起间隔与最后一个写响应。MTE 价格优先使用目标每核 B/cycle；SIMD 价格按目标 FP32 每核每周期操作数缩放。首次、扩核、已覆盖与立即重复的分段参数取自已有 950DT 逐笔 DMA 标定。其余芯片不读目标 CA。冻结诊断中的 one-slot row 表示每个 token 的通道宽只切一片；底层输入队列仍可重叠连续两行，队列深度不由这个标签推断。','',
        '双片沿用从 DT 已验证的两槽依赖：一笔写发起取前一写发起加最小发起间隔、两片前相同输入槽释放后读就绪加向量处理，两者的较晚值。把某条件的首写时间分为一次性到达、控制、最晚输入就绪与向量处理，以单片真实分段之差把 DT 已校准的“已覆盖”首写移至各可行起点；末写响应也按本地条件调整。中间的各片只付排程算出的发起间隔，分片迭代数很大时采用对数步的 max-plus 递推。对物理双片只见过普通换形/立即重复，“首次/扩核的首发差可从单片迁移”是待检验假设。','',
        '令某端点的最慢核处理 R 行，单片的成本为 `首行写发起 + 后续 R−1 行暴露间隔 + 末写响应 + 退出`。双片的后续间隔数为 2R−1。另以数据量除以目标共享带宽提供的服务下限约束整个窗口。未知历史时，token 保留 v0.33 既定的端点几何中值，通道分核保留原四笔写模型和限幅规则。全程只传 shape、dtype 和芯片。','',
        '更具体地，令 $W_j$ 为第 j 片写发起的周期，$I_j$ 为相邻两片可持续的最短发起间隔，$L$ 为同一输入槽下一次可发起读取相对两片前写发起的提前量，$A_j$ 为新一片输入读就绪的跨度，$C_j$ 为输入就绪到能够写结果的计算跨度。从第三片起 $W_j=\\max(W_{j-1}+I_j,W_{j-2}-L+A_j+C_j)$。在首写 $W_0$ 上加入对应状态估计与已覆盖状态估计的差。$T_s=\\max(T_{shared},(W_{last}+D_s+E_s)/(1000 f))$，D 为末写响应周期，E 为最后一写完成到物理退出的周期，f 以 GHz 计，结果以 µs 计。这里的 s 只是可行前序的条件假设，不是设备调用的可见输入。','',
        '读写响应的有效服务单价由既有 DT 指令片段给出 18.396 和 16.018 周期/KiB，另计每种状态的固定响应；目标若有 MTE 端口带宽即替换。31.775 周期/64通道向量组来自 DT 第一笔输出写前的向量工作，按目标每核每周期 FP32 峰值反比缩放。双片槽的 L=242 周期来自宽行 DT 实际复用同 UB 地址时的指令发起差，两条状态不共付这一差。首发态/热态控制到首笔操作数读的参考值为 3105.5 / 1542.0 周期，稳态读发起间隔参考值为 706 / 555 周期，写间隔 617 / 554.5 周期；负载吞吐下限仍优先约束。逐项零点及源数据见[单片发起标定](../../v0.25/post/prices.json)和[两槽周转标定](../../v0.30/post/prices.json)。','',
        '例如宽行 [65,4,5120] 的 DT 条件预算为普通换形 5.661 µs、热重复 5.041 µs，两者后续三次写的预算同为 2.504 µs。PR 条件下模型给 2.566 µs，但实际仍是 2.504 µs：当前对 PR 重新计算每核 FP32 吞吐的缩放在这段略高。970DT 有 120 个可用 AIV，输入只需每个忙核一行，点值外推 3.797 µs；没有其设备校准。','',
        '原先登记的十个同合同形状，任意单点要求最大≤50%时的最低 WAPE 为 DT 28.46%、PR 29.43%。其中 DT 的 [8,4,896] 只有一窗，真正有多个前序的九形 DT 下界为 29.41%。该下界不能用于声称全量阈值不可达。已知调用顺序的系统仍可使用公开的 OperatorStream，其耗时与无状态候选分开评估。','',
        '## 冻结后验证与成本','',
        '新短序列每端一个新进程依序执行 [69,4,176] 首发与紧邻重复，再执行 [2,4,368] 换形与重复。前者两端各 35 个 AIV、最忙核两轮，后者四个 AIV 切通道。真实运行沿用已退休的 v0.25/26 批次锁，每端最多 40 分钟、2 GiB，固定六个物理宿主核、低优先级执行。每个输出元素、真实 SKU、安装包 .text、各活动 AIV 的 END、官方核表与输出总字节均需通过。其余验证复用既有采集任务；1024-token DT 端在本模型冻结前已开始，只能计回放。','',
        '| shape / 芯片 | 数据角色或预登记状态 | 模型 µs | CA µs | 误差 | 970DT 外推 µs |','|---|---|---:|---:|---:|---:|'])
    for r in fwd['rows']:
        ca=r['ca_us'];state=r['role']+(' · '+r['sequence_role'] if 'sequence_role' in r else '')
        lines.append(f"| {r['shape']} / {'PR' if r['chip']==P else 'DT'} | {state} | {r['us']:.3f} | {ca if ca is None else f'{ca:.3f}'} | {fmt(r['error_pct'])} | {r['forecast_970dt_us']:.3f} |".replace(' | None |',' | N/A |'))
    from lower_bound import markdown
    lines+=markdown(e['replay'])
    from operator_model import load_chip
    from operator_model.post_resource import resource_post
    large=[]
    lines+=['','## 真实 hidden、单卡 B8 整输入的条件外推','','| 模型宽 D | 每请求输入 | 扁平 T | DT / PR / 970DT 预测设备 ms | DT / PR / 970DT 最忙核行数 |','|---:|---:|---:|---|---|']
    for title,width in [('DeepSeek-V4.1-Flash',5120),('DeepSeek-V4-Pro',7168)]:
        cfg=read(ROOT/'configs/models'/(title+'.json'))
        assert cfg.get('text_config',cfg)['hidden_size']==width
        for n in (32768,131072,262144):
            sh=[8*n,4,width];by_chip={}
            for c in (D,P,F):
                target=load_chip(c);v=schedule(sh,c);diag=resource_post(sh,target,{})[-1]
                by_chip[c]=dict(us=v['us'],local_stages_us_by_state=v['prices'],shared_floor_us=diag['shared_floor_us'],rows=v['rows'],
                      active=diag['target_plan']['active_aiv'],channel_tiles=v['fragments'],data_role='unmeasured resource extrapolation')
            large.append(dict(model_width=title,shape=sh,context=n,batch=8,by_chip=by_chip))
            lines.append(f"| {width} | {n} | {sh[0]} | "+' / '.join(f"{by_chip[c]['us']/1000:.3f}" for c in (D,P,F))+' | '+' / '.join(str(by_chip[c]['rows']) for c in (D,P,F))+' |')
    lines+=['','假设将所有未分块 token 合成一次内置 aclnnMhcPost 调用，实际 chunked prefill 须按调度块分别估算，vLLM HcPost 另计。DT 随每核轮次增加而接近线性；在 Flash B8×32Ki 上，DT 每核4096行、共享内存服务下限6.011 ms，模型给6.839 ms。PR 每核4682行，模型达到配置的共享带宽服务下限17.188 ms；970DT 每核2185行、共享下限1.665 ms，模型给3.342 ms。共享下限沿用公开资源估算的流量与带宽情景，不能由此认定真机时哪一级发生了堵塞。[逐项资源与行数](whole-input.json)']
    save(OUT/'whole-input.json',dict(rows=large,scope='Model hidden widths with a conditional built-in CANN Post at N=4. No CA for these end-to-end tensor sizes; Kimi uses a different residual mechanism.'))
    tm=e['query_ms']
    lines+=['',f"239 次已加载估时器在线查询的中位 {tm['median']:.3f} ms、P95 {tm['p95']:.3f} ms、最大 {tm['maximum']:.3f} ms。设备微秒、宿主查询和离线 CA 墙钟是三个量。970DT 全部外推。",'',
        '[完整历史逐窗、旧模型与所有阶段](replay.json) · [v0.34 安装包同集对齐](baseline-wheel.json) · [真实末核 DMA、地址和 SHA](physical.json) · [候选冻结](freeze.json) · [新留出、预算与查重](fresh-freeze.json) · [新 CA 状态](fresh-state.json) · [端点及资格](forward.json) · [算例](../../../docs/examples/mhc_post.md)','']
    tmp=OUT/'report.tmp';tmp.write_text('\n'.join(lines));tmp.replace(OUT/'report.md')
if __name__=='__main__':main()
