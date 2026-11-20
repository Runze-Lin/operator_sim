"""Refresh paired forward scores only from the registered, qualified native router receipts."""
import fcntl,sys,time,os
from math import ceil
from statistics import median
from campaign import REG,RAW,OLD,TARGET,ROOT,OUT,DT,PR,FUT,read,sha,rel,write,now,forecast,predict,load_chip,structural

def stats(rs,key):
    if not rs:return dict(windows=0,different_shapes=0,wape=None,maximum=None)
    return dict(windows=len(rs),different_shapes=len({tuple(r['shape']) for r in rs}),wape=100*sum(abs(r[key]-r['ca_us']) for r in rs)/sum(r['ca_us'] for r in rs),maximum=100*max(abs(r[key]/r['ca_us']-1) for r in rs))

def refresh():
    f=read(REG);phys=read(OUT/'physical.json');base=read(ROOT/'results/v0.34/replay.json')['rows'];hist=[];forward=[];agg={}
    for r in phys['rows']:
        old,=[z for z in base if z['op']=='moe_gating_top_k' and z['chip']==r['chip'] and z['identity']==[r['trace_sha256']]]
        model=predict(r['shape'],load_chip(r['chip']));assert model['active_aiv']==r['aiv'] and all(x['ub_batch_rows']==model['batch_capacity'] for x in r['cores']);
        hist.append(dict(chip=r['chip'],shape=r['shape'],kernel=old['kernel'],trace_sha256=r['trace_sha256'],ca_us=r['ca_us'],public=old['us'],candidate=model['us'],role=r['role'],aiv=r['aiv'],max_rows=max(x['rows'] for x in r['cores']),cap=model['batch_capacity'],
            source=rel(OUT/'physical.json'),future_extrapolation=predict(r['shape'],load_chip(FUT))['us']))
    for chip in (DT,PR):
        rs=[r for r in hist if r['chip']==chip];agg[chip]={scope:{key:stats(ss,key) for key in ('public','candidate')} for scope,ss in [('all archive',rs),('phase sources',[r for r in rs if r['role']=='primitive-price development']),('other archive',[r for r in rs if r['role']!='primitive-price development'])]}
    for j in f['cases']:
        shape=j['shape']
        for chip,tag in ((DT,'dt'),(PR,'pr'),(FUT,'extrap')):
            current=j['forecasts'][chip];path=OUT/(tag+'.json') if shape==TARGET else ROOT/f['existing'][chip] if chip!=FUT and shape[0]==4096 else None
            row=dict(shape=shape,chip=chip,public=current['public_us'],candidate=current['us'],ca_us=None,role='prospective pending' if path and chip!=FUT else 'extrapolation',capture_s=None,active_aiv=current['active_aiv'],critical_rows=current['rows_per_core'],critical_batches=current['critical_batches'],tail_rows=current['tail_rows'],evidence=None)
            if path and chip!=FUT and path.exists():
                q=read(path);v=q.get('verdict',q);assert q['chip']==chip and q['shape']==shape and q['freeze_sha256']==(sha(REG) if shape==TARGET else f['previous_sha']);cost=q.get('cost',{});row.update(evidence=rel(path),role='terminal incomplete',capture_s=cost.get('seconds'))
                if v.get('qualified'):
                    assert v['instruction_us']>0 and v['kernel'].startswith('MoeGatingTopK_') and v['released_text_sha256'] and len(v['trace_sha256'])==64
                    row.update(ca_us=v['instruction_us'],trace_sha256=v['trace_sha256'],kernel=v['kernel'],role='independent frozen before acquisition' if cost['started_utc']>f['utc'] else 'replay (acquisition started before forecast)',mechanism=q.get('mechanism'))
                    row['same_kernel_key']=v['kernel'].endswith('_10005')
                    if shape[0]==4096 and row['same_kernel_key']:
                        artifact=OUT/f'4096-{tag}-mechanism.json'
                        if not artifact.exists():
                            try:mechanism=structural(ROOT/v['trace'],shape,current)
                            except (OSError,ValueError,KeyError,IndexError,AssertionError) as exc:mechanism=dict(all_lanes_match=False,error=repr(exc),trace_sha256=v['trace_sha256'])
                            write(artifact,mechanism)
                        row['mechanism_evidence']=rel(artifact)
                    row['signed_pct_public']=100*(row['public']/row['ca_us']-1);row['signed_pct_candidate']=100*(row['candidate']/row['ca_us']-1)
            forward.append(row)
    write(OUT/'score.json',dict(freeze_sha256=sha(REG),archive=hist,metrics=agg,forward=forward,extrapolation_scope='Every 970DT result is target-only. Large 950 geometries without receipts are also projection-only.'))
    money=lambda x:'N/A' if x is None else f'{x:.3f}'
    state=read(OUT/'state.json') if (OUT/'state.json').exists() else {};old=read(OLD/'state.json');m=next(x for x in f['cases'] if x['shape']==TARGET)['forecasts']
    lines=['# MoE sigmoid TopK 的多行与多批验证','',f"旧档有 22 个完整时间窗、284 个活动 AIV。新收据为 {sum(r['ca_us'] is not None for r in forward)} / 4。1536 行采集状态为 `{state.get('phase','queued')}`，4096 行沿用原任务 `{old.get('phase')}`。单卡 FP32 得分、FP32 偏置、sigmoid、单专家组、返回 FP32 权重及 int32 IDs，实际入口 `aclnnMoeGatingTopK`。","",'| 数据角色 | SKU | 窗 | 公开 WAPE / 最大 | 候选 WAPE / 最大 |','|---|---|---:|---:|---:|']
    for s,lab in [('all archive','完整历史'),('other archive','排除三项 DT 阶段取价来源')]:
        for c in (DT,PR):
            r=agg[c][s];a,b=r['public'],r['candidate'];lines.append(f"| {lab} | {c} | {a['windows']} | {a['wape']:.2f}% / {a['maximum']:.2f}% | {b['wape']:.2f}% / {b['maximum']:.2f}% |")
    lines+=['','## 数据校验与可移植的价格','',
        '全部 22 窗的实际库 ELF 与 CANN 9.2.1 `ops_transformer/moe_gating_top_k_apt` 相同，tiling key 10005 对应包内 `MoeGatingTopKWithoutGroupRegbase`。按 token 行分核。首个 GM 描述符取 tiling，bias 每核读一次。每批一笔 logits 输入，逐行 sigmoid、加偏置排序，再用原 sigmoid 分值 gather + 归一化，最后分别对 weights 与 IDs 各发一笔写。全部核对照 `VBS32`、`VMS4V2`、`MOV_UB_TO_UB` 的次数，行总数与实际 token 相等。实际排序使用 32 元组和四路归并，模型没有加入直方图阶段。', '',
        '令 A=32⌈E/32⌉，J=8⌈K/8⌉，T/E/K 分别为 token、专家和选出的专家数。每核最多 r=⌈T/C⌉ 行，活动核数 ⌈T/r⌉，C 为芯片 AIV 数。预测批上限 `B=min(128,floor((U−4×17×A)/(8×A+16×J)))`。U 是目标可用 UB 字节。分母是双缓冲 logits 加双缓冲两份输出的逐行字节。17 是由已知片上地址反推的 host *有效预留*，与源代码可见的全部物理缓冲大小分别解释。首笔权重输出的 UB 源地址除以 `8A` 给出实际 B。E=128/256/384/512/896 分别得到 128/108/71/51/26，覆盖 284 个活动核；跨代 host 上限与预留仍是假设。','',
        '令 m=⌈E/32⌉。每行先执行一条含 m 个 repeat 的 `VBS32`。四路归并每层将长度相同的完整四元组合并成一次可重复发射，剩下的 2–4 路再单发，单路尾进入下一层；每层末及首层前各复制整行的 8A 字节键值对。E=896 的四条归并按（元素,重复）为 (896,7)、(512,1)、(384,1)、(896,1)。E=128 仅为 (128,1)。对应 10005 程序是确定性的。[安装包的完整排序循环](../../../toolchains/cann/9.2.1/cann-9.2.1/aarch64-linux/asc/impl/basic_api/dav_3510/kernel_operator_proposal_impl.h)','',
        '同核物理执行周期来自 DT 单行 E128、896 的分条 trace。`VBS32 = 23+0.5A`，每次 UB 拷贝 `29+A/32`；`VMS4V2(n,h)=40.20+0.51625n+21.7067(h−1)`，n 为本条处理元素数，h 为该条的重复次数。后者用五个真实归并窗口拟合三个物理项。VF 首行 norm 到 sort 为 `111.167+0.451823A` 周期，末行 K≤64 gather/norm 为119周期。只有 VF 计算随每核 FP32 算力/频率比换算，专用排序与 UB 引擎暂保持每周期吞吐；目标独立价格可以替换。已核验的专家宽度为 128、256、384、512 和 896，均对齐 32；非对齐专家数、K>64、hash、分组、不同激活和输出物化需要独立路径证据。', '',
        '一批多行时，末行写之前的完整响应不必逐行重复。DT [65,128,8] 含32个双行核，连续两次 VBS32 中位发起间隔 528 周期；先前 select 发起到下行 norm 发起中位24周期，后续 norm 到排序比孤立首行多77周期。每个批付一次首行及 rᵢ−1 次稳态发起间隔。首次 bias 读准备为1243周期，后续 logits 响应代理855周期；界面以 `max(读响应, 数据/有效带宽, 前批写尾)` 建模。每核 MTE2/MTE3 价格限于本核端口和 HBM/活动核数，块到达偏斜由同一 DT 多核 trace 的 bias 发起跨度定价。多个完整 UB 批之间的间隙尚无可评分的原始新窗。这是冻结候选，公开后端继续使用旧策略。[物理价格与取价 trace](prices.json) · [逐核指令、读写与批大小](physical.json)','',
        '## 真实几何与窗口','',
        'Kimi 配置有 896 个路由专家、每 token 选 16 个、sigmoid、1 组。取单卡 batch=8，每次每请求 192 行，得 1536 行。在 32Ki 输入上这是切块的一次调用；4096 行对应每请求 512 行。无切块的 32/128/256Ki 也给出目标价，包括有 token 多轮的一侧。未取得完整窗的长形保持 N/A。实际服务的调度、router GEMM、EP 分发与专家计算需由系统另行调度。','',
        '| T/E/K | 芯片 | 活动 AIV / 忙核行 / 批数 / 最后批行 | 公开 µs | 候选 µs | 完整 CA µs | 口径 |','|---|---|---|---:|---:|---:|---|']
    for r in forward:
        lines.append(f"| {r['shape']} | {r['chip']} | {r['active_aiv']}/{r['critical_rows']}/{r['critical_batches']}/{r['tail_rows']} | {money(r['public'])} | {money(r['candidate'])} | {money(r['ca_us'])} | {r['role']} |")
    lines+=['',f"1536 形的 DT/PR 预测分别为 {m[DT]['us']:.3f}/{m[PR]['us']:.3f} µs。DT 满核 24 行，一批；PR 忙核 28 行，两批。该形每个输出张量严格检查 24,576 个条目，核级身份、输出和流水整窗分别留证；每端采集预算 4 小时和 16 GiB。原 4096 双端保持原预约，没有重新采集。[采前价格与核验合同](freeze.json) · [1536 行冻结元数据更正](freeze-erratum.md) · [逐 shape 对照](score.json) · [原队列](../../v0.29/large/vector/state.json)"]
    tmp=OUT/f'.report.{os.getpid()}.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(OUT/'report.md')

def watch():
    f=read(REG);paths=[OUT/f'{s}.json' for s in ('dt','pr')]+[ROOT/p for p in f['existing'].values()];seen=None
    with (RAW/'report.lock').open('a') as l:
        fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            sig=tuple(p.stat().st_mtime_ns if p.exists() else None for p in paths+[OUT/'state.json',OLD/'state.json'])
            if sig!=seen:refresh();seen=sig
            if all(p.exists() for p in paths):return
            time.sleep(45)

if __name__=='__main__':{'refresh':refresh,'watch':watch}[sys.argv[1]]()
