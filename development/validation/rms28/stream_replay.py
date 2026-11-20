"""Reprice archived RMSNorm device windows through the current public API; run no CA."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate, load_chip, OperatorStream, rmsnorm_pipeline
from development.validation.rms_ordered_verify.verify import digest, events

OUT = ROOT/"results/v0.28/rms"
DT, PR = 'Ascend950DT_9582', 'Ascend950PR_957b'
def read(path): return json.loads((ROOT/path).read_text())
def persist(path, value):
    (OUT/path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def measure(rows, key):
    eligible = [r for r in rows if r.get(key) is not None]
    if not eligible:return None
    delta = [abs(r[key]-r['ca_us']) for r in eligible]
    return dict(n=len(delta),shapes=len({tuple(r['shape']) for r in eligible}),
                wape=100*sum(delta)/sum(r['ca_us'] for r in eligible),
                max=100*max(d/r['ca_us'] for d,r in zip(delta,eligible)))
def table(rows, axes, metrics):
    by = defaultdict(list)
    for r in rows:
        by[tuple(r[key] for key in axes)].append(r)
    return [{**dict(zip(axes,g)),**{m:measure(items,m) for m in metrics}} for g,items in sorted(by.items())]

def address_contract(row):
    """Hash each logical row-range's four operands, including every UB block on each AIV."""
    trace=ROOT/row['trace']
    assert digest(trace)==row['sha256']
    dispatch=trace.parent.parent
    pc0=int((dispatch/'dump/pc_start_addr.txt').read_text(),16)
    operators={0x4878:('gamma','src_addr'),0x4a48:('x','src_addr'),0x4f38:('rstd','dst_addr'),0x5128:('y','dst_addr')}
    lanes=[]
    for shard in sorted(trace.parent.glob('core*/trace.json')):
        ops=defaultdict(list)
        for e in events(shard):
            if not e['name'].startswith('MOV_SRC_TO_DST'):continue
            pc=int(e['args']['pc_addr'],16)-pc0
            if pc in operators:
                key,which=operators[pc]
                detail=json.loads(e['args']['detail'])
                ops[key].append((e['ts'],detail[which],detail['processed_bytes']))
        by_issue={k:[(addr,count) for _,addr,count in sorted(ops[k])] for k in ('gamma','x','rstd','y')}
        assert len(by_issue['gamma'])==1
        assert len(by_issue['x'])==len(by_issue['rstd'])==len(by_issue['y'])>0
        lanes.append(by_issue)
    t,w=row['shape'];lanes.sort(key=lambda r:r['rstd'][0][0])
    base={k:lanes[0][k][0][0] for k in ('gamma','x','rstd','y')}
    assert len(lanes)==row['active_aiv']
    nrow=0
    for z in lanes:
        assert z['gamma']==[(base['gamma'],w*2)]
        for (x,nx),(r,nr),(y,ny) in zip(z['x'],z['rstd'],z['y']):
            assert (r,y,x)==(base['rstd']+4*nrow,base['y']+2*w*nrow,base['x']+2*w*nrow)
            assert nr%4==0 and nx==ny==(nr//4)*w*2
            nrow+=nr//4
    assert nrow==t
    payload=json.dumps(lanes,sort_keys=True,separators=(',',':')).encode()
    return dict(sha256=sha256(payload).hexdigest(),logical_blocks=len(lanes),ub_blocks=sum(len(r['x']) for r in lanes),
                operands=base,task=row['task']['full_task_id'],raw_stream_id=row['task']['raw_stream_id'])

def paired():
    collections={
        'v0.23':('results/rms-unknown23/','independent.json','rows','process'),
        'v0.27':('results/v0.27/rms/','score.json','measured','group'),
        'v0.28':('results/v0.28/rms/','evaluation.json','measured','group'),
    }
    rows=[];addr={};pairs=[]
    for batch,(directory,name,key,group_key) in collections.items():
        measured=read(directory+name)[key]
        gold={(r['chip'],r[group_key],r['i']):r for r in measured}
        plan=read(directory+'freeze.json')
        streams={}
        for p in plan['rows']:
            sku=p['chip']
            if sku not in (DT,PR):continue
            group=p[group_key]
            flight=(sku,group)
            stream=streams.setdefault(flight,OperatorStream(sku))
            s=p['shape'];h=p['prior_aiv'];role=p['role'].replace('immediate_repeat','repeat')
            i=p['i']
            k=(sku,group,i)
            curr=stream.advance('rmsnorm',s,dtype='bf16')
            manual=estimate('rmsnorm',s,{'resident_aiv':h},chip=sku,dtype='bf16')
            assert curr.latency_us==manual.latency_us and abs(curr.latency_us-p['known_us'])<1e-11
            if role=='prefix':continue
            target=gold[k]; physical=curr.projection['target_plan']['iterations']
            direct=rmsnorm_pipeline(s,chip=sku,resident_aiv=h,immediate_repeat=role=='repeat')['us']
            if batch=='v0.28':
                assert abs(direct-p['specific_context_us'])<1e-11
                if '--addresses' in sys.argv:
                    addr[k]=address_contract(target)
                    if role=='repeat':
                        prev=addr[sku,group,i-1];now=addr[k]
                        assert now['task']==prev['task']+1
                        assert target['first_tick']>gold[sku,group,i-1]['last_tick']
                        assert gold[sku,group,i-1]['shape']==s
                        pairs.append(dict(chip=sku,shape=s,group=group,same_addresses=prev['sha256']==now['sha256'],previous=prev,repeat=now))
            t=dict(batch=batch,chip=sku,group=group,ordinal=i,shape=s,role=role,H=h,A=p['active_aiv'],
                ub_iterations=physical,ca_us=target['ca_us'],trace=target['trace'],
                unknown_us=estimate('rmsnorm',s,chip=sku,dtype='bf16').latency_us,
                prior_unknown_us=p.get('previous_us',p.get('old_us')),known_us=manual.latency_us,
                stream_us=curr.latency_us,explicit_context_us=direct)
            assert t['prior_unknown_us'] is not None
            rows.append(t)
    if pairs:
        assert len(pairs)==20
        verify_runner=read('results/v0.28/rms/freeze.json')['files']['build/vector-v8/NORM']
        assert digest(ROOT/'build/vector-v8/NORM')==verify_runner
        persist('stream-addresses.json',dict(scope='Published v0.28 frozen pairs, common physical entry/END. All full-row x/y/rstd tiles and shared gamma were checked by GM address and byte count on both dispatches. These addresses were not public predict inputs.',pairs=pairs,runner_sha256=verify_runner,
            application_stream='smoke/vector_v8.cpp creates one aclrtStream before the batch; smoke/extension_norm_addmm.cpp passes that handle to aclnnRmsNorm and synchronizes each launch. CA TASK_BEGIN raw_stream_id advances with task ordinal. No claim of a single unchanged internal CA queue ID.'))
    return rows

def archive():
    rows=[]
    archive=read('results/v0.22/rms-unknown/audit.json')
    for z in archive['rows']:
        sku=z['chip'];s=z['shape'];h=z['prior_aiv']
        new=estimate('rmsnorm',s,chip=sku,dtype='bf16')
        explicit=estimate('rmsnorm',s,{'resident_aiv':h},chip=sku,dtype='bf16')
        rows.append(dict(chip=sku,provider=z['provider'],source=z['source'],trace=z['trace'],generator=z['generator'],
            shape=s,H=h,state='first' if h==0 else 'expansion' if h<z['active_aiv'] else 'covered',
            previous_unknown_us=z['unknown_us'],unknown_us=new.latency_us,historical_known_us=z['known_us'],
            known_us=explicit.latency_us,ca_us=z['ca_us'],trace_id=z['trace_id']))
    # Keep previously reported ten-shape cohorts as separate views. A trace can
    # legitimately occur in two historical reports; never pool those views.
    named=[]
    for z in read('results/v0.22/evaluation.json')['rows']:
        if z['op']!='rmsnorm':continue
        q=estimate('rmsnorm',z['shape'],chip=z['chip'],dtype='bf16')
        named.append(dict(chip=z['chip'],cohort=z['cohort'],shape=z['shape'],ca_us=z['ca_us'],
            unknown_us=q.latency_us,prior_unknown_us=z['us'],source=z['source'],trace=z['receipt']['trace']))
    late=[]
    for z in read('results/v0.24/rms-prefix/evaluation.json')['rows']:
        s=z['shape'];sku=z['chip'];h=z['true_H']
        late.append(dict(chip=sku,shape=s,H=h,observed_lower=z['observed_lower'],ca_us=z['ca_us'],trace=z['trace'],
            known_us=estimate('rmsnorm',s,{'resident_aiv':h},chip=sku).latency_us,prior_known_us=z['full_us'],
            unknown_us=estimate('rmsnorm',s,chip=sku).latency_us))
    return rows,named,late

def main():
    joint=paired();a,n,l=archive()
    early=read('results/v0.22/rms-unknown/audit.json')
    flagged={v['trace_id'] for v in early['unknown_over_40']}
    older=[z for z in a if z['trace_id'] in flagged]
    assert len(older)==len(flagged)
    large=[]
    for r in read('results/v0.20/rms/records.json'):
        if r['shape'][0]<1024:continue
        sku=r['chip'];shape=r['shape'];q=estimate('rmsnorm',shape,chip=sku)
        large.append(dict(chip=sku,shape=shape,ca_us=r['ca_us'],us=q.latency_us,
            mode=q.projection['timing_diagnostics']['model'],ub_iterations=q.projection['target_plan']['iterations'],
            first_us=estimate('rmsnorm',shape,{'resident_aiv':0},chip=sku).latency_us,source=r['trace']))
    data=dict(scope='Offline public API replay. v0.23/v0.27 stage-training, v0.28 previously frozen paired predictions; this audit adds zero independent shapes and zero CA hours. Cohorts overlap; no union summary across archive views.',
        pairs=joint,pair_groups=table(joint,['batch','chip','role'],['unknown_us','prior_unknown_us','known_us','stream_us','explicit_context_us']),
        archived=a,by_provider=table(a,['chip','provider'],['unknown_us','previous_unknown_us','known_us','historical_known_us']),
        archived_by_observed=table(a,['chip','provider','state'],['unknown_us','previous_unknown_us','known_us']),
        flagged=older,flagged_summary=table(older,['chip','provider'],['unknown_us','previous_unknown_us','known_us']),
        named=n,named_summary=table(n,['chip','cohort'],['unknown_us','prior_unknown_us']),
        prefix24=l,prefix24_summary=table(l,['chip'],['unknown_us','known_us','prior_known_us']),
        larger=large,larger_summary=table(large,['chip'],['us','first_us']))
    persist('stream-replay.json',data)

def markdown():
    q=read(OUT/'stream-replay.json')
    addr=read(OUT/'stream-addresses.json')
    def fmt(d):
        return f"{d['wape']:.2f} / {d['max']:.2f}"
    text=['# RMSNorm 经过真实公共入口的前序回放','',
        '|950 端点|v0.28 同序十形，旧有序流复跑 WAPE / 最大|专项显式复跑 WAPE / 最大|切换后 WAPE 变化|',
        '|---|---:|---:|---:|']
    for r in q['pair_groups']:
        if r['batch']=='v0.28' and r['role']=='repeat':
            text.append(f"|{r['chip'][6:11]}|{fmt(r['stream_us'])}%|{fmt(r['explicit_context_us'])}%|{r['explicit_context_us']['wape']-r['stream_us']['wape']:+.2f} 个百分点|")
    text+=['','**保留当前有序流默认估时。** 当上一调用成功时，`OperatorStream.advance` 记住已用 AIV 的确切覆盖或可证明的下界。它对 RMS 使用相同的占用 H 解释普通已覆盖调用和立即复跑，既不接受输入存储身份，也不保存上一次 RMS 的合同。v0.28 的确切复跑价格在 PR 降了最大误差，却使 WAPE 上升。下面两个旧批次在 PR 也有相同方向。把下一次相同形状一律切换缺少双端净收益。','',
        '本页只做回放。v0.23、v0.27 的完整窗曾用于本轮阶段定价，v0.28 的 40 个是之前已经冻结和公布的十形双次窗；此次没有新 CA 或新增独立样本。下列多个历史报告可引用同一原始窗，各表独立统计。','',
        '## 已标记为超限的归档窗','',
        'v0.22 审计事先标出的十次旧公共无 H 大误差都来自 DT compact 导出。把同一批重新通过 v0.28 公共入口计价，WAPE / 最大由 **44.75 / 54.94%** 降到 **30.99 / 44.60%**，确认了最大误差回到 50% 以内，这一筛选组的 WAPE 仍超过 25%。给出当时实际 H 后为 **6.64 / 13.88%**。','',
        '|shape|H / A|CA µs|旧无 H|v0.28 无 H|实际 H 的公共估计|无 H 有符号误差|','|---|---:|---:|---:|---:|---:|---:|']
    for r in q['flagged']:
        a=estimate('rmsnorm',r['shape'],chip=r['chip']).active_aiv
        text.append(f"|{r['shape'][0]} × {r['shape'][1]}|{r['H']} / {a}|{r['ca_us']:.3f}|{r['previous_unknown_us']:.3f}|{r['unknown_us']:.3f}|{r['known_us']:.3f}|{100*(r['unknown_us']/r['ca_us']-1):+.1f}%|")
    text+=['','## 校准来源与前序协议分开计分','',
        '|芯片及旧采集器|前序|窗口数|旧无 H|当前无 H|当前已知 H|','|---|---|---:|---:|---:|---:|']
    for r in q['archived_by_observed']:
        text.append(f"|{r['chip'][6:11]} · {r['provider']}|"+{'first':'进程首发','expansion':'扩核','covered':'活动核覆盖'}[r['state']]+f"|{r['unknown_us']['n']}|{fmt(r['previous_unknown_us'])}|{fmt(r['unknown_us'])}|{fmt(r['known_us'])}|")
    text+=['','三张上游采集视图包含 199 个有 H 归档窗。旧有序公共价格在全部这些记录里与当前 `estimate(...,resident_aiv=H)` 的最大算差为 7.11×10⁻¹⁵ µs。无 H 的线上调整缓解部分热态高估，在冷态产生了新的低估。窗口来源和方向可从逐形文件复核。','',
        '|配对旧批次 / 数据角色|协议|DT 公共无 H|PR 公共无 H|DT 有序流|PR 有序流|DT 显式专项|PR 显式专项|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    by={(r['batch'],r['chip'],r['role']):r for r in q['pair_groups']}
    for version in ('v0.23','v0.27','v0.28'):
        for role in ('entry','repeat'):
            cells=[fmt(by[version,c,role][k])for k in ('unknown_us','stream_us','explicit_context_us')for c in (DT,PR)]
            text.append('|'+version+' · '+('既有冻结留出' if version=='v0.28' else '阶段开发回放')+'|'+('首遇/扩核' if role=='entry' else '立即复跑')+'|'+'|'.join(cells)+'|')
    text+=['','每个配对旧批次、芯片和协议各十个形状。当输入是按宿主提交顺序的 AIV 历史时，`OperatorStream.advance` 的全部 120 次有序目标结果均与显式 H 相同。v0.24 的额外十形/端中，已知 H 的 DT、PR 分别为 3.56/9.07%、4.65/11.89%，无 H 为 9.07/23.54%、9.03/21.79%。','',
        '## 原来报告的十形与大输入','',
        '|旧报告命名队列|芯片|形数|旧无 H|当前无 H|','|---|---|---:|---:|---:|']
    for r in q['named_summary']:
        text.append(f"|{r['cohort']}|{r['chip'][6:11]}|{r['unknown_us']['shapes']}|{fmt(r['prior_unknown_us'])}|{fmt(r['unknown_us'])}|")
    text+=['','同一旧十形可能给出较大的新误差。例如在 v0.20 十形上，无 H 回放从 DT 10.77% 增至 12.65%，PR 从 13.27% 增至 14.39%，仍在 25% 目标内。筛出其中每个 SKU 的 6 个 T≥1024 原归档完整窗，最大者是 41,943,040 个元素。','',
        '|shape|DT CA / 无H / 首次 µs|PR CA / 无H / 首次 µs|','|---|---:|---:|']
    by_large=defaultdict(dict)
    for r in q['larger']:by_large[tuple(r['shape'])][r['chip']]=r
    for shape,skus in by_large.items():
        text.append(f'|{shape[0]} × {shape[1]}|'+'|'.join(' / '.join(f"{skus[c][k]:.3f}" for k in ('ca_us','us','first_us')) for c in (DT,PR))+'|')
    text+=['','这六个大输入的无 H WAPE / 最大为 DT 11.39/14.66%、PR 13.80/24.07%。整批 B8×32k–256k、D=5120/7168 的全行 RMS 仍只有模型推演。[每核轮次和三端外推](report.md)','',
        '## 哪种复用证据可以作为输入','',
        f"已有 v0.28 的 {len(addr['pairs'])} 个跨两侧实测对中，所有逻辑行块的 x、γ、y、rstd 的 GM 地址和字节数均逐块复核，{sum(v['previous']['sha256']==v['repeat']['sha256'] for v in addr['pairs'])} 对完全相同。宿主在 batch 循环前只创建一个 ACL 应用流，每次 `aclnnRmsNorm` 后同步，目标任务号相邻且设备窗互不重叠。CA 导出的内部 raw_stream_id 随任务序号递增，该字段不能用于断言两次使用同一个内部硬件队列。实验只验证这个实际调度协议，没有新缓冲区、覆盖后重用或跨流的对应对照。[逐块地址摘要和任务身份](stream-addresses.json)",'',
        '系统拥有实际张量存储与派发记录时，可以在调用方比较设备、应用流、前一条成功提交的 kernel 合同、dtype、形状和缓冲区身份，再显式走现有专项接口。只有形状相同或只知道 AIV 已覆盖时，直接使用普通流。新配对中 A≤4 的同形首遇和复跑存在不可消除的歧义：即使逐形事后选取最佳点，两态同时 WAPE 的理想下界也达到 DT 37.21%、PR 36.93%。这里的 gamma/向量/写回诊断也适用于解释精度，两种合法的估时在不同域各有误差。','',
        '```python','from operator_model import rmsnorm_pipeline','',
        '# 已核对上一条成功提交的调用与本次具有相同合同及存储布局',
        'estimate = rmsnorm_pipeline([539, 4000], chip="Ascend950PR_957b", dtype="bf16",',
        '                            resident_aiv=54, immediate_repeat=True, diagnostics=True)',
        '# estimate["us"] = 4.839484；同序 CA 为 4.880 µs',
        '```','',
        '两个 UB 块的确切复跑在三批已观测形状中均改善，但每端只有一个 v0.28 独立形属于这个分组。其他单块样本上可能变差，当前证据不足以凭此新建自动路由。各模型既有价格均保留。','',
        '[逐形公共入口回放、来源及误差](stream-replay.json) · [专项公式与流水图](method.md)']
    (OUT/'stream-replay.md').write_text('\n'.join(text).rstrip()+'\n')

if __name__=='__main__':
    if '--report-only' not in sys.argv:
        main()
    markdown()
