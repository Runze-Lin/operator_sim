"""Check exact API equivalence to the pre-CA policy, then score only qualified, unseen windows."""
from math import ceil
from pathlib import Path
from statistics import median
import sys,time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import read,write,now
from development.validation.post24.freeze import sha
from development.validation.post24.report import stats,fmt
from development.validation.post26.freeze import OUT,OLD
from operator_model import OperatorStream,estimate,load_chip
from operator_model.post_repeat import channel_tiles

def registrations():
    previous=read(OLD/'freeze.json');policy=read(OLD/'repeat-policy-freeze.json');ext=read(OUT/'registration.json')
    p={(r['chip'],r['group'],r['i']):r for r in policy['rows']}
    rows=[]
    for r in previous['rows']:
        k=(r['chip'],r['group'],r['i']);q=p.get(k)
        rows.append(dict(**r,version='v0.25',included=q is not None,
                    policy_us=q['policy_us'] if q else (r['us'] if r['mode']=='repeat' else r['old_known_us']),
                    registered_at=policy['utc'] if q else previous['utc']))
    for r in ext['rows']:rows.append(dict(**r,version='v0.26',included=True,policy_us=r['us'],registered_at=ext['utc']))
    return rows,ext

def main():
    pairs,reg=registrations();streams={};checks=[];costs=[];preds={};measurements=[];failed=[];host_costs=[]
    for r in pairs:
        key=(r['version'],r['chip'],r['group'])
        if key not in streams:streams[key]=(OperatorStream(r['chip']),OperatorStream(r['chip'],experimental_post_repeats=True))
        current,trial=streams[key]
        start=time.perf_counter_ns();v=trial.advance('mhc_post',r['shape'],dtype='bf16');wall=(time.perf_counter_ns()-start)/1e6
        pub=current.advance('mhc_post',r['shape'],dtype='bf16');unknown=estimate('mhc_post',r['shape'],chip=current.chip,dtype='bf16')
        deltas=dict(policy=abs(v.latency_us-r['policy_us']),public=abs(pub.latency_us-r['old_known_us']),unknown=abs(unknown.latency_us-r['old_unknown_us']))
        assert max(deltas.values())<1e-10
        changed=v.projection['timing_diagnostics']['model']=='post_token_immediate_repeat'
        assert changed==(r['role']=='repeat' and r['plan']['route']=='post_token')
        if not changed:assert pub==v
        checks.append(dict(sku=r['chip'],group=r['group'],i=r['i'],shape=r['shape'],eligible_repeat=changed,wall_ms=wall,drift_us=deltas))
        preds[key+(r['i'],)]=dict(version=r['version'],chip=r['chip'],group=r['group'],i=r['i'],shape=r['shape'],role=r['role'],state=r['state'],previous_shape=r['previous_shape'],resident=r['prior_aiv'],eligible_repeat=changed,policy_us=v.latency_us,public_us=pub.latency_us,unknown_us=unknown.latency_us,active_aiv=v.active_aiv,registered_at=r['registered_at'],independent=r['included'])
    # Repeat the serving entry, including model selection and telemetry, with setup excluded.
    for k,(base,trial) in streams.items():
        _,sku,g=k
        for _ in range(4):
            trial.reset();base.reset()
            for r in [r for r in pairs if (r['version'],r['chip'],r['group'])==k]:
                start=time.perf_counter_ns();q=trial.advance('mhc_post',r['shape'],dtype='bf16');ms=(time.perf_counter_ns()-start)/1e6
                assert abs(q.latency_us-r['policy_us'])<1e-10
                costs.append(ms)
    for ver,folder in [('v0.25',OLD),('v0.26',OUT)]:
        for f in sorted((folder/'receipts').glob('*.json')):
            o=read(f);host_costs.append(dict(version=ver,chip=o['chip'],group=o['group'],**o['cost']))
            for r in o['rows']:
                key=(ver,r['chip'],r['group'],r['i']);p=preds[key]
                if not p['independent']:continue
                assert o['cost']['started_utc']>p['registered_at'] and p['shape']==r['shape']
                if not r['timing_qualified']:
                    failed.append(dict(sku=r['chip'],group=r['group'],i=r['i'],shape=r['shape'],reason=r.get('reason','missing complete timing')));continue
                s=r['stage'];future=preds[(ver,reg['extrapolated'][0],r['group'],r['i'])]
                geometry=channel_tiles(r['shape'][-1],load_chip(r['chip']));trace_data=r.get('ca_milestones');structure=None
                if trace_data and geometry and s['kernel'].endswith('_6'):
                    observed_reads=[z['bytes'] for z in trace_data['dma_read']]
                    observed_writes=[z['bytes'] for z in trace_data['dma_write']]
                    widths=[geometry['inner']]*(geometry['count']-1)+[geometry['tail']]
                    per_row=[64,16]+[z for d in widths for z in (2*d,8*d)]
                    expected_writes=[8*d for d in widths]
                    rows_observed=len(observed_writes)//geometry['count']
                    matched=observed_reads==[96]+per_row*rows_observed and observed_writes==expected_writes*rows_observed
                    structure=dict(terminal_rows=rows_observed,channel_tiles=geometry,
                            actual_matches_arch35_loop=matched,whole_row_response_resolves_actual_descriptors=matched and geometry['count']==1)
                measurements.append(dict(**p,ca_us=s['ca_us'],kernel=s['kernel'],actual_aiv=s['active_aiv'],trace=s['trace'],trace_sha256=s['trace_sha256'],golden_elements=r['all_output_elements'],modeled_970_us=future['policy_us'],terminal_structure=structure))
    a,b=reg['measured'];columns=('policy_us','public_us','unknown_us');summary={};good={}
    modes=['all','repeat','entry','first','expand','reuse']
    for sku in (a,b):
        rr=[r for r in measurements if r['chip']==sku]
        def selected(r,m):
            return m=='all' or (m in ('repeat','entry') and r['role']==m) or (m in ('first','expand','reuse') and r['role']=='entry' and r['state']==m)
        summary[sku]={m:{c:stats([r for r in rr if selected(r,m)],c) for c in columns} for m in modes}
        good[sku]=len({tuple(r['shape']) for r in rr if r['role']=='repeat' and r['eligible_repeat'] and r['actual_aiv']==r['active_aiv'] and r['kernel'].endswith('_6')})
    indexed={(r['chip'],r['version'],r['group'],r['i']):r for r in measurements}
    paired=[]
    for r in measurements:
        p=indexed.get((b,r['version'],r['group'],r['i']))
        if r['chip']==a and p:paired.append(dict(shape=r['shape'],group=r['group'],i=r['i'],role=r['role'],ca_us=r['ca_us']/p['ca_us'],**{k:r[k]/p[k] for k in columns}))
    ratios={m:{c:stats([r for r in paired if m=='all' or r['role']==m],c) for c in columns} for m in ('all','repeat','entry')}
    passed=all(good[s]>=10 and summary[s]['repeat']['policy_us']['wape']<=20 and summary[s]['repeat']['policy_us']['maximum']<=40
              and summary[s]['repeat']['policy_us']['wape']<summary[s]['repeat']['public_us']['wape'] for s in (a,b))
    costs.sort();bench=dict(n=len(costs),p50_ms=median(costs),p95_ms=costs[ceil(.95*len(costs))-1],max_ms=max(costs),max_first_call_per_registered_query_ms=max(s['wall_ms'] for s in checks))
    write(OUT/'score.json',dict(utc=now(),policy='explicit opt-in, default remains published',independent=measurements,summary=summary,all_core_repeat_shapes=good,unqualified=failed,
         support_threshold_met=passed,ratio_rows=paired,ratios=ratios,query=bench,host_costs=host_costs))
    write(OUT/'api-equivalence.json',dict(replayed_before_or_after_collection='all registered numeric predictions independently frozen before CA; this is a functional equivalence audit, not new measurement',maximum_drift_us=max(max(x['drift_us'].values()) for x in checks),rows=checks))
    md=['# 内置 MhcPost 的同流精确重复','', '| 芯片与前序 | 合格窗 | 可选策略 WAPE / 最大 | 默认 v0.25 WAPE / 最大 | 不传前序 WAPE / 最大 |','|---|---:|---:|---:|---:|']
    for sku in (a,b):
        for m,z in summary[sku].items():
            if z['policy_us']:md.append(f"| {sku} / {m} | {z['policy_us']['n']} | "+' | '.join(fmt(z[c]) for c in columns)+' |')
    md+=['',f"合格的独立重复形状为 950DT {good[a]}、950PR {good[b]}。全核窗口及机制身份同时合格才计入十形目标。共 {len(measurements)} 个合格的重复与非重复调用，全状态 WAPE 和精确重复分别汇报。目标为两端精确重复分别 WAPE≤20%、最大≤40%、WAPE 优于默认，且各至少 10 个不同新形。当前判定为 **"+('达到门槛**。' if passed else '尚未通过门槛**。'),'',
         '公开入口提供显式选择的实验模式。每个 OperatorStream 绑定一个 rank、设备和有序流，成功模拟一条实际派发后使用 advance；探索候选价格使用 preview。复用核数、核函数实现、dtype、残差与布局合同及完整 shape 均匹配时，且实际采用 token 分核，连续两个相同调用的后一个使用冻结的逐行模型。缺失具体 FP32 吞吐、通道分核、普通热态、首次和未知前序都沿用已有点估计。裸 estimate 调用保持无状态。','',
         '示例中两次调用传入相同 BF16 内置 Post。preview 多次不会把首次预测变成重复预测。未知的外部成功派发用 forget_history() 使旧合同失效。','',
         '```python','from operator_model import OperatorStream','','stream = OperatorStream("Ascend950DT_9582", experimental_post_repeats=True)','first = stream.advance("mhc_post", [67, 4, 384], dtype="bf16")','repeat = stream.preview("mhc_post", [67, 4, 384], dtype="bf16")','print(first.latency_us, repeat.latency_us, repeat.projection["stages_us"])','```','',
         f"公共 API 已与 DT、PR、970DT 的 {len(checks)} 条冻结预测逐条比对，最大差 {max(max(x['drift_us'].values()) for x in checks):.3g} µs。{bench['n']} 次实际 advance（含调度选择、工作量与速率返回）墙钟 p50/p95/最大 {bench['p50_ms']:.3f}/{bench['p95_ms']:.3f}/{bench['max_ms']:.3f} ms。970DT 全部为外推。",'',
         '## 同集跨芯片比值','', '| 前序 | 同组双端窗口对 | 可选策略的加速比 WAPE / 最大 | 默认加速比 WAPE / 最大 |','|---|---:|---:|---:|']
    for m,z in ratios.items():
        if z['policy_us']:md.append(f"| {m} | {z['policy_us']['n']} | {fmt(z['policy_us'])} | {fmt(z['public_us'])} |")
    md+=['','比值均为 DT 耗时 / PR 耗时。未知前序的点估计与默认一致。同形首次和紧邻重复可能出现不同设备时间；小 token、宽通道的 channel kernel 走单独的分片路线。[小宽组历史配对、无法消除的误差及可观测状态](channel-observability.md)','',
         '## 所有合格新窗口','', '| 芯片 | T×4×D | 前序 | CA µs | 可选 µs | 默认 µs | 有符号误差 | 核数预测/CA | 970DT 外推 µs |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in measurements:md.append(f"| {r['chip']} | {'×'.join(map(str,r['shape']))} | {r['state']} / {r['role']} | {r['ca_us']:.3f} | {r['policy_us']:.3f} | {r['public_us']:.3f} | {100*(r['policy_us']/r['ca_us']-1):+.2f}% | {r['active_aiv']}/{r['actual_aiv']} | {r['modeled_970_us']:.3f} |")
    md+=['','所有纳入样本来自同一流上的 CANN 9.2.1 内置 ops_transformer/mhc_post_apt 实际 ELF；逐核需有 END 与终止标记且匹配官方时长，写回字节及完整 FP32 参考舍入 BF16 金标逐元素检查。首批 5 形取自 v0.25 的 width/many 组；此前 boundary、steady 组仅作开发观察。补充 5 形和两种备用形的所有价格均在测量前登记。备用条件只看可评分形状覆盖数。','',
         '[首批独立五形的策略冻结](../../v0.25/post/repeat-policy-freeze.json) · [第二批注册和预算](registration.json) · [逐形误差与加速比](score.json) · [三芯片公共 API 数值与查询开销](api-equivalence.json)']
    if (OUT/'split-width-repeat.json').exists():
        m=read(OUT/'split-width-repeat.json');z=m['tiles'];al=100*(m['model']['us']/m['CA_us']-1)
        md+=['','## 宽度 5120 的独立流水对照','',
             f"950DT 的 [65,4,5120]，{m['active']} 个 AIV。预测 {m['model']['us']:.3f} µs，实测 {m['CA_us']:.3f} µs（{al:+.2f}%）。末核按实际输出 GM 地址负责 token "+'、'.join(str(k['token']) for k in m['items'])+"。每个 token 的写回分为两个 2560 通道片，每片四次各 5120 B 的 burst 之间相隔 10240 B。全张量 520 个 burst 的覆盖经地址排序检查。",'',
             r'同一套 arch35 RegBase 参考 host 按 $U_{req}=320+52d_{inner}<U_{usable}$ 选择最少通道片。两组 FP32 系数各按 8 元素对齐并双缓冲，合计固定 320 B；每列 hOut 双缓冲需 4 B、残差流需 16 B、输出和中间量需 32 B。$d_{inner}$ 按 16 个 BF16 元素对齐且至少 32。取 253952 B 时 4864 列能整行放入，4880 列开始需要两片；5120 列分为两片各 2560。安装包的设备源码在 token 内先读两笔系数，再在 dOuter 循环里逐片读 hOut/残差、计算和写回。该处 trace 与参考 host 判据一致。','',
             '可选预测沿用预登记的整行价格。诊断中的 channel_tiles 指出核内有两片，schedule_granularity=whole_row_aggregate 表示当前耗时公式仍合并片间响应和发起间隔。各窗口同时报告数据搬运结构是否吻合。不同设备的可用 UB 可能变化，不能把单次整窗的接近程度当作未来分片调度精度。','',
             '![上栏整行资源价格，下栏地址核验的两片实际 DMA](split-width-repeat.png)','',
             '[图的逐笔时间与归属](split-width-repeat.json) · [参考库 arch35 host 的 UB 选择器](https://gitcode.com/cann/ops-transformer/blob/14d022b1a4c3df263bf697534e3c25c7c992ff5d/mhc/mhc_post/op_host/op_tiling/arch35/mhc_post_tiling_regbase_arch35.cpp)']
    if (OUT/'wide-stages.md').exists():
        md+=['','[宽度 5120 在两端的阶段复核](wide-stages.md)：常规复用的 MTE2 互斥计价比 CA 高约 2.51 µs / 2.63 µs，吃掉大部分总偏差。']
    if (OUT/'large-api.json').exists():
        large=read(OUT/'large-api.json')['rows'];md+=['','## 极多轮时的 API 开销','',
        f"对 B=8、S=32k/128k/256k、D=5120/7168 的整段 BF16 输入实调 950DT、950PR、970DT 接口。每个最忙 AIV 最多处理 {max(r['rows_per_longest_aiv'] for r in large):,} 行，18 个重复查询最大墙钟 {max(r['query_ms'] for r in large):.3f} ms。这里验证闭式模型的查询成本。这些形状的设备耗时在三颗芯片上均为外推；实际 vLLM DeepSeek 使用另一套 HcPost 绑定。",'', '[逐形工作量、耗时外推与查询墙钟](large-api.json) · [上游实际 HcPost 绑定](../../../docs/examples/hc_post_vllm.md)']
    if (OUT/'warm-schedule-freeze.json').exists():
        from development.validation.post26.warm import evaluate as evaluate_warm
        evaluate_warm()
        md+=['','[普通热态改变 shape 的预登记机制试验](warm-schedule.md)']
    (OUT/'report.md').write_text('\n'.join(md).rstrip()+'\n')
    print(len(measurements),'eligible repeat shapes',good,'pass',passed,flush=True)
if __name__=='__main__':main()
