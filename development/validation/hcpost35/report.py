"""Plot the frozen cohort schedule against the critical lane's actual per-row output descriptors."""
from collections import Counter
import json
from pathlib import Path
from statistics import median
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.35/hcpost'
sys.path.insert(0,str(ROOT))
from development.validation.rms_ordered_verify.verify import events

def rd(p):return json.loads(p.read_bytes())

def rows(r):
    core=r['critical_core']
    stage=(ROOT/r['global_trace_path']).parent
    lane=r['all_shards'][core]
    first=lane['shard_ticks'][0]-lane['global_ticks'][0]+r['global_ticks'][0]
    writes=[]
    for e in events(stage/core/'trace.json'):
        if e['name']!='MOV_SRC_TO_DST_ALIGNv2':continue
        d=json.loads(e['args']['detail'])
        if d['src_mem']=='UB' and d['dst_mem']=='OUT' and not any(d.get(k) for k in ('is_wait','is_sync','is_duration_excluded')):
            writes.append(dict(offset=d['dst_addr'],unit_B=d['processed_bytes'],start_us=(e['ts']*1650-first)/1650,
                               end_us=((e['ts']+e['dur'])*1650-first)/1650))
    base=min(w['offset'] for w in writes)
    grouped={}
    row_bytes=r['shape'][2]*r['shape'][3]*2
    for w in writes:
        index=(w.pop('offset')-base)//row_bytes
        grouped.setdefault(index+1,[]).append(w)
    assert sorted(grouped)==list(range(1,len(grouped)+1))
    assert all(sorted(w['unit_B'] for w in group)==[8192,32768] for group in grouped.values())
    return [grouped[k] for k in sorted(grouped)]

def plot(r,actual,model,model_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,(a,b)=plt.subplots(2,1,figsize=(10,7),sharex=True)
    y=lambda k: model_rows+1-k
    first2=model['first_two_rows_ticks']/1650;ii=model['warm_ii_ticks']/1650
    a.broken_barh([(0,first2)],(y(2)-.42,1.84),facecolors='#bbc7cd')
    for k in range(3,model_rows+1):
        a.broken_barh([(first2+(k-3)*ii,ii)],(y(k)-.42,.84),facecolors='#37768e')
    for k,ws in enumerate(actual,1):
        for w in ws:
            shade='#1c784c' if w['unit_B']==32768 else '#dc8c36'
            off=-.28 if w['unit_B']==32768 else .01
            b.broken_barh([(w['start_us'],w['end_us']-w['start_us'])],(y(k)+off,.29),facecolors=shade)
    for ax,endpoint in [(a,r['candidate_us']),(b,r['ca_us'])]:
        ax.axvline(endpoint,color='#303642',ls='--',lw=1);ax.grid(axis='x',color='#d5dde3');ax.set_axisbelow(True)
        ax.set_ylim(.2,model_rows+.8);ticks=sorted(set([1,2,3,6,9,12,15,len(actual),model_rows]));ax.set_yticks([y(i) for i in ticks],labels=[f'row {i}' for i in ticks],fontsize=8)
        ax.spines[['top','right']].set_visible(False)
    a.set_title(f'Frozen DT-calibrated schedule · {model_rows} rows per slowest cohort   {r["candidate_us"]:.3f} µs',loc='left',fontweight='bold',fontsize=11)
    b.set_title(f'CA · {r["critical_core"]} · {len(actual)} actual rows   {r["ca_us"]:.3f} µs',loc='left',fontweight='bold',fontsize=11)
    a.text(.01,.06,'Gray: first 2 rows jointly priced   Blue: steady row service interval',transform=a.transAxes,fontsize=8,color='#344550')
    b.text(.01,.09,'Green: 4 × 4096 BF16 values   Orange: 4 × 1024 BF16 values',transform=b.transAxes,fontsize=8,color='#344550')
    b.set_xlabel('Time from earliest instruction across the full launch (µs)');fig.suptitle('950PR  |  vLLM HcPost  |  1024 tokens × 4 channels × hidden 5120',fontsize=14,fontweight='bold',y=.97)
    fig.subplots_adjust(left=.09,right=.97,bottom=.13,top=.88,hspace=.42)
    fig.text(.09,.038,'Upper bars show modeled row completion budgets. Lower bars show real MTE3 descriptors, including dependency waits. Vertical lines mark full windows.',fontsize=7.6,color='#44545a')
    fig.savefig(OUT/'pr-rows.png',dpi=150,bbox_inches='tight');plt.close(fig)

def main():
    r=rd(OUT/'pr-full.json')
    calibr=rd(ROOT/'results/v0.32/post/hc-pr-freeze.json')
    p,=[c for c in calibr['results'] if c['sku']==r['chip']]
    co=max(p['cohorts'],key=lambda c:c['local_ticks'])
    actual=rows(r)
    observed=[max(a['end_us'] for a in g) for g in actual]
    slope=median(b-a for a,b in zip(observed[1:],observed[2:]))
    row_bytes=r['shape'][2]*r['shape'][3]*2
    histogram=Counter(c['output_bytes']//row_bytes for c in r['all_shards'].values())
    assert sum(n*c for n,c in histogram.items())==r['shape'][1] and all(v['output_bytes']%row_bytes==0 for v in r['all_shards'].values())
    model_rows=co['rows']
    assert model_rows>=len(actual)
    delta_first=observed[1]-co['first_two_rows_ticks']/1650
    delta_steady=observed[-1]-observed[1]-(len(actual)-2)*co['warm_ii_ticks']/1650
    drain=r['ca_us']-observed[-1]
    assert abs(delta_first+delta_steady+drain-(r['ca_us']-r['candidate_us']))<1e-5
    plot(r,actual,co,model_rows)
    data=dict(chip=r['chip'],shape=r['shape'],model_cohort=co,model_us=r['candidate_us'],all_core_ca_us=r['ca_us'],critical_core=r['critical_core'],
              observed_row_counts=histogram,write_descriptors=actual,causal_note='Model prices first two rows jointly. The observed windows are descriptors on the actual critical lane. A descriptor can contain dependencies.',
              median_steady_write_completion_interval_us=slope,delta_us=dict(first_2=delta_first,steady=delta_steady,terminal_drain=drain))
    (OUT/'pr-rows.json').write_text(json.dumps(data,indent=2)+'\n')
    wf=r['acquisition_cost'];tot=wf['seconds']/60;d=rd(ROOT/'results/v0.33/hcpost/partial.json')
    txt=f'''# vLLM HcPost 的 1024-token PR 全核退休与写回

| 芯片 / 证据 | 冻结公开模型 µs | DT 读响应候选 µs | CA 全窗 µs | 公开 / 候选有符号误差 |
|---|---:|---:|---:|---|
| 950PR，同次完整内核导出 | {r['public_baseline_us']:.3f} | {r['candidate_us']:.3f} | **{r['ca_us']:.3f}** | {r['public_signed_pct']:+.2f}% / {r['candidate_signed_pct']:+.2f}% |
| 950DT，3 个核缺终止指令 | {d['public_us']:.3f} | {d['candidate_us']:.3f} | N/A，至少 {d['conservative_lower_us']:.3f} | 公开 N/A；候选至少低估 {d['candidate_absolute_error_at_least_pct']:.2f}% |
| 970DT，外推 | 24.696 | 20.899 | N/A | N/A |

输入 BF16 激活，每轮 1,024 个 token、4 个连接分支，hidden=5,120。实际调用 vLLM Ascend 的 aclnnHcPost，经 arch35 适配库进入 {r['kernel']}。逐元素核对了 {r['physical_outputs']:,} 个 BF16 输出，全部与 RNE 金标一致。源文件已按原冻结登记核对，捕获的设备 ELF 的 .text 指纹与安装组件相同。该算例的公开模型、DT 校准候选均按冻结值引用。候选冻结时 PR 已在运行，结果尚未揭盲；这是同一大 shape 的跨芯片在途盲评。

安装版 CA 在真实 950PR、LPDDR5 配置上运行，发布了唯一一次 kernel 的独立目录。原顶层采集以 raw cap 退出，墙钟 {tot:.2f} 分钟，峰值磁盘 {wf['peak_bytes']/2**30:.2f} GiB，原登记上限 24 GiB。退出发生在打印模型停止、全数值验证和官方逐核时间表以后。当时每次 kernel 的目录已有全部 56 个核，顶层重复导出仅有 34 个核的 trace 文件。保留原始失败回执，另从同次 kernel 的数据核验设备时间。此路径没有原生 SQLite 指令库。逐核 CSV 给出 END 次数，官方控制台只给两位小数的核内耗时，均缺各核绝对首发与退休时钟。逐核设备窗取自同次完整导出；本次新增 CA 采集为零。

全部 56 个 AIV 的逐核时间线、全片 trace、二进制画图包中的时间线与指令统计、每核的指令 CSV 逐项交叉。每个核都有恰好一个 END；END 可早于已发起的异步指令最后退休。用全局最早首发和最晚退休求设备窗，得到 ({r['global_ticks'][1]:,}−{r['global_ticks'][0]:,}) / 1,650 = {r['ca_us']:.5f} µs。全核 GM 写出直接验证了 {r['store_runs']:,} 个地址区间，连续无洞、无重叠，共 {r['output_payload_bytes']:,} B，恰好覆盖所有输出。官方输出的逐核时长表按 0.01 µs 的打印精度与时间线相符。完整离线复核用时 {r['offline_validation_wall_s']:.1f} 秒。

## 算例的行流水

将 1,024 个 token 按 56 AIV 分配，{histogram[19]} 个核各执行 19 行，{histogram[18]} 个核各执行 18 行，行数由实际输出写回字节独立计出。每行 4 个分支都分为 4,096+1,024 个 BF16 元素的两段。模型对慢组联合估算前两行 {co['first_two_rows_ticks']:.1f} 周期，以后每行间隔 {co['warm_ii_ticks']:.1f} 周期。代入 {model_rows} 行得到 ({co['first_two_rows_ticks']:.1f} + ({model_rows}−2)×{co['warm_ii_ticks']:.1f})/1,650 = {co['local_ticks']/1650:.3f} µs。全片共享资源下界 {p['physical_floor_us']:.3f} µs；取两者最大值。

实测最后退休的是 {r['critical_core']}，共完成 {len(actual)} 行。第二行最后一笔输出描述符在 {observed[1]:.3f} µs 退休，余下各行输出完成间隔中位数 {slope:.3f} µs；模型慢组的稳态服务间隔为 {co['warm_ii_ticks']/1650:.3f} µs。总残差 {r['ca_us']-r['candidate_us']:.3f} µs 可按实际行边界直接拆分，前两行贡献 {delta_first:.3f} µs，随后 17 行累计增量 {delta_steady:.3f} µs，最后一笔输出退休至全核末端另有 {drain:.3f} µs。约 {100*delta_first/(r['ca_us']-r['candidate_us']):.0f}% 的差值在第二行写完前已出现。描述符包含依赖等待，其中的 HBM、SIMD 和队列等待仍未逐项分离。DT 大形仍只有下界，不能给出正式 DT/PR 加速比。

![HcPost 每行的冻结模型预算和 PR 临界核实际写退休](pr-rows.png)

[完整核验与逐核端点](pr-full.json) · [对应行事件](pr-rows.json) · [原预算失败回执](../../v0.31/post/vllm_hc_post-Ascend950PR_957b.json) · [950DT 的下界](../../v0.33/hcpost/partial.json) · [预测冻结及参数来源](../../v0.32/post/hc-pr-freeze.json)
'''
    (OUT/'report.md').write_text(txt)
if __name__=='__main__':main()
