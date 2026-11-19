"""Render the current resource walk-through against a reused double-buffer trace."""
from pathlib import Path
import sys
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate,load_chip

chips={'dt':'Ascend950DT_9582','pr':'Ascend950PR_957b','future':'Ascend970DT'}
docs=ROOT/'docs/examples'
archive=ROOT/'results/v0.18/sku'
shape=[448,6592]
pair=next(r for r in json.loads((archive/'pairs.json').read_text()) if r['op']=='quant' and r['shape']==shape)
evidence=[r for r in json.loads((archive/'trace-evidence.json').read_text())['rows'] if r['op']=='quant' and r['shape']==shape]
samples=[r for r in json.loads((ROOT/'results/v0.19/quant-dma-samples.json').read_text()) if r['shape']==shape]


def run(target,shape):
    return estimate('quant',shape,dtype='bf16',chip=target,timing_mode='resource')


def main():
    page=['# DynamicMXQuant 算例 · 两个 UB tile 的预取与写回','',
    '输入取单卡 BF16 [448,6592]，末轴 block size 32。调用 CANN 9.2.1 ops-nn 的 aclnnDynamicMxQuant，kernel 为 DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv。两个 950 端的 E4M3FN 数据和 E8M0 scale 均通过逐字节检查。该组旧 trace 已用于阶段校准，角色为开发回放。','',
    '```python','from operator_model import estimate',
    "r = estimate('quant', [448,6592], dtype='bf16', chip='Ascend950PR_957b', timing_mode='resource')",
    "print(r.latency_us, r.active_aiv, r.projection['timing_diagnostics'])",'```','',
    '每个元素读 2 B、写 1 B，每行写 2⌈6592/64⌉=206 B 的 scale。全片读 5,906,432 B，逻辑写 3,045,504 B。组最大值归约与取绝对值合计 2RC−R⌈C/32⌉ 个有用标量操作，Cube 为零。scale 编码和 VF 控制另外定价。','',
    '| 平台 | v0.18 µs | v0.19 µs | CA µs | v0.19 误差 | 活跃 AIV 与最大轮数 |',
    '|---|---:|---:|---:|---:|---|']
    results={}
    for side,name in chips.items():
        r=results[side]=run(name,shape)
        ca=pair.get(side+'_ca_us')
        e=f'{r.latency_us/ca-1:+.2%}' if ca else 'N/A'
        ca_s=f'{ca:.4f}' if ca else '无，情景外推'
        page.append(f"| {name} | {pair[side+'_us']:.4f} | {r.latency_us:.4f} | {ca_s} | {e} | {r.active_aiv} × {r.projection['timing_diagnostics']['target_plan']['iterations']} |")
    page+=['','![两颗芯片最晚核上的 CA 描述符时间与模型窗口](figures/quant.png)','',
           '图中横坐标从该次 kernel 的最早事件起算。每条横带是真实在途的 DMA 描述符，竖线为 CA 全核窗口终点。两个读同时在途时，第二个请求已发起，两个读的字节最终共同占用整片带宽。']
    fig,axs=plt.subplots(2,1,figsize=(8.4,4.7),layout='constrained',sharex=True)
    for ax,e in zip(axs,evidence):
        side='dt' if 'DT' in e['chip'] else 'pr'
        rs=next(s for s in samples if s['side']==side)
        c=rs['cores'][e['terminal_core']]
        for queue,y,color in [('read',1,'#187a72'),('write',0,'#c9792a')]:
            for i,(start,end,b) in enumerate(c[queue]):
                ax.broken_barh([(start/1650,(end-start)/1650)],(y+(i%2)*.23,.19),facecolors=color)
        ax.axvline(e['ca_us'],label='CA window',color='#333333')
        ax.axvline(results[side].latency_us,label='v0.19 prediction',linestyle='--',color='#6865ad')
        ax.set_yticks([.3,1.3],['MTE3 writes','MTE2 reads'])
        ax.set_title(f"{e['chip']}   |   {e['terminal_core']}",fontsize=11)
        ax.spines[['top','right']].set_visible(False)
        ax.set_xlim(left=0)
        rd=c['read'][1:]
        page+=['',f"{e['chip']} 实测尾核 {e['terminal_core']} 的首笔有效读为 {rd[0][2]:,} B，发起周期 {rd[0][0]}，完成周期 {rd[0][1]}。"
               + (f"第二笔读在 {rd[1][0]} 已提交，距首笔 {rd[1][0]-rd[0][0]} cycles。" if len(rd)>1 else '')
               +f" trace SHA-256 前 12 位为 {rs['sha256'][:12]}。"]
    axs[0].legend(loc='upper left',fontsize=8,ncol=2)
    axs[1].set_xlabel('Time since earliest core event (µs)')
    fig.savefig(docs/'figures/quant.png',dpi=140)
    plt.close(fig)
    for side,name in chips.items():
        r=results[side];d=r.projection['timing_diagnostics'];t=d['target_plan'];a=d['first_tile_cycles'];clock=load_chip(name).frequency_ghz*1000
        page+=['',f'## {name} 的计算','',
              f"UB 容量 {load_chip(name).details['ub_bytes']:,} B，分出 {r.active_aiv} 个 AIV。最忙核承担 {t['rows_per_core']}×{t['cols_per_core']}，UB tile 是 {t['tile'][0]}×{t['tile'][1]}，共 {t['iterations']} 轮。每轮分别按实际行列数裁边。",
              '',f"首块读 {a['read_bytes']:,} B，写 {a['write_bytes']:,} B，其中 scale {a['scale_bytes']:,} B。每核本地读响应 {a['local_read_cycles']:.1f} cycles，共享波扣除发起跨度后的读完成约束 {a['shared_read_floor_cycles']:.1f}，取大得 {a['read_latency']:.1f}；下笔读的本地发起间隔为 {a['read_ii']:.1f}。",
              '',f"组数 {a['groups']}，VF 轮数 ⌈组数/8⌉+⌈组数/128⌉={a['loops']}。每轮 {d['effective_vec_ops_per_loop']:.2f} 等效操作，目标每核吞吐 {d['vec_ops_per_cycle_per_core']:.2f} 操作/cycle。加 SIMD 固定控制后需 {a['compute_cycles']:.1f} cycles。读等待和前端准备可重叠，首块从发读到可算为 {a['simd_ready']:.1f} cycles。数据写需 {a['data_write_cycles']:.1f}，scale 写含相邻生产者的等待需 {a['scale_write_cycles']:.1f}，两笔在途后的写窗口为 {a['write']:.1f} cycles。",
              '',f"双槽依赖是输入槽消费后才可再发读、输出槽释放后才可复用；同一 SIMD 上顺序计算。在本次冷读、写回 L2 的情景下，{d['output_retained_bytes']:,.0f} B 的输出可驻留，算子退休前需要的片外输出估计 {d['assumed_offchip_write_bytes']:,.0f} B。整片下界含末次读之后最短的算写，扣除已经过的核间发起跨度后为 {d['chip_floor_after_launch_cycles']:.1f} cycles。",
              '',f"总周期 = 启动 {d['stages_us']['dispatch']*clock:.1f} + 本地双槽流水和共享下界的最大值 {d['stages_us']['ub_pipeline']*clock:.1f} + 排空 {d['stages_us']['drain']*clock:.1f}。除以 {clock:g} cycles/µs 得 **{r.latency_us:.4f} µs**。"]
    ca_s=pair['dt_ca_us']/pair['pr_ca_us'];pred=results['dt'].latency_us/results['pr'].latency_us
    page+=['','## 误差来自哪里','',
          f"该例 DT/PR 实测加速比 {ca_s:.3f}×，v0.19 为 {pred:.3f}×。PR 上全片低带宽开始支配局部流水；v0.18 对同一组预取读同时使用均分 HBM 约束其发起，又要求输出全部写至 HBM 后算子才能退休。旧预测 {pair['pr_us']:.3f} µs，新预测 {results['pr'].latency_us:.3f} µs，CA {pair['pr_ca_us']:.3f} µs。余下约 {abs(results['pr'].latency_us-pair['pr_ca_us']):.2f} µs 涉及跨核读完成离散度、无写回计数器时的 L2 驻留假设和最终到达的 AIV；仅凭 MTE3 时戳无法分摊三者。",
          '', '## 长 prefill 输入','',
          '若每卡 8 个请求各贡献 32,768 个 token，并在一次调用中合批，全卡行数为 262,144。以下示例取 hidden=4096，沿用相同 FP8 合同，整次调用只作资源外推。系统若每步采用 2048-token 预算，应按每个实际 chunk 单独输入。','',
          '| 情景 | AIV | 最忙核轮数 | 输入/逻辑输出 GiB | 预计退休前 HBM 输出 GiB | 预测 µs |','|---|---:|---:|---:|---:|---:|']
    for name in chips.values():
        q=run(name,[262144,4096]);d=q.projection['timing_diagnostics'];p=d['target_plan']
        page.append(f"| {name} | {q.active_aiv} | {p['iterations']} | {d['assumed_offchip_read_bytes']/2**30:.3f} / {d['logical_write_bytes']/2**30:.3f} | {d['assumed_offchip_write_bytes']/2**30:.3f} | {q.latency_us:.2f} |")
    page+=['','[价格推导、可辨识性及校准源](../../results/v0.19/quant-method.md) · [当前独立验证与逐例结果](../error-analysis.md)']
    (docs/'quant.md').write_text('\n'.join(page)+'\n')


if __name__ == '__main__':
    main()
