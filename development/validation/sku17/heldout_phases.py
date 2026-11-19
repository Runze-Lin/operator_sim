"""Post-freeze diagnostics; only read the already-exported critical core, no CA reruns."""
import json
from statistics import median
from campaign import ROOT,OUT,PR,read,write,sha
from operator_model import estimate,load_chip


def quant(row,stage,chip):
    core=max(stage['windows'],key=lambda c:stage['windows'][c][1])
    path=(ROOT/stage['trace']).parent/core/'trace.json'
    reads=[];writes=[]
    for e in read(path)['traceEvents']:
        if e.get('ph')!='X':continue
        if e['name']!='MOV_SRC_TO_DST_ALIGNv2':continue
        detail=json.loads(e['args']['detail'])
        item=(e['ts'],e['dur'],detail['processed_bytes'])
        (reads if detail['instr_type']=='MTE2' else writes).append(item)
    reads.sort();writes.sort()
    # First descriptor transports host tiling data; a real input can also happen to be 160 B.
    if not(reads and reads[0][2]==160 and len(reads)>1 and len(writes)==2*(len(reads)-1)):
        return dict(qualified=False,reason='descriptor pattern differs',core=core)
    first=reads[1];w0,w1=writes[:2]
    d=estimate('quant',row['shape'],chip=chip,timing_mode='resource').projection['timing_diagnostics']
    z=d['first_tile_cycles'];freq=load_chip(chip).frequency_ghz*1000
    return dict(qualified=True,core=core,trace=str(path.relative_to(ROOT)),sha256=sha(path),
        comparable_bytes=(first[2]==z['read_bytes'] and w0[2]+w1[2]==z['write_bytes']),
        observed_payload_B=[first[2],w0[2]+w1[2]],predicted_payload_B=[z['read_bytes'],z['write_bytes']],
        actual_read_us=first[1],actual_write_us=max(w0[0]+w0[1],w1[0]+w1[1])-w0[0],
        model_read_us=z['read_latency']/freq,model_write_us=z['write']/freq,
        actual_tiles=len(reads)-1,model_tiles=d['target_plan']['iterations'],
        writers=d['scale_peak_writers'],shared_read_floor_us=z['shared_read_floor_cycles']/freq,
        local_read_us=z['local_read_cycles']/freq,
        observed_write_gap_us=w1[0]-w0[0],estimated_scale_queue_us=d['scale_queue_cycles']/freq)


def process():
    rows=[r for r in read(OUT/'pairs.json') if r['op'] in ('quant','matmul')]
    collected=[]
    for r in rows:
        stage=r['pr']['stages'][0]
        if r['op']=='quant':
            details=quant(r,stage,PR)
        else:
            z=estimate('matmul',r['shape'],chip=PR).projection['timing_diagnostics'];p=z['target_plan']
            details=dict(waves=p['waves'],original_tiles=p['output_tiles'],tail_split=p['tail_split_mn'],
                    predicted_mmads=z['mmads_total'],observed_mmads=stage['mmad'],
                    exposed_wave_us=z['stages_us']['additional_output_waves'],
                    write_window_us=z['stages_us']['write_tail'],visible_hbm_write_bytes=z['shared_traffic']['hbm_write_bytes_exposed'])
        collected.append(dict(op=r['op'],shape=r['shape'],v16_us=r['old_pr_us'],v17_us=r['pr_us'],ca_us=r['pr_ca_us'],diagnostics=details))
    dest=ROOT/'results/v0.17'
    write(dest/'heldout-phases.json',dict(role='frozen PR holdout; diagnostics computed after capture, no refit',rows=collected))
    lines=['# 独立样本上的 MTE 和跨波诊断','','使用已采集的 950PR trace 与冻结预测。单位 µs。Quant 看最晚结束 AIV 的首个有效 tile；Matmul 核对所有核的 MMAD 指令总数。','',
           '## DynamicMXQuant 首个有效 tile','','读耗时取 payload 描述符自身的开始到完成；另一次 160 B 先导描述符传 tiling 参数。写窗从第一条 payload 写的发起到数据和 scale 两条写均完成。最后结束核有时分到较小的边界块，须先比较字节再解释相位差。','',
           '| [行,列] | 实测 / 预测读字节 | 读 CA / 模型 | 实测 / 预测写字节 | 写 CA / 模型 | 同粒度写者 | 总 CA / v0.16 / v0.17 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in collected:
        if r['op']!='quant':continue
        d=r['diagnostics'];tot=f'{r["ca_us"]:.3f} / {r["v16_us"]:.3f} / {r["v17_us"]:.3f}'
        if not d['qualified']:lines.append(f'| {r["shape"]} | N/A | N/A | N/A | N/A | N/A | {tot} |');continue
        a,p=d['observed_payload_B'],d['predicted_payload_B']
        lines.append(f'| {r["shape"]} | {a[0]} / {p[0]} | {d["actual_read_us"]:.3f} / {d["model_read_us"]:.3f} | {a[1]} / {p[1]} | {d["actual_write_us"]:.3f} / {d["model_write_us"]:.3f} | {d["writers"]} | {tot} |')
    quant_rows=[r['diagnostics'] for r in collected if r['op']=='quant' and r['diagnostics'].get('comparable_bytes')]
    if quant_rows:
        read_bias=median(d['model_read_us']-d['actual_read_us'] for d in quant_rows)
        write_bias=median(d['model_write_us']-d['actual_write_us'] for d in quant_rows)
        lines+=['',f'在首块字节吻合的 {len(quant_rows)} 个样例，读、写有符号相位偏差中位数分别是 {read_bias:+.3f} 与 {write_bias:+.3f} µs。它们是首块完成窗偏差；多 tile 上的发起间隔与 UB 重叠另计，不能直接乘以迭代数推算整算子误差。']
    for large,label in [(False,'输入低于 1 MiB'),(True,'输入至少 1 MiB')]:
        group=[r for r in collected if r['op']=='quant' and (2*r['shape'][0]*r['shape'][1]>=2**20)==large]
        if group:
            w=lambda col:sum(abs(r[col]-r['ca_us']) for r in group)/sum(r['ca_us'] for r in group)
            lines+=['',f'{label}：{len(group)} 例。PR 总时延 WAPE，v0.16 为 {w("v16_us"):.2%}，v0.17 为 {w("v17_us"):.2%}。1 MiB 用于描述数据量，模型不使用这一阈值选路。']
    lines+=['','## MatMulV3 整波与末波','','| [M,N,K] | 波数 | 父块数 | 最后 M/N 切分 | 全核 MMAD 预测 / CA | 增量波暴露 µs | 总 CA / v0.16 / v0.17 µs |','|---|---:|---:|---|---:|---:|---:|']
    for r in collected:
        if r['op']!='matmul':continue
        d=r['diagnostics']
        lines.append(f'| {r["shape"]} | {d["waves"]} | {d["original_tiles"]} | {d["tail_split"]} | {d["predicted_mmads"]} / {d["observed_mmads"]} | {d["exposed_wave_us"]:.4f} | {r["ca_us"]:.3f} / {r["v16_us"]:.3f} / {r["v17_us"]:.3f} |')
    for single,label in [(True,'单波'),(False,'多波')]:
        group=[r for r in collected if r['op']=='matmul' and (r['diagnostics']['waves']==1)==single]
        if not group:continue
        w=lambda v:sum(abs(r[v]-r['ca_us']) for r in group)/sum(r['ca_us'] for r in group)
        matched=sum(r['diagnostics']['observed_mmads']==r['diagnostics']['predicted_mmads'] for r in group)
        lines+=['',f'{label} {len(group)} 例，v0.16 / v0.17 的时延 WAPE 为 {w("v16_us"):.2%} / {w("v17_us"):.2%}。{matched}/{len(group)} 例全核 MMAD 总数吻合。条数吻合仍允许不同的启动、服务和 drain 时间。']
    lines+=['','[逐例原始数据及来源摘要](heldout-phases.json) · [大 Matmul 的两芯片逐核 MMAD 尺寸直方图](sku/campaign-mmad.json) · [完整误差表](../../docs/error-analysis.md)']
    (dest/'heldout-phases.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    process()
