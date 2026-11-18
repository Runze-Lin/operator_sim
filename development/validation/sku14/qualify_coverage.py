"""Describe independent shape coverage without altering the frozen estimator or rows."""
from campaign import OUT, load, write
from math import fsum


def describe(rows):
    def bytes_(r):return r['work']['logical_read_bytes']+r['work']['logical_write_bytes']
    def mean_abs(rows, col): return fsum(abs(r[col]-r['pr_ca_us']) for r in rows)/fsum(r['pr_ca_us'] for r in rows)
    out={}
    for op in ('quant','matmul','fa'):
        cuts=sorted((r for r in rows if r['op']==op),key=lambda r:(bytes_(r),r['id']))
        if len(cuts)!=10: raise ValueError('Need complete independently qualified holdout')
        shape_span=lambda r: {'id':r['id'],'shape':r['shape'],'logical_bytes':bytes_(r),'cube_ops':r['work']['cube_ops'],'vec_ops':r['work']['vec_ops'],'pr_ca_us':r['pr_ca_us'],'pr_us':r['pr_us']}
        def group(rs):return dict(n=len(rs),min_bytes=bytes_(rs[0]),max_bytes=bytes_(rs[-1]),pr_wape=mean_abs(rs,'pr_us'),reuse_dt_wape=mean_abs(rs,'dt_us'),ids=[r['id'] for r in rs])
        out[op]=dict(n=len(cuts),smallest=shape_span(cuts[0]),largest=shape_span(cuts[-1]),small_half=group(cuts[:5]),large_half=group(cuts[5:]),unique_pr_kernels=sorted({r['kernel'] for r in cuts}),active_match=sum(r['pr_aic']==r['active_aic'] and r['pr_aiv']==r['active_aiv'] for r in cuts))
    return out

if __name__=='__main__':
    rows=load(OUT/'holdout.json');data=describe(rows);write(OUT/'coverage.json',data)
    lines=['# 独立集的工作量覆盖','','每个算子都在实际进入 PR 误差表的十个不同 shape 内，按逻辑读+写字节数升序拆为较小、较大两半（各 5 个）。逻辑字节不等同于物理 DRAM 流量；此表只描述采样范围，不能证实数据量继续增大后一定线性。','','| 算子 | 最小 → 最大逻辑 IO | 较小组：PR / 复用 DT WAPE | 较大组：PR / 复用 DT WAPE | AIC 和 AIV 均命中 |','|---|---|---|---|---|']
    for op,g in data.items():
        a,b=g['smallest'],g['largest'];x,y=g['small_half'],g['large_half']; lines.append(f"| {op} | `{a['shape']}` {a['logical_bytes']:,} B → `{b['shape']}` {b['logical_bytes']:,} B | {x['pr_wape']:.2%} / {x['reuse_dt_wape']:.2%} | {y['pr_wape']:.2%} / {y['reuse_dt_wape']:.2%} | {g['active_match']}/10 |")
    lines+=['','## 保持其余维度不变的数据量对照','', '| 两个输入（其余属性相同） | 有用工作量倍数 | PR CA 时间倍数 | 冻结预测时间倍数 |','|---|---:|---:|---:|']
    index={r['id']:r for r in rows}
    for lo,hi in [('q3_test_one_ub','q2_test_ub_loop'),('matmul_256_512_256','matmul_512_512_256'),('f2_cal_q128','f2_test_q129')]:
        if lo not in index or hi not in index:continue
        a,b=index[lo],index[hi]; op=a['op'];w='vec_ops' if op=='quant' else 'cube_ops'
        lines.append(f"| {op} `{a['shape']}` → `{b['shape']}` | {b['work'][w]/a['work'][w]:.3f}× | {b['pr_ca_us']/a['pr_ca_us']:.3f}× | {b['pr_us']/a['pr_us']:.3f}× |")
    lines+=['','quant 使用工具记录的向量有用工作量；matmul 和 FA 使用有用 Cube 操作数。跨行分块和核数变化均可能使设备时间偏离工作量倍数。不能从这两个测量点确定跨分块的渐近斜率。']
    lines+=['','数据角色：此处 PR shape 均在模型冻结后首次采集；历史 DT 结果仅用于配对分解，不能计为 DT 的新独立验证。每个点使用其冻结前的 dtype/attrs、相同输入和对应历史 runner。库与开源家族的对应不能证明某个开源提交与二进制逐行一致。']
    (OUT/'coverage.md').write_text('\n'.join(lines)+'\n')
