"""Post-freeze comparison. Read qualified PR timelines only; never launch CA."""
from pathlib import Path
import hashlib
import json
import sys
from collections import Counter,defaultdict
from itertools import product

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.15'
sys.path.insert(0,str(ROOT))
from operator_model.target_tiling import matmul_plan as legacy_tile


def partition(n,b):
    full,tail=divmod(n,b)
    return [(v,c) for v,c in ((b,full),(tail,1)) if v and c]


def extract(r):
    plan=r['projection']['target_plan'];assert plan['waves']==1
    d=r['projection']['timing_diagnostics'];shape=r['shape'];tile=plan['tile']
    a,b,k=(partition(n,t) for n,t in zip(shape,tile))
    old=legacy_tile(shape,28)  # v0.14 used the PR capacity and doubled the output tile dimensions.
    oldpart=[partition(n,t) for n,t in zip(shape,old['tile'])]
    oldhist={(x,y,z):i*j*h for (x,i),(y,j),(z,h) in product(*oldpart)}
    predicted={(x,y,z):i*j*h for (x,i),(y,j),(z,h) in product(a,b,k)}
    percore=Counter()
    for (x,i),(y,j) in product(a,b):percore[2*x*y*shape[2]]+=i*j
    path=ROOT/r['pr']['trace'];data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
    assert digest==r['pr']['trace_sha256']
    es=[e for e in json.loads(data)['traceEvents'] if e['ph']=='X']
    end=lambda e:e['ts']+e['dur']
    origin=min(e['ts'] for e in es)
    critical=max(es,key=end)['pid'];last=[e for e in es if e['pid']==critical]
    mm=[e for e in last if e['name']=='MMAD'];dr=[e for e in last if e['name']=='MOV_OUT_TO_L1_MULTI_ND2NZ']
    pair=sorted(dr,key=lambda e:e['ts'])[:2];assert len(pair)==2
    points=[origin,pair[0]['ts'],max(map(end,pair)),min(e['ts'] for e in mm),max(map(end,mm)),max(map(end,last))]
    keys=('prefix','first_pair','l0_lead','mmad_window','drain')
    ca=dict(zip(keys,(b-a for a,b in zip(points,points[1:]))))
    expected=dict(zip(keys,[d['prefix_us'],d['first_pair_us'],d['l0_us'],d['page_intervals_us']+d['last_compute_us'],d['drain_us']]))
    assert min(ca.values())>=-.000001 and abs(sum(ca.values())-r['pr_ca_us'])<1e-5
    assert abs(sum(expected.values())-r['pr_us'])<1e-6
    by=defaultdict(int);obs=Counter();counts=Counter()
    for e in es:
        if e['name']!='MMAD':continue
        args=json.loads(e['args']['detail']);x,y,z=[args[n] for n in ('m','n','k')]
        obs[(x,y,z)]+=1;by[e['pid']]+=2*x*y*z;counts[e['pid']]+=1
    stringkeys=lambda hist:{'x'.join(map(str,key)):count for key,count in sorted(hist.items())}
    return dict(id=r['id'],shape=shape,trace=r['pr']['trace'],trace_sha256=digest,kernel=r['kernel'],
                tile=tile,ca_us=r['pr_ca_us'],dt_ca_us=r['dt_ca_us'],pr_v15_us=r['pr_us'],
                pr_v14_us=r['prior_pr_us'],reuse_dt_us=r['dt_us'],active_aic=[r['pr_aic'],r['active_aic']],
                v14_structural_tile=old['tile'],v14_structural_aic=old['active_aic'],v14_mmad_hist=stringkeys(oldhist),
                predicted_mmad_hist=stringkeys(predicted),observed_mmad_hist=stringkeys(obs),
                mmads_per_core_hist=sorted(Counter(counts.values()).items()),
                predicted_ops_per_core_hist=sorted(percore.items()),observed_ops_per_core_hist=sorted(Counter(by.values()).items()),
                critical_core=critical,critical_phase_us=ca,predicted_phase_us=expected)


def main():
    rs=[r for r in json.loads((OUT/'sku/holdout.json').read_text()) if r['op']=='matmul']
    assert len(rs)==len({tuple(r['shape']) for r in rs})==10, 'wait for exactly ten qualified unique PR shapes'
    result=[extract(r) for r in rs]
    (OUT/'matmul-independent.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'matmul-independent.md').write_text(markdown(result))

def markdown(rs):
    keys=['pr_v15_us','pr_v14_us','reuse_dt_us'];den=sum(r['ca_us'] for r in rs)
    metrics={k:100*sum(abs(r[k]-r['ca_us']) for r in rs)/den for k in keys}
    exact=sum(r['predicted_mmad_hist']==r['observed_mmad_hist'] for r in rs)
    loads=sum(r['predicted_ops_per_core_hist']==r['observed_ops_per_core_hist'] for r in rs)
    cores=sum(a==b for a,b in (r['active_aic'] for r in rs))
    lines=['# Matmul：10 个新 PR shape 的冻结后诊断','',f'WAPE：v0.15 **{metrics[keys[0]]:.2f}%**，v0.14 {metrics[keys[1]]:.2f}%，直接复用 DT {metrics[keys[2]]:.2f}%。活跃 AIC 命中 {cores}/10；全部 MMAD 的 (m,n,k) 及次数完全一致 {exact}/10；逐核计算量的分布完全一致 {loads}/10。后两项比核数严格：即使总 FLOP 相同，也会发现尾块大小或 K 分片次数错了。均是 PR 完整 CA，DT 仅作对照。','',
    '| PR [M,N,K] | AIC 预/实 | 实测 MMAD 总数 | PR CA µs | v0.15 µs | v0.14 µs | 直接 DT µs | 首对读 预/实 µs | 最后 MMAD 后尾段 预/实 µs |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rs:
        e,c=r['predicted_phase_us'],r['critical_phase_us']
        lines.append('| '+' | '.join(['×'.join(map(str,r['shape'])),'/'.join(map(str,r['active_aic'])),str(sum(r['observed_mmad_hist'].values())),*[f'{r[k]:.3f}' for k in ['ca_us',*keys]],*[f'{e[k]:.3f}/{c[k]:.3f}' for k in ['first_pair','drain']]])+' |')
    lines.extend(['','首对读指尾核首批两条 GM→L1 指令从首发到最后一条完成的窗口；尾段指该核最后一条 MMAD 完成到全局最后一条指令结束，并非原始 FIXP 指令时长。尾核由全核结束时间选取。预测与实测的完整分段、真实与预测 MMAD 尺寸直方图、每核工作量（2mnk ops）和指令次数分布见 [JSON](matmul-independent.json)。每核工作比较不要求虚拟和真实 core ID 一致。'])
    picks=[('最大绝对百分比误差',max(rs,key=lambda r:abs(r[keys[0]]/r['ca_us']-1))),('最大逻辑工作量',max(rs,key=lambda r:r['shape'][0]*r['shape'][1]*r['shape'][2]))]
    lucky=[x for x in rs if x['v14_mmad_hist']!=x['observed_mmad_hist'] and x['predicted_mmad_hist']==x['observed_mmad_hist'] and abs(x['pr_v14_us']-x['ca_us'])<abs(x['pr_v15_us']-x['ca_us']) and all(x['id']!=y['id'] for _,y in picks)]
    if lucky:picks.append(('补充反例：时延更接近并不证明分块正确',min(lucky,key=lambda x:abs(x['pr_v14_us']-x['ca_us']))))
    for title,r in picks:
        e,c=r['predicted_phase_us'],r['critical_phase_us'];diff={k:e[k]-v for k,v in c.items()}
        hist=', '.join(f'{k}：{v} 条' for k,v in r['observed_mmad_hist'].items());load=', '.join(f'{ops} ops × {n} 核' for ops,n in r['observed_ops_per_core_hist'])
        lines.extend(['',f"**{title}：{r['shape']}。** 实测 MMAD 分布 {hist}；核间工作为 {load}。",'',
        '|阶段|首读之前|首对读|读完成至 MMAD|MMAD 窗口|尾段|','|---|---:|---:|---:|---:|---:|','| 预测−实测 µs | '+' | '.join(f'{diff[k]:+.3f}' for k in ('prefix','first_pair','l0_lead','mmad_window','drain'))+' |',
        '',f"v0.15、v0.14、直接复用 DT 的有符号误差依次为 "+'、'.join(f'{100*(r[k]/r["ca_us"]-1):+.2f}%' for k in keys)+'。'])
        if r['v14_mmad_hist']!=r['predicted_mmad_hist']:
            lines.append(f"v0.14 的分块 {r['v14_structural_tile']}、预测 {r['v14_structural_aic']} 核；v0.15 实际采用候选块 {r['tile']}，预测/实测核数 {r['active_aic']}。分块变化会同时改变读窗和单核计算量，误差变化不能全算成核数效应。")
        elif abs(r['pr_v15_us']-r['pr_v14_us'])<1e-6:
            lines.append("按真实尾块裁切后，两版的计算分块与时延预测相同；当前误差来自连续阶段定价，改变分块没有改善这个形状。")
        if r['dt_ca_us'] is not None:
            lines.append(f"历史 DT CA 为 {r['dt_ca_us']:.3f} µs，实测 PR−DT 为 {r['ca_us']-r['dt_ca_us']:+.3f} µs；预测迁移增量为 {r['pr_v15_us']-r['reuse_dt_us']:+.3f} µs。")
        else:
            lines.append("此例未提供配对 DT CA；直接 DT 是预测基线，不能据此断言目标硬件快慢。")
    return '\n'.join(lines)+'\n'

if __name__=='__main__':main()
