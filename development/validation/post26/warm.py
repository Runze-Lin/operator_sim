"""Independent shadow of changed-shape warm launches, using only pre-existing hardware stage prices."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post24.freeze import sha
from development.validation.post24.report import stats,fmt
from development.validation.post26.freeze import OUT,OLD
from development.validation.sku19.campaign import now,read,write
from operator_model import load_chip
from operator_model.post_resource import resource_post
from operator_model.post_token_pipeline import token_pipeline
FILE=OUT/'warm-schedule-freeze.json';SRC=OUT/'registration.json'

def price(r,prices):
    chip=load_chip(r['chip'])
    baseline,_,_,diag=resource_post(r['shape'],chip,{'resident_aiv':r['prior_aiv']})
    use=(r['previous_shape'] is not None and tuple(r['previous_shape'])!=tuple(r['shape'])
         and diag['context']=='reuse' and diag['target_plan']['route']=='post_token')
    if not use:return dict(changed=False,shadow_us=baseline,baseline_us=baseline)
    q=token_pipeline(r['shape'],chip,diag['target_plan'],'reuse',prices,diag['shared_floor_us'])
    return dict(changed=True,shadow_us=q['us'],baseline_us=baseline,stages=q['stages_us'],gaps_cycles=q['gap_cycles'])
def register():
    if FILE.exists():raise FileExistsError(FILE)
    raw=ROOT/'runs/post26';folders=sorted(p.name for p in raw.iterdir() if p.is_dir())
    assert not folders,'CA for prospective v0.26 has already started'
    reg=read(SRC);prices=read(OLD/'prices.json');rows=[]
    for r in reg['rows']:
        rows.append(dict(**{k:r[k] for k in ('chip','shape','previous_shape','prior_aiv','active','group','i','role','state')},**price(r,prices)))
    inputs=[SRC,OLD/'prices.json',ROOT/'operator_model/post_token_pipeline.py',ROOT/'operator_model/post_resource.py',Path(__file__).resolve()]
    write(FILE,dict(utc=now(),raw_directories_at_registration=folders,inputs={str(f.relative_to(ROOT)):sha(f) for f in inputs},
        hypothesis='Sufficient AIV coverage and different immediately preceding shape can permit cross-token read/write overlap. Uniformly choose the frozen generic reuse phases learned from changed-shape warm dispatches. Exact-repeat prices depend on a different antecedent and are not used. Device schedule uses identical closed-form first/steady/last overlap and configured per-resource floors. No hardware or shape exceptions.',
        eligibility='Same stream, preceding successful builtin MhcPost, previous known full shape different, H>=A>0, token split; all other shapes retain default. Direction and thresholds are unchanged at all endpoints. No new simulator tasks.',
        metrics='WAPE/max and paired DT:PR where both endpoints are eligible; unchanged rows excluded as non-informative. Wide [65,4,5120] and all v0.25 groups are development only. Frozen v0.26 groups yield two eligible DT rows, one PR row; this is a mechanism check, not sufficient coverage for deployment.',
        rows=rows))
def evaluate():
    frozen=read(FILE)
    assert all(sha(ROOT/f)==h for f,h in frozen['inputs'].items()),'frozen file changed'
    index={(r['group'],r['chip'],r['i']):r for r in frozen['rows']};out=[];rejects=[]
    for p in sorted((OUT/'receipts').glob('*.json')):
        src=read(p);assert src['cost']['started_utc']>frozen['utc']
        for r in src['rows']:
            k=index[(src['group'],src['chip'],r['i'])]
            assert k['shape']==r['shape']
            if not k['changed']:continue
            if not r['timing_qualified']:
                rejects.append(dict(**k,reason=r.get('reason')));continue
            out.append(dict(**k,ca_us=r['stage']['ca_us'],kernel=r['stage']['kernel'],actual_aiv=r['stage']['active_aiv'],trace=r['stage']['trace'],trace_sha256=r['stage']['trace_sha256']))
    prices=read(OLD/'prices.json');dev=[]
    for p in sorted((OLD/'receipts').glob('width-*.json')):
        for r in read(p)['rows']:
            if r['timing_qualified'] and r['shape']==[65,4,5120] and r['role']=='entry':
                val=price(r,prices);assert val['changed']
                dev.append(dict(chip=r['chip'],shape=r['shape'],previous_shape=r['previous_shape'],ca_us=r['stage']['ca_us'],**val))
    ordered=read(SRC)['measured'];summary={s:{key:stats([x for x in out if x['chip']==s],key) for key in ('shadow_us','baseline_us')} for s in ordered}
    pairs=[];a,b=ordered
    for u in out:
        if u['chip']==a:
            v=next((x for x in out if x['chip']==b and x['group']==u['group'] and x['i']==u['i']),None)
            if v:pairs.append(dict(shape=u['shape'],ca_us=u['ca_us']/v['ca_us'],**{key:u[key]/v[key] for key in ('shadow_us','baseline_us')}))
    state='pending' if not out else 'partial' if len(out)+len(rejects)<3 else 'complete'
    write(OUT/'warm-schedule.json',dict(state=state,utc=now(),qualified=out,unqualified=rejects,summary=summary,paired_dt_pr=pairs,development_replay=dev))
    md=['# 普通热态换形时，后续 token 能否提前发起','',f"独立结果 **{state}**。候选只作用于前次内置 Post 合同已知、shape 改变且活动 AIV 已覆盖的 token 路线。在既有 v0.26 序列里仅有 DT 两次、PR 一次符合；其余照用默认价格，不能计作对候选的验证。970DT 只登记外推场景。", '',
        '全部符合条件的形状统一选择开发档案已经冻结的普通 reuse 阶段价格。普通 warm 与立即重复的区别是可由调用方直接观测的上一条 shape。两者可以共用读写流水递推，发起间隔和首次响应仍由各自状态的记录定价。该对照在采集前冻结，并且没有新增 CA、重新拟合参数或修改公开 opt-in 的语义。', '',
        '| 芯片 | 完整新窗 | 影子 WAPE / 最大 | 默认 WAPE / 最大 |','|---|---:|---:|---:|']
    for s,v in summary.items():md.append(f"| {s} | {v['shadow_us']['n'] if v['shadow_us'] else 0} | {fmt(v['shadow_us'])} | {fmt(v['baseline_us'])} |")
    md+=['','## 已见过的宽度 5120，仅作开发回放','', '| 芯片 | CA µs | 现行 µs | 共享跨行排程、普通 reuse 价格 µs | 有符号偏差 |','|---|---:|---:|---:|---:|']
    for d in dev:md.append(f"| {d['chip']} | {d['ca_us']:.3f} | {d['baseline_us']:.3f} | {d['shadow_us']:.3f} | {100*(d['shadow_us']/d['ca_us']-1):+.2f}% |")
    md+=['','[65,4,5120] 从 [121,4,1216] 切来，DT、PR 两侧都有充足 AIV。按默认互斥阶段定价，MTE2 分别多算 2.513/2.629 µs。实际 5120 列按 UB 容量拆成两半片，每组系数只读一次；完整输出地址和分片读写起止可直接确认，后一行发起可以早于上一行写完。模型的整行线性字节服务与逐行发起间隔可分别核算，但单个设备窗口无法反推 HBM/L2 命中、单条 DMA 的独占带宽或具体物理分配的缓存复用。即使回放改善，上表仍不能算作独立证据。','',
        '[冻结前序/价格与时间](warm-schedule-freeze.json) · [随回执更新的逐形及加速比](warm-schedule.json) · [实际流水和分块](report.md) · [互斥阶段计价差](wide-stages.md)']
    (OUT/'warm-schedule.md').write_text('\n'.join(md).rstrip()+'\n')
if __name__=='__main__':
    if 'register' in sys.argv:register()
    evaluate()
