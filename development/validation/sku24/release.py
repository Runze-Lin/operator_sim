"""Assemble evidence for the public release; retain role and implementation at each row."""
import json
from hashlib import sha256
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import __version__, estimate

OUT=ROOT/'results/v0.24'

def read(path):
    return json.loads(path.read_bytes())

def write(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def stamp(p):
    return dict(path=str(p.relative_to(ROOT)),sha256=sha256(p.read_bytes()).hexdigest())

def main():
    freezes=[OUT/name for name in ('rms-prefix/freeze.json','kda-prefetch/register.json','post/freeze.json','kda-v1/register.json')]
    freezes.append(ROOT/'results/v0.23/pro-prefill/freeze.json')
    write(OUT/'model.json',dict(version=__version__,implementation='CANN 9.2.1 built-ins and explicit FLA 26.9.1 V2',freezes=[stamp(f) for f in freezes],
                                sources=[stamp(f) for f in sorted((ROOT/'operator_model').glob('*.py'))],
                                hardware=[stamp(f) for f in sorted((ROOT/'configs/chips').glob('Ascend*.json'))]))
    lines=['# 本版逐 shape 核对','','耗时均为设备 µs。950DT / 950PR 的误差取 100 × (预测 / CA − 1)，970DT 只做外推。首次、重复和开发回放各自保留。','','|算子及来源|设备|输入|数据角色|模型 / CA|误差|970DT 外推|','|---|---|---|---|---:|---:|---:|']
    evidence=[]
    def add(op,chip,shape,role,pred,actual,future,receipt):
        evidence.append(dict(op=op,chip=chip,shape=shape,role=role,predicted_us=pred,ca_us=actual,future_970_us=future,receipt=receipt))
        gap=100*(pred/actual-1)
        lines.append(f'| {op} | {chip[6:11]} | {shape} | {role} | {pred:.3f} / {actual:.3f} | {gap:+.2f}% | {future:.3f} |')
    pro=ROOT/'results/v0.23/pro-prefill'
    for f in sorted(pro.glob('pro-b8-*-blocks.json')):
        z=read(f)
        assert read(pro/f.name.replace('-blocks',''))['qualified']
        res=estimate('cast_bf16_fp32',z['shape'],chip=z['chip'])
        assert abs(res.latency_us-z['model_us'])<1e-8
        add('Cast · aclnnCast',z['chip'],z['shape'],'Pro 整批 prefill，独立采集',z['model_us'],z['ca_us'],z['projection_970dt_us'],str(f.relative_to(ROOT)))
    rms=read(OUT/'rms-prefix/evaluation.json')
    for r in rms['rows']:
        name=r['group']+'-'+r['chip']+'-windows.json' if r.get('source') is None else f"recovery-{r['group']}-{r['chip']}-{r['i']}.json"
        add('RMSNorm · aclnnRmsNorm',r['chip'],r['shape'],f"可见下界 {r['observed_lower']}，完整 H={r['true_H']}，独立物理窗",r['partial_us'],r['ca_us'],r['future_partial_us'],'results/v0.24/rms-prefix/'+name)
    kda=read(OUT/'kda-prefetch/comparison.json')
    for r in kda['rows']:
        add('KDA · aclnnChunkKdaFwdV2',r['sku'],r['shape'],'本版新形' if r['role']=='independent' else '开发回放',r['candidate'][-1],r['actual'][-1],r['prediction_970_us'],'results/v0.24/kda-prefetch/comparison.json')
    post=read(OUT/'post/evaluation.json')
    for r in post['rows']:
        add('内置 Post · aclnnMhcPost',r['chip'],r['shape'],'独立有序，同集候选 '+r['role'],r['candidate_us'],r['ca_us'],r['future_us'],'results/v0.24/post/evaluation.json')
    refs={'prior_all_operators':'results/v0.22/evaluation.json','previous_release':'results/v0.23/evaluation.json',
          'partial_prefix':'results/v0.24/rms-prefix/evaluation.json','kda_pipeline':'results/v0.24/kda-prefetch/comparison.json',
          'post_candidate':'results/v0.24/post/evaluation.json','fused_v1_diagnostics':'results/v0.24/kda-v1/README.md'}
    write(OUT/'evaluation.json',dict(version=__version__,receipts=refs,rows=evidence))
    lines+=['','[RMSNorm 的两种设备口径、加速比和全部金标](rms-prefix/README.md) · [KDA 按三段分开、两代方法同集](kda-prefetch/report.md) · [Post 分前序以及旧预测与候选](post/report.md) · [KDA V1 正确性与同步](kda-v1/README.md)',
            '','[原有全算子的逐形结果](../v0.22/shape-errors.md) · [上版新增算子的逐形结果](../v0.23/shape-errors.md) · [当前全量误差总结](../../docs/error-analysis.md)']
    (OUT/'shape-errors.md').write_text('\n'.join(lines)+'\n')
    current = ROOT/'docs/error-analysis.md'
    text = current.read_text()
    def replace_row(title,row,after_title):
        nonlocal text
        entries = text.splitlines()
        entries = [l for l in entries if not l.startswith('| '+title+' |')]
        i = next(i for i,l in enumerate(entries) if l.startswith('| '+after_title+' |'))
        entries.insert(i+1,'| '+title+' | '+row+' |')
        text = '\n'.join(entries)+'\n'
    chips = ('Ascend950DT_9582','Ascend950PR_957b')
    cells=[]
    for c in chips:
        g=rms['summary'][c]['partial_us']
        cells.append(f"{g['n']} · {g['wape_pct']:.2f}% / {g['max_abs_pct']:.2f}%")
    status='十形补采中'
    if all(rms['summary'][c]['partial_us']['n']==10 for c in chips):
        status=' / '.join('本组达标' if rms['summary'][c]['partial_us']['wape_pct']<=20 and rms['summary'][c]['partial_us']['max_abs_pct']<=40 else '超限' for c in chips)
    replace_row('RMSNorm（可见部分前序）',' | '.join(cells)+' | 新形冻结，仅给下界 L，全核物理退休 | '+status,'RMSNorm（给前序 H）')
    cells=[]
    for c in chips:
        g=post['summary'][c]['all']['baseline_us']
        count=len({tuple(r['shape']) for r in post['rows'] if r['chip']==c})
        cells.append(f"{count} · {g['wape']:.2f}% / {g['maximum']:.2f}%")
    replace_row('mHC Post（公开模型，有序前序）',' | '.join(cells)+' | 新形独立，首次、重复及回访，含导出终止标签 | 详见分态及候选对照','mHC Post')
    current.write_text(text)


if __name__=='__main__':
    main()
