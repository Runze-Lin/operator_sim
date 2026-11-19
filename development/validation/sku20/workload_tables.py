"""Publish the entire pinned B8/TP8 prefill tensor grid in browsable Markdown."""
from collections import defaultdict
import gzip
from itertools import groupby
import json
from pathlib import Path

NAMES={'DeepSeek-V4.1-Flash':'deepseek-flash','DeepSeek-V4-Pro':'deepseek-pro','Kimi-K3':'kimi-k3'}
CHIPS=('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT')


def ranges(items):
    numbers=sorted(x for x in items if isinstance(x,int))
    out=[]
    for _,g in groupby(enumerate(numbers),lambda pair:pair[1]-pair[0]):
        group=[v for _,v in g]
        out.append(str(group[0]) if len(group)==1 else f'{group[0]}–{group[-1]}')
    out.extend(str(x) for x in items if not isinstance(x,int))
    return ', '.join(out)


def number(v):
    return '—' if v is None else f'{v:.3f}'


def contract_key(call):
    # These public routes each have a fixed dtype unless overridden in attrs.
    return call['op'], tuple(call['shape']), call.get('dtype'), json.dumps(call.get('attrs',{}),sort_keys=True)


def grid(root,rows):
    v=root/'results/v0.20'
    observed={}
    for r in rows:
        key=contract_key(r)
        assert key not in observed
        observed[key]=r
    models=defaultdict(list)
    with gzip.open(v/'workloads/contracts.jsonl.gz','rt') as source:
        for case in map(json.loads,source):
            s=case['scenario']
            if s['batch']==8 and s['chunk']==0 and s['tp']==8 and s['dp']==1 and s['ep']==8:
                models[s['model']].append(case)
    assert set(models)==set(NAMES)
    out=root/'docs/workloads';out.mkdir(exist_ok=True)
    for model,cases in models.items():
        cases.sort(key=lambda c:c['scenario']['seq'])
        assert [c['scenario']['seq'] for c in cases]==[32768,131072,262144]
        lines=[f'# {model} · 整批 prefill 的逐节点单卡张量','',
            '每个表对应 8 条等长请求一次提交，attention TP8、DP1，MoE EP8。层号从 0 开始。'
            '权重写为源码的逻辑形状，KV 表同时包含持久化或条件性缓冲。专家行数采用本情景的均衡路由。'
            '表内给出完整轴长度，可以按具体节点直接查询。','',
            '三个时延列的单位是 µs。950 的单元格是“模型 / 同形 CA”，970DT 是配置外推。'
            '只有算子、shape 和属性一致的独立库调用才显示正式指令窗。原生 wrapper 和单独检查过的 BF16 ND 投影另有证据。'
            '此情景的残差激活为 BF16，beta Sigmoid 是 FP32→FP32；动态 MX 量化是 BF16→FP8 E4M3FN 和 E8M0 scale。'
            '“仅形状”表示只导出了结构，没有独立后端计时合同。视部署而定的算子可能被融合。'
            '[调度与参数来源](../prefill-shapes.md) · [独立库采集的角色及误差](../error-analysis.md) · '
            '[完整配置、条件和其他并行拓扑](../../results/v0.20/workloads/contracts.jsonl.gz)','']
        for case in cases:
            seq=case['scenario']['seq'];nodes=case['contracts']
            lines += [f'## 每请求 {seq:,} token · 单卡 {case["tokens_per_rank"]:,} token','',
                '| 节点与层号 | 输入、输出及参数形状 | 单算子合同 | DT 模型 / CA | PR 模型 / CA | 970DT |',
                '|---|---|---|---:|---:|---:|']
            for n in nodes:
                tensors='<br>'.join(f'{k} <code>{json.dumps(a,separators=(",",":"),ensure_ascii=False)}</code>' for k,a in n['tensors'].items())
                name=f"`{n['op']}`<br>{ranges(n['layers'])}"
                call=n.get('standalone_call')
                condition='<br>'+n['condition'].replace('|','／') if n.get('condition') else ''
                if not call:
                    lines.append(f'| {name} | {tensors} | 仅形状{condition} | — | — | — |')
                    continue
                evidence=observed.get(contract_key(call),{})
                values=[n['standalone_prediction'][c]['us'] for c in CHIPS]
                label='`'+call['op']+'`'
                if call.get('attrs'):
                    label+='<br>'+', '.join(f'{k}={json.dumps(v)}' for k,v in call['attrs'].items())
                label+=condition
                cells=[number(values[i])+' / '+number(evidence.get(s+'_ca_us')) for i,s in enumerate(('dt','pr'))]
                lines.append(f'| {name} | {tensors} | {label} | {cells[0]} | {cells[1]} | {number(values[2])} |')
            lines.append('')
        lines+=['[返回三个模型的对照表](../prefill-shapes.md) · [模型与 CA 流水图集](../pipeline-comparisons.md)','']
        (out/(NAMES[model]+'.md')).write_text('\n'.join(lines))


if __name__=='__main__':
    root=Path(__file__).resolve().parents[3]
    rows=json.loads((root/'results/v0.20/evaluation.json').read_text())['rows']
    grid(root,rows)
