"""Replace the two-chip summary at the start of the user-facing error guide."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BEGIN = '<!-- release32-summary -->'
END = '<!-- /release32-summary -->'
NAMES = {
    'quant': 'DynamicMXQuant', 'matmul': 'Matmul', 'fa': 'FA',
    'lightning_indexer': 'Lightning Indexer', 'addmm': 'Addmm', 'rmsnorm': 'RMSNorm',
    'mhc_pre': 'mHC Pre', 'mhc_sinkhorn': 'mHC Sinkhorn', 'mhc_pre_sinkhorn': 'mHC PreSinkhorn',
    'mhc_post': 'mHC Post', 'mhc': 'mHC 拆分链', 'vllm_hc_post': 'vLLM HcPost',
    'kda_recurrent': 'KDA recurrent', 'kda_chunk': 'KDA chunk', 'fla_kda_v2': 'KDA chunk FLA V2',
    'gmm2': 'MoE GMM2', 'grouped_matmul_swiglu_quant_v2': 'GMM1–SwiGLU–MXQuant',
    'moe_ffn': 'MoE FFN', 'moe_gating_top_k': 'MoE sigmoid Top-K', 'cast_bf16_fp32': 'Cast BF16→FP32',
    'sigmoid_fp32': 'KDA beta Sigmoid', 'attn_res_fwd': 'Kimi AttnResFwd'
}
CONTRACT = {'known history': '已知前序', 'stream': '有序流', 'shape': '仅 shape', 'visible stream': '可见前序',
            'dense multi-chain': '多状态链', 'compact CA': 'compact CA', 'fused V1': '融合 V1'}


def update():
    path = ROOT/'docs/error-analysis.md'
    s = path.read_text()
    if '<!-- release34-summary -->' in s:
        return
    original, tail = s.split(BEGIN)
    _, tail = tail.split(END)
    stats = json.loads((ROOT/'results/v0.32/evaluation.json').read_text())['summary']
    scope = json.loads((ROOT/'results/v0.33/acceptance-scope.json').read_text())
    ledger = json.loads((ROOT/'results/v0.32/replay.json').read_text())['rows']
    rows = ['| 算子与输入/计时口径 | 950DT 形/窗 · WAPE / 最大 | 950PR 形/窗 · WAPE / 最大 | 统计样本角色 | 校准状态 DT / PR |',
            '|---|---:|---:|---|---|']
    chips = ('Ascend950DT_9582','Ascend950PR_957b')
    for route, pair in sorted(stats.items(),key=lambda x: (list(NAMES).index(x[0].split(' · ')[0]),x[0])):
        if route in scope['excluded_error_rows']:
            continue
        name, _, mode = route.partition(' · ')
        title = NAMES[name] + ('，'+CONTRACT.get(mode,mode) if mode else '')
        if route == 'kda_recurrent':
            title += '，单状态链'
        cells, passed = [], []
        for chip in chips:
            st = pair[chip]['full']
            cells.append(f"{st['shapes']}/{st['n']} · {st['wape']:.2f}% / {st['maximum']:.2f}%" if st['n'] else '0 · N/A')
            passed.append('N/A' if not st['n'] else ('达标' if st['wape']<=25 and st['maximum']<=50 else '超限'))
        subset = [r for r in ledger if r['op']==route and r['us'] is not None]
        role = '无合格窗' if not subset else '历史回放'
        roles = {r['data_role'] for r in subset}
        if roles == {'independent'}:
            role = '冻结后新窗'
        elif 'independent' in roles:
            role = '回放及冻结后新窗'
        rows.append('| '+' | '.join([title,*cells,role,' / '.join(passed)])+' |')
    content = original+BEGIN+'\n\n'+'\n'.join(rows)+'\n\n'+END+tail
    if content != s:
        path.write_text(content)


if __name__=='__main__':
    update()
