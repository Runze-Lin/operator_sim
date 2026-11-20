"""Price one registered geometry per retained implementation and join exact CA receipts."""
import argparse
from functools import lru_cache
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/v0.35/large'
DT, PR, FUT = 'Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'
BASE = 'v0.32/large/coverage'
LOG = {}
FUSION = 'v0.29/large/fusion/freeze'
CUBE = 'v0.29/large/cube/freeze'
VECTOR = 'v0.29/large/vector/freeze'
POST = 'v0.31/post/freeze'


def ref(name):
    return 'results/' + name + '.json'


@lru_cache(None)
def load(name):
    content = (ROOT / ref(name)).read_bytes()
    LOG[ref(name)] = sha256(content).hexdigest()
    return json.loads(content)


def pick(items, **keys):
    return next(row for row in items if all(row.get(k) == v for k, v in keys.items()))


def cases():
    src = load(CUBE)['jobs']
    fuse = load(FUSION)['jobs']
    registered = []

    def add(op, shape, attrs, scenario, origin, alias=None, boundary=False):
        registered.append(dict(route=op, public_op=alias or op, shape=shape, attrs=attrs,
                               scenario=scenario, registration=ref(origin), boundary=boundary))

    for op, shape, attrs, context in [
        ('cast_bf16_fp32', [2097152, 64], {}, 'Pro B8×256Ki，64 列输入转换'),
        ('sigmoid_fp32', [2097152, 12], {}, 'Kimi B8×256Ki，每 token 12 个本地头'),
        ('quant', [262144, 7168], {}, 'Kimi B8×32Ki，hidden=7168'),
        ('rmsnorm', [8192, 5120], {}, 'Flash B4×2048 调度 token，hidden=5120'),
        ('fa', [9, 131072, 64, 1, 1], {}, '单头稠密 BF16，对 128Ki KV 做 Q9 校验，D=64 控制形'),
        ('matmul', [1048576, 128, 512], {}, 'Flash B8×128Ki，indexer.wk 的 BF16 投影几何'),
        ('mhc_sinkhorn', [4097, 4], {'num_iters': 4}, 'N=4，I=4，超过整批 4096 的尾行控制；历史 resident_aiv 为 DT57/PR52')
    ]:
        origin = (VECTOR if op in ('rmsnorm', 'quant') else
                  'v0.32/large/archive' if op == 'mhc_sinkhorn' else
                  'v0.29/large/fa/freeze' if op == 'fa' else BASE)
        add(op, shape, attrs, context, origin)
    for op in ('gmm2', 'grouped_matmul_swiglu_quant_v2', 'moe_ffn'):
        j = pick(src, id='moe')
        add(op, j['shape'], j['attrs'], 'Pro B8×32Ki，top6 均衡 EP384、专家 TP8，每卡单专家 4096 行', CUBE)
    j = pick(src, id='kda-v2')
    add('fla_kda_v2', j['shape'], j['attrs'], 'Kimi TP8，单请求 prefill 分块 1024 token、12 本地头，强制 V2', CUBE, 'kda_chunk')
    for op, job, context in [('mhc', 'flash-chunk4096', 'Flash B8×512，拆分库三段合计'),
                             ('mhc_pre', 'flash-chunk4096', 'Flash B8×512，拆分库前处理，导出中间量'),
                             ('mhc_pre_sinkhorn', 'fused-t512', '库融合分支 512 token、宽 4096、20 次迭代控制形')]:
        o = pick(pick(fuse, id=job)['outputs'], op=op)
        add(op, o['shape'], o['attrs'], context, FUSION)
    for op, context in [('mhc_post', 'Flash 宽 5120，独立库 MhcPost，带 h_res'),
                        ('vllm_hc_post', 'Flash 宽 5120，vLLM HcPost 自定义实现')]:
        j = pick(load(POST)['jobs'], id=op)
        add(op, j['shape'], j['attrs'], context, POST, 'mhc_post')
    j = load('v0.34/addmm/freeze')
    add('addmm', j['shape'], j['attrs'], 'Flash B8×8192 调度 token，显式列 bias 的单算子变体',
        'v0.34/addmm/freeze')
    j = pick(load('v0.34/attn/freeze')['jobs'], shape=[512, 7168, 8, 1])
    add(j['op'], j['shape'], j['attrs'], 'Kimi 512 token prefill，7168 通道，1 个有效 cached residual',
        'v0.34/attn/freeze')
    j = pick(load(VECTOR)['jobs'], op='moe_gating_top_k')
    add(j['op'], j['shape'], j['attrs'], 'Kimi B8×512，896 专家中取 16', VECTOR)
    j = load('v0.34/indexer/freeze')['job']
    attr = pick(pick(fuse, id='index-32ki')['outputs'], op='lightning_indexer')['attrs']
    add(j['op'], j['shape'], attr, 'Q9、8192 个可用 key，稠密 top512；容量对应 32Ki 的 4:1 压缩',
        'v0.34/indexer/freeze')
    j = load('v0.34/recurrent/freeze')['job']
    add('kda_recurrent (single chain)', j['shape'], j['attrs'], '单链最多 8 步，V=256 为库边界；Kimi V=128 单列在下方',
        'v0.34/recurrent/freeze', 'kda_recurrent', boundary=True)
    if {x['route'] for x in registered} != set(load('v0.33/large/coverage')['large_shapes']):
        raise ValueError('retained implementations changed; revisit the registered geometry for each')
    return sorted(registered, key=lambda r:r['route'])


def qualified(value, protocol, source, evidence, role):
    if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'invalid qualified clock in {source}')
    primary = ROOT / source
    if not primary.is_file():
        raise ValueError(f'missing qualified source {source}')
    if source not in LOG:
        LOG[source] = sha256(primary.read_bytes()).hexdigest()
    return dict(ca_us=value if protocol != 'wrapper' else None,
                wrapper_us=value if protocol == 'wrapper' else None,
                lower_us=None, protocol=protocol, source=source, index=evidence, role=role)


def measurement(r, chip):
    op, shape = r['route'], r['shape']
    tag = 'dt' if chip == DT else 'pr'
    key = (op, tuple(shape), chip)
    large_rows = load(BASE)['rows']
    matches = [x for x in large_rows if (x['op'], tuple(x['shape']), x['chip']) == key]
    result = dict(ca_us=None, wrapper_us=None, lower_us=None, protocol=None, source=None,
                  index=ref(BASE), role='无合格同形收据')
    for x in matches:
        if x.get('status') == 'qualified' and x['measured_us'] is not None:
            result = qualified(x['measured_us'], x['protocol'], x['source'], ref(BASE), '既有采集回放')
    if op == 'rmsnorm':
        path = f'v0.33/large/rms-8192-5120-{tag}'
    elif op == 'vllm_hc_post':
        path = 'v0.35/hcpost/pr-full' if chip == PR else 'v0.33/hcpost/partial'
    elif op == 'kda_recurrent (single chain)':
        path = f'v0.34/recurrent/{tag}'
    elif op == 'lightning_indexer':
        path = f'v0.34/indexer/{chip}'
    else:
        path = None
    if path and (ROOT/ref(path)).is_file():
        x = load(path)
        if x['chip'] != chip or x['shape'] != shape:
            raise ValueError(f'wrong endpoint: {path}')
        verdict = x.get('verdict', x)
        if verdict.get('qualified') and verdict.get('ca_us') is not None:
            role = '原始逐 launch 回执离线恢复' if op == 'vllm_hc_post' else ('逐核复核旧采集' if op=='rmsnorm' else '冻结后完整指令')
            time_us = verdict['ca_us'][op] if op == 'lightning_indexer' else verdict['ca_us']
            return qualified(time_us, 'full', ref(path), ref(path), role)
        if 'conservative_lower_us' in x:
            return dict(result, lower_us=x['conservative_lower_us'], source=ref(path), index=ref(path),
                        role='原始 trace 不完整，三核缺 END', protocol='partial')
    if op == 'kda_recurrent (single chain)':
        result.update(index=ref('v0.34/recurrent/freeze'), role='已登记序列/状态上界，等待合格端点')
    if op == 'mhc':
        src = 'v0.29/large/fusion/score'
        x = pick(load(src)['rows'], op=op, shape=shape)['target'][chip]
        if x['ca_us'] and x['ca_receipt']:
            return qualified(x['ca_us'], 'serialized stages', x['ca_receipt'], ref(src), '分解链逐段全窗之和')
        return dict(result,index=ref(src),role=x['qualification'])
    # Only use clocks admitted by each route's original qualification reducer.
    sources = {
        'attn_res_fwd': ('v0.34/attn/score', 'direct'),
        'addmm': ('v0.34/addmm/score', 'rows'),
        'lightning_indexer': ('v0.34/indexer/score', 'prospective'),
        'moe_gating_top_k': ('v0.35/gating/score', 'forward'),
        'mhc_post': ('v0.31/post/score', 'rows'),
        'mhc_pre': ('v0.29/large/fusion/stage-score', 'rows'),
        'mhc_pre_sinkhorn': ('v0.29/large/fusion/stage-score', 'rows'),
        'grouped_matmul_swiglu_quant_v2': ('v0.29/large/cube/matrix','rows'),
        'gmm2': ('v0.29/large/cube/matrix','rows'), 'moe_ffn': ('v0.29/large/cube/matrix','rows'),
        'fla_kda_v2': ('v0.29/large/cube/matrix','rows')
    }
    if op in sources:
        src, field = sources[op]
        for x in load(src)[field]:
            if x.get('chip') != chip or x.get('shape') != shape or x.get('op', op) not in (op, r['public_op']):
                continue
            if x.get('qualified') is False or x.get('ca_us') is None:
                if not result['ca_us']:
                    result['index'] = ref(src)
                    result['role'] = x.get('stage') or x.get('role') or x.get('status') or result['role']
                continue
            source = x.get('receipt') or x.get('source') or x.get('receipt_path') or x.get('evidence') or ref(src)
            if not (ROOT/source).is_file():
                source = str((Path(ref(src)).parent/source))
            result = qualified(x['ca_us'], 'full', source, ref(src), '原登记的合格端点')
    return result


def numbers(value):
    return f'{value:.3f}' if value is not None else 'N/A'


def price(api, op, shape, attrs, chip):
    start = time.perf_counter()
    p = api.estimate(op, shape, attrs, chip=chip)
    elapsed = 1000 * (time.perf_counter() - start)
    return dict(us=p.latency_us, status=p.status, reason=p.reason, query_ms=elapsed, dtype=p.work.get('dtype'))


def matmul_sizes(api):
    scored = load('v0.32/matmul/score')
    rows = []
    sizes = (32768,65536,131072,262144)
    for size in sizes:
        shape = [8*size,128,512]
        entry = dict(context_per_request=size,shape=shape,times={},role='开发回放' if size in (32768,131072) else '冻结的几何')
        for c in (DT,PR):
            actual = None
            source = None
            for row in scored['rows']:
                if row['shape'] == shape and row['chip'] == c:
                    actual, source = row['ca_us'], row['source']
            for row in scored['holdouts']:
                if row['shape'] == shape and row['sku'] == c and row.get('ca'):
                    actual, source = row['ca'], ref('v0.32/matmul/score')
            final = 'v0.33/matmul/64ki-final'
            if size==65536 and c==DT and (ROOT/ref(final)).is_file():
                receipt=load(final)
                if receipt.get('phase') == 'qualified' and receipt.get('reduced_certificate_sha256'):
                    actual, source=receipt['ca_us'],ref(final)
            entry['times'][c] = dict(public_us=price(api,'matmul',shape,{},c)['us'], ca_us=actual,source=source)
        entry['extrapolated_970dt_us'] = price(api,'matmul',shape,{},FUT)['us']
        rows.append(entry)
    return rows


def atomic(path, text):
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('installed_prefix', type=Path)
    args = parser.parse_args()
    prefix = args.installed_prefix.resolve()
    if not sys.flags.isolated:
        raise ValueError('use python -I and pass an installed wheel prefix')
    sys.path.insert(0, str(prefix))
    import operator_model as api
    if not Path(api.__file__).resolve().is_relative_to(prefix):
        raise ValueError('public API must come from the installed package')
    entries = cases()
    for r in entries:
        r['price'] = {chip:price(api, r['public_op'], r['shape'], r['attrs'], chip) for chip in (DT, PR, FUT)}
        r['measured'] = {c:measurement(r,c) for c in (DT, PR)}
        for c, v in r['measured'].items():
            pred = r['price'][c]['us']
            v['signed_error_pct'] = 100*(pred/v['ca_us']-1) if pred and v['ca_us'] else None
        dt, pr = [r['measured'][c]['ca_us'] for c in (DT,PR)]
        r['dt_over_pr'] = dict(measured=dt/pr if dt and pr else None,
                              predicted=r['price'][DT]['us']/r['price'][PR]['us'] if r['price'][DT]['us'] and r['price'][PR]['us'] else None)
    paired = sum(all(r['measured'][c]['ca_us'] for c in (DT,PR)) for r in entries if not r['boundary'])
    n_full = sum(q['ca_us'] is not None for r in entries for q in r['measured'].values())
    costs = [x['query_ms'] for r in entries for x in r['price'].values()]
    ext = []
    for src, field, op, shape in [
        ('v0.32/replay', 'rows', 'kda_recurrent', [1,8,1,1,128,128]),
        ('v0.34/indexer/score','rows', 'lightning_indexer', [1,1,8192,8,128]),
        ('v0.34/indexer/score','rows', 'lightning_indexer', [1,9,4096,8,128])]:
        for x in load(src)[field]:
            if x.get('op',op) == op and x['shape'] == shape and x.get('ca_us'):
                attrs = x.get('attrs',pick(entries,route='lightning_indexer')['attrs'])
                ext.append(dict(route=op,shape=shape,chip=x['chip'],ca_us=x['ca_us'],source=ref(src),
                                trace_sha256=x.get('identity') or x['trace_sha256'],
                                price=price(api,op,shape,attrs,x['chip']),role='相邻几何回放，与主表分别计数'))
    flash = matmul_sizes(api)
    package_record, = prefix.glob('ascend_operator_model-*.dist-info/RECORD')
    output = dict(version=api.__version__, wheel_record_sha256=sha256(package_record.read_bytes()).hexdigest(),
                  counts=dict(routes=len(entries), large=sum(not r['boundary'] for r in entries), paired_large=paired,
                  qualified_instruction_endpoints=n_full, wrappers=sum(q['wrapper_us'] is not None for r in entries for q in r['measured'].values())),
                  chip_ids=[DT, PR, FUT], rows=entries, additional=ext, flash_matmul_contexts=flash,source_sha256=LOG,
                  query_ms=dict(n=len(costs),max=max(costs),median=statistics.median(costs)))
    lines = ['# 已登记大形的逐算子预测与 CA','',
             f'当前公开后端 {api.__version__}。21 条路线中，20 条有登记的大形，另列单链 KDA 的合法序列上界。本表所选大形中有 {paired} 条取得同 shape、同实现的 DT/PR 双端完整指令计时，合格指令端点共 {n_full} 个。表中耗时单位均为 µs。','',
             '| 路线 | 本地输入 shape | 950DT 预测 | DT 完整 CA | DT 实测下界 | DT wrapper | 950PR 预测 | PR 完整 CA | 970DT 外推 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in entries:
        dt,pr = [r['measured'][c] for c in (DT,PR)]
        nums = [r['price'][DT]['us'], dt['ca_us'],dt['lower_us'],dt['wrapper_us'],
                r['price'][PR]['us'],pr['ca_us'], r['price'][FUT]['us']]
        lines.append('| '+r['route']+' | '+str(r['shape'])+' | '+' | '.join(map(numbers,nums))+' |')
    lines += ['', '完整 CA 取原合格端点的设备指令窗。FA 的 DT 来自 compact CA，PR 来自官方逐指令采集。Quant 的 DT 数值通过且 wrapper 已结束，指令整窗缺失，因此单列。vLLM HcPost 的 DT 缺三个核的结束标记，PR 已从同一采集的逐 launch 副本恢复完整窗。相同数值四舍五入后可能出现在不同算子，具体来源按行给出。970DT 全部为相同实现假设下的估计。','',
              '## 几何与证据','',
              '| 路线 | 工作负载与适用范围 | DT 指令误差 | PR 指令误差 | 原始登记 / 端点证据 |',
              '|---|---|---:|---:|---|']
    def link(label,path):
        return f'[{label}]({os.path.relpath(ROOT/path,OUT)})'
    for r in entries:
        links = [link('登记',r['registration'])]
        errors = []
        for c,label in zip((DT,PR),('DT','PR')):
            obs = r['measured'][c]
            links.append(link(label,obs['source'] or obs['index']))
            e = obs['signed_error_pct']
            errors.append(f'{e:+.2f}%' if e is not None else 'N/A')
        lines.append('| '+r['route']+' | '+r['scenario']+' | '+' | '.join(errors)+' | '+' · '.join(links)+' |')
    lines += ['', '误差为 100 × (公开预测 / 完整 CA − 1)。价格按表中合同调用公开 API，不传实测的核数、tiling 或历史 resident_aiv。4097 行 Sinkhorn 的设备历史和 4 次迭代是该样本的条件，生产 20 次迭代须单独验证。N/A 表示本表未取得相应证据；每条采集的预算、完整性和 kernel 身份沿端点链接可查。','',
              '## 同一路线的其他实测','',
              '| 路线 | shape | 目标 | 当前预测 µs | 完整 CA µs | 收据 |',
              '|---|---|---|---:|---:|---|']
    for x in ext:
        lines.append(f"| {x['route']} | {x['shape']} | {x['chip']} | {numbers(x['price']['us'])} | {numbers(x['ca_us'])} | {link('来源',x['source'])} |")
    lines += ['', '## Matmul 整批 prefill 的数据量阶梯','',
              'Flash 的 indexer.wk 在 B=8 时，本地形状为 [8 × 每请求输入长度,128,512]。以下按整批一次提交，均为连续 BF16。32Ki 和 128Ki 两档参与过模型开发。64Ki 的 DT 在预测冻结时已开跑且未读取结果，256Ki 为登记留出。使用 chunk 的部署按实际调度行数另行查询。','',
              '| 每请求 token | 本地 M | DT 预测 / 完整 CA µs | PR 预测 / 完整 CA µs | 970DT 外推 µs |',
              '|---:|---:|---:|---:|---:|']
    for f in flash:
        outputs = []
        for c in (DT, PR):
            t = f['times'][c]
            clock = link(numbers(t['ca_us']),t['source']) if t['ca_us'] and t['source'] else 'N/A'
            outputs.append(numbers(t['public_us'])+' / '+clock)
        lines.append(f"| {f['context_per_request']//1024}Ki | {f['shape'][0]:,} | "+' | '.join(outputs)+f" | {numbers(f['extrapolated_970dt_us'])} |")
    lines += ['', '同一投影输入长度从 32Ki 增至 128Ki，行数增至 4 倍。双端实测时间均增长超过 4 倍，不能仅凭这两个开发点断定大形严格线性。64Ki 的原始采集预算为 6 小时，只有完整数值、设备资源和逐核退休均通过才计入。[64Ki 运行与终态](../../v0.33/matmul/64ki-state.json) · [三模型 B=1/9/11/16 的 decode 与 MTP 逐形表](../../v0.32/matmul/decode.md)']
    lines += ['', f"63 次公开查询仅计 estimate 调用，当前主机进程内中位 {statistics.median(costs):.3f} ms、最大 {max(costs):.3f} ms。离线 CA 沿用原始采集。本表可用同一生成器重新读取已更新的端点计分文件以及完整 HcPost 收据。所有预测、精确参数、端点资格、源文件哈希与查询时间见[机读表](coverage.json)。",'']
    OUT.mkdir(exist_ok=True)
    atomic(OUT/'coverage.json', json.dumps(output,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    atomic(OUT/'report.md','\n'.join(lines))


if __name__=='__main__':
    main()
