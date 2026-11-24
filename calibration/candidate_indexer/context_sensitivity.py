"""Isolate candidate-indexer pricing on the context sweep's original rank-0 DES graphs.

Use only trusted local pickles from calibration.context_sweep. 'prepare' validates
the two archived runtimes and caches per-node prices outside the repository.
'report' changes candidate durations only, preserving the graph and its resources.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import replace
from functools import partial
import gzip
import json
import logging
import math
from pathlib import Path
import pickle
import tomllib

from calibration.context_report import linear, write
from calibration.context_sweep import ACCEPT, OUTPUT, case_id, inputs, profiles
from calibration.roofline_audit import nominal_engine_us
from calibration.system_compare import near

CANDIDATE = 'inference::candidate_filtered_indexer'
FULL = 'operator_sim_full'
ROOT = Path(__file__).resolve().parents[1]
LANES = ('full', 'peak', 'effective')
PA_BLOCK = 128  # explicit deployment scenario, in positions


def page_table(scratch, spec):
    with (scratch/case_id(spec)/'task.toml').open('rb') as f:
        capacity=tomllib.load(f)['inference']['runtime']['kv_capacity']
    check(capacity>=spec[5]+spec[6], 'KV capacity is smaller than this captured stage')
    return capacity,(capacity+PA_BLOCK-1)//PA_BLOCK


def check(condition, message):
    if not condition:
        raise ValueError(message)


def quote(scratch, spec):
    q = json.loads((scratch/case_id(spec)/'quote.json').read_text())
    check(q['spec'] == list(spec) and not q['failures'] and not q['below'], case_id(spec))
    check(sum(q['operators'].values()) == sum(s['nodes'] for s in q['stages']), 'Incomplete operator census')
    return q


def bundle(scratch, spec):
    with gzip.open(scratch/case_id(spec)/'graph.pkl.gz', 'rb') as f:
        return pickle.load(f)


def hardware(options):
    from sim_platform.hardware.loading import load_hardware
    hw = options.hardware
    if not hasattr(getattr(hw, 'chip', None), 'hbm'):
        hw = hw.to_model_hardware() if hasattr(hw, 'to_model_hardware') else load_hardware(hw).to_model_hardware()
    return hw


def geometry(n):
    (rows,h,d),(b,l,dkey),weights,positions,blocks = (t.shape for t in n.inputs)
    out = n.outputs[0].shape
    check(dkey == d and rows % b == 0 and tuple(weights) == (rows,h) and tuple(positions)==(rows,)
          and blocks[0] == out[0] == rows, 'Unexpected candidate contract')
    return b, rows//b, h, d, l, blocks[1], out[1]


def caps(options):
    resources = dict(options.capacities)
    if not options.cube_vec_overlap:
        resources['compute'] = 1
    return resources


def prepare(spec, scratch, cache):
    from sim_platform.api.inference_evaluation import evaluate_inference_workload
    from sim_platform.backend.factory import build_predictor
    from sim_platform.backend.l1 import simulate_graph
    from sim_platform.config.execution import PredictionOptions
    from operator_model.system_routes import STRUCTURAL, communication
    logging.disable(logging.WARNING)  # legacy backend has known, unrelated misses
    q = quote(scratch, spec)
    if q['operators'].get(CANDIDATE, 0) == 0:
        return False  # the original full, peak and effective DES are unchanged
    workload,o,_,_ = bundle(scratch,spec)
    hw = hardware(o)
    cubes, vecs = (dict(engine.tflops_by_bits) for engine in (hw.chip.cube, hw.chip.vec))
    def eval_all(name):
        model = build_predictor(PredictionOptions((name,),8192), (), hardware=hw)
        return evaluate_inference_workload(workload, predictor=model, options=o)
    receipts = []
    for fine, old, archived in zip(eval_all(FULL), eval_all('roofline'), q['stages'], strict=True):
        costs = dict(fine.execution_costs)
        pred, oldpred = dict(fine.predictions), dict(old.predictions)
        oldcost = dict(old.execution_costs)
        check(near(fine.duration_us, archived['full_us']) and near(old.duration_us, archived['legacy_us']),
              f'{case_id(spec)}: pricing installation differs from frozen capture')
        amounts = {lane: costs.copy() for lane in LANES[1:]}
        nodes = {}
        for n in fine.graph.nodes:
            ref = n.ref
            cost = costs[ref]
            if n.raw_name not in STRUCTURAL and n.raw_name not in communication.NAMES:
                detail = pred[ref].estimate.details
                tc = nominal_engine_us(detail.cube_work_by_dtype, cubes)
                tv = nominal_engine_us(detail.vec_work_by_dtype, vecs)
                tm = (detail.read_bytes+detail.write_bytes)/(hw.chip.hbm.peak_bytes_per_second/1e6)
                amounts['peak'][ref] = replace(cost,duration_us=max(tc,tv,tm))
                amounts['effective'][ref] = replace(cost,duration_us=max(tc/hw.chip.cube.utilization.ratio,
                        tv/hw.chip.vec.utilization.ratio, tm/hw.chip.hbm.utilization.ratio))
            candidate = None
            if n.raw_name == CANDIDATE:
                p = oldpred[ref]
                check(p.status == 'success' and p.estimate.details.model_name == 'candidate-filtered-indexer-roofline'
                    and oldcost[ref].duration_us > 0 and pred[ref].backend_name == FULL, 'Candidate pricing failed')
                candidate = (geometry(n),oldcost[ref].duration_us)
            nodes[ref.node_id] = (cost,amounts['peak'][ref].duration_us,amounts['effective'][ref].duration_us,candidate)
        for lane,prices in amounts.items():
            check(near(simulate_graph(fine.graph,prices,caps(o)).duration_us,archived[lane+'_us']),lane)
        receipts.append(nodes)
    check(sum(r[-1] is not None for stage in receipts for r in stage.values()) == q['operators'][CANDIDATE],
          'Candidate census mismatch')
    path = cache/(case_id(spec)+'.gz')
    with gzip.open(path.with_suffix('.tmp'), 'wb', compresslevel=1) as f:
        pickle.dump(receipts, f)
    path.with_suffix('.tmp').replace(path)
    return True


def jobs(fn, specs, workers):
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(fn,specs))


def replay(spec, scratch, cache, packed):
    from sim_platform.backend.l1 import simulate_graph
    from operator_model.sim_backend import OperatorSimBackend
    q=quote(scratch,spec)
    n=q['operators'].get(CANDIDATE,0)
    stage_counts = [4] if spec[4]=='prefill' else [4,2]
    check(len(q['stages']) == len(stage_counts), 'Stage count changed')
    if n == 0:
        # q is the complete per-op census from this frozen graph, not a model-name inference.
        return [(dict(ref={k:r[k+'_us'] for k in LANES}, legacy_whole=r['legacy_us'],
                      restore=r['full_us'], nodes=0, old_nodes_us=0, full_nodes_us=0,
                      **({'mx':{k:r[k+'_us'] for k in LANES},'mx_nodes_us':0,'prior':r['full_us'],'prior_nodes_us':0} if packed else {})))
                for r in q['stages']]
    if packed:
        from operator_model.backend import _rates
        from operator_model.roofline import cold_io_window
        from operator_model.operators.attention.flash_sparse import price
    workload,options,_,_=bundle(scratch,spec)
    hw=hardware(options)
    target=OperatorSimBackend()._resolve_chip(hw.name,hw.chip)
    capacity,pages=page_table(scratch,spec)
    capacities=caps(options)
    with gzip.open(cache/(case_id(spec)+'.gz'),'rb') as f:
        all_nodes=pickle.load(f)
    stages=[]
    for graph,stored,r,num in zip(workload.graphs,all_nodes,q['stages'],stage_counts,strict=True):
        old_prices={}
        prior_prices={}
        prior_node_us=0.0
        mx_prices={k:{} for k in LANES}
        full_node_us=old_node_us=mx_node_us=0
        shapes=set()
        hits=0
        check(len(stored)==len(graph.nodes)==r['nodes'],f'Node count changed for {case_id(spec)}')
        for node in graph.nodes:
            fine,peak,effective,candidate=stored[node.ref.node_id]
            check((candidate is not None) == (node.raw_name == CANDIDATE), 'Cache/node mismatch')
            old_prices[node.ref]=fine
            if candidate is not None:
                shape,legacy=candidate
                shapes.add('x'.join(map(str,shape)))
                check(shape==geometry(node), 'Shape changed after preparation')
                old_prices[node.ref]=replace(fine,duration_us=legacy)
                hits+=1
                old_node_us+=legacy
                full_node_us+=fine.duration_us
            if packed:
                prior=fine.duration_us
                durations=(fine.duration_us,peak,effective)
                if candidate is not None:
                    latency,work,_=price(shape,target,pa_block_size=PA_BLOCK,table_pages=pages)
                    prior,prior_work,_=price(shape,target,pa_block_size=PA_BLOCK,table_pages=pages,scalar_lookup_cycles=64)
                    check(prior_work==work,'Timing-only calibration changed the work ledger')
                    prior=cold_io_window(prior,work,target)['duration_us']
                    prior_node_us+=prior
                    # Same cold-I/O contract as the other system nodes. Leave any
                    # computation discrepancy visible rather than clipping to peak.
                    latency=cold_io_window(latency,work,target)['duration_us']
                    logical=_rates(work,latency,target)['logical_roofline']
                    tc,tv,tm=(logical[k] for k in ('cube_us','vector_us','logical_io_us'))
                    check(all(v is not None and math.isfinite(v) for v in (tc,tv,tm,latency)),
                          f'Incomplete packed-route quote: {shape}')
                    price_peak=max(tc,tv,tm)
                    check(latency>=price_peak or near(latency,price_peak),f'Packed route below aligned peak: {shape}')
                    price_eff=max(tc/hw.chip.cube.utilization.ratio,tv/hw.chip.vec.utilization.ratio,
                                  tm/hw.chip.hbm.utilization.ratio)
                    durations=(latency,price_peak,price_eff)
                    mx_node_us+=latency
                prior_prices[node.ref]=replace(fine,duration_us=prior)
                for k,us in zip(LANES,durations):
                    mx_prices[k][node.ref]=replace(fine,duration_us=us)
        check(hits==num, 'Unexpected hit count')
        def des(costs):
            return simulate_graph(graph,costs,capacities).duration_us
        stage=dict(ref={k:r[k+'_us'] for k in LANES},legacy_whole=r['legacy_us'],restore=des(old_prices),
                   nodes=hits,full_nodes_us=full_node_us,old_nodes_us=old_node_us,shapes=';'.join(sorted(shapes)),
                   kv_capacity=capacity,pa_block_size=PA_BLOCK,table_pages=pages)
        if packed:
            stage.update(mx={k:des(prices) for k,prices in mx_prices.items()},mx_nodes_us=mx_node_us,
                         prior=des(prior_prices),prior_nodes_us=prior_node_us)
        stages.append(stage)
    check(sum(stage['nodes'] for stage in stages)==n,'Receipt node count mismatch')
    return stages


def scalar_sensitivity(scratch, cache):
    """One observed terminal prefill block: change just the address-lookup service."""
    from operator_model.operators.attention.flash_sparse import price
    from operator_model.roofline import cold_io_window
    from operator_model.sim_backend import OperatorSimBackend
    from sim_platform.backend.l1 import simulate_graph
    spec=('v41-flash',1048576,1,32,'prefill',1044480,4096)
    q=quote(scratch,spec)
    workload,o,_,_=bundle(scratch,spec)
    graph,=workload.graphs
    hw=hardware(o)
    target=OperatorSimBackend()._resolve_chip(hw.name,hw.chip)
    with gzip.open(cache/(case_id(spec)+'.gz'),'rb') as f:
        stored,=pickle.load(f)
    rows=[]
    for cycles in (0,64,256):
        capacity,pages=page_table(scratch,spec)
        costs={}
        candidates=[]
        for n in graph.nodes:
            fine,_,_,candidate=stored[n.ref.node_id]
            costs[n.ref]=fine
            if candidate is not None:
                shape,legacy=candidate
                us,work,plan=price(shape,target,pa_block_size=PA_BLOCK,table_pages=pages,scalar_lookup_cycles=cycles)
                us=cold_io_window(us,work,target)['duration_us']
                costs[n.ref]=replace(fine,duration_us=us)
                candidates.append(dict(node=n.ref.node_id,shape='x'.join(map(str,shape)),
                       full_scan_bf16_us=fine.duration_us,legacy_candidate_us=legacy,
                       packed_candidate_us=us,kv_capacity=capacity,pa_block_size=plan['pa_block_size'],table_pages=plan['table_pages'],
                       address_path=plan['address_path'],interval_cycles=plan['interval_cycles']))
        check(len(candidates)==4,'Single block should contain four consumers')
        baseline=q['stages'][0]['full_us']
        hybrid=simulate_graph(graph,costs,caps(o)).duration_us
        rows.extend(dict(case=case_id(spec),scalar_lookup_cycles=cycles,
                         original_bf16_stage_us=baseline,packed_stage_us=hybrid,**r) for r in candidates)
    return rows


def integrate(series,phase,profile):
    _,length,_,_,chunk=profile
    if phase=='prefill':
        return sum(series[i] if i in series else linear(series,i) for i in range(length//chunk))/1e6
    # Match context_report, including the aligned three-point extrapolation at 1 Mi.
    return sum(linear(series,length+i*ACCEPT) for i in range((OUTPUT-2)//ACCEPT+1))/1e6


def anchor_error(points, phase, length, chunk):
    """Check internal anchors against endpoints; not a device accuracy metric."""
    xs = sorted(points)
    if phase == 'prefill':
        count = length // chunk
        if len(xs) == count:
            return None  # no interpolation used
        anchors = {i:points[i] for i in (1,count//2,count-1)}
        checks = [i for i in xs if i and i not in anchors]
    else:
        if len(xs) != 3:
            return None  # all decode rounds enumerated for this profile
        anchors = {i:points[i] for i in (xs[0],xs[-1])}
        checks = [xs[1]]
    return max(100*abs(linear(anchors,i)/points[i]-1) for i in checks)


def write_note(path,rows,anchors):
    flash = sorted((r for r in rows if r['model']=='v41-flash'), key=lambda r:(r['prompt'],r['batch'],r['ep']))
    has_mx = 'request_packed_full_s' in flash[0]
    lanes_label = '原图 / 仅候选旧价' + (' / MX 场景' if has_mx else '')
    def prices(r,phase):
        lanes = ['reference_full','restore_candidate']+(['packed_full'] if has_mx else [])
        return ' / '.join(f"{r[phase+'_'+lane+'_s']:.3f}" for lane in lanes)
    note = ['# 候选 Indexer 对整批请求耗时的影响（950DT）',
            '比较对象是同一份 rank 0 系统捕获。只更改候选节点的耗时，所有其他节点、依赖、执行资源和容量不变；分别执行离散事件调度，再沿相同的上下文位置对 prefill 与 decode 积分。秒数是部署情景的预测，没有端到端设备实测。',
            '“原图”使用已发布的 BF16 全表评分资源排程；“仅候选旧价”在精细图上替换候选耗时，时长来自系统原 Roofline 对同一节点的报价。CSV 的 `legacy_whole` 才是连同其他算子、缺项和占位都按原后端运行的整图，不能与“仅候选旧价”等同。']
    if has_mx:
        note += ['“MX 场景”改用 MXFP4 稀疏候选评分，假定 Q 已量化、索引 cache 的 key 和 scale 已折叠；不新增上游格式转换，不删减图上已有节点，也不重新计算 KV 分配。PA 分页按每页 128 个位置的部署假设，表宽取任务配置的预留容量（prompt + 1,042），即先按请求选表行，保留所有已分配列；32Ki、128Ki、512Ki、1Mi 分别预留 265、1,033、4,105、8,201 列。后两组从第一个 prefill 块起即选标量地址路径，此前先验取每条记录 64 周期；现默认在宽表同一行片段的相邻 tile 上采用 950DT 单团队 31 个 MMAD 发起间隔的算术平均数 109,555/31≈3,534 周期，片段的首发和尾部仍由排程定价。单 tile 的片段没有相邻间隔。这 31 个间隔的前 8、中 7、末 16 段中位数分别为 5,428、2,749、2,459 周期；全行均值包含起步与续发，不代表硬件恒定带宽。[1Mi/B1 末块另算 0、64、256 周期](context-sensitivity-scalar.csv)，三者是参数情景，不是误差上下限。这是替换 consumer 的条件价格，不表示原 BF16 捕获图已经执行了生产内核。“同账本峰值／原效率”均在这个 MX 场景上重跑 DES；仅候选更换工作量与位宽，其余节点分别复用原图相应账本。原效率采用 Cube 60%、Vector 60%、HBM 80%。']
    note += ['Flash 有 11 组。原始采样锚点中 prefill 共 72 张图、288 个候选调用，decode 共 33 对 target/draft 图、198 个候选调用。另 22 组 Pro/Kimi 的完整算子清单没有候选调用，三条资源账本的阶段与请求结果逐组不变。表内每行代表整批请求，生成 1,024 个输出 token，包含 prefill 的第一个 token；TP 都是 8。',
            f'| 输入 | 本地 B | EP | Prefill 秒：{lanes_label} | Decode 秒：{lanes_label} | 请求秒：{lanes_label} |'+(' MX 64 周期先验 / 同账本峰值 / 原效率（请求秒） |' if has_mx else ''),
            '|---|---:|---:|---:|---:|---:|'+('---:|' if has_mx else '')]
    for r in flash:
        length = '1Mi*' if r['prompt']==1048576 else str(r['prompt']//1024)+'Ki'
        note.append(f"| {length} | {r['batch']} | {r['ep']} | {prices(r,'prefill')} | {prices(r,'decode')} | {prices(r,'request')} |"+
                    (f" {r['request_packed_prior_full_s']:.3f} / {r['request_packed_peak_s']:.3f} / {r['request_packed_effective_s']:.3f} |" if has_mx else ''))
    if has_mx:
        tail,=[a for a in anchors if (a['prompt'],a['batch'],a['phase'],a['history'])==(1048576,1,'prefill',1044480)]
        n=tail['candidate_nodes']
        note.append(f"\n1Mi/B1 最后一个 prefill 块：候选 consumer 平均单次 {tail['candidate_packed_prior_us']/n:,.1f} → {tail['candidate_packed_us']/n:,.1f} µs（64 周期先验 → 指令间隔模型）；整块 DES {tail['packed_prior_full_us']:,.1f} → {tail['packed_full_us']:,.1f} µs。实测包括[单团队开发窗](qsli-one-team.json)与冻结后的[32 行并发单波](prefill-wave.json)。直接用 84.792 µs 的独立单波外推 128 波，按两种启动假设得到 0.484 s 与 0.487 s，[同图复算](wave-chunk.json)。4,096 行整块仍为外推。")
        pmax = max(r['prefill_packed_full_anchor_check_max_pct'] or 0 for r in flash)
        dmax = max(r['decode_packed_full_anchor_check_max_pct'] or 0 for r in flash)
        note.append(f'\nFlash 的 MX 情景对已采样内部位置进行插值检查，prefill 的最大偏差 {pmax:.3f}%，decode 为 {dmax:.3f}%。已采样位置之间没有逐块验证。')
    note += ['', '32Ki 的 prefill 逐块枚举，其余长度的 prefill 和 Flash decode 沿采样点分段线性积分。*1Mi 的 decode 在模型预置上限外，故总时间也含外推。CSV 的 anchor_check 是用保留的内部位置检查插值所得的相对偏差；它不衡量对真机的误差，也不能验证预置上限外的行为。',
             '[33 组逐配置原始秒数、完整历史后端与插值诊断](context-sensitivity-requests.csv) · [Flash 逐阶段锚点、候选输入与微秒账](context-sensitivity-flash-anchors.csv) · [原系统配置和积分方法](../../docs/context-sweep.md)',
             '复算时先按原实验生成可信的本机图和 quote，再从仓库根目录运行（SIMPlatform 环境，缓存放在库外）：',
             '```sh\npython -m calibration.candidate_indexer.context_sensitivity prepare "$scratch" "$cache"\npython -m calibration.candidate_indexer.context_sensitivity report "$scratch" "$cache" --packed --output "$report"\n```']
    # Consecutive Markdown table rows need a single newline.
    text='\n\n'.join(note)
    path.write_text(text.replace('|\n\n|','|\n|')+'\n',encoding='utf-8')


def report(args):
    specs=[s for profile in profiles() for s in inputs(profile)]
    results=jobs(partial(replay,scratch=args.scratch,cache=args.cache,packed=args.packed),specs,args.workers)
    by_spec=dict(zip(specs,results,strict=True))
    with (ROOT/'reference/context-950dt/requests.csv').open() as f:
        frozen={(r['model'],int(r['prompt']),int(r['batch']),int(r['ep'])):r for r in csv.DictReader(f)}
    rows,anchors=[],[]
    for profile in profiles():
        m,l,b,ep,chunk=profile
        original=frozen[profile[:4]]
        row=dict(model=m,prompt=l,output=OUTPUT,batch=b,tp=8,ep=ep,
                 decode_sampling=original['decode_sampling'],decode_positions=original['decode_positions'],
                 fits_144GB=original['fits_144GB'])
        for phase in ('prefill','decode'):
            series={}
            hits=0
            for spec in (s for s in inputs(profile) if s[4]==phase):
                history=spec[5]
                stages=by_spec[spec]
                for i,stage in enumerate(stages):
                    terms={'reference_'+k:stage['ref'][k] for k in LANES}
                    terms.update(legacy_whole=stage['legacy_whole'],restore_candidate=stage['restore'])
                    if args.packed:
                        terms['packed_prior_full']=stage['prior']
                        terms.update({'packed_'+k:stage['mx'][k] for k in LANES})
                    pos=history//chunk if phase=='prefill' else history
                    hits+=stage['nodes']
                    for k,v in terms.items():
                        points=series.setdefault(k,{})
                        points[pos]=points.get(pos,0.)+v
                    if stage['nodes']:
                        anchors.append(dict(model=m,prompt=l,batch=b,ep=ep,phase=phase,
                            stage='prefill' if phase=='prefill' else ('target','draft')[i],history=history,
                            kv_capacity=stage['kv_capacity'],pa_block_size=stage['pa_block_size'],table_pages=stage['table_pages'],
                            candidate_nodes=stage['nodes'],candidate_shapes_bqhd_lck=stage['shapes'],candidate_full_us=stage['full_nodes_us'],
                            candidate_legacy_us=stage['old_nodes_us'],
                            **({'candidate_packed_us':stage['mx_nodes_us'],'candidate_packed_prior_us':stage['prior_nodes_us']} if args.packed else {}),
                            **{k+'_us':v for k,v in terms.items()}))
            row[phase+'_sampled_candidate_nodes']=hits
            for k,points in series.items():
                row[phase+'_'+k+'_s']=integrate(points,phase,profile)
                if k=='restore_candidate' or k=='packed_full':
                    row[phase+'_'+k+'_anchor_check_max_pct']=anchor_error(points,phase,l,chunk)
            for k,cmp in [('legacy_whole','legacy')]+[('reference_'+k,k) for k in LANES]:
                check(near(row[phase+'_'+k+'_s'],float(original[phase+'_'+cmp+'_s'])),
                      f'Archived request total differs for {profile}, {phase}/{k}')
        for k in series:
            row['request_'+k+'_s']=row['prefill_'+k+'_s']+row['decode_'+k+'_s']
        rows.append(row)
    check(len(rows)==33 and len(set(by_spec))==len(specs), 'Incomplete sweep')
    args.output.mkdir(exist_ok=True,parents=True)
    if args.packed:
        write(args.output/'context-sensitivity-scalar.csv',scalar_sensitivity(args.scratch,args.cache))
    write_note(args.output/'context-sensitivity.md',rows,anchors)
    write(args.output/'context-sensitivity-requests.csv',rows)
    write(args.output/'context-sensitivity-flash-anchors.csv',anchors)

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','report'))
    p.add_argument('scratch',type=Path)
    p.add_argument('cache',type=Path)
    p.add_argument('--packed',action='store_true',help='also price the MXFP4 QSLI deployment scenario')
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args=p.parse_args()
    specs=[s for profile in profiles() for s in inputs(profile)]
    if args.mode=='prepare':
        args.cache.mkdir(parents=True,exist_ok=True)
        result=jobs(partial(prepare,scratch=args.scratch,cache=args.cache),specs,args.workers)
        print(f'{sum(result)} of {len(result)} graph captures have candidate nodes')
    else:
        report(args)
