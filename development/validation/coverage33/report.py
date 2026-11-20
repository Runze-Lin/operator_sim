"""Rebuild the public large-route matrix and RMS case from qualified receipts."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.33/large'
OLD=ROOT/'results/v0.32/large/coverage.json'
CHIPS=('Ascend950DT_9582','Ascend950PR_957b')
FUT='Ascend970DT'

def get(path):
    return json.loads(path.read_bytes())

def md(path):
    return str(path.relative_to(ROOT))

def save(path,text):
    temp=path.with_suffix('.tmp')
    temp.write_text(text)
    temp.replace(path)

def render(prefix):
    if not sys.flags.isolated:
        raise ValueError('use Python -I and an installed wheel prefix')
    sys.path.insert(0,str(prefix))
    import operator_model as api
    if not Path(api.__file__).is_relative_to(prefix):
        raise ValueError('installed wheel not used')
    old=get(OLD)
    states={op:sorted(shapes) for op,shapes in old['complete_same_shape_pairs_by_op'].items()
            if op not in ('kda_chunk_fused_v1','kda_recurrent')}
    costs=[]
    for t,d in ((8192,5120),(32768,1280)):
        endpoints=[OUT/f'rms-{t}-{d}-{tag}.json' for tag in ('dt','pr')]
        if not all(p.is_file() for p in endpoints):
            continue
        pair=[]
        for p,c in zip(endpoints,CHIPS):
            r=get(p)
            if not(r['qualified'] and r['shape']==[t,d] and r['chip']==c):
                raise ValueError('unmatched receipt '+str(p))
            model=api.estimate('rmsnorm',[t,d],chip=c,dtype='bf16')
            future=api.estimate('rmsnorm',[t,d],chip=FUT,dtype='bf16')
            if not (model.latency_us and future.latency_us):
                raise ValueError('registered RMS geometry missing a forecast')
            hist=Counter((core['rows'],sum(core['y_tiles_by_row_count'].values())) for core in r['cores'].values())
            pair.append(dict(chip=c,shape=r['shape'],ca_us=r['ca_us'],price_us=model.latency_us,
                signed_pct=100*(model.latency_us/r['ca_us']-1),extrapolated_970dt_us=future.latency_us,
                historical_price_us=r['public_v032_us'],original_ca_minutes=r['cost']['seconds']/60,
                recheck_wall_s=r['archive_audit_wall_s'],active_aiv=r['active_aiv'],
                rows_tiles_cores=sorted([[x,y,z] for (x,y),z in hist.items()]),trace_sha256=r['trace_sha256'],
                trace=r['trace'],receipt=md(p),work=model.work,
                plan=model.projection['target_plan'],stage_budgets_us=model.projection['stages_us']))
        costs.append(pair)
        if [t,d]==[8192,5120] and [t,d] not in states['rmsnorm']:
            states['rmsnorm'].append([t,d])
    count=sum(bool(v) for v in states.values())
    registered=len(states)
    # The remaining implementation is ordinary single-chain recurrent. The only
    # previously registered large KDA recurrent geometries had multiple chains.
    states['kda_recurrent (single chain)']=[]
    lines=['# 真实大形 CA 覆盖','',
        f'本轮保留 {len(states)} 条实现，其中 {count} 条已有同一大形、相同合同的双端完整设备计时。历史大形登记中仍在范围内的有 {registered} 条，其中 {count} 条有双端完整窗。第 21 条是单状态链 recurrent KDA，其已测 S=8 边界样本与大形验收分别统计。已豁免多链 recurrent 与 chunk fused V1，V2 仍在表中。','',
        'RMSNorm 新增的是对过往真实采集的严格复核。原 large29 清单已经登记了 Flash 每次调度 8192 个 token、hidden=5120；此处沿用它的大形口径。已测 8192 行，B8 不切 chunk 时的 262144、1048576、2097152 行仍待测。','',
        '| 路线 | 已有双端合格大形 | 核对依据 |','|---|---|---|']
    for op,items in sorted(states.items()):
        shown='、'.join('['+','.join(map(str,s))+']' for s in items) if items else 'N/A'
        source='[本轮逐核复核](#rmsnorm-大形)' if op=='rmsnorm' else '[上一版逐形资格与端点](../../v0.32/large/report.md)'
        if op=='kda_recurrent (single chain)':
            source='[S≤8 的单链控制，旧登记无单链大形](#单状态链-kda)'
        lines.append(f'| {op} | {shown} | {source} |')
    lines+=['', '### 其它优先排查的 Vector 路线', '',
        '| 算子 | 已查到的对照 | 尚缺的双端大形 |',
        '|---|---|---|',
        '| DynamicMxQuant | [2048,5120] 双端指令窗，Kimi [262144,7168] 的 DT 已有数值和 wrapper 整窗，独立于指令窗口径 | [16384,7168] 双端指令、[262144,7168] 同合同双端完整计时 |',
        '| MoE gating top-k | 两端真实 896 专家、top-16 的既有控制为 24 token | 既有登记 [4096,896,16] |',
        '| AttnResFwd | 既有 7168 通道小形使用 materialize 或不同写回属性；不能替代 Kimi 的 in-place prefill 合同 | 已登记 [512,7168,8,1]、write_idx=-1、materialize=false |',
        '', '本轮优先找回已登记且已有真实测量的调用。[原 Vector 合同、真实参数和既有采集队列](../../v0.29/large/vector/report.md)。']
    lines+=['','## 单状态链 KDA','',
        '单链解码合同 B=Hq=Hv=1、K=128、V=128 或 256、S≤8。Kimi 的 V=128 路径已测到 [1,8,1,1,128,128]，DT/PR 完整窗 35.396/39.450 µs；这是单核八步序列边界，状态张量为 64 KiB。旧大形登记采用的是 B6、16、128 且每请求含 12 条状态链。单链在此表保留为大形 N/A；保留的单链准确性与适用域见[逐形回放](../../v0.32/replay.json)。','',
        '## RMSNorm 大形','',
         '表内 950DT/PR 的 CA 是从全核最早指令发起到最后指令退休的微秒数。耗时预测使用隔离安装的 '+api.__version__+'，按调用方只传 shape、dtype、芯片的入口定价。所有实测属于开发回放，970DT 一列只作外推。','',
         '| 形状 | 芯片 | 模型 / CA µs | 有符号误差 | 970DT 外推 µs | 参与 AIV | 单核实测行数 / 块数 × 核数 | 原 CA 墙钟 |',
         '|---|---|---:|---:|---:|---:|---|---:|']
    for rows in costs:
        for r in rows:
            dist='，'.join(f'{x} / {y} × {z}' for x,y,z in r['rows_tiles_cores'])
            lines.append(f"| {r['shape']} | {r['chip']} | {r['price_us']:.3f} / {r['ca_us']:.3f} | {r['signed_pct']:+.2f}% | {r['extrapolated_970dt_us']:.3f} | {r['active_aiv']} | {dist} | {r['original_ca_minutes']:.1f} min |")
    ratios=[]
    lines+=['','| 几何 | 950DT → PR 实测加速比 | 模型加速比 | 加速比误差 |',
              '|---|---:|---:|---:|']
    for dt,pr in costs:
        observed=dt['ca_us']/pr['ca_us'];predicted=dt['price_us']/pr['price_us']
        ratios.append(dict(shape=dt['shape'],profile='PR over DT speedup = DT time / PR time',actual=observed,
                           predicted=predicted,signed_pct=100*(predicted/observed-1)))
        lines.append(f"| {dt['shape']} | {observed:.4f}× | {predicted:.4f}× | {ratios[-1]['signed_pct']:+.2f}% |")
    lines+=['',
         '第一个 shape 对应 Flash 归一化的实际 hidden 和 B4 × 2048 已调度 token，也可组成 B8 × 1024 的单步调度。后者是将同一输入形状映射到另一个请求调度组合。[32768,1280] 具有相同元素总数，是宽度/行数控制样本，不能代表对 32Ki 整上下文的全 hidden 归一化验证。','',
         '每个端点实查安装包 ops-nn 的 RmsNorm_normal_all_bf16_high_performance_5000、采集对象 .text、物理 SKU 与内存模型、完整 BF16 Y 与逐行 FP32 rstd 结果。原始合并 trace 的 SHA-256、每个 AIV 唯一物理 END、供应商各核 duration、实际 MTE3 描述符写出的 Y 和 rstd 总量均通过。[32768,1280] 的 PR 有核缺 profiler 的 END_LABEL 附注，对该核另完整扫描所有 X 事件，以最早发起和最晚退休交叉核对。逐核指令汇总 CSV 会把尾块与满块计到同一 PC，只能从原逐条描述符取实际 burst_number 和 processed_bytes。','',
         '计数遗漏来自原 large29 聚合只导入 vector/archive 下的 wrapper 和特定 Cast/Sigmoid 的大形回放，而后续分级表没有联入 v0.28 的 RMSNorm msopprof 大形库。这里复用同一次 CA 的已验证物理身份；数据库的历史误差分母与独立样本数不增加。两条旧采集登记已写跳过回执，明确本次未发起 CA、没有本次耗时，原预测与外部合格 trace 分别保存。[DT 去重回执](../../v0.29/large/vector/instruction-rmsnorm-8192-5120-Ascend950DT_9582.json) · [PR 去重回执](../../v0.29/large/vector/instruction-rmsnorm-8192-5120-Ascend950PR_957b.json)','',
         '[冻结的大形判据与原采集来源](freeze.json) · [计算过程与具体案例](../../../docs/examples/rmsnorm-large.md) · [重建脚本](../../../development/validation/coverage33/rms_archive.py)','']
    save(OUT/'report.md','\n'.join(lines))
    save(OUT/'coverage.json',json.dumps(dict(version=api.__version__,counts=dict(paired=count,registered=registered,retained=len(states)),
        base=md(OLD),large_shapes=states,rms_pairs=costs,rms_speedups=ratios,registration_sha256=sha256((OUT/'freeze.json').read_bytes()).hexdigest(),
        case_role='archived replay; no new independent samples; 970DT extrapolation'),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
if __name__=='__main__':
    render(Path(sys.argv[1]).resolve())
