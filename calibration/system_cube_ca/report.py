"""Render matrix development and frozen holdout rows from complete receipts."""
import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
COMPARISON = '## 同形状双端 CA 对照'
DT,PR,TARGET = 'Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'


def read(name):
    return json.loads((BASE/name).read_text())


def report():
    frozen = read('frozen.json')
    parts = [(read('measured.json'),frozen['development'],'开发')]
    if (BASE/'independent.json').is_file():
        parts.append((read('independent.json'),frozen['independent'],'冻结留出'))
    if (BASE/'router.json').is_file():
        pred = {sku:[row] for sku,row in read('router-current.json')['development'].items()}
        parts.append((read('router.json'),pred,'Stream-K 开发'))
    if (BASE/'router-independent.json').is_file():
        pred = {sku:[row] for sku,row in read('router-current.json')['independent'].items()}
        parts.append((read('router-independent.json'),pred,'Stream-K 新留出'))
    lines=[COMPARISON,'','下表的误差为 (模型 − CA)/CA。同一组行使用相同的逻辑 M,N,K、转置视图和输出精度。两端的芯片时钟都为 1.65 GHz；最后一列始终为无目标 CA 的外推。','',
           '| 数据角色、接口、输入→输出 | M,N,K | DT 模型 / CA µs | DT 误差 | PR 模型 / CA µs | PR 误差 | DT / PR 实际工作 AIC | 等效逻辑 Roofline U，DT / PR | 970DT 外推 µs |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for group,preds,role in parts:
        for i,(d,p) in enumerate(zip(group[DT]['cases'],group[PR]['cases'])):
            assert d['shape_MNK']==p['shape_MNK']==list(preds[DT][i]['shape_MNK'])
            da,pa=d['ca_us'],p['ca_us']
            dm,pm=preds[DT][i]['predicted_us'],preds[PR][i]['predicted_us']
            dt = d['dtype'].replace('_','→')
            util = 'N/A'
            if d['dtype'] in ('BF16','BF16_FP32'):
                m,n,k = d['shape_MNK']
                size = 2*(m+n)*k + m*n*(2 if d['dtype']=='BF16' else 4)
                chips=Path(__file__).resolve().parents[2]/'src/operator_model/data/chips'
                ratios=[]
                for sku,model in ((DT,dm),(PR,pm)):
                    cfg=json.loads((chips/(sku+'.json')).read_text())
                    floor=max(2*m*n*k/(cfg['cube_tops']['16']*1e6),size/(1000*cfg['hbm_bandwidth_gbs']))
                    ratios.append(f'{floor/model:.1%}')
                util=' / '.join(ratios)

            lines.append(f"| {role} · {d['api']} {dt} | {','.join(map(str,d['shape_MNK']))} | {dm:.3f} / {da:.3f} | {(dm/da-1):+.1%} | {pm:.3f} / {pa:.3f} | {(pm/pa-1):+.1%} | {d['productive_aic']} / {p['productive_aic']} | {util} | {preds[TARGET][i]['predicted_us']:.3f} |")
    lines += ['','U 取逻辑 Cube 理想时间与输入输出字节在峰值 HBM 带宽下时间的最大值，再除以模型设备时间。内置两款 950 缺少严格 FP32 峰值，该精度的 U 为 N/A。逻辑输入字节数并非实测 HBM 流量。','','| 数据组 | 双端完整窗数 | DT WAPE / 最大绝对相对误差 | PR WAPE / 最大绝对相对误差 | CA 采集跨度，DT / PR 秒 |',
              '|---|---:|---:|---:|---:|']
    for group,preds,role in parts:
        stats=[]
        for sku in (DT,PR):
            pred=[r['predicted_us'] for r in preds[sku]]
            truth=[r['ca_us'] for r in group[sku]['cases']]
            err=[abs(x-y) for x,y in zip(pred,truth)]
            stats.append(f'{sum(err)/sum(truth):.1%} / {max(e/y for e,y in zip(err,truth)):.1%}')
        lines.append(f"| {role} | {2*len(pred)} | {stats[0]} | {stats[1]} | {group[DT].get('profiling_wall_seconds',group[DT]['profiler_log_span_seconds']):.0f} / {group[PR].get('profiling_wall_seconds',group[PR]['profiler_log_span_seconds']):.0f} |")
    replay=read('replay.json')
    unique=replay['unique_normalized_contracts']
    counts=sum(sum(t.values()) for case in replay['cases'].values() for t in case.values())
    wall=[r['model_query_us'] for role in frozen.values() if isinstance(role,dict) for cases in role.values() for r in cases]
    lines+=['',f'27 配置的 {replay["captures"]} 张 target/draft 图含 {counts:,} 个矩阵节点。去除模块路径与分区标签的重复后，共 {sum(unique.values())} 个归一化输入，覆盖 {len(unique)} 种合同，均返回有价窗口。核对使用系统的实际 OpFeatures 转换；FP4 的物理字节槽、逻辑 shape 和 stride 都参与检查。此回放检验覆盖，耗时误差只来自上面的同合同校准。冻结时每次 Python 排程测得 {min(wall):.0f}–{max(wall):.0f} µs（已载入芯片配置，不含图规范化和 CA）。[逐配置、逐阶段、逐合同回放计数](../../calibration/system_cube_ca/replay.json) · [回放程序](../../calibration/system_cube_ca/replay.py)',
            '', '两种 FP32 Mm 开发投影在 DT/PR 均观测到跨核分 K。以 DT 的 [9,4,20480] 为例，32 个 AIC 各计算 K640，实测每核一片 K512、一片 K128。7.535 µs 整窗中最早 MMAD 发起在 4967 cycles，最后 MMAD 结束在 8173 cycles，随后还有 4259 cycles。模型用统一启动和退休预算，低估该窄列分核调用。N384 的 [9,384,5120] 在 DT 的物理 MMAD 为 [9,192,32]，PR 上共享存储时间对体阶段的影响更大。实际共享层命中、各核到达和 MMAD 之间的间隔仍能产生误差。',
            '', '验证程序对每个输出元素逐位检查。A 的每一行选择不同的 K，W 的数据同时依赖列和 K；Addmm 另外加入非零 bias。探针记录实际 SKU、内存模型、每核首指令与唯一 END 退休点，校验 dump ELF 与 OPP 中文件的 SHA-256，并统计真正执行 MMAD 的 AIC。只计作一次稀疏输入正确性检查，尚未覆盖一般数值分布、累加顺序和特殊浮点数。宿主采集跨度含模拟及解析，独立留出期间机器并发负载较高。',
            '', 'CANN 9.2.1 的 FP32 两条 API 在开发窗均调到 ops-nn 的 `MatMulV3_ND_ND_ND_ND_FP32_FP32_FP32_FP32_high_performance_4162_mix_aic`。BF16→BF16 开发窗为 `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_66`。各行的 kernel、ELF、MMAD 维度与计数、时间戳摘要都在对应 receipt 中。冻结后对系统张量角色归一化的修复和外推状态标记不改变表中的排程或冻结价格。[开发 CA](../../calibration/system_cube_ca/measured.json) · [冻结预测、工作量与分核](../../calibration/system_cube_ca/frozen.json) · [CA 探针](../../calibration/system_cube_ca/probe.cpp) · [窗口与 ELF 核查](../../calibration/system_cube_ca/receipt.py)']
    if (BASE/'independent.json').is_file():
        lines+=['','留出集中，DT 的 Addmm [128,1,5376] 低估 32.4%。该次首个 MMAD 与末次 MMAD 后退休分别为 2786/2575 cycles，高于模型继承的小行数 bias 启动/退休 1932/2240；每 AIC 三次 MMAD，核间错位与发起间隔也未完全暴露在各段预算里。实测该 shape 为 PR 更快，模型判成 DT 更快。BF16 [144,128,512] 两端都用 24 个 AIC，各 MMAD [48,16,256]；当前启发式选择 16 个 AIC 并给出相同的双端时间。留出误差保留原冻结结果。','',
                  '独立 CA 明细保留各 SKU 的内核、启动核数、生产核数和整窗。单次分核规则可能在新几何选错，样本数未达到每合同十个独立 shape。[冻结留出 CA](../../calibration/system_cube_ca/independent.json)']
    if (BASE/'router.json').is_file():
        lines+=['','Kimi [9,896,7168] 第一次冻结采用 BF16 输出的分核近似，DT/PR 为 4.654/10.693 µs，低估 53.2%/18.9%。两端 kernel 实际均为 `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_FP32_high_performance_4162_mix_aic`，有 36 个 AIV 执行 RV_VADD。这个样本用于识别并校准正式版的 FP32 输出 Stream-K，表中“Stream-K 开发”为当前方法在原样本上的回放。独立数据另取 27 组里尚未使用的最小 M=144。[首次冻结，历史对照](../../calibration/system_cube_ca/router-frozen.json) · [当前开发和留出冻结价格](../../calibration/system_cube_ca/router-current.json) · [开发原生 CA](../../calibration/system_cube_ca/router.json)']
    if (BASE/'router-independent.json').is_file():
        lines+=['','M144 按冻结结果评估，两端各 129,024 个 FP32 输出逐位正确，DT/PR 均正确预测 32/28 个 AIC。两端实际 MMAD 是 [144,224,64]，模型把 M 按 128+16 两块分工，从而给每个 M,N 任务配置更少的 K 分片，代价偏高。实测 64/56 个 AIV 上有 RV_VADD；在线模型保留归并操作量和时序预算，生产 AIV 数未知。DT/PR 的跨芯片加速比按 PR 时间除以 DT 时间计算，预测 1.154、实测 1.243。[Stream-K 新留出 CA](../../calibration/system_cube_ca/router-independent.json)']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('doc',type=Path)
    doc=parser.parse_args().doc
    doc.write_text(doc.read_text().split(COMPARISON,1)[0].rstrip()+'\n\n'+report())
