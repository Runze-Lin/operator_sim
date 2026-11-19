"""Render the verification summary from the frozen candidate and archived receipts."""
from audit import observations, benchmark, save, check_freeze
from model import OUT, ROOT, DT, read
from pathlib import Path

def title(chip):
    return chip.removeprefix('Ascend').split('_')[0]

def join(xs):return ' / '.join(f'{x:.3f}' for x in xs)
def text(case):return '×'.join(str(v) for v in case)
if __name__=='__main__':
    check_freeze()
    rows=observations();save('development-check.json',rows)
    pre=read(OUT/'prefill-projections.json')
    report=[
        '# FLA KDA V2 候选的设备资源与误差',
        '',
        '| 算子 | 950DT 合格开发形状 | DT 开发 WAPE / 最大 | 950PR 合格开发形状 | PR 开发 WAPE / 最大 | 独立验证 |',
        '|---|---:|---:|---:|---:|---|']
    forward=read(ROOT/'results/kda-v23/holdout-comparison.json')
    assert forward['prediction_freeze_sha256']==read(OUT/'frozen-annotations.json')['freeze_sha256']
    independent=[]
    for sku in (DT,'Ascend950PR_957b'):
        items=[r for r in forward['rows'] if r['qualified'] and r['chip']==sku]
        if not items:
            independent.append(title(sku)+' 0 / N/A')
            continue
        wape=100*sum(abs(r['prediction_us']-r['ca_us']) for r in items)/sum(r['ca_us'] for r in items)
        worst=max(abs(r['signed_pct']) for r in items)
        independent.append(f'{title(sku)} {len(items)} 形 {wape:.2f}% / {worst:.2f}%')
    metrics=[]
    for sku in (DT,'Ascend950PR_957b'):
        valid=[r for r in rows if r['qualified'] and r['case']['sku']==sku]
        err=[abs(r['predict_us']-r['full_window_us']) for r in valid]
        metrics.append(f"{100*sum(err)/sum(r['full_window_us'] for r in valid):.2f}% / {100*max(e/r['full_window_us'] for e,r in zip(err,valid)):.2f}%")
    report.extend([f'| Chunk KDA V2，三段设备服务 | {sum(r["qualified"] and r["case"]["sku"]==DT for r in rows)} | {metrics[0]} | {sum(r["qualified"] and r["case"]["sku"]!=DT for r in rows)} | {metrics[1]} | {"，".join(independent)}（WAPE / 最大） |',
        '', 'S2H1 的预测 DT/PR 加速比为 0.9987，CA 为 1.0462。按相同 shape 两端独立预测，当前候选把这个 shape 的 DT 和 PR 快慢方向判反。S65 另有同形双端开发窗口，见下表；两种小输入不足以验证跨芯片加速比。正式发布库、设备代码和数值资格由每行的原回执保存。公开入口仅在调用方显式选择 FLA 26.9.1 V2 时启用，结果标记为实验外推。[模型和公式](modeling.md) · [逐形回放与计数](development-check.json)',
        '', '![950DT 的 KDA 模型与实测三段时序](dt-s2-stage-comparison.png)',
        '', '上面展示的 V0、V1、C2 等行对应源码工作单元。上图按冻结模型的源码相位依赖展开读、写、Vector 及 Cube 服务预算，底图同色显示相应单元的实际指令时间。每列的零点是该 kernel 所有活动核的最早指令，下方虚线止于最后退休。实测下图七条紫色刻线对应七次 MMAD。上图 Prepare C7 的同价 W/U 显示为一组 Cube 服务预算。CA 整链第一条指令至最后退休为 28.034 µs，含两段 dispatch 间隔。模型所比口径为三段窗口之和 23.241 µs。[原始 trace SHA 和阶段明细](dt-primitive-services.json)',
        '', '## 每个形状的证据',
        '',
        '| SKU | B,S,Hq,Hv,K,V | 资格 | 模型 Prepare/FwdH/Finalize µs | 实测或诊断窗 µs | 模型/合格 CA 总 µs | 3 段核数 / MMAD 维度与次数 / 各路径字节一致 |',
        '|---|---|---|---|---|---|---|'])
    for r in sorted(rows,key=lambda r:(r['case']['shape'],r['case']['sku'])):
        c=r['case'];z=r['stages']
        equal=[all(s['observed_aic_aiv']==s['predicted_aic_aiv'] for s in z),all(s['equal_mmad'] for s in z),all(s['equal_dma'] for s in z)]
        prefix='官方 BF16 精度协议，' if r['case']['id'].startswith('development-') else '首轮协议，'
        status=prefix+('合格开发' if r['qualified'] else '只计结构诊断')
        rel='../kda-v23/'+Path(r['receipt']).name
        ca=f"{r['full_window_us']:.3f}" if r['qualified'] else 'N/A'
        report.append(f"| {title(c['sku'])} | [{text(c['shape'])}]({rel}) | {status} | {join(s['predict_us'] for s in z)} | {join(s['ca_us'] for s in z)} | {r['predict_us']:.3f} / {ca} | {' / '.join('是' if x else '否' for x in equal)} |")
    report.extend(['', 'S65 的 Prepare 和 Finalize 用两组，递归 FwdH 仍用一组。该形状在两端的 FwdH MMAD/搬运字节匹配，DT 预测 9.123 µs，诊断窗 7.277 µs。递归状态生成、前后块的 W/K 预读和数据槽重用可重叠，当前相位模型只重叠部分 S2 和 Stage3。CA 的 RVEC 子流水共 7,395 条，模型近似 6,368 条。模型低估记录数却高估时延，说明单靠调整向量吞吐不能解释差异。数值失败例仍排除于端到端误差。原始失败记录与更正参考后新运行的 BF16 逐操作协议分别保存，同 shape 的两个回执没有互相覆盖。[原数值诊断和开发资格](../kda-v23/report.md)',
        '', '950DT S2 Prepare 总前端 9,952 周期后开始首笔 payload。此后 V0/V1 至 C2 的实际数据依赖使C2 的 MMAD 发起比模型靠后，模型低估 Prepare 1.612 µs。FwdH 的 W 读取早于首轮初态转换完成，部分 S2 等待与其他通道推进并行；这说明简单串行队列预算偏保守。PR 与 DT 物理外存分别为 LPDDR5、HBM3，两端核数相同的小例中主要差异来自前端和首搬运响应。候选复用了 DT 启动和响应参数，PR 的 Prepare/Finalize 因而没有得到实测中的全部提前量。这里能定位偏差发生的时间段，更细的冷态、队列发起和测量环境各自贡献尚未识别。',
        '', '## Kimi 长 prefill 外推',
        '',
        '下面按一个完整、不切 chunk 的调度批次推导，单卡 attention DP 副本 B=8，每请求 S 取 32/128/256 Ki token，采用 Kimi K3 KDA 本地 Hq=Hv=12（TP8），K=V=128；64 是 KDA 内部 chunk。若 serving 采用 chunked prefill，每次传入的是该批实际调度的 token 和 cu_seqlens，不能将下表当作一次线上 kernel 的 CA。长表采用多头串行服务近似和可替换芯片资源，AIV 参与的 Finalize、长循环和重叠均待独立检验。970 全部为外推。',
        '',
        '| SKU | 单请求 S | 准备 / 递归 / 输出 ms | 合计 ms | 最大每核 head×chunk（同顺序） | 模型墙钟中位 ms |',
        '|---|---:|---:|---:|---|---:|'])
    for r in pre:
        z=r['stages'];report.append(f"| {title(r['chip'])} | {r['shape'][1]:,} | {join(s['device_us']/1000 for s in z)} | {r['device_us']/1000:.3f} | {' / '.join(str(s['max_core_head_chunks']) for s in z)} | {r['median_query_ms']:.3f} |")
    report.extend(['',f"各组合查询 30 次，本机已加载候选和 JSON 参数，观测到最慢 {max(r['max_query_ms'] for r in pre):.3f} ms。计算只遍历目标核数和有限 tile 类型，长上下文不会逐 token 展开。表中在固定 batch、每核多轮、满块占主导时近似线性；这是工作计数的可解释结果，当前缺同形 CA 来证明这条斜率。Prepare 和 Finalize 多头交错以及 HBM 工作集跨 L2 的行为可改变斜率。[每个情景的阶段、核数、工作量、带宽下界和查询开销](prefill-projections.json)",
        '', '## 冻结与复算',
        '',
        '首批冻结清单包含四个新 shape 的双端和 970 推演、实现与参数 SHA256。补充八个 shape 在新采集前另行封存，建模参数保持原冻结。[补充选择及预测](../kda-v23/supplement-predictions.json)T3H1 和 T5H2 的双端 CA 由独立验证作业管理。开发系列中的 T15/T63/T64/T65 一律保持其原登记角色。[冻结清单](freeze.json) · [原始采集](../kda-v23/report.md)',
        '',
        '在仓库根目录运行以下命令可以核对冻结 SHA，并重建离线结果。第一条命令仅提取已有 trace。图使用环境中的 Matplotlib。',
        '', '```bash','python3 development/validation/kda_resource23/extract.py','python3 development/validation/kda_resource23/audit.py','python3 development/validation/kda_resource23/report.py','toolchains/envs/cann-9.2.1/bin/python development/validation/kda_resource23/timeline.py','```',''])
    section=['## 冻结形状独立验证','','每条合格的新登记与冻结文件中的 SKU 和完整 shape 逐项匹配。逐行的全输出 golden、窗口、真实 ELF 和核数由原回执保存。[全部独立回执及 SHA](../kda-v23/holdout-comparison.json)','',
             '| SKU | B,S,Hq,Hv,K,V | 固定预测 µs | 合格 CA µs | 有符号误差 |','|---|---|---:|---:|---:|']
    for item in forward['rows']:
        if item['qualified']:
            section.append(f"| {title(item['chip'])} | [{text(item['shape'])}](../kda-v23/{item['receipt']}) | {item['prediction_us']:.3f} | {item['ca_us']:.3f} | {item['signed_pct']:+.2f}% |")
    section.append('')
    pairs={}
    for it in forward['rows']:
        if it['qualified']:pairs.setdefault(tuple(it['shape']),{})[it['chip']]=it
    for shape,skus in sorted(pairs.items()):
        if DT in skus and 'Ascend950PR_957b' in skus:
            dt=skus[DT];pr=skus['Ascend950PR_957b']
            rat=dt['prediction_us']/pr['prediction_us'];obs=dt['ca_us']/pr['ca_us']
            section.extend([f"独立 T{shape[1]} H{shape[2]} 的 DT/PR 耗时比预测 {rat:.4f}，实测 {obs:.4f}，相对误差 {(rat/obs-1)*100:+.2f}%。", ''])
    for label in ('950dt','950pr'):
        for tokens,heads in ((3,1),(5,2)):
            fn=f'{label}-t{tokens}-holdout-comparison.png'
            if (OUT/fn).is_file():
                section.extend([f'![正式冻结 T{tokens} H{heads} 的 {label.upper()} 三段模型与实测]({fn})',''])
    index=report.index('## Kimi 长 prefill 外推')
    report[index:index]=section
    (OUT/'report.md').write_text('\n'.join(report))
