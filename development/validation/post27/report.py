"""Produce development, prospective and exact-repeat deployment evidence on separate tables."""
from pathlib import Path
from statistics import median
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post27.campaign import OUT,score
from development.validation.post24.report import stats,fmt
from development.validation.sku19.campaign import read

def main():
    from development.validation.post27.physical import main as recompute_physical
    recompute_physical()
    src,r,summary,pairs,precise=score();a,b=src['measured'];dev=read(OUT/'archive.json')['rows'];p=read(OUT/'parameters.json')
    md=['# 内置 Post 的派发入口、首轮搬运和多轮发起','', '| 芯片 | 精确重复的独立不同 shape | v0.27 默认重复 WAPE / 最大 | v0.26 旧默认重复 WAPE / 最大 | 新的非重复影子完整窗 | 新影子 WAPE / 最大 | 默认非重复 WAPE / 最大 |','|---|---:|---:|---:|---:|---:|---:|']
    for s in (a,b):
        orig=precise['summary'][s]['repeat'];row=summary[s];md.append(f"| {s} | {precise['all_core_repeat_shapes'][s]} | {fmt(orig['policy_us'])} | {fmt(orig['public_us'])} | {len([x for x in r if x['chip']==s])} | {fmt(row['us'])} | {fmt(row['base'])} |")
    from development.validation.post27.deployment import main as deployment
    gate=deployment();passed=gate['qualified_for_default'];md+=['',f"精确重复的发布门槛是每端至少 10 个合格的新形，重复样本 WAPE≤20%、最大≤40%、WAPE 优于旧默认。当前{'满足' if passed else '尚未满足'}。该判定只针对同一有序流上前一次成功派发完全相同的内置 BF16 Post。普通热态不会凭 AIV 计数变成精确重复，未知前序另行计价。新非重复影子只用 v0.26 已注册且在本次冻结时两端都尚未开始的 multi/reserve，备用组是否执行只看原队列覆盖。当前同序号双端前向对照 {len(pairs)} 对。", '',
        '## 可观察的前序与价格','',r'同一个已知 AIV 驻留数 $H$ 可以对应不同前序。库的通道分核入口为 ELF 后缀 _5，token 分核为 _6。对当前 token 路线将连续成功派发映射到前一入口类型，再结合 $H=0$、$0<H<A$ 或 $H\ge A$ 得到首次、扩核或复用。A 是当前分配的活动 AIV 核数；具体先前 shape 的值只用于计算前一入口，不作为时间索引。整个提交历史不完整时不套用这次校准。','',
        r'把每行从 MTE2 发起到最后 MTE3 完成的响应与下一行可以发起的间隔分开。第一到第二行的发起可能还含入口切换、初始化等等待；按当前状态与前一入口独立标定。第三行起使用跨上下文共享的稳态价格 $I_q=\max(I_{q,0},C_{vec}+\tau_q)$，q 为读或写队列。$C_{vec}=31.7754\lceil D/64\rceil\rho$ 是已冻结的每 64 通道向量组服务周期，$\rho$ 为参考与目标每 AIV 每周期 FP32 吞吐比。$I_{q,0}$ 是描述符/队列前端的最低发起周期，$\tau_q$ 是向量计算后的交接尾项。各自估计自多个完整窗口相邻有效描述符的发起时刻。全芯片共享 L2/HBM 及每 AIV 本地 MTE 字节服务仍作为独立下界。', '',
        f"26 个可观测稳态窗口给出读 $I_0$={p['curves']['shared']['read_ii']['minimum']:.1f} 周期、尾项 {p['curves']['shared']['read_ii']['fixed_compute_tail']:.1f} 周期；写对应 {p['curves']['shared']['write_ii']['minimum']:.1f}、{p['curves']['shared']['write_ii']['fixed_compute_tail']:.1f} 周期。每个窗口先取后续间隔中位数，避免有很多行的单次采集支配参数。不同芯片使用各自频率将周期转为 µs。拟合对象始终是具体阶段或描述符的发起差。", '',
        '## 历史轮次证据与开发回放','', '| 当前状态/前次入口 | 设备窗 | 有真实首个读发起间隔的窗口 | 首读到下一行首读中位 µs | 默认 WAPE / 最大 | v0.25 逐行 WAPE / 最大 | 新影子 WAPE / 最大 |','|---|---:|---:|---:|---:|---:|---:|']
    for k in ['all']+sorted({v['cohort']for v in dev}):
        ss=dev if k=='all' else [v for v in dev if v['cohort']==k];g=[x['read_gaps_us'][0] for x in ss if x['read_gaps_us']]
        md.append(f"| {k} | {len(ss)} | {len(g)} | "+(f'{median(g):.3f}' if g else 'N/A')+' | '+' | '.join(fmt(stats(ss,key)) for key in ['public_us','old_pipeline_us','candidate_us'])+' |')
    md+=['','这些数据已在以前的迭代重复使用。每次评价新的发起曲线时，整个待评 shape 在两端的测量、以及待评进程的全部形状一起剔除；更早的固定端口响应仍沿用 v0.25 价格。本表属于开发回放。不同前次入口样本的宽度和活动核数也有变化，观测到的间隔差不能唯一归因于 I-cache 或某种特定缓存。','',
       '以 D=1536 说明。在 950DT 上，由通道入口 _5 切到 _6 的 [128,4,1536]、[256,4,1536]，第一笔到第二行首读间隔分别 1.037 和 0.622 µs；连续 _6 的 [196,4,1536] 则为 0.460 µs。旧 [128/192/256,4,1024] 的前一 shape 同为 [4,4,6144]、走 _5，先前只看 H=64 会将这些不同的派发状态混合定价。例中当 D=1536 时共有 24 个 64 通道组，同 950 每核条件下 $C_{vec}$≈762.6 周期，稳态读预算 max(554.6,762.6+189.0)/1650≈0.577 µs，写对应约 0.530 µs。','',
       '## 真正未见过的形状','', '| SKU | T×4×D | 前序合同 | 新影子 µs | v0.25 逐行 µs | 默认 µs | CA µs | 新影子偏差 |','|---|---|---|---:|---:|---:|---:|---:|']
    for x in r:md.append(f"| {x['chip']} | {'×'.join(map(str,x['shape']))} | {x.get('cohort',x['state'])} | {x['us']:.3f} | {x['old_pipeline_us']:.3f} | {x['base']:.3f} | {x['ca_us']:.3f} | {100*(x['us']/x['ca_us']-1):+.2f}% |")
    if not r:md+=['','当前这些新进程尚无合格窗口，全部预测已在运行前登记。']
    md+=['','现有窗不另启动 CA，可回放的实际任务、启动协议与耗时见 v0.26 采集。970DT 所列仅为预冻结的架构外推。对于超出一行 UB 的宽度，内置 kernel 会沿通道继续切片，本次单行总价仍合并片内搬运与计算。','',
      '[新影子冻结、适用前序及三端价格](registration.json) · [分阶段周期](parameters.json) · [开发回放与来源](archive.json) · [独立窗口与双端加速比](score.json) · [精确重复的单独部署验证](../../v0.26/post/report.md)']
    md+=['','## 精确重复的默认资格按物理分组核查','',
         '| 芯片 | 实际末核轮数、每行通道片数 | 实测不同形状 | 已发布重复 WAPE / 最大 | v0.26 旧默认 WAPE / 最大 |','|---|---|---:|---:|---:|']
    for sku,data in gate['regions'].items():
        for name,x in data['groups'].items():md.append(f"| {sku} | {name} | {x['different_shapes']} | {fmt(x['policy'])} | {fmt(x['old'])} |")
    md+=['','十形目标按每芯片的整个精确重复阶段计算。已测各物理分组均检查 WAPE≤20%、最大≤40%；多通道片的价格只对完整行总耗时负责。普通热态、首次和无法辨识的前序各自保留现有路径，精确重复的窗口不能用于证明它们的准确度。[默认资格与分组数据](deployment.json)']
    if (OUT/'same-token-history.json').exists():
        data=read(OUT/'same-token-history.json');md+=['','## 同一个 AIV 上的同一组四行','',
         '![上模型、下 CA；左普通换形、右同形立即重复，四栏时标一致](same-token-history.png)','',
         '| 可观察前序 | 全设备模型 / CA µs | 第一到第二个 token 首读发起间隔，模型 / CA µs | 当前观察核是否全芯片末核 |','|---|---:|---:|---|']
        for x in data['entries']:
            q=x['model'];c=x['ca'];model_dt=q['timeline'][1]['reads_us'][0][0]-q['timeline'][0]['reads_us'][0][0];actual_dt=c['rows'][1]['reads'][0]['s']-c['rows'][0]['reads'][0]['s']
            md.append(f"| {q['scope']} | {q['us']:.3f} / {c['device_us']:.3f} | {model_dt:.3f} / {actual_dt:.3f} | {'是' if c['critical_lane'] else '实际末核 '+c['terminal']} |")
        md+=['','两列逻辑输出行均为 token 108–111，位于 core17.veccore0；全芯片 196 个 token 的输出区域与每行四个操作数的连续步长均已核实。左列以新影子回放已见样本，右列采用在此次 CA 之前已冻结的精确重复完整行时序。紫色的向量实际活动仅归到 AIV，本图保留这个粒度。[逐条 DMA、向量区间、核与原始 trace 身份](same-token-history.json)']
    from development.validation.post26.freeze import OUT as previous
    device_receipt=previous/'receipts'/'multi-Ascend950DT_9582.json'
    prospective=OUT/'unseen-same-token.json'
    if device_receipt.exists() and not prospective.exists():
        same=[z for z in read(device_receipt)['rows'] if z['shape']==[157,4,1088]]
        if len(same)==2 and all(z['timing_qualified'] for z in same):
            import importlib.util,subprocess
            python=sys.executable if importlib.util.find_spec('matplotlib') else ROOT/'toolchains/envs/cann-9.2.1/bin/python'
            subprocess.run([str(python),str(Path(__file__).with_name('figure.py')),'prospective'],check=True,cwd=ROOT)
    if prospective.exists():
        u=read(prospective);md+=['','## 同一行的独立前向验证','',
           '![950DT 上两种已冻结前序的同一组三行，上为预测，下为 CA](unseen-same-token.png)','',
           '| 950DT [157,4,1088] 的前序 | 模型 / CA µs | 第 1→2 行读发起间隔，模型 / CA µs | 末核 |','|---|---:|---:|---|']
        for e in u['entries']:
            m=e['model'];c=e['ca'];reads=[x['reads_us'][0][0] for x in m['timeline']];actual=[x['reads'][0]['s'] for x in c['rows']]
            md.append(f"| {m['scope']} | {m['us']:.3f} / {c['device_us']:.3f} | {reads[1]-reads[0]:.3f} / {actual[1]-actual[0]:.3f} | {c['terminal']} |")
        changed,repeat=u['entries']
        def first(item):
            return item['model']['timeline'][0]['reads_us'][0][0],item['ca']['rows'][0]['reads'][0]['s']
        cm,cc=first(changed);rm,rc=first(repeat)
        md+=['',f'图中换形后的首行首操作数读发起，模型 / CA 为 {cm:.3f} / {cc:.3f} µs，模型偏晚 {cm-cc:.3f} µs。立即重复则为 {rm:.3f} / {rc:.3f} µs。首行到第二行的读发起间隔误差小于 0.004 µs，换形与重复的设备时延差仍含首读之前的等待、单行响应及末核选择。']
        md+=['','两种前序的时间都已在该进程开始前登记并固定。实线搬运区间来自相同输出地址行的 MTE2/MTE3，向量活动独立画在核级一行，虚线为全设备末指令。[真实行地址、启动核及每条指令窗口](unseen-same-token.json)']
    if (OUT/'physical-summary.json').exists():
        ph=read(OUT/'physical-summary.json');md+=['','## 去掉合成标签的辅助物理窗口','',
        '该口径对每个旧、新窗口都要求完整 ELF 入口和真实物理 END。所有核均排除工具生成的 END_LABEL，分母、设备终点和各端样本从头按这一口径计算。精确重复的主发布判定仍用预登记的全标签口径。','',
        '| SKU | 完整物理重复不同 shape | 候选 WAPE / 最大 | 旧默认 WAPE / 最大 |','|---|---:|---:|---:|']
        for sku,st in ph['summary'].items():md.append(f"| {sku} | {st['different_shapes']} | {fmt(st['policy'])} | {fmt(st['old'])} |")
        md+=['','[全体物理窗口、缺失标签与真实终点证明](physical-summary.json)']
    if (OUT/'published.json').exists():
        deployed=read(OUT/'published.json')
        speed=precise['ratios']['repeat'];spot=read(OUT/'unseen-same-token.json')['entries'][1]
        md+=['','## 系统仿真入口和打包校验','',
             '每个 rank、每条有序流各持有一个 `OperatorStream`。在真实成功派发后调用 `advance`，候选调度调用 `preview`。默认路径在前一调用与当前调用均为相同 shape、相同内置四流 BF16 残差合同且当前按 token 分核时，应用逐行流水。若显式传入历史，提供的 AIV 覆盖必须与复用相符。未成功的查询不会更新派发状态；`reset()`、`forget_history()`、执行其他实现或换形都会影响后继的资格。跨通道片只输出整行总价。','',
             '```python','from operator_model import OperatorStream','',
             'stream = OperatorStream("Ascend950DT_9582")','first = stream.advance("mhc_post", [157, 4, 1088])',
             'repeat = stream.advance("mhc_post", [157, 4, 1088])',
             'print(repeat.latency_us, repeat.projection["timing_diagnostics"]["model"])','```','',
             f'这里第二次调用给出 {spot["model"]["us"]:.3f} µs，独立 CA 窗为 {spot["ca"]["device_us"]:.3f} µs。要复现 v0.26 可指定 `experimental_post_repeats=False`。首次、普通换形、无历史和 vLLM 的自定义 HcPost 分开定价。换形的新候选在两端分别有 {len([x for x in r if x["chip"]==a])} / {len([x for x in r if x["chip"]==b])} 个独立窗，继续保留作对照。','',
             f'同形、同序号的 DT/PR 有 {speed["policy_us"]["n"]} 对精确重复完整窗，预测耗时比相对 CA 的 WAPE / 最大为 {fmt(speed["policy_us"])}，旧入口为 {fmt(speed["public_us"])}。970DT 仅为资源参数外推。当前通用目标 WAPE≤20%、最大≤50%，本轮精确重复使用更早冻结的 WAPE≤20%、最大≤40% 门槛。','',
             f'在源码目录外离线安装 {deployed["wheel"]}。两种已测 SKU 及 970DT 一共 {deployed["total"]} 次流查询，包含 {deployed["repeat"]} 次可用精确重复的派发。默认、显式旧版和无历史三种查询分别对照事前冻结的估值，最大差 {deployed["maximum_delta_us"]:.3f} µs。此进程的默认入口墙钟中位数 {deployed["median_ms"]:.3f} ms，p95 {deployed["p95_ms"]:.3f} ms，包含首次加载的最大值 {deployed["maximum_ms"]:.3f} ms。[逐次安装包回放和墙钟](published.json)']
    (OUT/'report.md').write_text('\n'.join(md).rstrip()+'\n')
if __name__=='__main__':main()
