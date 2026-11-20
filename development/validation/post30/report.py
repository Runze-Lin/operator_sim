"""Render Post resource evidence and both pending prospective tests."""
from math import ceil
from statistics import median
from study import ROOT,OUT,OLD,REG,DT,PR,FUT,read,save,forward,now


def f(x):
    return '—' if x is None else f'{x:.3f}'
def m(d):
    return f"{d['wape']:.2f}% / {d['maximum']:.2f}%"
def write_report():
    score=read(OUT/'score.json');new=forward()
    save(OUT/'forward-score.json',dict(updated_utc=now(),source_freeze=str(REG.relative_to(ROOT)),qualified=new,
        complete=len(new)==4,pass_preregistered=(all(
        len(rs:= [r for r in new if r['chip']==sku])==2 and
        (aa:=sum(abs(r['conditional_us']-r['ca_us'])for r in rs))<=.25*sum(r['ca_us']for r in rs)
        and max(abs(r['conditional_us']/r['ca_us']-1)for r in rs)<=.5
        and all(aa<sum(abs(r[k]-r['ca_us'])for r in rs)for k in ('baseline_us','frozen_aggregate_us')) for sku in (DT,PR)) if len(new)==4 else None)))
    channel=read(ROOT/'results/v0.28/post/channel-score.json')
    channel_reg=read(OUT/'channel-register.json')
    channel_new=[]
    for c in (DT,PR):
        source=OUT/('channel-'+c+'.json')
        if source.exists():
            j=read(source)
            for r in j.get('rows',[]):
                if r['role']=='exact-repeat' and r['timing_qualified']:
                    channel_new.append(dict(chip=c,shape=r['shape'],ca_us=r['stage']['ca_us'],candidate_us=r['candidate_us'],old_us=r['public_ordered_us'],
                        trace_sha=r['stage']['trace_sha256'],source=str(source.relative_to(ROOT))))
    sup={}
    for c in (DT,PR):
        old=[dict(chip=c,shape=r['shape'],ca_us=r['ca_us'],candidate_us=r['candidate_us'],old_us=r['public_known_us'],trace_sha=r['trace_sha'])
             for r in channel['rows'] if r['role']=='repeat' and r['chip']==c]
        extra=[r for r in channel_new if r['chip']==c];merged=old+extra
        def metrics(a,key):
            return (dict(wape=100*sum(abs(r[key]-r['ca_us'])for r in a)/sum(r['ca_us']for r in a),
                    maximum=100*max(abs(r[key]/r['ca_us']-1)for r in a)) if a else None)
        assert len(merged)==len({r['trace_sha']for r in merged})
        sup[c]=dict(unique_shapes=len({tuple(r['shape'])for r in merged}),previous_cohort=len(old),new_qualified=len(extra),
                    combined=metrics(merged,'candidate_us'),baseline=metrics(merged,'old_us'),supplement_only=metrics(extra,'candidate_us'))
    save(OUT/'channel-score.json',dict(updated_utc=now(),new=channel_new,by_sku=sup,
        enough=all(v['unique_shapes']>=10 and v['new_qualified']==2 and v['combined']['wape']<=25 and v['combined']['maximum']<=50 and v['combined']['wape']<v['baseline']['wape'] for v in sup.values()),
        original_v028_gate=channel['qualifies_to_enable_on_ordered_channel_repeat']))
    ys=score['rows'];times=sorted(r['stream_query_ms']for r in ys);cap=read(REG)
    lines=['# 内置 BF16 mHC Post 的宽行两片流水','',
      '| 芯片、样本口径 | 完整窗 / 形 | v0.29 有序 WAPE / 最大 | 两槽排程开启后 WAPE / 最大 | 仅 shape WAPE / 最大 |',
      '|---|---:|---:|---:|---:|']
    for group,name in [('all','全部历史回放'),('changed','普通换形双片，DT 开发 / PR 历史迁移'),('channel','通道分核全历史'),('exact_token_repeat','token 紧邻重复全历史')]:
        for sku in (DT,PR):
            s=score['summary'][sku][group]
            lines.append(f"| {sku[6:11]} · {name} | {s['windows']} / {s['shapes']} | {m(s['ordered_us'])} | {m(s['candidate_us'])} | {m(s['stateless_us'])} |")
    lines+=['',
     '上述数据按实际成功派发顺序用公开 OperatorStream 回放。相同全核指令窗按 SHA 去重。239 个窗口只有 2 个会选用新条件排程，覆盖同一几何在两个芯片上的调用。v0.29 默认仍是工具的默认估时；双分片新路径须显式启用。历史窗口见逐例数据，独立资格见后文。额外保留的物理退休窗 DT54/PR56 使用不同结束事件，不能混入主表。',
     f'239 次查询墙钟中位 {median(times):.3f} ms，P95 {times[ceil(.95*len(times))-1]:.3f} ms，最大 {times[-1]:.3f} ms。表中时间是设备微秒。[逐窗计算与 970DT 外推](score.json)','',
     '## [65,4,5120] 的真实关键 AIV','',
     '前一条操作为 [121,4,1216] 的同族 token Post。本次有 33 个 AIV，32 个处理两行，另一个处理一行。库核是安装包 ops-transformer 的 aclnnMhcPost，ELF 入口后缀为 6。四路 BF16 的主输入和 residual 沿通道切成两个 2560 列片。片内每个输出要做四项加权，片间可以独立计算。完整输出的 520 个连续写区间覆盖 2,662,400 B。','',
     '| SKU | v0.29 有序 | 旧整行条件模型 | 新双片模型 | CA | 新模型误差 | 额外实测 |','|---|---:|---:|---:|---:|---:|---|']
    for sku,tag in [(DT,'dt'),(PR,'pr')]:
        v=read(OUT/f'fragments-{tag}.json');new_us=v['conditional_us'];ca=v['ca_us']
        lines.append(f"| {sku[6:11]} | {f(v['original_ordered_us'])} | {f(v['aggregate_us'])} | {f(new_us)} | {f(ca)} | {100*(new_us/ca-1):+.2f}% | {v['core']}，2 行 |")
    lines+=['','32 个双行核上的相邻写发起间隔一致，为 1377 个参考周期。读 hOut 和 X 的初始间隔是 49 周期，槽开始周转后为 2 周期。进一步对齐每个核的输入 UB 地址发现，初始两个片的 hOut 发起相差 299 周期；稳态槽内的下一次 hOut 发起早于两片前对应输出的 issue 242 周期。多片的输入等待、逐片 SIMD 以及上一笔未完成的输出写存在重叠。截取应用预测中对应的槽周转时间可以复现临界核后两笔输出发起。242 是已观测的时间差；尚无直接硬件事件证明“写发起”本身控制槽释放。DT 每核时间和分位数、PR 真实数据在下图和明细中单列。','',
     '![950DT 普通换形，两条 token 上相同输出片的资源与 CA 对照](fragments-dt.png)','',
     '![950PR 普通换形，两条 token 上相同输出片的资源与 CA 对照](fragments-pr.png)','',
     '[DT 33 核的 operand issue/ack 和 GM/UB 地址](operands-dt.json) · [DT 预测与 CA](fragments-dt.json) · [PR 预测与 CA](fragments-pr.json)','',
     '## 可用于系统流的公式及参数','',
     '接口限定四路 BF16、含 residual、连续 ND。需已知同一有序流上一条成功 Post 也是 token 分核，当前换形且活跃 AIV 已覆盖；只有根据目标 UB 算出恰为两个片才会启用。片宽按 16 列对齐，最后片用其实际列数。同形重复、单片、通道分核、未知前序及 vLLM HcPost 各自沿用对应入口。','',
     '```python','from operator_model import OperatorStream',"stream = OperatorStream('Ascend950DT_9582', track_post_history=True,", '                        experimental_post_fragments=True)',"stream.advance('mhc_post', [121, 4, 1216])","pred = stream.advance('mhc_post', [65, 4, 5120])","print(pred.latency_us, pred.projection['timing_diagnostics']['stages_us'])",'```','',
     '示例仅两次调用时，工具从这两次成功形状重建同族 AIV 高水位。也可直接用 estimate_post_fragments(shape, chip=..., previous_shape=..., resident_aiv=..., inspect_rows=(0,1)) 获取逐片时间线。前一调用失败、仅做 preview 或中间真正派发其它算子均按公开流语义处理。','',
     '令 d 为当前片宽，g=ceil(d/64) 为向量组数，f 为目标 GHz，V 为目标 FP32 每 AIV 每周期运算数。每片主输入、residual、输出分别为 2d、8d、8d 字节。每行额外读取 64 B 和 16 B 系数。MTE 读写单价 k 的单位为参考周期/KiB；目标芯片给出带宽 B GB/s 时使用 k=1024f/B，缺失时继承 950 价格。每笔响应为固定等待加 payload·k/1024。两个输入均到齐后进入 SIMD。向量成本 v 取实际工作 36d/V 与 31.775g·Vref/V 的较大值；31.775 源于早期独立宽度的 DT 阶段斜率。每片写发起增量 I 取向量成本加 105.984 周期、本片输入服务、上一片输出服务的最大值。[原斜率和带宽基价](../../v0.25/post/prices.json)','',
     r'从第 3 片起，最后一笔输出描述符发起为 $W_j=\max(W_{j-1}+I_j,\ W_{j-2}-L+A_j+C_j)$。L 是同输入槽被下一片使用到之前那片发写的可重叠周期；A 是本槽发读至最晚依赖就绪，C 是就绪至能发写的向量工作。头两个片单独计算系数就绪、299 周期 lookahead 和首轮就绪。末笔写加入固定响应和字节服务，最后加 10 周期退休，并检查共享 HBM/L2 下界。两片一行形成一个重复的 2×2 转移；快速幂只需 O(log 忙核行数)，请求的逐行诊断上限 8 行。','',
     '新参数仅取 950DT 一个普通换形过程的物理事件中位数。首片就绪到写发起扣除向量组工作约 141 周期，末写扣除字节的响应中位 136 周期。响应在不同核间有漂移；终端确认前的倒数第二笔写没有独立的末笔服务价格，图中模型仅标发起。目标 SKU 的 HBM、L2、Vector、核数及 UB 各自由芯片配置决定。槽归还差、固定控制和发起周期沿用本次 DT；新芯片队列深度、ISA 或算子分块改变时需要新的机制证据。970DT 均为外推。[所有价格、样本范围与单位](prices.json) · [排程实现](../../../operator_model/post_fragment_pipeline.py)','',
     '## 采前冻结与硬下界','',
     '| 新双片普通换形 | DT 公开 / 条件 / CA | PR 公开 / 条件 / CA | 970DT 条件外推 |','|---|---:|---:|---:|']
    for sh in [r['shape'] for r in cap['plans'] if r['chip']==DT]:
        values=[];future=None
        for c in (DT,PR,FUT):
            rr=next(r for r in cap['plans'] if r['chip']==c and r['shape']==sh)
            if c!=FUT:
                observation=next((v for v in new if v['chip']==c and v['shape']==sh),{})
                values.append(f"{f(rr['baseline_us'])} / {f(rr['conditional_us'])} / {f(observation.get('ca_us'))}")
            else:future=rr['conditional_us']
        lines.append(f"| {sh} | {values[0]} | {values[1]} | {f(future)} |")
    lines+=['', '以上沿用已登记的 [112,4,768] → [67,4,4912] → [69,4,5680] 两端流，登记时没有测量结果。本轮给这项试验新增的 CA 启动为零。在两端完整返回前，默认准入保持 N/A。[采前预测、模型哈希与验收条件](freeze.json) · [已有队列](../../v0.29/post/split-state.json) · [独立回执](forward-score.json)','',
       '通道路径另有两种前向形状 [13,4,3520] 和 [19,4,2720]，每种在同一流各成功派发两次。使用 v0.24 的原阶段价格，双端每个进程 4 次调用，预算各 1 h / 4 GiB。先前 DT/PR 原冻结通道重复已有 '+str(sup[DT]['previous_cohort'])+'/'+str(sup[PR]['previous_cohort'])+' 种 shape；之前超时的 [7,4,3328] 保留原失败记录。PR 的 [23,4,1984] 已在原队列，补充组等候既有采集释放后按端点共用一个 CA 槽。新的条件价格、公开基线和 970DT 外推均逐次预登记。[补充冻结](channel-register.json) · [补充得分与原登记资格](channel-score.json)','',
       '同一芯片、dtype、shape、库核和输入合同只有一个可输出的点估计时，让每种 shape 的预测恰取其真实时间中位数，就得到此数据集最小 WAPE。若还要求每个样本最大相对误差 ≤50%，每形可行交集为 [0.5·最大真实时间，1.5·最小真实时间]，把中位数截入交集可得联合下界。该结论分别使用同形冷/热 10 形 21 和 22 个真实窗；不能把理想中位数当作线上预测。','',
       '| SKU | 冷/热窗口 | 最低可能 WAPE | 最大误差≤50%时最低 WAPE | 最低可能最大误差 |','|---|---:|---:|---:|---:|']
    for c in (DT,PR):
        b=score['bounds'][c];lines.append(f"| {c[6:11]} | {b['windows']} | {b['unconstrained_min_WAPE_pct']:.2f}% | {b['minimum_WAPE_at_max50_pct']:.2f}% | {b['minimax_lower_bound_pct']:.2f}% |")
    lines+=['','在这组明确配对上仅 shape 无法同时满足 WAPE ≤25% 与最大 ≤50%。调用方可以传真实前序，或用 estimate_post_history 单独查看各状态及自己提供的概率。全历史混合权重与此配对试验的权重不同，二者的 WAPE 分列。[逐形交集、理论中心和真实底数](score.json)']
    (OUT/'report.md').write_text('\n'.join(lines).rstrip()+'\n')
if __name__=='__main__':write_report()
