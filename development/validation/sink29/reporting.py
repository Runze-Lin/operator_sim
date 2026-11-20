"""Render current standalone Sinkhorn inputs and sequence-specific timing evidence."""
from pathlib import Path
from statistics import median
from audit import ROOT,OUT,V26,V28,DT,PR,FUT,read,REG


def m(v):return 'N/A' if not v else f"{v['wape']:.2f} / {v['maximum']:.2f}"
def output(d):
    lines=['# mHC Sinkhorn · 同一批输入上的无状态与同流估时','',
      '### Summary','',
      '|950 芯片及数据角色|窗 / 不同形|现行无状态 WAPE / 最大 %|现行从真实首发跟踪|现行中途接入|候选只看 shape|候选使用可见前缀|',
      '|---|---:|---:|---:|---:|---:|---:|']
    old=[v for v in d['history'] if v['role'].startswith('original')]
    order=[v for v in d['history'] if v['role'].startswith('v0.26')]
    for sku in (DT,PR):
        for k,label,count in [('original-slice','旧开发、T>64/I10',3 if sku==DT else 1),
                ('v26-ordered-T-le256','v0.26 冻结十形，T≤256',10),('v26-same-I10-T>64','该冻结集 T>64/I10',2),
                ('v26-joined-H-partially-known','中途接入、扩核与热态均可能',None),
                ('v26-joined-H-proves-reuse','中途接入、已证实全部覆盖',None),
                ('v28-earlier-registration-complete','v0.28 原登记已完成',None),
                ('v29-guard-eligible-previously-unseen','本次候选采前冻结新窗',None)]:
            z=d['metrics'][sku][k];n=z['ordered_us']['n'] if z['ordered_us'] else 0
            if k.startswith('v28'):count=len({(r['t'],r['iters']) for r in d['forward_complete'] if r['sku']==sku and r['group']!='paired'})
            elif k.startswith('v29'):count=len({(r['t'],r['iters']) for r in d['forward_complete'] if r['sku']==sku and not r['source_already_seen'] and r['unknown_guard_us'] is not None})
            lines.append(f"|{sku[6:11]} · {label}|{n} / {count if count is not None else n}|"+'|'.join(m(z[k]) for k in ('default_us','ordered_us','lost_us','unknown_guard_us','prefix_guard_us'))+'|')
    lines+=['','**25.64% 确实可在当前无状态入口复现。** 它来自 950DT 的三个旧开发窗口，首次一个、同族覆盖后的热态两个。PR 存档只有 T68 首次；两端旧组的协议构成不同。已知完整成功执行顺序时，现版系统后端即可使用 `OperatorStream(chip,track_sinkhorn_history=True)`，每次以实际同流顺序调用 `advance("mhc_sinkhorn",[T,4],{"num_iters":I})`。这一推荐对应 v0.26 登记前已冻结的条件估时，每端十种 T≤256 的新形；v0.29 对它们做入口一致性回放。未经区分的单次调用仍需保留前序未知。','',
      '刚附到已有流时先调用 `forget_history()`，执行完成的同族算子可以恢复覆盖核数的下界。不同流、未观测的调用、其他算子的 AIV 不能当作同族已经覆盖。只有方案已选定且实际 dispatch 才推进，查询候选用 `preview`。','',
      '## 逐形回放与外推','',
      '|数据和协议|SKU|T / I|H / AIV / R|全核 CA µs|现行无 H / 有序 µs|无状态保护点 / 中途接入 µs|970DT 同协议外推 无 H / 有序 µs|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in old+[r for r in order if r['t']>64 and r['iters']==10]:
        f=r['future_970'];tag='旧三形及 PR 对照' if r in old else 'v0.26 冻结'
        lines.append(f"|{tag} · {r['state']}|{r['sku'][6:11]}|{r['t']} / {r['iters']}|{r['H']} / {r['active']} / {r['rows']}|{r['ca_us']:.3f}|{r['default_us']:.3f} / {r['ordered_us']:.3f}|{r['unknown_guard_us']:.3f} / {r['lost_us']:.3f}|{f['default_us']:.3f} / {f['ordered_us']:.3f}|")
    lines+=['','T65 的真实宿主输入此前依序为 T2、4、8、9、16、31、32、63，T66 之前为 T3、5、7、11、15、21、33、47。两者调用参数和宿主成功输出逐次核实。T68 两端单独启动。未来芯片使用相同的宿主顺序，再按 120 个 AIV 计算 H 和实际分核。','',
      '## 哪段流水随前序变化','',
      r'CANN 9.2.1 `aclnnMhcSinkhorn`，真实 FP32 key 0、4×4、无中间输出。$R=\lceil T/V\rceil$ 是可用 $V$ 个 AIV 下的每核最多矩阵数，实际活跃核 $A=\lceil T/R\rceil$。$G=\lceil R/8\rceil$ 由 256 B 寄存器包含 8 个 32 B UB 块确定。本组全为 G=1，每核每次有 $2I+2=22$ 次 VF，包括前处理、交错行/列归一化与末尾 scatter。已知 H=0 选首次，0<H<A 选扩核，H≥A 选覆盖。tiling 描述符、输入读取、首次 VF 与最后写回有独立的发起/等待价；按目标频率换算循环周期，用目标每核 FP32 的每周期能力重定价。254 周期稳态间隔从 DT 的连续 row 发起读到，不把整段 VF 生命周期按迭代数相加。[阶段校准参数](../../v0.26/sink/prices.json) · [实际源码与硬件边界](../../v0.28/fusion/eligibility.md)','',
      '|算例|首发 tiling|输入发起|首次列|首次行|稳态行间隔|scatter|写回发起 / 写描述符窗口|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in old:
        a=r['marks_cycles']
        def fmt(k):return f"{a[k][0]:.0f}/{a[k][1]:.0f}"
        lines.append(f"|{r['sku'][6:11]} T{r['t']} {r['state']}|"+'|'.join(fmt(k) for k in ('tiling_issue','input_issue','first_column','first_row','steady_row_to_row','scatter'))+'|'+fmt('write_issue')+' ; '+fmt('write_lifetime')+'|')
    lines+=['','表内是“模型 / 实际周期”，零点为本次启动各核最早指令，后两列包含等待。旧热态两形与首次形的主要差距出现在首次读/发起、scatter 前的累积偏移及末写回；迭代数相同的稳态行间隔保持 254 周期。不同 T 的末核可能处理尾块，R 是最忙核上限。以下是同一 T67 的首次、热态模型和实测，时序差异亦可从同形配对直接看到。','',
      '![950DT T67 两种前序](../../v0.27/fusion/950dt-same-input.png)','',
      '![950PR T67 两种前序](../../v0.27/fusion/950pr-same-input.png)','',
      '## 只有部分同族历史时的物理区间与单值','',
      r'维护已观察到的同族最大活跃核数 $h$。若流在观测前已有调用，真实 H 仅知落在 [h,V]。条件模型最多枚举 first、expand、resident 三类，一类只算一次。设 $L,U$ 是可达阶段时间的最低和最高值；它们是模型端点。若业务给出状态概率 $p_s$，批量总时的期望为 $\sum_s p_s T_s$，可由调度器的历史策略估计概率。没有这样的信息时，不把旧批次首次/热态出现频率搬到新业务。','',
      r'只剩一种状态即可直接定价。多种状态尚存时，新增离线对照保留原价 $t_0$，并将它投影到 $[(1-\delta)U,(1+\delta)L]$，其中 $\delta=\max(0.25,(U-L)/(U+L))$。0.25 沿用了 v0.28 对可达条件模型的对称相对预算；样本 WAPE 是验收指标，两者不等价。与早先直接取等状态均值不同，旧价在可达预算内就原样返回。','',
      '同形 T67 首次/热态在 DT 为 5.232/3.260 µs、PR 为 5.068/3.285 µs。任何无 H 单值的这两个点的最大相对误差下界分别为 23.22%、21.34%，来自 $p=2LU/(L+U)$ 与真实两个窗口；两者都小于 50% 门槛。DT 的三旧形回放降低到 21.91%，代价是在原 v0.26 十形上的无 H WAPE 16.64%→18.67%；PR 的对应首次 T68 单窗 8.55%→16.81%。把保护点用在“确定有前序、上限未知”的 DT 四个扩核窗口也从 4.36% 升到 9.99%。因此现在可推荐的是已经冻结验证的有序接口；新增单值继续作为诊断，对无 H 不改变公共后端。','',
      '## 冻结、覆盖与开销','',
      'v0.28 四形流的 T69 首次两侧均已测，不能充作 v0.29 的新留出。仍未开跑的有序序列与 T233 寄存器试验已在本轮登记了所有可观察场景。T69 再出现是已有几何的新状态；T137、T233 是本轮内单寄存器可区分的全新几何；T1153、T2257 属于多寄存器，沿既有专项单列。这次新增 CA 为 0，收集结果以设备全核退出、逐元素数值、AIV 集合及 VF 个数核验。当前收到的本轮合格数见开头表，旧资格不顺延给新增保护点。','',
      '[逐形、完整前序、物理阶段、全核回执及实测与外推](score.json) · [本轮观测前冻结](freeze.json) · [现行有序入口的十形验证](../../v0.27/fusion/stream.md)']
    (OUT/'report.md').write_text('\n'.join(lines).rstrip()+'\n')
