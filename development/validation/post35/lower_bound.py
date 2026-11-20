"""Diagnostic lower bound under an exact-input equality and maximum-relative-error constraint."""
from collections import defaultdict
from statistics import median
from registration import D,P,OUT,ROOT,read,save

def oracle(rows,cap=.50):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r['shape'])].append(r)
    cols=[]
    for sh,points in groups.items():
        ys=[r['ca_us'] for r in points];lo,hi=min(ys),max(ys);med=median(ys);L=(1-cap)*hi;U=(1+cap)*lo
        point=min(U,max(L,med)) if L<=U else None
        cols.append(dict(shape=list(sh),windows=len(ys),lo=lo,hi=hi,ca_sum=sum(ys),unconstrained_median=med,
            best_unconstrained_abs_sum=sum(abs(med-y) for y in ys),minimax_joint_relative_percent=100*(hi-lo)/(hi+lo),
            max50_admissible_us=[L,U] if point is not None else None,best_achievable_point_us_at_max50=point,
            unavoidable_absolute_us=sum(abs(point-y) for y in ys) if point is not None else None,
            protocol_windows=[dict(us=r['ca_us'],history_for_diagnosis_only=r['validation_only_prior'],sha256=r['sha256'],original_source=r['trace']) for r in points]))
    denom=sum(v['ca_sum'] for v in cols);feasible=all(x['max50_admissible_us'] is not None for x in cols)
    constrained=sum(x['unavoidable_absolute_us'] for x in cols) if feasible else None
    for v in cols:v['same_shape_wape_constrained']=100*v['unavoidable_absolute_us']/v['ca_sum'] if v['unavoidable_absolute_us'] is not None else None
    cols.sort(key=lambda r:(r['minimax_joint_relative_percent'],r['shape']),reverse=True)
    return dict(n=len(rows),shapes=len(cols),singleton_shapes=sum(c['windows']==1 for c in cols),sum_ca_us=denom,
        unconstrained_wape_pct=100*sum(x['best_unconstrained_abs_sum'] for x in cols)/denom,
        minimum_achievable_max_pct=max(x['minimax_joint_relative_percent'] for x in cols),
        cap_feasible=feasible,cap_pct=100*cap,min_wape_at_max50_pct=100*constrained/denom if feasible else None,
        oracle_joint_25_50_feasible=feasible and 100*constrained/denom<=25,groups=cols)
def calculate(rows):
    original=read(ROOT/'results/v0.32/post/replay.json')['same_contract_oracle'];result={}
    for c in (D,P):
        rs=[r for r in rows if r['chip']==c];lookup={(tuple(r['shape']),r['ca_us']):r for r in rs}
        exact=[lookup[tuple(g['shape']),y] for g in original[c]['shapes'] for y in g['measured_us']]
        scopes={'all':rs,'post_channel':[r for r in rs if r['route']=='post_channel'],
                'post_token':[r for r in rs if r['route']=='post_token'],'strict_same_contract':exact}
        from collections import Counter
        freq=Counter(tuple(v['shape']) for v in exact)
        scopes['strict_with_two_or_more_windows']=[v for v in exact if freq[tuple(v['shape'])]>=2]
        result[c]={k:oracle(v) for k,v in scopes.items()}
    info=dict(definition='Same SKU+BF16+with-hRes, built-in aclnnMhcPost, identical shape -> one point. WAPE denominator is sum of observed window times. Visible dispatch history excluded. Different acquisition regimes remain as recorded in all-archive and restricted same-scheme subset is separate. Retrospective use of all measured times only diagnoses identifiability, no forecasts fitted to these medians.',by_chip=result)
    save(OUT/'oracle.json',info);return info

def markdown(rows):
    result=calculate(rows)['by_chip']
    lines=['','## 已有窗口对仅形状模型的误差下界','',
    '给定同一芯片、同 dtype/属性和同一 shape 的真实设备时间 $y_i$，任何无前序输入的确定性模型都必须给同一个点 $p$。设 $h=\\max y_i,l=\\min y_i$。要求每次绝对相对误差≤50%，等价于 $0.5h\\le p\\le1.5l$。区间有解的条件是 $h\\le3l$。如不限制最大误差，样本中位数最小化总绝对偏差；加上约束，只需将中位数截入上述区间。按每形分别求最小绝对偏差、再除以这一组所有实测时间之和，恰好得到 WAPE 下界。单形可达到的最小最大相对误差是 $(h-l)/(h+l)$；整份数据取所有形的最大值。','',
    '| 芯片 / 分组 | shape / 窗 | 仅 1 窗的 shape | 无约束最低 WAPE | 理论最低最大误差 | 最大≤50% 时最低 WAPE | 同时≤25% / ≤50% 可行 |','|---|---:|---:|---:|---:|---:|---|']
    for chip,tag in ((D,'950DT'),(P,'950PR')):
        for k,name in [('all','全档'),('post_channel','通道分核'),('post_token','token 分核'),('strict_same_contract','原登记同合同形'),('strict_with_two_or_more_windows','仅保留至少两个合格窗的原登记形')]:
            a=result[chip][k]
            lines.append(f"| {tag} / {name} | {a['shapes']} / {a['n']} | {a['singleton_shapes']} | {a['unconstrained_wape_pct']:.2f}% | {a['minimum_achievable_max_pct']:.2f}% | {a['min_wape_at_max50_pct']:.2f}% | {'可行' if a['oracle_joint_25_50_feasible'] else '不可行'} |")
    lines.extend(['','全档在两端均有理论解。只观测一次的 shape 在离线求下界时可以恰好取到那一个真实值，因此全档下界偏乐观；它不能证明未见 shape 的可预测性。原严格协议每端登记 10 个同合同形状。DT 的 [8,4,896] 只收录一个合格时窗，9 个形真正配成两窗或三窗；PR 的 10 个形均有至少两个窗。剔除该 DT 单窗再计算，子集仍不能同时通过 25% 和 50% 两项指标。全档有很多没有冷热成对的形状，两个结论可以并存。','',
     '按同一通道 shape 在两种实际前序间最难兼顾的最大相对误差排序，各片列前三形。区间是最大误差必须≤50%时允许的所有预测值，WAPE 下界只针对该 shape 的已观测窗口。','',
     '| 芯片 | shape | 窗 | 最短 / 最长 CA µs | 最大≤50% 可选区间 µs | 最低最大误差 | 限制后的 WAPE 下界 |','|---|---|---:|---:|---:|---:|---:|'])
    for chip,tag in ((D,'950DT'),(P,'950PR')):
        for g in result[chip]['post_channel']['groups'][:3]:
            a,b=g['max50_admissible_us'];lines.append(f"| {tag} | {g['shape']} | {g['windows']} | {g['lo']:.3f} / {g['hi']:.3f} | {a:.3f}–{b:.3f} | {g['minimax_joint_relative_percent']:.2f}% | {g['same_shape_wape_constrained']:.2f}% |")
    lines+=['','同形下界独立于预测器；上述中位数及区间只在回放标签上计算。在线候选的资源参数和点值均来自冻结公式，未引用本诊断文件。[逐形真实窗口、最优点与资格 SHA](oracle.json)']
    return lines
