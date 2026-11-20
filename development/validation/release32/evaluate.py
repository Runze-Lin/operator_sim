"""Replay the installed v0.32 wheel, retaining measurement contracts and unpriced rows."""
from collections import defaultdict
from hashlib import sha256
import json
from math import ceil, isclose
from pathlib import Path
from statistics import median
import sys
import os
from collections import Counter
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/v0.32'
DT, PR, FUT = 'Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'
CHIPS = (DT,PR)


def read(path):
    return json.loads((ROOT/path).read_bytes())


def save(name, value):
    dest = OUT/name
    tmp = dest.with_suffix('.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    tmp.replace(dest)


def summary(rows, key):
    valid = [r for r in rows if r[key] is not None]
    if not valid:
        return dict(n=0, shapes=0, wape=None, maximum=None, bias=None, faster_fraction=None)
    total = sum(r['ca_us'] for r in valid)
    diffs = [r[key]-r['ca_us'] for r in valid]
    return dict(n=len(valid), shapes=len({tuple(r['shape']) for r in valid}),
                wape=100*sum(abs(d) for d in diffs)/total, maximum=max(100*abs(d)/r['ca_us'] for d,r in zip(diffs,valid)),
                bias=100*sum(diffs)/total, faster_fraction=sum(d<0 for d in diffs)/len(diffs))


def evaluate(prefix):
    if not sys.flags.isolated:
        raise RuntimeError('Use python -I with the isolated v0.32 installation directory')
    sys.path.insert(0,str(Path(prefix).resolve()))
    import operator_model as api
    assert api.__version__=='0.32.0' and Path(api.__file__).is_relative_to(Path(prefix))
    # -I deliberately excludes checkout/relative modules; only tools live here.
    sys.path.append(str(Path(__file__).resolve().parent))
    from receipts import early_registered, matmul_windows, post_windows, other_full_windows, standalone_hcpost
    from manifest import fingerprint
    manifest = fingerprint(Path(prefix).resolve())
    previous_release = read('results/v0.31/replay.json')['rows']
    previous_ids = {(r['op'], r['chip'], tuple(r['trace_sha256']) if not isinstance(r['trace_sha256'],str) else (r['trace_sha256'],)) for r in previous_release}
    by_id, cache, wall = {}, {}, []
    def query(op, shape, attrs, chip):
        k=(op,tuple(shape),json.dumps(attrs,sort_keys=True),chip)
        if k not in cache:
            start=perf_counter_ns()
            cache[k]=api.estimate(op,shape,attrs,chip=chip)
            wall.append((perf_counter_ns()-start)/1e6)
        return cache[k].latency_us
    def add(op,shape,chip,attrs,kernel,source,identity,ca,old,now,future,role,category='replay',why='archived_or_not_proved_blind'):
        identity = (identity,) if isinstance(identity,str) else tuple(identity)
        if chip not in CHIPS or ca is None or not identity:
            raise ValueError((op,shape,chip,ca,identity))
        key = op,chip,identity
        if key in previous_ids and category=='independent':
            category,why='replay','same qualified trace already used by v0.31'
        if key in by_id:
            other = by_id[key]
            assert list(shape)==other['shape'] and isclose(ca,other['ca_us'],abs_tol=.001),key
            assert (now is None and other['us'] is None) or (now is not None and other['us'] is not None and isclose(now,other['us'],abs_tol=1e-7)),key
            # A frozen receipt may also be visible through the asynchronous Matmul scorer.
            # Upgrade a generic role only after joining to the exact registered trace.
            if category=='independent' and key not in previous_ids:
                other.update(data_role=category,role=role,classification=why,source=source)
            elif (other['data_role']=='replay' and other['classification']=='archived_or_not_proved_blind'
                  and why!='archived_or_not_proved_blind'):
                other.update(role=role,classification=why,source=source)
            return
        by_id[key]=dict(op=op,shape=shape,chip=chip,attrs=attrs,kernel=kernel,source=source,
                        identity=list(identity),ca_us=ca,previous_us=old,us=now,
                        extrapolated_970dt_us=future,role=role,data_role=category,classification=why)
    def public(label,op,shape,chip,attrs,kernel,source,identity,ca,old,role,check=None,category='replay',why='archived_or_not_proved_blind'):
        now=query(op,shape,attrs,chip)
        if check is not None:
            assert now is not None and isclose(now,check,abs_tol=1e-7), (op,shape,chip,now,check)
        add(label,shape,chip,attrs,kernel,source,identity,ca,old,now,query(op,shape,attrs,FUT),role,category,why)
    for r in read('results/v0.31/replay.json')['rows']:
        name=r['op']
        if name in ('mhc_sinkhorn','matmul','mhc_post'):
            continue
        op={'fla_kda_v2':'kda_chunk','vllm_hc_post':'mhc_post'}.get(name,name)
        public(name,op,r['shape'],r['chip'],r['attrs'],r['kernel'],r['source'],r['trace_sha256'],
               r['ca_us'],r['current_us'],'archived: '+r['role'],r['current_us'])
    for r in read(OUT/'matmul/score.json')['rows']:
        public('matmul','matmul',r['shape'],r['chip'],{},r['kernel'],r['source'],r['hash'],
               r['ca_us'],r['old_us'],r['role'],r['new_us'])
    for r in read('results/v0.30/sink/score.json')['rows']:
        attrs={'num_iters':r['iters']}
        shape=[r['t'],4]
        public('mhc_sinkhorn','mhc_sinkhorn',shape,r['sku'],attrs,r['provider'],r['source'],r['trace_sha256'],
               r['ca_us'],r['public_us'],'archived: '+r['context'])
        if r['h'] is not None:
            public('mhc_sinkhorn · known history','mhc_sinkhorn',shape,r['sku'],dict(attrs,resident_aiv=r['h']),
                   r['provider'],r['source'],r['trace_sha256'],r['ca_us'],r['known_us'],'archived known prefix')
    for name,op,shape,chip,attrs,kernel,path,h,ca,old,expected,cat,why in (*early_registered(),*matmul_windows()):
        public(name,op,shape,chip,attrs,kernel,path,h,ca,old,'prospective join: '+why,expected,cat,why)
    for label,op,sh,chip,attrs,k,src,h,ca,old,cat,why in (*other_full_windows(),*standalone_hcpost()):
        same = old is not None and query(op,sh,attrs,chip) is not None and isclose(old,query(op,sh,attrs,chip),abs_tol=1e-7)
        if not same:cat,why='replay','no identical frozen public price on this geometry'
        public(label,op,sh,chip,attrs,k,src,h,ca,old,'registered earlier: '+why,None,cat,why)
    post_report=OUT/'post/replay.json';post=read(post_report)
    ordered_check=read(OUT/'ordered-wheel-check.json')
    assert ordered_check['wheel_sha256']==manifest['wheel_sha256'] and ordered_check['original_sha256']==sha256(post_report.read_bytes()).hexdigest()

    for r in post['rows']:
        for label,old,new,future in [('mhc_post · shape','v031_shape_us','shape_us','extrapolated_970_shape_us'),
                                      ('mhc_post · stream','old_us','stream_us','extrapolated_970_stream_us')]:
            if new=='shape_us':
                assert isclose(query('mhc_post',r['shape'],r['explicit_attrs'],r['chip']),r[new],abs_tol=1e-7)
            add(label,r['shape'],r['chip'],r['explicit_attrs'],r['kernel'],r['trace'],r['sha256'],
                r['ca_us'],r[old],r[new],r[future],'archived sequential replay: '+r['scenario'])
    rms=read('results/v0.28/rms/stream-replay.json')
    for r in rms['archived']+rms['pairs']:
        compact=r.get('provider')=='npusim compact'
        source=r.get('source','results/v0.28/rms/stream-replay.json')
        public('rmsnorm · '+('compact CA' if compact else 'msopprof'),'rmsnorm',r['shape'],r['chip'],{},
               'aclnnRmsNorm',source,r['trace'],r['ca_us'],r['unknown_us'],'archived full device window',r['unknown_us'])
        if 'stream_us' in r:
            public('rmsnorm · visible stream','rmsnorm',r['shape'],r['chip'],{'resident_aiv':r['H']},'aclnnRmsNorm',
                   source,r['trace'],r['ca_us'],r['stream_us'],'archived verified dispatch order',r['stream_us'])
    for chip,r in read('results/v0.28/cast/score.json')['heldout'].items():
        if r['qualified']:
            rec=read(r['receipt'])
            old=read('results/v0.28/cast/register.json')['prices'][chip]['pressure']['us']
            public('cast_bf16_fp32','cast_bf16_fp32',rec['shape'],chip,{},rec['stage']['kernel'],r['receipt'],
                   rec['stage']['trace_sha256'],r['ca_us'],old,'archived 165 MiB memory-pressure test',old)
    for p,ca,kernel,h,source,future,cat,why in post_windows(api):
        sh,chip=p['shape'],p['chip']
        public('mhc_post · shape','mhc_post',sh,chip,{},kernel,source,h,ca,p['v031_shape_us'],'post prospective '+why,p['shape_us'],cat,why)
        add('mhc_post · stream',sh,chip,{},kernel,source,h,ca,p['v031_visible_stream_us'],p['ordered_us'],future,'post prospective '+why,cat,why)
    for sku in CHIPS:
        path=OUT/'hcpost'/(sku+'.json')
        if not path.is_file():continue
        r=read(path)
        assert r['chip']==sku and r['physical_complete'] and r['output_bytes']==2*r['good_outputs']
        chip=r['chip']
        j,=[v for v in read('results/v0.31/post/freeze.json')['jobs'] if v['op']==r['op']]
        public('vllm_hc_post · physical full window','mhc_post',r['shape'],chip,j['attrs'],r['kernel'],
               str(path.relative_to(ROOT)),r['trace_sha256'],r['ca_us'],j['predictions'][chip]['us'],
               'frozen device estimate; per-launch trace and complete output, outer export capped')
    all_rows=list(by_id.values())
    empty=['kda_chunk · fused V1','kda_recurrent · dense multi-chain']
    labels=sorted(set(r['op'] for r in all_rows).union(empty))
    results={}
    lines=['# v0.32 全算子误差','',
           '| 算子与测量口径 | 950DT 形 / 窗 · WAPE / 最大 · 有符号偏差 | 950PR 形 / 窗 · WAPE / 最大 · 有符号偏差 | 绝对误差门槛 |',
           '|---|---:|---:|---|']
    for label in labels:
        cols=[]
        results[label]={}
        for chip in CHIPS:
            rows=[r for r in all_rows if (r['op'],r['chip'])==(label,chip)]
            stats=summary(rows,'us')
            matched=[r for r in rows if r['us'] is not None and r['previous_us'] is not None]
            results[label][chip]=dict(full=stats,prior_same_windows=summary(matched,'previous_us'),
                                      current_same_windows=summary(matched,'us'),
                                      unpriced=sum(r['us'] is None for r in rows))
            cols.append(f"{stats['shapes']}/{stats['n']} · {stats['wape']:.2f}% / {stats['maximum']:.2f}% · {stats['bias']:+.2f}%"
                        if stats['n'] else '0 · N/A')
        statuses=[]
        for chip in CHIPS:
            s=results[label][chip]['full']
            statuses.append('N/A' if not s['n'] else '达标' if s['wape']<=25 and s['maximum']<=50 else '超限')
        lines.append(f"| {label} | {' | '.join(cols)} | {' / '.join(statuses)} |")
    lines+=['','### 冻结后独立验证与开发/历史回放分列','',
            '| 算子/协议 | DT 独立：形/窗，WAPE / 最大 | PR 独立：形/窗，WAPE / 最大 | DT 开发/回放：形/窗，WAPE / 最大 | PR 开发/回放：形/窗，WAPE / 最大 |',
            '|---|---:|---:|---:|---:|']
    for label in labels:
        columns=[]
        for role in ('independent','replay'):
            for chip in CHIPS:
                records=[r for r in all_rows if r['op']==label and r['chip']==chip and r['data_role']==role]
                s=summary(records,'us')
                compared=[r for r in records if r['us'] is not None and r['previous_us'] is not None]
                results[label][chip].setdefault('by_role',{})[role]=dict(**s,records=len(records),unpriced=len(records)-s['n'],sources=dict(Counter(r['classification'] for r in records)),prior_matched=summary(compared,'previous_us'),current_matched=summary(compared,'us'))
                columns.append(f"{s['shapes']}/{s['n']} · {s['wape']:.2f}% / {s['maximum']:.2f}%" if s['n'] else '0/0 · N/A')
        lines.append(f"| {label} | {' | '.join(columns)} |")
        for chip in CHIPS:
            v=results[label][chip];assert sum(a['n'] for a in v['by_role'].values())==v['full']['n']
    lines+=['','独立窗必须有同一输入的采前冻结、精确芯片和合格收据，且之前没有用于报告/校准。已使用的旧留出、采集中才冻结、未知角色全部归入开发/回放。不能用全量回放通过门槛来证明独立泛化。非默认协议分别列行，不能相加作为不同物理窗。候选算法不计入当前公开预测。',
            '','WAPE = Σ|预测−CA| / ΣCA，最大为单窗最大绝对相对误差，有符号偏差 = Σ(预测−CA) / ΣCA。'
            '各端分别要求 WAPE ≤25%、最大 ≤50%。完整历史回放包含开发样本；新窗的数据角色和跨芯片外推逐项保留。'
            '无合格 CA 的设备不计误差。','',
            '[每个 shape 的模型、CA、接口、诊断来源和 970DT 外推](replay.json) · [同一组窗口的旧版对照](evaluation.json) · '
            '[各算子大形覆盖](large/report.md) · [总误差分析](../../docs/error-analysis.md)','']
    save('replay.json',dict(rows=all_rows,units='us',version=api.__version__))
    save('evaluation.json',dict(summary=results,threshold_pct=dict(wape=25,maximum=50)))
    tmp=OUT/('acceptance.'+str(os.getpid())+'.tmp');tmp.write_text('\n'.join(lines));tmp.replace(OUT/'acceptance.md')
    save('model.json',manifest)
    w,=(ROOT/'build/wheels/v032').glob('*.whl')
    save('wheel-check.json',dict(wheel_sha256=sha256(w.read_bytes()).hexdigest(),version=api.__version__,
         distinct_queries=len(wall),all_under_100ms=max(wall)<100,in_process_first_query_ms=dict(p50=median(wall),p95=sorted(wall)[ceil(.95*len(wall))-1],maximum=max(wall))))
    print(len(all_rows),'windows across protocols;',len(wall),'distinct full API calls')


if __name__=='__main__':
    evaluate(sys.argv[1])
