"""Reconstruct output rectangle and K coordinate for each participating Cube core."""
from collections import defaultdict,Counter
from pathlib import Path
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from operator_model.matmul_multisplit import estimate_matmul_retile
from development.validation.matmul28.streamk_audit import read,write,metrics,DT,PR,FUT
OUT=ROOT/'results/v0.28/matmul/multi';PARENT=OUT.parent;REG=OUT/'freeze.json'

def geometry(stage,shape):
    m,n,k=shape;base=(ROOT/stage['trace']).parent;cores=[];vbytes=[]
    for c in sorted(stage['windows']):
        f=base/c/'trace.json';raw=f.read_bytes();mm=[];moves=[];fix=[];out=[]
        for e in json.loads(raw)['traceEvents']:
            if e.get('ph')!='X':continue
            name=e['name']
            if name not in ('MMAD','MOV_OUT_TO_L1_MULTI_ND2NZ','FIX_L0C_TO_DST','MOV_SRC_TO_DST_ALIGNv2'):continue
            d=json.loads(e['args']['detail'])
            if any(d.get(v) for v in ('is_sync','is_wait','is_duration_excluded')):continue
            if name=='MMAD':mm.append((d['m'],d['n'],d['k']))
            elif name=='MOV_OUT_TO_L1_MULTI_ND2NZ':moves.append((d['src_addr'],d['n'],d['d']))
            elif name=='FIX_L0C_TO_DST':fix.append((d['dst_addr'],d['processed_bytes']))
            elif d['src_mem']=='UB' and d['dst_mem']=='OUT':out.append((d['dst_addr'],d['processed_bytes']))
        if '.cube' in c:
            if not mm:continue
            cores.append(dict(core=c,mm=mm,moves=moves,fix=fix,sha256=hashlib.sha256(raw).hexdigest()))
        else:vbytes.extend(out)
    assert cores and vbytes
    a0=min(a for p in cores for a,_,_ in p['moves'])
    b0=min(a for p in cores for a,_,_ in p['moves'] if a>=a0+2*m*k)
    rectangles=defaultdict(list);hist=Counter();count_mm=0;partial_addr=set();per_core=[]
    for p in cores:
        starts=[a for a,_,_ in p['moves']]
        ma=min(a for a in starts if a<b0);mb=min(a for a in starts if a>=b0)
        mstart,kstart=divmod((ma-a0)//2,k);kcheck,nstart=divmod((mb-b0)//2,n)
        assert kstart==kcheck
        dims={(v[0],v[1]) for v in p['mm']};assert len(dims)==1
        dm,dn=dims.pop();depth=sum(v[2] for v in p['mm']);end=kstart+depth
        rectangles[mstart,nstart,dm,dn].append((kstart,end))
        hist.update(p['mm']);count_mm+=len(p['mm']);assert len(p['fix'])==1
        partial_addr.add(p['fix'][0][0]);per_core.append(dict(core=p['core'],sha256=p['sha256'],m0=mstart,n0=nstart,m=dm,n=dn,k0=kstart,k1=end,mmads=len(p['mm']),fixp_bytes=p['fix'][0][1]))
    assert len(partial_addr)==len(cores)
    rects=sorted(rectangles);exact_area=0
    for j,(mi,ni,dm,dn) in enumerate(rects):
        assert 0<=mi<mi+dm<=m and 0<=ni<ni+dn<=n
        exact_area+=dm*dn
        assert all(max(mi,mj)>=min(mi+dm,mj+em) or max(ni,nj)>=min(ni+dn,nj+en) for mj,nj,em,en in rects[:j])
        i=0
        for lo,hi in sorted(rectangles[mi,ni,dm,dn]):assert lo==i;i=hi
        assert i==k
    assert exact_area==m*n and count_mm==stage['mmad']
    stores=sorted(vbytes);offset=stores[0][0]
    for a,b in stores:assert a==offset;offset+=b
    assert offset-stores[0][0]==2*m*n
    return dict(output_rectangles=[dict(m_start=r[0],n_start=r[1],m=r[2],n=r[3],k_spans=sorted(rectangles[r])) for r in rects],
        groups=len(rects),splits_per_output=sorted({len(v) for v in rectangles.values()}),productive_aic=len(cores),
        active_aic=stage['active_aic'],active_aiv=stage['active_aiv'],actual_mmad=count_mm,
        mmads=[dict(m=a,n=b,k=c,count=v)for (a,b,c),v in sorted(hist.items())],cores=per_core,all_output_bytes=2*m*n)

def score():
    reg=read(REG);rows=[];miss=[];fn=OUT/'score.json';cached={}
    vendor_sha=next(v for f,v in reg['source_sha256'].items() if f.endswith('.o'))
    if fn.exists():cached={r['trace_sha256']:r.get('geometry') for r in read(fn)['rows']}
    sources=[]
    orig=read(PARENT/'freeze.json')
    for j in orig['rows']:
        for chip,tag in ((DT,'dt'),(PR,'pr')):
            if j['lane']!='decode':continue
            pred=estimate_matmul_retile(j['shape'],chip=chip)
            if pred['route']!='retiled_output_split_k':continue
            slug='-'.join(map(str,j['shape']))+'-'+tag;path=PARENT/(slug+'-drain.json')
            if not path.exists():path=PARENT/(slug+'.json')
            sources.append((j['shape'],chip,pred,j['predictions'][chip]['us'],path,'used in development'))
    for j in reg['rows']:
        for chip,tag in ((DT,'dt'),(PR,'pr')):
            p=j['predictions'][chip]
            sources.append((j['shape'],chip,p['new'],p['public']['latency_us'],OUT/('-'.join(map(str,j['shape']))+'-'+tag+'.json'),'frozen prospective'))
    for shape,chip,pred,old,path,role in sources:
        if not path.exists():miss.append(dict(shape=shape,sku=chip,role=role,reason='pending'));continue
        d=read(path)
        if not d.get('qualified'):miss.append(dict(shape=shape,sku=chip,role=role,reason=d.get('reason')));continue
        if role=='frozen prospective':
            assert hashlib.sha256(REG.read_bytes()).hexdigest()==d['register_sha256'] and d['cost']['started_utc']>reg['utc']
            source=path.with_name(path.stem+'-acquisition.json')
            assert hashlib.sha256(source.read_bytes()).hexdigest()==d['original_sha256']
            if d.get('acquisition_sha256'):assert d['acquisition_sha256']==d['original_sha256']
        st,=d['stages'];assert st['binary_sha256']==vendor_sha
        ca=st['ca_us'];geom=cached.get(st['trace_sha256'])
        geometry_error=None
        if not geom and st['kernel'].endswith('_4098_mix_aic'):
            try:geom=geometry(st,shape)
            except (AssertionError,KeyError,OSError,ValueError,IndexError) as exc:geometry_error=repr(exc)
        ok=bool(geom and pred['route']=='retiled_output_split_k' and geom['groups']==pred['output_tiles'] and geom['splits_per_output']==[pred['k_splits']] and geom['productive_aic']==pred['productive_aic'] and (geom['active_aic'],geom['active_aiv'])==(pred['active_aic'],pred['active_aiv']))
        rows.append(dict(shape=shape,sku=chip,role=role,receipt=str(path.relative_to(ROOT)),trace_sha256=st['trace_sha256'],kernel=st['kernel'],ca_us=ca,
            old_us=old,pred_us=pred['latency_us'],geometry_matches=ok,mmad_matches=bool(geom and geom['actual_mmad']==pred.get('mmads_total')),
            geometry=geom,geometry_error=geometry_error,prediction=pred,original_acquisition=(str(source.relative_to(ROOT)) if role=='frozen prospective' else None),
            original_sha256=d.get('original_sha256',d.get('acquisition_sha256'))))
    groups={}
    for sku in (DT,PR):
        for role in ('used in development','frozen prospective'):
            w=[r for r in rows if (r['sku'],r['role'])==(sku,role)]
            groups[sku+' '+role]=dict(old=metrics(w,'old_us'),candidate=metrics(w,'pred_us'),all_geometry_and_mmad=bool(w)and all(r['geometry_matches'] and r['mmad_matches'] for r in w))
    forward=[groups[sku+' frozen prospective'] for sku in (DT,PR)]
    complete=all(g['candidate']['n']==len(reg['rows']) for g in forward)
    passed=complete and all(g['all_geometry_and_mmad'] and g['candidate']['wape']<=25 and g['candidate']['max_ape']<=50 and g['candidate']['wape']<g['old']['wape'] for g in forward)
    result=dict(rows=rows,missing=miss,groups=groups,
        forward_gate='insufficient completed windows' if not complete else 'passed narrow frozen gate' if passed else 'failed narrow frozen gate',
        default_route='separate experimental candidate; promotion requires separate review of historical replays')
    write(fn,json.dumps(result,indent=2)+'\n');report(result)
def report(d):
    s=['# 重选输出块后切 K','',
       '| SKU | 样本角色 | 完整窗 | 旧 WAPE / 最大 | 冻结候选 WAPE / 最大 | 每个分块与 MMAD 都对齐 |','|---|---|---:|---:|---:|---|']
    for name,v in d['groups'].items():
        fmt=lambda x:f"{x['wape']:.2f}% / {x['max_ape']:.2f}%" if x['n'] else 'N/A'
        a=v['candidate'];s.append(f"| {name[:11]} | {name.split(' ',1)[1]} | {a['n']} | {fmt(v['old'])} | {fmt(a)} | {('是' if v['all_geometry_and_mmad'] else '否') if a['n'] else 'N/A'} |")
    s+=['','| [M,N,K] | SKU | 来源 | 实测输出矩形 × K 段 | 生产 µs | 候选 µs | CA µs | 真实/预测 MMAD |','|---|---|---|---|---:|---:|---:|---:|']
    for r in d['rows']:
        g=r['geometry'];v=r['prediction'];s.append(f"| {r['shape']} | {r['sku'][6:11]} | {r['role']} | {str(g['groups'])+'×'+str(g['splits_per_output']) if g else '其他库 key'} | {r['old_us']:.3f} | {r['pred_us']:.3f} | {r['ca_us']:.3f} | {g['actual_mmad'] if g else 'N/A'} / {v.get('mmads_total','N/A')} |")
    s+=['','## 资源机制','',
        '上游普通 Matmul 的输出 tile 不能直接复用于切 K。这里在常规方案只能发一波且原来不止一个输出块时，从目标 L0A、L0B 的 BF16 双缓冲和 64 深度最低要求得到输出分块的轴上限。AM、AN 分别为均分后输出轴的 16 元素对齐宽度。L0C 容纳 4AM·AN 字节才有资格。投影场景观察到的调度是尽量把输出合并，令 G=ceil(M/AM)ceil(N/AN)，每块最多 S=floor(C/G) 个 K 段，其中 C 是目标 AIC 数。各段深度为 ceil(K/S)，最后一段保留真实尾数。实际 MMAD 深度由 min(L0A/(4AM),L0B/(4AN)) 向下对齐 64 得出，另受 L1 双缓冲页限制。','',
        '按候选矩形重算源 host 的准入成本。usedCore/totalCore 是 HBM 与 L2 可达带宽的比例，比较两种几何的 max(有效传输成本,计算成本) 以及分段带来的归约开销。这与最终整窗的设备排程分开。最终时间为 launch + max(局部页流水,芯片冷 HBM 输入,L2 到各核的实际输入) + merge。merge 计价沿用已冻结的每个 K 分片发起与 FP32 归约机制。缺单独 L2 峰值时用旧 Matmul 已使用的 100 字节/核/周期。开源 DAV_3510 会枚举多种切 K 候选、并按平台实施再平衡，本候选只描述已经看到的单波饱和模式。','',
        '以 DT [144,64,7168] 为例，原普通模型推 9 个 tile，实际是整个输出一块。32 个 AIC 各处理 K=224，四个 MMAD 的深度为 64、64、64、32。[9,1280,5120] 则真实分成五个 [9,256]，各自的 K 连续覆盖 5120。DT 每块 6 段，PR 每块 5 段。启动 AIC 为 32/28，实际承担 MMAD 的 AIC 为 30/25。逐核输入地址、输出矩形、K 区间、结果写出与原始 SHA 可复算。[源数据](score.json)','',
        '新留出只用两种从真实模型推导的几何，每端各两窗，仍未达到十形。必须按前瞻登记的数值、物理退休、核数，以及输出分块与 K 边界资格审查。已看过的历史窗只作开发回放。970DT 仅作外推。后台严格等窄矩形前序队列完成。[冻结的三芯片预测和准入条件](freeze.json) · [状态](state.json)','']
    if (OUT/'dt-144-retile.png').exists():
        s+=['','## 同轴算例','',
            '![950DT 重选单个 144×64 输出矩形，模型与 CA 的四个 MMAD 对齐](dt-144-retile.png)','',
            '上图列出已建模的阶段预算，下图读取各自最后结束的 AIC 与 AIV。实测 FIXP 描述符可先发起再等待 Cube 写完，末尾归约预算只表征聚合发起和增量流量。在图中独立标注这些计时口径。[每条发起、整窗原点和原始 SHA](dt-144-retile.json)','']
    write(OUT/'report.md','\n'.join(s))
if __name__=='__main__':score()
