"""Score immutable pre-acquisition predictions and verify physical K ranges independently."""
from collections import Counter, defaultdict
from dataclasses import replace
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import re
import sys
from statistics import median
from time import perf_counter

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import load_chip
from operator_model.matmul_splitk import estimate_matmul_splitk
from development.validation.matmul28.streamk import DT,PR,FUT,OUT

BASE='MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance'
PROVIDER=ROOT/'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_nn/mat_mul_v3'/BASE
def read(p):return json.loads(p.read_bytes())
def digest(p):return sha256(p.read_bytes()).hexdigest()
def write(p,s):q=p.with_suffix('.tmp');q.write_text(s);q.replace(p)
def metrics(rows,pred='candidate_us',gold='ca_us'):
    if not rows:return dict(n=0,wape=None,max_ape=None)
    err=[abs(r[pred]-r[gold]) for r in rows]
    return dict(n=len(rows),wape=100*sum(err)/sum(r[gold] for r in rows),max_ape=100*max(e/r[gold] for e,r in zip(err,rows)))

def shards(stage,shape):
    m,n,k=shape
    directory=(ROOT/stage['trace']).parent;parts=[];vec=[];files=[]
    for core in sorted(stage['windows']):
        p=directory/core/'trace.json';file_bytes=p.read_bytes();files.append(dict(core=core,sha256=sha256(file_bytes).hexdigest()))
        mm=[];reads=[];writes=[];fix=[];work_reads=[]
        for e in json.loads(file_bytes)['traceEvents']:
            if e.get('ph')!='X':continue
            name=e['name']
            if name not in ('MMAD','MOV_OUT_TO_L1_MULTI_ND2NZ','FIX_L0C_TO_DST','MOV_SRC_TO_DST_ALIGNv2'):continue
            x=json.loads(e['args']['detail'])
            if any(x.get(v) for v in ('is_wait','is_sync','is_duration_excluded')):continue
            if name=='MMAD':mm.append([x['m'],x['n'],x['k']])
            elif name=='MOV_OUT_TO_L1_MULTI_ND2NZ':reads.append(dict(addr=x['src_addr'],n=x['n'],d=x['d']))
            elif name=='FIX_L0C_TO_DST':fix.append(dict(addr=x['dst_addr'],size=x['processed_bytes']))
            elif x['src_mem']=='UB' and x['dst_mem']=='OUT':writes.append(dict(addr=x['dst_addr'],size=x['processed_bytes']))
            elif x['src_mem']=='OUT' and x['dst_mem']=='UB' and x['burst_number']>1:work_reads.append(x['processed_bytes'])
        if '.cube' in core:
            assert len(mm)==1 and mm[0][:2]==[m,n] and len(fix)==1 and len(reads)==2
            depth=mm[0][2]
            a,=tuple(q for q in reads if (q['n'],q['d'])==(m,depth))
            b,=tuple(q for q in reads if (q['n'],q['d'])==(depth,n))
            parts.append(dict(core=core,actual_mmad=mm[0],a_address=a['addr'],b_address=b['addr'],workspace_address=fix[0]['addr'],workspace_bytes=fix[0]['size']))
        else:vec.append(dict(core=core,partial_read_bytes=sum(work_reads),write_bytes=sum(w['size'] for w in writes),output_writes=writes))
    # Matrix A is row-major: successive contiguous K slices of its first row.
    # Matrix B is row-major: the corresponding offset is K_start * N * sizeof(BF16).
    parts.sort(key=lambda q:q['a_address']);a0=parts[0]['a_address'];b0=parts[0]['b_address'];start=0
    for item in parts:
        assert item['a_address']==a0+start*2 and item['b_address']==b0+start*n*2
        item['k_start']=start;start+=item['actual_mmad'][2];item['k_end']=start
    assert start==k and len(parts)==stage['active_aic'] and len(vec)==stage['active_aiv']
    assert len({p['workspace_address'] for p in parts})==len(parts)
    stores=sorted((w['addr'],w['size']) for v in vec for w in v['output_writes'])
    offset=stores[0][0]
    for addr,size in stores:assert addr==offset;offset+=size
    assert offset-stores[0][0]==m*n*2
    return dict(all_shard_sha256=files,k_partitions=parts,unique_actual_k=sorted({q['actual_mmad'][2] for q in parts}),
        distinct_workspace_starts=len({p['workspace_address']for p in parts}),total_partial_write_bytes=sum(p['workspace_bytes'] for p in parts),
        aiv_counts_with_output=sum(v['write_bytes']>0 for v in vec),partial_vec_payload_read_bytes=sum(v['partial_read_bytes'] for v in vec),
        complete_nonoverlapping_final_bytes=offset-stores[0][0],vec=vec)

def audit():
    freeze=OUT/'streamk-freeze.json';orig=OUT/'freeze.json';f=read(freeze);o=read(orig)
    assert all(digest(ROOT/p)==v for p,v in f['source_sha256'].items())
    blob=digest(PROVIDER.with_suffix('.o'));manifest=read(PROVIDER.with_suffix('.json'));desc=next(q for q in manifest['kernelList'] if q['tilingKey']==4098)
    rows=[];ratios=[];queries=[]
    existing=OUT/'streamk-score.json'
    cached={r['trace_sha256']:r['mechanism'] for r in read(existing)['rows'] if r.get('mechanism')} if existing.exists() else {}
    for j,old in zip(f['rows'],o['rows']):
        assert j['shape']==old['shape']
        estimates=j['predictions'];s=j['shape']
        item={}
        for sku,tag in ((DT,'dt'),(PR,'pr'),(FUT,None)):
            p=estimates[sku];z=old['predictions'][sku]
            t=perf_counter();production=estimate_matmul_splitk(s,chip=sku);queries.append((perf_counter()-t)*1000)
            assert isclose(production['latency_us'],p['us'],rel_tol=1e-12) and (production['active_aic'],production['active_aiv'])==(p['aic'],p['aiv']) and production['route']==p['route']
            row=dict(sku=sku,shape=s,lane=old['lane'],model=old['model'],node=old['node'],source_batch=old['requests'],source_q=old['query_per_request'],
                role='extrapolation only' if tag is None else 'already started' if p['already_started'] else 'prospective endpoint',
                ordinary_us=z['us'],candidate_us=p['us'],predicted_route=p['route'],predicted_cores=[p['aic'],p['aiv']],candidate=production)
            rows.append(row)
            if not tag:continue
            label='-'.join(map(str,s))+'-'+tag;receipt=OUT/(label+'.json');sup=OUT/(label+'-drain.json')
            if not receipt.exists():row['status']='pending';continue
            d=read(receipt);assert d['freeze_sha256']==digest(orig) and d['shape']==s and d['chip']==sku
            assert (d['cost']['started_utc'] <= f['utc'])==p['already_started'],'registration ordering mismatch'
            q=read(sup) if sup.exists() else d
            row.update(receipt=str(receipt.relative_to(ROOT)),original_sha256=digest(receipt),original_verdict=d.get('qualified'),original_reason=d.get('reason'),
                       wall_s=d['cost']['seconds'],started_utc=d['cost']['started_utc'])
            if not q.get('qualified'):
                row['status']='no complete qualified receipt';continue
            if sup.exists():
                assert q['acquisition_sha256']==digest(receipt)
                row.update(drain_receipt=str(sup.relative_to(ROOT)),drain_sha256=digest(sup))
            st,=q['stages'];key=int(re.search(r'_high_performance_(\d+)',st['kernel'])[1])
            assert st['binary_sha256']==blob
            row.update(status='qualified',ca_us=q['ca_us'],observed_key=key,kernel=st['kernel'],trace_sha256=st['trace_sha256'],aic=st['active_aic'],aiv=st['active_aiv'],actual_mmads=st['mmad'],
                       correct_core_counts=row['predicted_cores']==[st['active_aic'],st['active_aiv']],correctness=q.get('correctness'))
            if p['route']=='one_output_split_k':
                assert key==4098 and st['kernel'].endswith('_4098_mix_aic')
                row['mechanism']=cached.get(st['trace_sha256']) or shards(st,s)
            item[sku]=row
        if len(item)==2:
            a,b=item[DT],item[PR];ratios.append(dict(shape=s,role='prospective both' if a['role']==b['role']=='prospective endpoint' else 'at least one endpoint already started',
                predicted_route=a['predicted_route'],ca_us=a['ca_us']/b['ca_us'],candidate_us=a['candidate_us']/b['candidate_us'],ordinary_us=a['ordinary_us']/b['ordinary_us']))
    groups={}
    for sku in (DT,PR):
        for role in ('already started','prospective endpoint'):
            for branch in ('one_output_split_k','ordinary'):
                sel=[x for x in rows if x['sku']==sku and x['role']==role and x['predicted_route']==branch and x.get('status')=='qualified']
                groups[sku+'/'+role+'/'+branch]=dict(ordinary=metrics(sel,'ordinary_us'),candidate=metrics(sel),correct_core_counts=sum(x['correct_core_counts'] for x in sel))
    dt=load_chip()
    scenarios={};shape=[16,64,7168]
    for name,chip in [('dt',dt),('same-config-new-name',replace(dt,name='anonymous')),
        ('1.2x frequency and peak rates',replace(dt,frequency_ghz=dt.frequency_ghz*1.2,cube_tops={k:(v*1.2 if v else v) for k,v in dt.cube_tops.items()},vector_tops={k:(v*1.2 if v else v) for k,v in dt.vector_tops.items()})),
        ('half AIC/AIV, same per-core rates',replace(dt,cube_cores=dt.cube_cores//2,vector_cores=dt.vector_cores//2,
        cube_tops={k:(v/2 if v else v) for k,v in dt.cube_tops.items()},vector_tops={k:(v/2 if v else v) for k,v in dt.vector_tops.items()})),
        ('10% HBM',replace(dt,hbm_bandwidth_gbs=dt.hbm_bandwidth_gbs*.1)),('explicit L2 at 10% of default service rate',replace(dt,details={**dt.details,'l2_bandwidth_gbs':(dt.details.get('l2_bandwidth_gbs') or dt.hbm_bandwidth_gbs)*.1})),('half FP32 Vector',replace(dt,vector_tops={**dt.vector_tops,'32':dt.vector_tops['32']*.5})),('970 resource scenario',load_chip(FUT))]:
        scenarios[name]=estimate_matmul_splitk(shape,chip=chip)
    for k in (15000,15104,15105,15360,16384):scenarios['970 K='+str(k)]=estimate_matmul_splitk([16,64,k],chip=FUT)
    out=dict(freeze_sha256=digest(freeze),utc=f['utc'],old_freeze_sha256=digest(orig),installed_elf_sha256=blob,
             installed_manifest_key4098=desc,source_branch='DAV_3510',rows=rows,groups=groups,dt_over_pr_ratios=ratios,
             sensitivities=scenarios,online_ms=dict(median=median(queries),max=max(queries),queries=len(queries)))
    write(existing,json.dumps(out,indent=2,allow_nan=False)+'\n');report(out)

def report(a):
    head=['# MatMulV3 单输出矩形的 Stream-K 独立核查','',
          '| SKU | 数据角色 | 完整窗 | 普通 WAPE / 最大 | 冻结 Stream-K WAPE / 最大 | AIC、AIV 均正确 |','|---|---|---:|---:|---:|---:|']
    def fmt(z):return 'N/A' if not z['n'] else f"{z['wape']:.2f}% / {z['max_ape']:.2f}%"
    for sku in (DT,PR):
        for role,label in [('already started','冻结前已开始'),('prospective endpoint','真正前瞻端')]:
            k=a['groups'][sku+'/'+role+'/one_output_split_k'];head.append(f"| {sku[6:11]} | {label} | {k['candidate']['n']} | {fmt(k['ordinary'])} | {fmt(k['candidate'])} | {k['correct_core_counts']} |")
    head+=['','四形来自 DeepSeek-V4-Pro 的 indexer.weights_proj 几何，TP=8 时逻辑 B[K,N]=[7168,64]。B=1、Q=9 与 B=9、Q=1 都给 M=9。同一连续 BF16、三维几何与属性只计一个样本。宿主走 aclnnMatmul，量化、转置、非连续步长或融合的应用需要另行确认库派发。','',
           '| M | SKU | 角色 | 普通 µs | 冻结 µs | CA µs | 冻结偏差 | 实测 AIC / 启动 AIV / 写出 AIV | 真 MMAD K | 预算 K |','|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in a['rows']:
        if not r.get('mechanism'):continue
        d=r['mechanism'];ret=r['ca_us'];head.append(f"| {r['shape'][0]} | {r['sku'][6:11]} | {'前瞻' if r['role']=='prospective endpoint' else '已开始'} | {r['ordinary_us']:.3f} | {r['candidate_us']:.3f} | {ret:.3f} | {100*(r['candidate_us']/ret-1):+.2f}% | {r['aic']}/{r['aiv']}/{d['aiv_counts_with_output']} | {d['unique_actual_k'][0]} | {r['candidate']['max_aligned_depth']} |")
    ratios=[r for r in a['dt_over_pr_ratios'] if r['predicted_route']=='one_output_split_k']
    if ratios:
        head+=['','| M，DT/PR 时间比 | CA | 普通 | 冻结候选 | 来源 |','|---:|---:|---:|---:|---|']
        for r in ratios:head.append(f"| {r['shape'][0]} | {r['ca_us']:.4f} | {r['ordinary_us']:.4f} | {r['candidate_us']:.4f} | {'双端前瞻' if r['role']=='prospective both' else '至少一端已开始'} |")
    outside=[r for r in a['rows'] if r['predicted_route']=='ordinary' and r.get('status')=='qualified']
    if outside:
        head+=['','## 同队列中的其他矩形和分支','',
               '| [M,N,K] | SKU | 实测 key | 公共普通 AIC/AIV | 真 AIC/AIV | 普通 µs / CA µs | 误差 |','|---|---|---:|---:|---:|---:|---:|']
        for r in outside:
            p,c=r['candidate_us'],r['ca_us'];m,n,k=r['shape'];sc,sv=r['predicted_cores']
            head.append(f"| [{m},{n},{k}] | {r['sku'][6:11]} | {r['observed_key']} | {sc}/{sv} | {r['aic']}/{r['aiv']} | {p:.3f} / {c:.3f} | {100*(p/c-1):+.2f}% |")
        if any(r['observed_key']==4098 for r in outside):
            head+=['','表中存在一个实际多输出 4098 反例。即使无偏置、相同 ND/BF16、较长 K 且启动相同核数，输出被拆成多个矩形后仍需建模各矩形分得的 K 段数、哪个 AIC 同时处理多个矩形以及 reduce 的跨核任务。单矩形校准无法覆盖此路径。']
    head+=['','## 机制、来源和公式','',
        '实际安装的同一多入口 BF16 二进制由两端 dump 与安装包 SHA-256 一致性确认。普通 prefill 的 tiling key 131074 与这里的 4098 共享 ELF。正式 4098 kernelList 项标为 MIX_AIC、crossCoreSync=1、taskRation=1:2。清单的内部 kernelName 使用 ND_NZ 别名，宿主 BF16 连续 ND_ND 的实际执行名记录为 ND_ND 和 4098_mix_aic。按 arch35 位布局拆解 4098 得 API_LEVEL=2、MODEL=1 即 STREAM_K。每核真实 MMAD、分段源地址、FP32 部分和的不同写地址与 AIV 最终写回共同确认派发。','',
        '各 AIC 的一条 MMAD 都处理相同 M、N 和不同 K 区间。按 A 首地址排序，逐段核查 A_addr=A₀+2k_start、B_addr=B₀+2Nk_start，要求各区间恰覆盖 [0,K)，部分和写地址独立，AIV 最终写回恰覆盖 2MN 字节。原普通模板沿 MN 切，单一矩形只分给一个 AIC，因而四形预测均为约 15 µs。[逐核 MMAD、地址、写回与散列](streamk-score.json) · [开源 host 分支参考](../../../third_party/ops-nn/matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_streamk_tiling.cpp) · [CANNBot 950 切分机制](../../../third_party/cannbot-knowledge/knowledge/ops/ascendc/optimizations/opstune/mechanisms/ascendc_tiling.md)','',
        'C、V 为目标 AIC、AIV 数，f 为 GHz。先要求普通分块只有一个输出矩形、ceil(K/256)×256 ≥ max(8192,256C)/2。参考 DAV_3510 host 还会比较多个候选。本窄合同试 S=min(C,ceil(K/64)) 个部分和，每份以 D=64 ceil(ceil(K/S)/64) 计价。D 是资源预算的上取整。DT 实际执行 224，预算 256。','',
        'T_split = T_launch + max(T_partial, 2K(M+N)/(1000B_HBM)) + T_merge。B_HBM 是整芯标称 GB/s，T_partial 复用 BF16 Matmul 对 [M,N,D] 的 MTE2、MTE1、Cube 与尾写阶段。T_launch=(c_start+c_arrival log₂S)/(1000f)，c_start、c_arrival 是已有资源模型的发起周期。T_merge=(1000+60S×256/p)/(1000f) + 2MN(4S−1)/(1000B_L2)。p 是目标 AIV 的 FP32 ops/cycle，256 是参考值。1000 与 60 是参考 host 的 GetNegPerf 经验 cycles。2MN(4S−1) 表示额外归约流量估计。L2 缺省时退至 HBM 服务率，以 L2 给部分和定价属于本模型假设，开源 GetNegPerf 原式使用有效 HBM 带宽。T_split 低于普通价才选择切 K。','',
        '## 独立性与部署边界','',
        'M=1/9/11/16 共享 N=64、K=7168。冻结时 M=1/9 两端、11/DT 都有 raw 目录，全部列作开发/已开始。11/PR、16/DT、16/PR 是真正前瞻端，其中 16 是唯一双端都未开跑的几何。初始资格失败的端保留原回执。补充认证读取原始同一次采集，要求每核物理 END、所有描述符退休、聚合窗与全输出 correctness 一致。其他输出矩形的采集继续按旧队列完成。遇到多矩形仍走 4098 时，仅表示本候选尚未涵盖该路由。','',
        '只查询实验分支的调用方式如下。默认 estimate 继续使用其原路径。','',
        '§§§python','from operator_model.matmul_splitk import estimate_matmul_splitk','out = estimate_matmul_splitk([16, 64, 7168], chip="Ascend950PR_957b")','print(out["route"], out["latency_us"], out["active_aic"], out["active_aiv"])','§§§','',
        '## 970DT 门槛与资源敏感性','',
        '| 配置变动 | 分支 | 对齐后 K 门槛 | µs |','|---|---|---:|---:|']
    for label,r in a['sensitivities'].items():head.append(f"| {label} | {r['route']} | {r['threshold_aligned_k']:.0f} | {r['latency_us']:.3f} |")
    head+=['','970DT 的 120 AIC 将对齐后的 K 下限推至 15360，K=7168 时返回普通路径。原始 K=15105 向上对齐达门槛后仍需满足单矩形及成本判定。970 的 AIC:AIV=1:1，若模型预测切 K，还须假定未来库提供此比例的混合实现。950 的 4098 MIX 实测为 1:2，970 全为外推。[候选登记](streamk-freeze.json) · [原预测登记](freeze.json)','',
           f"进程内重算 3 配置 × 16 形的中位 {a['online_ms']['median']:.3f} ms、最大 {a['online_ms']['max']:.3f} ms。三种便携配置均与冻结的路由、核数、分段预算及总耗时一致。",'']
    write(OUT/'streamk-analysis.md','\n'.join(head).replace('§',chr(96)))
if __name__=='__main__':audit()
