"""Check that every AIC partial-sum byte is read exactly once by the final AIV merge."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from development.validation.matmul31.device.campaign import ca, OUT, REG


def flattened(intervals):
    """Return the union without padding. Repeated or overlapping accesses are significant."""
    runs=[]; overlap=0
    for start,end in sorted(intervals):
        if end<=start:raise ValueError('empty trace interval')
        if runs and start<=runs[-1][1]:
            overlap+=max(0,min(end,runs[-1][1])-start);runs[-1][1]=max(end,runs[-1][1])
        else:runs.append([start,end])
    return runs,overlap


def audit(stage,shape,splits):
    m,n,k=shape
    shards=(ROOT/stage['trace']).parent
    fixed=[]; loads=[]; stores=[]; src_moves=[];active_readers=set();active_writers=set();metadata_loads=[]
    for core in stage['windows']:
        for e in json.loads((shards/core/'trace.json').read_bytes())['traceEvents']:
            name=e['name']
            if e.get('ph')!='X' or name not in ('FIX_L0C_TO_DST','MOV_SRC_TO_DST_ALIGNv2','MOV_OUT_TO_L1_MULTI_ND2NZ'):continue
            x=json.loads(e['args']['detail'])
            if any(x.get(k) for k in ('is_wait','is_sync','is_duration_excluded')):continue
            if name=='FIX_L0C_TO_DST':
                data=x['processed_bytes']; assert data==4*x['m']*x['n']; start=x['dst_addr'];fixed.append((start,start+data))
            elif name=='MOV_OUT_TO_L1_MULTI_ND2NZ':src_moves.append(x['processed_bytes'])
            elif x['src_mem']=='OUT' and x['dst_mem']=='UB':
                count,xlen=x['burst_number'],x['burst_len'];assert count*xlen==x['processed_bytes']
                loads.append((core,x['src_addr'],count,xlen,x['src_stride']))
            elif x['src_mem']=='UB' and x['dst_mem']=='OUT':
                count,xlen=x['burst_number'],x['burst_len'];assert count*xlen==x['processed_bytes'];active_writers.add(core)
                for i in range(count):
                    start=x['dst_addr']+i*x['dst_stride'];stores.append((start,start+xlen))
    if not fixed:raise ValueError('No AIC partial writes in trace')
    fix_union,fix_overlap=flattened(fixed); assert not fix_overlap
    reads=[]; unassigned=[]
    for core,start,count,xlen,stride in loads:
        intervals=[(start+i*stride,start+i*stride+xlen) for i in range(count)]
        hits=[any(a<=lo<hi<=b for a,b in fix_union) for lo,hi in intervals]
        if all(hits):reads.extend(intervals);active_readers.add(core)
        elif any(hits) or any(max(lo,a)<min(hi,b) for lo,hi in intervals for a,b in fix_union):unassigned.append((core,start,count,xlen,stride))
        else:metadata_loads.extend(intervals)
    writes,wover=flattened(stores);read_union,rover=flattened(reads)
    tot=lambda pairs:sum(hi-lo for lo,hi in pairs)
    checks=dict(no_unassigned_partial_overlap=not unassigned,exact_readback=fix_union==read_union and not rover,
                partial_matches_logical=tot(fixed)==4*m*n*splits,
                final_matches_logical=len(writes)==1 and not wover and tot(writes)==2*m*n)
    return dict(checks=checks,actual_partial_write_bytes=tot(fixed),actual_partial_read_bytes=tot(reads),actual_output_write_bytes=tot(stores),
                total_logical_A_B_bytes=2*k*(m+n),actual_GM_to_L1_descriptor_bytes=sum(src_moves),outside_workspace_load_bytes=tot(metadata_loads),
                aiv_reading_partials=len(active_readers),aiv_writing_result=len(active_writers),partial_address_intervals=len(fix_union),
                partial_read_address_intervals=len(read_union),unassigned_partial_overlap=unassigned,duplicate_read_bytes=rover,
                scope='Actual GM-side descriptors, not physical HBM bus transactions. An AIV read counts only if its full addressed byte range is inside a traced FP32 FIXP destination. Outside-workspace reads recorded separately.')


def build():
    reg=ca.read(REG);rows=[]
    cached=ca.read(OUT/'traffic.json')['rows'] if (OUT/'traffic.json').exists() else []
    by_hash={r['trace_sha256']:r for r in cached if r.get('checks')}
    received=0
    selected={(tuple(j['shape']),sku):(j,sku) for j in reg['rows'] for sku in (ca.DT,ca.PR)}
    for key,(j,sku) in selected.items():
        case='-'.join(map(str,j['shape']))+('-dt' if sku==ca.DT else '-pr');path=OUT/(case+'.json')
        if not path.exists():continue
        received+=1
        d=ca.read(path)
        if not d['qualified']:continue
        s,=d['stages']
        if s['trace_sha256'] in by_hash:
            rows.append(by_hash[s['trace_sha256']]);continue
        try:parts=audit(s,j['shape'],j['predictions'][sku]['geometry']['k_splits'])
        except (KeyError,ValueError,AssertionError) as exc:parts={'failure':repr(exc)}
        rows.append(dict(shape=j['shape'],sku=sku,receipt=path.name,trace_sha256=s['trace_sha256'],**parts))
    ca.save(OUT/'traffic.json',dict(freeze_sha256=ca.digest(REG),rows=rows))

    lines=['# 切 K 部分和的地址核对','',
           '| [M,N,K] | 芯片 | AIC 实际写入的 FP32 部分和 B | AIV 实际读回 B | 最终 BF16 写回 B | 真正读部分和的 AIV | 区间逐字节闭合 |',
           '|---|---|---:|---:|---:|---:|---|']
    for r in rows:
        if 'checks' not in r:
            lines.append(f"| {r['shape']} | {r['sku'][6:11]} | — | — | — | — | 待核验 |")
            continue
        passed=all(r['checks'].values())
        lines.append(f"| {r['shape']} | {r['sku'][6:11]} | {r['actual_partial_write_bytes']:,} | {r['actual_partial_read_bytes']:,} | {r['actual_output_write_bytes']:,} | {r['aiv_reading_partials']} | {'是' if passed else '否'} |")
    lines+=['','从各 AIC 的 FIXP 描述符取得每个 FP32 部分和的目标地址。对 AIV 的 MTE2 源地址按 burst 个数、长度和 stride 展开，逐字节核对部分和写入、读回的区间集合与重叠。独立核对最终 MTE3 写出区间与整张 BF16 输出。统计口径是核侧发出的 GM 访问，物理 HBM 流量还依赖缓存命中和写回。只有正式完整窗的行进入该表。','',
            '[逐形核对、额外读取和 MTE2 输入量](traffic.json) · [计时与几何预测](report.md)','']
    (OUT/'traffic.md').write_text('\n'.join(lines))
    return received==len(selected)

if __name__=='__main__':
    if '--watch' in sys.argv:
        while not build():time.sleep(30)
    else:build()
