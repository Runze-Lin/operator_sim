"""Audit 950DT launch isolation without invoking CA or fitting to holdouts.

Reads raw per-process inputs and logs, helper flush records and SQLite instruction
intervals; joins to pre-CA predictions by operator/shape and to accepted rows by
exact DB path. Audit of preparatory and failed launches never enlarges the cohort.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
VERSION = ROOT / 'results/v0.12'
OPS = {'rmsnorm':'rms','mhc_post':'post'}
APIS = {'rmsnorm':'aclnnRmsNorm','mhc_post':'aclnnMhcPost'}
read = lambda p: json.loads(p.read_text())


def union(xs):
    v=[]
    for lo,hi in sorted(xs):
        if hi<=lo: continue
        if v and lo<=v[-1][1]: v[-1]=(v[-1][0],max(v[-1][1],hi))
        else: v.append((lo,hi))
    return v


def length(xs): return sum(b-a for a,b in xs)


def intersection(xs,ys):
    i=j=total=0
    while i<len(xs) and j<len(ys):
        a,b=xs[i]; c,d=ys[j]; total+=max(0,min(b,d)-max(a,c))
        if b<=d:i+=1
        else:j+=1
    return total


def read_db(path, op, shape):
    t,d=shape[0],shape[-1]
    rms=op=='rmsnorm'; expected={'y':t*d*2*(1 if rms else shape[1])}
    if rms:expected['rstd']=t*4
    bycore=defaultdict(lambda:defaultdict(list)); ranges=defaultdict(list); raw=Counter()
    events=defaultdict(list); burst_bytes_ok=True
    uri=path.resolve().as_uri()+'?mode=ro'
    with sqlite3.connect(uri,uri=True) as db:
        info=dict(db.execute('SELECT * FROM KernelInfo'))
        cores=list(db.execute('''SELECT ExecInstrCoreId,CoreTypeId,min(ExecInstrTickStart),max(ExecInstrTickEnd),count(*),sum(ExecInstrTickEnd<ExecInstrTickStart),sum(ExecInstrName GLOB 'RV_*') FROM ExecutedInstructions GROUP BY 1,2 ORDER BY 4 DESC,1,2'''))
        if not cores:return dict(db_rows=0,timing_db_ok=False,output_ok=False,launch_db=info.get('launch_id'),kernel=info.get('name')),[]
        c,u=cores[0][:2]; hz=float(info['frequency_ghz'])*1000
        for i,j,a,b,skip,params in db.execute("""SELECT ExecInstrCoreId,CoreTypeId,ExecInstrTickStart,ExecInstrTickEnd,ExecInstrIsDurationExcluded,ExtendParams FROM ExecutedInstructions WHERE ExecInstrName='MOV_SRC_TO_DST_ALIGNv2' AND json_extract(ExtendParams,'$.dst_mem')='OUT' ORDER BY ExecInstrTickStart"""):
            p=json.loads(params); width=p['burst_len']; n=p['burst_number']; key='rstd' if rms and width<2*d else 'y'
            if p['queue_type']!='MTE3' or width*n!=p['processed_bytes'] or min(width,n)<=0:burst_bytes_ok=False
            intervals=[(p['dst_addr']+k*p['dst_stride'],p['dst_addr']+k*p['dst_stride']+width) for k in range(n)]
            bycore[i,j][key].append((intervals,p['processed_bytes']))
            ranges[key]+=intervals;raw[key]+=p['processed_bytes']
            if (i,j)==(c,u) and not skip:events[key].append((a,b))
        simd=union(db.execute("""SELECT ExecInstrTickStart,ExecInstrTickEnd FROM ExecutedInstructions WHERE ExecInstrCoreId=? AND CoreTypeId=? AND ExecInstrIsDurationExcluded=0 AND json_extract(ExtendParams,'$.queue_type')='SIMD'""",(c,u)))
    db_rows=sum(x[4] for x in cores); ticks=max(x[3] for x in cores)-min(x[2] for x in cores)
    result=dict(kernel=info.get('name'), active=len(cores), db_rows=db_rows, reversed=sum(x[5] for x in cores), min_core_rv=min(x[6] for x in cores),
                launch_db=info.get('launch_id'), reference_us=ticks/hz, critical_core=c, critical_type=u,
                timing_db_ok=(hz==1650 and len(cores)==info.get('ai_core_active') and ticks==info.get('exec_time_ticks') and
                              db_rows==info.get('instrs_exec_num') and all(x[1] in (2,3) and x[5]==0 and x[6]>0 for x in cores)))
    valid=burst_bytes_ok; details=[]; bases={}
    for key,nbytes in expected.items():
        ur=union(ranges[key]); bases[key]=ur[0][0] if ur else 0
        result[key+'_expected_B']=nbytes; result[key+'_union_B']=length(ur); result[key+'_raw_B']=raw[key]
        valid &= len(ur)==1 and nbytes==raw[key]==length(ur)
        ages=union(events[key]);result[key+'_store_us']=length(ages)/hz; result[key+'_simd_overlap_us']=intersection(ages,simd)/hz
    # Check separate allocations, as well as complete coverage of each output.
    if rms:valid &= not intersection(union(ranges['y']),union(ranges['rstd']))
    for i,j,*_ in cores:
        core_row=dict(physical_core=i,core_type=j)
        for key in expected:
            copies=bycore[i,j][key]; valid &= bool(copies)
            spans=[union([(a-bases[key],b-bases[key]) for a,b in xs]) for xs,_ in copies]
            core_row[key+'_writes']=len(copies); core_row[key+'_write_bytes']=[n for _,n in copies]
            core_row[key+'_byte_ranges']=spans
        if rms:
            # Byte ranges for y/rstd on each core must describe the same logical rows.
            normal=lambda name,step:union([(a//step,b//step) for span in core_row[name+'_byte_ranges'] for a,b in span])
            valid &= (all(a%step==b%step==0 for name,step in [('y',2*d),('rstd',4)] for span in core_row[name+'_byte_ranges'] for a,b in span)
                      and normal('y',2*d)==normal('rstd',4))
        details.append(core_row)
    stores=union(events['y']+events['rstd']);simd_busy=length(simd); hidden=intersection(stores,simd)
    result.update(output_ok=bool(valid), critical_store_us=length(stores)/hz, critical_simd_us=simd_busy/hz,
                  critical_store_simd_overlap_us=hidden/hz, critical_store_minus_simd_us=(length(stores)-hidden)/hz)
    busiest=max(details,key=lambda x:sum(x['y_write_bytes']))
    result['busiest_y_write_B']=busiest['y_write_bytes']
    return result,details


def audit_launch(folder,index,op,shape,previous,path,log,outer,helper):
    db_name=str(path.relative_to(ROOT)) if path is not None else None
    row=dict(op=op,shape=shape,process=str(folder.relative_to(ROOT)),launch=index,source=db_name,prior=previous)
    causes=[]; segments=re.findall(fr'^BATCH_BEGIN {index}\s*$\n(.*?)^BATCH_END {index}\s*$',log,re.M|re.S)
    segment=segments[0] if len(segments)==1 else ''; dims='R C'.split() if op=='rmsnorm' else 'T N D'.split()
    header=re.search(rf'^case={op} api={APIS[op]} (.*)$',segment,re.M); attrs=dict(re.findall(r'(\w+)=(\w+)',header[1])) if header else {}
    exp=[math.prod(shape)]+([shape[0]] if op=='rmsnorm' else [])
    golden=re.findall(r'^correctness ([^\r\n]+)',segment,re.M)
    cf=[dict(re.findall(r'(\w+)=(\S+)',s)) for s in golden]
    row.update(api=APIS[op],golden_counts=[int(x.get('count',0)) for x in cf],golden_max_abs=[float(x['max_abs_error']) for x in cf if 'max_abs_error' in x],golden_ok=(all(x.get('bad')=='0' for x in cf) and [int(x.get('count',0)) for x in cf]==exp),
               shape_log_ok=[attrs.get(k) for k in dims]==list(map(str,shape)),task_count=len(set(re.findall(r'\[TASK_DONE\].*?stream_id:(\d+), task_id:(\d+) TASK_DONE',segment))))
    row['process_ok']=('SMOKE_PASS' in outer and 'Simulation SUCCESS' in outer and 'SMOKE_FAIL' not in outer and set(re.findall(r'runtime_soc=(\S+)',log))=={'Ascend950DT_9582'} and not re.search(r'\[ERROR\].*(execute_set_flag|execute_wait_flag|deadlock|invalid instruction)',log))
    part=re.split(r'\[cannsim_helper\] start launch: id=',helper)[1:]; matched=[p for p in part if p.startswith(str(index)+' ')]
    flush=re.findall(r'\binstruction handler flush: ([^\n]+)',matched[0]) if len(matched)==1 else []
    f=dict(re.findall(r'(\w+)=(\w+)',flush[0])) if len(flush)==1 else {}
    row['observed']=bool(matched or segments or path)
    row['helper']={k:(int(v) if v.isdigit() else v) for k,v in f.items()}
    details=[]
    if path is None:causes.append('DB absent/ambiguous')
    else:
        try:
            result,details=read_db(path,op,shape);row.update(result)
        except (OSError,sqlite3.Error,KeyError,ValueError) as exc:causes.append('DB: '+str(exc))
    row['helper_ok']=(path is not None and len(matched)==1 and f'db_path={path} ' in matched[0].splitlines()[0] and f.get('status')=='ok' and all(f.get(k)=='0' for k in ('invalid','mismatched','write_failed')) and str(row.get('db_rows'))==f.get('written'))
    for k in ('golden_ok','shape_log_ok','process_ok','helper_ok','timing_db_ok','output_ok'):
        if not row.get(k):causes.append(k)
    if row['task_count']!=1:causes.append('one runtime task required')
    if row.get('launch_db')!=index:causes.append('launch id mismatch')
    row['causes']=causes;row['timing_qualified']=not causes
    if causes and 'reference_us' in row:row['unqualified_db_span_us']=row.pop('reference_us')
    if previous is not None and row.get('active'):
        row['state']='first' if previous==0 else 'expand' if previous<row['active'] else 'reuse'
    return row,details


def scan(folders):
    audits=[];percore=[]
    for folder in folders:
        inputs=(folder/'inputs.txt').read_text().splitlines()
        logpaths=list(folder.rglob('npusim.log'));hpaths=list(folder.rglob('cannsim_helper.log'))
        logs=[p.read_text(errors='replace') if p is not None else '' for p in ((logpaths or [None])[0],folder/'run.log' if (folder/'run.log').exists() else None,(hpaths or [None])[0])]
        if max(len(logpaths),len(hpaths))>1:raise ValueError('Ambiguous process '+str(folder))
        before=0
        for i,line in enumerate(inputs):
            op,idx,*args=line.split();shape=[int(a) for a in args[:2 if op=='rmsnorm' else 3]]
            if int(idx)!=i:raise ValueError('non-ordinal launch '+line)
            candidates=list(folder.rglob(f'{i:04d}_*.db'));p=candidates[0] if len(candidates)==1 else None
            row,details=audit_launch(folder,i,op,shape,before,p,*logs);audits.append(row)
            percore.extend([dict(source=row['source'],**r) for r in details]); before=max(before,row['active']) if before is not None and row.get('timing_db_ok') and row.get('output_ok') else None
        known={r['source'] for r in audits if r['source']}
        for db in folder.rglob('*.db'):
            name=str(db.relative_to(ROOT))
            if name not in known:audits.append(dict(op=op,process=str(folder.relative_to(ROOT)),source=name,launch=db.name.split('_')[0],shape=[],prior=None,timing_qualified=False,causes=['unassigned extra or ambiguous DB']))
    return audits,percore


def csv_out(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r));compact=lambda v:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows({k:compact(v) for k,v in r.items()} for r in rows)


def stats(rows,key):
    errors=[abs(r[key]-r['reference_us']) for r in rows];return 100*sum(errors)/sum(r['reference_us'] for r in rows),100*max(e/r['reference_us'] for e,r in zip(errors,rows))


def group(row):
    return 'few_wide' if row['shape'][0]<=4 and row['shape'][-1]>=2048 else 'ordinary'


def add_cohort(audits,allow_incomplete=False):
    lookup={r['source']:r for r in audits if r['source']};accepted=[]
    for op,sub in OPS.items():
        old=read(ROOT/'results/v0.11'/sub/'rows.json');old_shapes={tuple(r['shape']) for r in old}
        freeze=read(VERSION/sub/'freeze.json'); plan={tuple(c['shape']):c for c in freeze['cases'] if c['op']==op and c.get('role','validation')=='validation'}
        current=read(VERSION/sub/'rows.json');fresh=[r for r in (current if isinstance(current,list) else current['rows']) if r['role']=='validation']
        shapes={tuple(r['shape']) for r in fresh}
        if len(plan)!=10 or len(fresh)!=len(shapes) or shapes-set(plan) or set(plan)&old_shapes or (not allow_incomplete and shapes!=set(plan)):
            raise ValueError('Missing, nonindependent or duplicated shape: '+op)
        for r in fresh:
            c=plan[tuple(r['shape'])]; a=lookup[str(r.get('db_source') or r['source'])];
            if (not a['timing_qualified'] or not math.isclose(r['reference_us'],a['reference_us']) or list(r['shape'])!=a['shape'] or r['kernel']!=a['kernel'] or ('prior_aiv' in r and r['prior_aiv']!=a['prior'])):raise ValueError('Unqualified or inconsistent accepted launch '+str(a))
            if not math.isclose(a['critical_store_minus_simd_us'],r['phase_us']['MTE3'],abs_tol=1e-7):
                raise ValueError('Critical output-store accounting differs from MTE3 exposed time')
            a['planned_prior']=c.get('prior_aiv'); a.update(role='new_holdout',group=group(a),previous_us=c['previous_us'],predicted_us=c['prediction']['predicted_us'],id=r['id']);accepted.append(a)
            if not math.isclose(a['predicted_us'],r['predicted_us']) or not math.isclose(a['previous_us'],r['previous_us']):raise ValueError('timed vs frozen predictions differ')
            if r.get('db_sha256'):
                sha=hashlib.sha256()
                with (ROOT/a['source']).open('rb') as raw:
                    while chunk:=raw.read(2**20):sha.update(chunk)
                if sha.hexdigest()!=r['db_sha256']:raise ValueError('DB digest changed: '+a['source'])
                a['db_sha256']=sha.hexdigest()
    for r in audits:r.setdefault('role','control_or_unused' if r['timing_qualified'] else 'failed')
    return accepted


def compare_plot(rows,output):
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for i,op in enumerate(OPS):
        by=[r for r in rows if r['op']==op]; left,right=axes[i]
        for idx,(key,label,color) in enumerate([('previous_us','v0.11','#657789'),('predicted_us','v0.12','#008574')]):
            groups=[[r for r in by if r['group']==g] for g in ('few_wide','ordinary')]; xy=[stats(g,key) for g in groups]
            for k,(w,m) in enumerate(xy):
                x=k+(idx-.5)*.22;left.bar(x,w,width=.20,color=color,label=label if k==0 else None)
                left.plot([x,x],[w,m],color=color);left.scatter([x],[m],marker='_',color=color)
        left.set(title=f'{op}: error on the same new holdouts',ylabel='Relative error (%)',xticks=[0,1],xticklabels=[f'{l} (n={len(g)})' for l,g in zip(['Few tokens, wide','Other shapes'],groups)])
        left.legend(title='Bar = WAPE, cap = max',fontsize=8)
        if op=='rmsnorm':
            by=sorted(by,key=lambda r:(r['shape'][-1],r['shape'][0]));x=list(range(len(by)));hidden=[r['critical_store_simd_overlap_us'] for r in by]; exposed=[r['critical_store_minus_simd_us'] for r in by]
            right.bar(x,exposed,color='#00776e',label='Outside SIMD');right.bar(x,hidden,bottom=exposed,color='#e3a82b',label='Overlaps SIMD')

            for idx,r in enumerate(by):
                chunks='+'.join(str(b//(2*r['shape'][-1])) for b in r['busiest_y_write_B'])
                right.annotate(chunks,(idx,hidden[idx]+exposed[idx]),xytext=(0,3),textcoords='offset points',ha='center',fontsize=7)
            right.margins(y=.18)
            right.set(title='Critical AIV: output-copy union; labels = rows per copy',ylabel='CA time (µs)',xticks=x,xticklabels=['×'.join(map(str,r['shape'])) for r in by]);right.tick_params(axis='x',labelrotation=38,labelsize=8);right.legend()
        else:
            by=sorted(by,key=lambda r:(r['process'],r['launch']));x=list(range(len(by)));palette={'first':'#914a68','expand':'#e89437','reuse':'#368583'}
            for state,color in palette.items():
                ii=[i for i,r in enumerate(by) if r['state']==state];right.scatter(ii,[by[i]['reference_us'] for i in ii],c=color,s=40,label='CA '+state)
            for key,mark,color in [('previous_us','x','#748396'),('predicted_us','+','#0e4840')]:right.scatter(x,[r[key] for r in by],marker=mark,s=52,c=color,label='v0.11' if key=='previous_us' else 'v0.12')
            right.set(title='Post: actual launch order within each process',ylabel='Kernel time (µs)',xticks=x,xticklabels=[f"{r['shape'][0]}×{r['shape'][-1]}\n{r['prior']}→{r['active']}" for r in by]);right.tick_params(axis='x',labelrotation=38,labelsize=8);right.legend(ncol=2,fontsize=8)
        for ax in (left,right):ax.grid(axis='y',alpha=.22);ax.set_axisbelow(True)
    figure.savefig(output,dpi=150);plt.close(figure)


def run_report(out,records,selected):
    def p(x):return f'{x:.3f}'
    md=['# v0.12 独立样本：计时审计与大 shape 对照','','| 算子 / 组 | 新 shape | v0.11 WAPE / 最大 | v0.12 WAPE / 最大 |','|---|---:|---:|---:|']
    for op in OPS:
        by=[r for r in selected if r['op']==op]
        for label,g in [('全部',by),('小 token、宽通道',[r for r in by if r['group']=='few_wide']),('普通',[r for r in by if r['group']!='few_wide'])]:
            previous=stats(g,'previous_us');now=stats(g,'predicted_us');md.append(f'| {op} / {label} | {len(g)} | {previous[0]:.2f}% / {previous[1]:.2f}% | {now[0]:.2f}% / {now[1]:.2f}% |')
    md+=['','两栏都是新冻结清单上的默认预测，不传 `resident_aiv`。小宽组为 T≤4 且 D≥2048；WAPE=Σ|预测−实测|/Σ实测，最大误差=max(|预测−实测|/实测)。旧留出已转开发，本表不计入。', '', '![默认模型同集误差、关键核写出交集及实测状态](comparison.png)','',
         '## 采集是否可用于计时','',f"在原始输入和 DB 中核对 {len(records)} 条记录（其中 {sum(r.get('observed', True) for r in records)} 条有 launch 证据）；{len(selected)} 个独立留出，{sum(r['role']=='control_or_unused' for r in records)} 个通过审计但未纳入独立误差，{sum(not r['timing_qualified'] for r in records)} 个失败或未完成。每个 launch 的原始相对路径、核、golden、helper 回调计数和失败原因见 [launches.csv](launches.csv)。",'',
         '从输入文件及分段日志独立核对 shape、唯一 runtime task、ACLNN、950DT、进程成功标志；BF16 主输出与独立 CPU 参考比较，RMSNorm 还检查每个 token 的 FP32 rstd。helper 的 flush 必须成功、invalid/mismatched/write_failed 为零、written 等于 DB 记录数；检查 KernelInfo、1.65 GHz 下全核指令区间首尾、有效 AIV、RV 指令与零逆序。helper 的 overwrite/start_only/end_only 是回调分类，不等于 DB 丢失指令数。', '',
         '把每条输出 MOV 的 burst 展开为半开字节区间；单张量的原始字节、去重并集、连续跨度都必须等于理论大小。每一 AIV 都须写输出。RMSNorm 分别审计 y 与 rstd 的地址范围，要求两输出分配不重叠、同核写回行集合相符。[per-core.csv](per-core.csv) 给每核每笔的字节范围（相对各自缓冲区最低写地址）与写次数。此门禁可捕捉重复和跨 launch 污染；它不能证明模拟器记录了所有内部微操作。', '']
    for op in OPS:
        changes=[r for r in selected if r['op']==op and r['planned_prior']!=r['prior']]
        if changes:
            md += ['重采改变了 '+op+' 以下点的进程历史（冻结计划→实际前序 AIV）：'+
                   '；'.join(f"{r['shape']}: {r['planned_prior']}→{r['prior']}" for r in changes)+
                   '。两版默认 shape-only 预测不变，按实际前序分组；不能把原计划的状态预测当作相同状态下的验证。','']
    failures=[r for r in records if not r['timing_qualified']]
    if failures:
        md+=['| 原进程 / launch | shape | 被剔除的实证 | 独立进程重采（launch，CA µs） |','|---|---|---|---|']
        for r in failures:
            detail=f"helper invalid={r['helper'].get('invalid')}，落库 {r['db_rows']} 条" if r.get('db_rows')==0 else f"主输出实写/应写 {r['y_raw_B']:,}/{r['y_expected_B']:,} B" if not r.get('output_ok') and 'y_raw_B' in r else ', '.join(r['causes'])
            recovered=next((a for a in selected if (a['op'],a['shape'])==(r['op'],r['shape'])),None)
            replacement=f"{Path(recovered['process']).name} / {recovered['launch']}，{p(recovered['reference_us'])}" if recovered else '未取得'
            md.append(f"| {Path(r['process']).name} / {r['launch']} | {r['shape']} | {detail} | {replacement} |")
        md.append('')
    md+=['## RMSNorm：输出分块与重叠','','最后完成的 AIV 上分别对主输出、rstd 的 MTE3 时间段求并，并与 SIMD 并集取交。末两列是两输出先取并，再与 SIMD 求交；同一周期不会重复计入；逐 shape 核对未重叠时间与完整流水独占 MTE3 之差小于 10⁻⁷ µs。分块列按写主输出字节最多的一核列出了指令实际覆盖行数，与最后完成核可以不同。', '', '| [T,D] | 最忙核 y 分块（行） | y 忙 / ∩SIMD (µs) | rstd 忙 / ∩SIMD (µs) | 写出并集 / ∩SIMD (µs) | 写出未与 SIMD 重叠 (µs) |', '|---|---|---:|---:|---:|---:|']
    for r in sorted((r for r in selected if r['op']=='rmsnorm'),key=lambda r:(r['shape'][-1],r['shape'][0])):
        chunks='+'.join(str(n//(2*r['shape'][-1])) for n in r['busiest_y_write_B']);md.append(f"| {r['shape']} | {chunks} | {p(r['y_store_us'])} / {p(r['y_simd_overlap_us'])} | {p(r['rstd_store_us'])} / {p(r['rstd_simd_overlap_us'])} | {p(r['critical_store_us'])} / {p(r['critical_store_simd_overlap_us'])} | {p(r['critical_store_minus_simd_us'])} |")
    md+=['','## Post：首次、扩核、热态','','状态由同一进程内前面所有 launch 的实际最大 AIV 数重建，包括控制 launch。首次 = 此进程还没有前序，扩核 = 当前活跃核数超过该高水位，热态 = 此次不必超过该高水位。这是调用历史分类，不等于直接测得芯片 cache 命中。','','| 进程 / launch | [T,N,D] | 先前→当前 AIV | 状态 | 完整 CA (µs) | v0.11 / v0.12 默认 (µs) |','|---|---|---|---|---:|---:|']
    for r in sorted((r for r in records if r['op']=='mhc_post'),key=lambda r:(r['process'],r['launch'])):
        label=f"{p(r['previous_us'])} / {p(r['predicted_us'])}" if r['role']=='new_holdout' else r['role'];md.append(f"| {Path(r['process']).name} / {r['launch']} | {r['shape']} | {r['prior']}→{r.get('active','?')} | {r.get('state','unknown')} | {p(r['reference_us']) if r['timing_qualified'] else 'N/A'} | {label} |")
    md+=['', '| Post 组 / 实际状态 | 新样例 | v0.11 WAPE / 最大 | v0.12 WAPE / 最大 |','|---|---:|---:|---:|']
    for group_name in ('few_wide','ordinary'):
        for state in ('first','expand','reuse'):
            ss=[r for r in selected if r['op']=='mhc_post' and r['group']==group_name and r['state']==state]
            if not ss:continue
            w0,m0=stats(ss,'previous_us');w1,m1=stats(ss,'predicted_us')
            md.append(f'| {group_name} / {state} | {len(ss)} | {w0:.2f}% / {m0:.2f}% | {w1:.2f}% / {m1:.2f}% |')
    md+=['','## 同宽、大工作量的实测趋势','','只列 T≥64，且 op、通道宽、流数、kernel、实际活跃 AIV、真实状态相同的相邻新样本。斜率是新增主输出每 MiB 对应的总耗时差（含计算和写回）；若分块或状态改变，不应拿单一斜率外推。', '', '| 条件 | T | 主输出 MiB | 完整 CA (µs) | 相邻 Δus/MiB |','|---|---|---|---|---|']
    g=defaultdict(list)
    for r in selected:
        if r['shape'][0]>=64:g[(r['op'],tuple(r['shape'][1:]),r['kernel'],r['active'],r['state'])].append(r)
    for key,rr in sorted(g.items()):
        rr.sort(key=lambda r:r['shape'][0]);us=[r['reference_us'] for r in rr];xs=[r['y_expected_B']/2**20 for r in rr];slopes=[(b-a)/(y-x) for a,b,x,y in zip(us,us[1:],xs,xs[1:])];md.append(f"| {key[0]} {key[1]}，{key[2]}，{key[3]} AIV，{key[4]} | {', '.join(str(r['shape'][0]) for r in rr)} | {', '.join(map(p,xs))} | {', '.join(map(p,us))} | {', '.join(map(p,slopes)) or 'N/A'} |")
    md+=['','这些是 950DT CA 有限区间的观测；组内仅一个点不能判定线性，两个点不能验证直线；即使三个点接近，也不能据此外推跨分支或跨代。图中的模型误差和对照都不使用旧 shape 充作新点。','', '复现：在仓库根目录执行 `python development/validation/audit_v12.py`（需要保留本轮原始 DB、日志及已发布的两份冻结记录和逐 shape 结果；绘图需要 matplotlib）。该命令只重读数据，不启动 CA。','']
    (out/'summary.md').write_text('\n'.join(md))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=VERSION/'audit');parser.add_argument('--folders',nargs='+',type=Path);parser.add_argument('--audit-only',action='store_true');parser.add_argument('--allow-incomplete',action='store_true',help='Report only accepted fresh rows when a frozen collection budget ends before the target is met');args=parser.parse_args()
    base_dirs=[*ROOT.glob('runs/*12*'),*ROOT.glob('development/validation/*12/raw'),*ROOT.glob('development/validation/*12/runs')]
    folders=args.folders or sorted({p.parent for base in base_dirs if base.is_dir() for p in base.rglob('inputs.txt')});folders=[p.resolve() for p in folders]
    if not folders:raise ValueError('No v0.12 capture folders; use --folders')
    audits,percore=scan(folders);selected=[] if args.audit_only else add_cohort(audits,args.allow_incomplete)
    out=args.output;out.mkdir(exist_ok=True,parents=True);csv_out(out/'launches.csv',audits);csv_out(out/'per-core.csv',percore)
    if args.audit_only:return
    with TemporaryDirectory() as cache:
        os.environ.setdefault('MPLCONFIGDIR',cache);os.environ.setdefault('MPLBACKEND','Agg');compare_plot(selected,out/'comparison.png')
    run_report(out,audits,selected)


if __name__=='__main__':main()
