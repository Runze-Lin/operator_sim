"""Preregister and capture fused/standalone large-shape operators under the shared CA lock."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT/'development/validation/sku17'),str(ROOT)]
from campaign_collect import stages,events,nbytes,sha
from development.validation.rms_ordered_verify.verify import text_bytes
from operator_model import estimate

OUT=ROOT/'results/v0.29/large/fusion';RAW=OUT/'raw';REG=OUT/'freeze.json';LOCK=ROOT/'runs/large29/shared.lock'
SKUS=['Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'];DEV=SKUS[:2]
BUILTIN=OUT/'build/builtin';HC=OUT/'build/hcpost'
VENDOR=ROOT/'build/hcpost23/vendors/hcpost'
INSTALL=ROOT/'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer'
attrs={'mhc':dict(dtype='bf16',implementation='split',num_iters=20),'mhc_pre':dict(dtype='bf16',has_gamma=False,alpha_count=3,return_intermediates=True),
       'mhc_sinkhorn':dict(dtype='fp32',num_iters=20,return_intermediates=False),'mhc_post':dict(dtype='bf16',has_h_res=True),
       'mhc_pre_sinkhorn':dict(dtype='bf16',num_iters=20,need_backward=False),'vllm_hc_post':dict(implementation='vllm_hc_post'),
       'lightning_indexer':dict(dtype='bf16',layout='BSND',sparse_count=512,sparse_mode=3,return_values=False)}
families={'mhc_pre':'MhcPre_','mhc_sinkhorn':'MhcSinkhorn_','mhc_post':'MhcPost_','mhc_pre_sinkhorn':'MhcPreSinkhorn_',
          'vllm_hc_post':'HcPost_','lightning_indexer':'LightningIndexer_'}
CASES=[
    ('flash-chunk512','mhc',[512,4,5120],10800,49152,'Flash N4/D5120 in a 512-token prefill chunk; installed three-operator reference alternative to the serving extension'),
    ('flash-hc512','vllm_hc_post',[1,512,4,5120],7200,24576,'Exact flat 512-token Flash row geometry and pinned vLLM custom implementation, one rank'),
    ('index-32ki','lightning_indexer',[1,9,32768,8,128],7200,24576,'BF16 dense BSND top-512 key candidate set with Q9. Logical full-length 32Ki indexer; neither compressed Flash pages nor the Pro top-1024 contract'),
    ('fused-t512','mhc_pre_sinkhorn',[512,4,4096],10800,49152,'Installed BF16 fused D4096/I20 reference geometry; T>=128 separate-team branch. D4096 is not the hidden width of Flash or Pro'),
    ('sink-chunk4096','mhc_sinkhorn',[4096,4],5400,16384,'B8 times 512 token chunk, row-local N4/I20; isolated cold launch'),
    ('flash-chunk4096','mhc',[4096,4,5120],21600,98304,'B8 times 512 token chunk or 4096 tokens/rank in one dispatch; installed split alternative to serving extension'),
    ('sink-b8-32ki','mhc_sinkhorn',[262144,4],21600,98304,'Full B8 x 32768 tokens on one rank, one original FP32 Sinkhorn launch with I20')]
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_bytes())
def save(path,j):
    f=path.with_suffix('.tmp');f.write_text(json.dumps(j,indent=2,ensure_ascii=False,allow_nan=False)+'\n');f.replace(path)
def file_id(p):return str(p.relative_to(ROOT))
def obs(op,sh,chip,a):
    t=time.perf_counter_ns();f=estimate('mhc_post' if op=='vllm_hc_post' else op,sh,a,chip=chip)
    return dict(us=f.latency_us,status=f.status,reason=f.reason,query_ms=(time.perf_counter_ns()-t)/1e6,
                active_aic=f.active_aic,active_aiv=f.active_aiv,work=f.work,plan=(f.projection or {}).get('target_plan'))
def outputs(op,sh):
    if op=='mhc':
        return [('mhc_pre',sh),('mhc_sinkhorn',sh[:2]),('mhc_post',sh),('mhc',sh)]
    return [(op,sh)]
def arguments(op,sh):
    if op=='vllm_hc_post':return [str(sh[1]),str(sh[3])]
    if op=='lightning_indexer':return [op,*map(str,sh[1:4])]
    if op=='mhc_sinkhorn':return [op,*map(str,sh),'20']
    return [op,*map(str,sh)]
def golden(op,sh):
    if op=='lightning_indexer':return {'topk_set':sh[1]*512}
    if op=='vllm_hc_post':return {'output':math.prod(sh)}
    if op=='mhc_sinkhorn':return {'output':sh[0]*16}
    t,n,d=sh
    groups={'hin':t*d,'hpost':t*n,'hres':t*n*n}
    if op=='mhc':
        groups.update(inv_rms=t,hmix=t*(n*n+2*n),hpre=t*n,chain_sinkhorn=t*n*n,chain_post=t*n*d,chain_end_to_end=t*n*d)
    return groups

def archives():
    rows=defaultdict(dict)
    def add(op,chip,sh,ca,trace,src):
        if not ca or not trace:return
        key=tuple(trace) if isinstance(trace,list) else trace
        rows[op,chip][key]=dict(shape=sh,ca_us=ca,source=src,trace_sha256=trace)
    for r in read(ROOT/'results/v0.29/replay.json')['rows']:
        if r['op'] in attrs:add(r['op'],r['chip'],r['shape'],r['ca_us'],r['trace_sha256'],'results/v0.29/replay.json')
    for r in read(ROOT/'results/v0.29/post/score.json')['rows']:
        add('mhc_post',r['chip'],r['shape'],r['ca_us'],r['sha256'],'results/v0.29/post/score.json')
    for file in (ROOT/'results/v0.23/hc-timing').glob('*.json'):
        r=read(file)
        if isinstance(r,dict) and r.get('qualified') and 'stage' in r:
            add('vllm_hc_post',r['chip'],r['shape'],r['ca_us'],r['stage']['trace_sha256'],file_id(file))
    for r in read(ROOT/'results/v0.27/fusion/public-sink.json')['rows']:
        add('mhc_sinkhorn',r['chip'],r['shape'],r.get('ca_us'),r.get('raw_sha256'),'results/v0.27/fusion/public-sink.json')
    for dir in ('results/v0.26/presink/raw','results/v0.26/presink','results/v0.27/fusion'):
        for f in (ROOT/dir).glob('*.json'):
            if not any(token in f.stem for token in ('score','replay','fused','endpoint','archive','public')):continue
            j=read(f)
            for r in j.get('rows',[]) if isinstance(j,dict)else []:
                if not isinstance(r,dict) or len(r.get('shape',[]))!=3 or r['shape'][1:]!=[4,4096]:continue
                sku=r.get('sku',r.get('chip'));has=r.get('ca_us')
                key=r.get('trace_sha256') or r.get('raw_sha256')
                if sku in DEV and has and key:add('mhc_pre_sinkhorn',sku,r['shape'],has,key,file_id(f))
    return {s:{op:dict(windows=len(rows[op,s]),maximum_by_tensor_elements=max(rows[op,s].values(),key=lambda x:math.prod(x['shape']),default=None),
                       maximum_by_token_or_kv=max(rows[op,s].values(),key=lambda x:(x['shape'][2] if op=='lightning_indexer' else x['shape'][1] if op=='vllm_hc_post'else x['shape'][0]),default=None)) for op in attrs}for s in DEV}


def register():
    if REG.exists() or RAW.exists():raise FileExistsError('previous register or active raw')
    seen=archives();jobs=[]
    for case,op,shape,seconds,mib,role in CASES:
        rr=[]
        for name,x in outputs(op,shape):
            p={s:obs(name,x,s,attrs[name])for s in SKUS}
            if name=='mhc_sinkhorn' and op=='mhc_sinkhorn':
                for s in SKUS:p[s]['explicit_empty_stream']=obs(name,x,s,dict(attrs[name],resident_aiv=0))
            rr.append(dict(op=name,shape=x,attrs=attrs[name],model=p))
        host=HC if op=='vllm_hc_post' else BUILTIN
        jobs.append(dict(id=case,op=op,shape=shape,role=role,index=arguments(op,shape),runner=file_id(host),wall_s=seconds,raw_mib=mib,
                         expected_numerics=golden(op,shape),outputs=rr))
    vendor={}
    for name,prefix in families.items():
        folders=[ROOT/'build/hcpost23/kernel']if name=='vllm_hc_post' else [INSTALL]
        fs=[f for folder in folders for f in folder.rglob(prefix+'*.o')if '_relocatable' not in f.name]
        assert fs,(name,INSTALL)
        vendor[name]=[{ 'file':file_id(f),'sha256':sha(f),'text_sha256':hashlib.sha256(text_bytes(f)).hexdigest()}for f in fs]
    assets=[Path(__file__),Path(__file__).with_name('build_hosts.py'),BUILTIN,HC,ROOT/'smoke/extension_indexer_mhc.cpp',
            ROOT/'development/validation/hcpost23/host.cpp',ROOT/'development/validation/sku17/campaign_collect.py',ROOT/'scripts/cann-run.sh',
            *[ROOT/'configs/models'/f'{m}.json'for m in ('DeepSeek-V4.1-Flash','DeepSeek-V4-Pro')]]
    save(REG,dict(utc=now(),jobs=jobs,archive=seen,model_skus=SKUS,measure=DEV,physical_clock_ghz=1.65,
                 provider=vendor,assets={file_id(f):sha(f)for f in assets},
                 protocol='Fair inter-campaign shared flock for each individual CA endpoint. Strict original numerical groups and counts, target SKU/DRAM, vendor table envelope per core, active shard inventory, physical END on each participating core and installation .text. No replay of incomplete acquisitions. Physical execution proof can be valid even if no matching timing model. MHC sequence excludes host gaps and is counted separately from components.',
                 unsupported_full={'mhc_pre/mhc_post/hc_post/mhc':'B8 x 32Ki x 4 x 5120 gives 5,368,709,120 elements, beyond signed-int product in the pinned golden generator. Full-width 512/4096-token chunks and the true full-token standalone Sinkhorn are registered separately.',
                                   'mhc_pre_sinkhorn':'The registered built-in D4096/I20 contract and the real separate extension at D5120 or D7168 are distinct. Future-chip branch timings have no CA.',
                                   'lightning_indexer':'Q9 and KV32768 denote an isolated dense candidate-index contract. Actual upstream uses compressed or filtered pages and runtime head placement, Pro top-k=1024. No equivalence to full three-model dispatch is implied.'}))
    render()

def status(phase,**kw):save(OUT/'state.json',dict(phase=phase,utc=now(),pid=os.getpid(),freeze_sha256=sha(REG),**kw))


def launch(j,chip,d):
    d.mkdir(exist_ok=False);args=['msopprof','simulator','--soc-version='+chip,'--dump=false',
          '--launch-count='+str(3 if j['op']=='mhc'else 1),'--timeout='+str(math.ceil(j['wall_s']/60)),'--output='+str(d/'profile'),str(ROOT/j['runner']),*j['index']]
    cmd=['/usr/bin/time','-f','%U %S','-o',str(d/'cpu.txt'),'bash',str(ROOT/'scripts/cann-run.sh')]
    if j['op']=='vllm_hc_post':
        cmd+=['bash','-c','export ASCEND_CUSTOM_OPP_PATH="$1"; export LD_LIBRARY_PATH="$1/op_api/lib:$LD_LIBRARY_PATH"; shift; exec "$@"','vendor',str(VENDOR)]
    cmd+=args
    t=time.monotonic();started=now();peak=0;stop=None
    with (d/'run.log').open('xb')as log:
        with subprocess.Popen(cmd,cwd=d,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)as worker:
            while worker.poll() is None:
                peak=max(peak,nbytes(d))
                if time.monotonic()-t>j['wall_s'] or peak>j['raw_mib']*2**20:
                    stop='wall cap' if time.monotonic()-t>j['wall_s']else 'raw cap';os.killpg(worker.pid,signal.SIGTERM)
                    try:worker.wait(10)
                    except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL)
                    break
                time.sleep(5)
            exit_code=worker.wait()
    save(d/'cost.json',dict(started_utc=started,finished_utc=now(),seconds=time.monotonic()-t,peak_bytes=max(peak,nbytes(d)),exit_code=exit_code,stop_reason=stop))


def verify(j,chip,d,reg):
    log=(d/'run.log').read_text(errors='replace')
    signed={k:int(v) for k,v in re.findall(r'^correctness name=(\w+) count=(\d+) bad=0\b',log,re.M)}
    stages_ops=[r['op']for r in j['outputs']if r['op']!='mhc']
    assert signed==j['expected_numerics'],('numeric counts',signed)
    timed,_=stages(dict(kernels=len(stages_ops),nchecks=len(signed)),chip,d)
    for op,st in zip(stages_ops,timed):
        p=ROOT/st['trace'];assert st['kernel'].startswith(families[op]),st['kernel']
        actual=hashlib.sha256(text_bytes(p.parents[1]/'dump/aicore_binary.o')).hexdigest()
        allowed=[v for v in reg['provider'][op] if actual==v['text_sha256']]
        assert allowed,'captured kernel .text differs from installation'
        shards={f.parent.name:f for f in p.parent.glob('core*/trace.json')}
        assert set(shards)==set(st['windows']),'raw vs vendor lane inventory'
        origins=[];endrecords={}
        for core,shard in shards.items():
            lo=math.inf;hi=-math.inf;ends=[]
            for e in events(shard):
                if e['ph']!='X':continue
                a=round(e['ts']*1650);b=round((e['ts']+e['dur'])*1650)
                lo=min(lo,a);hi=max(hi,b)
                if e['name']=='END':ends.append(a)
            assert len(ends)==1 and lo<=ends[0]<=hi,(core,ends)
            x,y=st['windows'][core];origins.extend([lo-round(1650*x),hi-round(1650*y)])
            endrecords[core]=[lo,ends[0],hi]
        assert max(origins)-min(origins)<=3,'aggregate/shard clocks'
        st.update(op=op,provider_matches=allowed,physical_end_all_lanes=True,terminal_ticks=endrecords)
    values={s['op']:s['ca_us']for s in timed}
    if j['op']=='mhc':values['mhc']=sum(values.values())
    return dict(qualified=True,ca_us=values,golden=signed,stages=timed)


def idle():
    def snap():
        return {int(x[0][3:]):(sum(map(int,x[1:9])),int(x[4])+int(x[5]))for line in Path('/proc/stat').read_text().splitlines() if re.match(r'cpu\d+ ',line)for x in [line.split()]}
    a=snap();quiet={c:0. for c in os.sched_getaffinity(0)}
    for _ in range(2):
        time.sleep(1);b=snap()
        for k in quiet:quiet[k]=max(quiet[k],1-(b[k][1]-a[k][1])/max(1,b[k][0]-a[k][0]))
        a=b
    seen=set();cores=[]
    for c in sorted(quiet,key=lambda c:quiet[c]):
        p=(Path('/sys/devices/system/cpu')/f'cpu{c}/topology/thread_siblings_list').read_text().strip()
        if quiet[c]<=.40 and p not in seen:cores.append(c);seen.add(p)
    available=int(re.search(r'^MemAvailable:\s*(\d+)',Path('/proc/meminfo').read_text(),re.M)[1])/2**20
    return cores[:max(0,min(64,len(cores)-8))],dict(quiet_distinct=len(cores),available_gib=available,disk_gib=shutil.disk_usage(ROOT).free/2**30)


def worker():
    reg=read(REG);RAW.mkdir(exist_ok=True);LOCK.parent.mkdir(parents=True,exist_ok=True)
    allowed=os.sched_getaffinity(0);os.nice(9);os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    with (RAW/'own.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for path,h in reg['assets'].items():assert sha(ROOT/path)==h,path
        for job in reg['jobs']:
            for chip in DEV:
                label=job['id']+'-'+chip;dst=OUT/(label+'.json');d=RAW/label
                if dst.exists():continue
                if d.exists() and not (d/'cost.json').exists():status('interrupted raw retained',case=label);return
                if not d.exists():
                    while True:
                        status('waiting for shared acquisition slot',case=label)
                        with LOCK.open('a')as slot:
                            fcntl.flock(slot,fcntl.LOCK_EX)
                            cpus,detail=idle()
                            if len(cpus)>=16 and detail['available_gib']>=48 and detail['disk_gib']>job['raw_mib']/1024+64:
                                try:
                                    os.sched_setaffinity(0,cpus);status('acquiring',case=label,cpus=cpus,headroom=detail)
                                    launch(job,chip,d)
                                finally:os.sched_setaffinity(0,allowed)
                                break
                        status('waiting for physical headroom; shared slot released',case=label,host=detail);time.sleep(23)
                status('verifying; shared CA slot released',case=label)
                cost=read(d/'cost.json');info=dict(id=job['id'],chip=chip,cost=cost,registration_sha256=sha(REG))
                try:
                    assert cost['started_utc']>reg['utc'] and cost['exit_code']==0 and not cost['stop_reason'], 'collection budget or process'
                    info.update(verify(job,chip,d,reg))
                except (OSError,ValueError,RuntimeError,AssertionError,KeyError,IndexError,StopIteration)as e:
                    info.update(qualified=False,reason=repr(e),ca_us={})
                save(dst,info);render();time.sleep(8)
        status('finished');render()


def render():
    reg=read(REG);rows=[];schedule=[];state=read(OUT/'state.json') if (OUT/'state.json').exists() else {}
    for j in reg['jobs']:
        for r in j['outputs']:
            row=dict(case=j['id'],op=r['op'],shape=r['shape'],interpretation=j['role'],cost_limit=[j['wall_s'],j['raw_mib']],target={})
            for s in SKUS:
                pred=r['model'][s];file=OUT/(j['id']+'-'+s+'.json');ca=read(file)if file.exists()else {}
                valid=ca.get('qualified',False);measured=ca.get('ca_us',{}).get(r['op']) if valid else None
                row['target'][s]=dict(model_us=pred['us'],status=pred['status'],reason=pred['reason'],active_aic=pred['active_aic'],active_aiv=pred['active_aiv'],
                    known_empty_us=pred.get('explicit_empty_stream',{}).get('us'),ca_us=measured,
                    signed_pct=100*(pred['us']/measured-1)if measured and pred['us']else None,
                    qualification='extrapolated'if s==SKUS[-1]else 'qualified'if valid else 'failed'if ca else 'pending',failure=ca.get('reason'),
                    ca_receipt=file.name if file.exists() else None)
            rows.append(row)
    save(OUT/'score.json',dict(freeze_sha256=sha(REG),rows=rows,summary_scope='Each chain includes the three already counted components. 970DT has no CA.'))
    def fmt(v):return 'N/A' if v is None else f'{v:.3f}'
    lines=['# 融合与路由算子的大形验证','',
      '| 算子 / 形状 | 真实部署几何、替代实现或库合同 | 950DT 模型 / 合格 CA µs | 950PR 模型 / 合格 CA µs | 970DT 外推 µs | DT / PR 取证 |','|---|---|---:|---:|---:|---|']
    description={'flash-chunk512':'Flash 实际 D5120，单个 512-token chunk；内置三段替代链','flash-hc512':'实际 Flash 宽度、512-token chunk 及 vLLM HcPost 实现','index-32ki':'Q9、KV32768、G8、D128 的密集 BF16 索引库合同，top-512',
       'fused-t512':'库合同 D4096、I20，T≥128 分组','sink-chunk4096':'8 请求 × 512 token，FP32 N4/I20 单调度',
       'flash-chunk4096':'8 请求 × 512 token，实际 Flash 宽度、内置替代链','sink-b8-32ki':'8 请求 × 32768 token，FP32 N4/I20 单调度'}
    for r in rows:
        dt,pr,fut=(r['target'][s]for s in SKUS)
        lines.append(f"| {r['op']} {r['shape']} | {description[r['case']]} | {fmt(dt['model_us'])} / {fmt(dt['ca_us'])} | {fmt(pr['model_us'])} / {fmt(pr['ca_us'])} | {fmt(fut['model_us'])} | {dt['qualification']} / {pr['qualification']} |")
    lines+=['','## 旧证据的尺度','', '| 算子 | 本次登记前 950DT 的最大张量 shape / CA µs | 950PR 的最大张量 shape / CA µs |','|---|---|---|']
    for op in attrs:
        def prior(sku):
            z=reg['archive'][sku][op]['maximum_by_tensor_elements']
            return f"{z['shape']} / {fmt(z['ca_us'])}"if z else 'N/A'
        lines.append(f'| {op} | {prior(DEV[0])} | {prior(DEV[1])} |')
    lines+=['','mHC 以每 rank 的实际 token 数 T 为行数。512 行、D5120 时，vLLM HcPost 的 950DT 64 个 AIV 最多每核 8 行，PR 56 个 AIV 最多 10 行；每行再按 4096+1024 通道分片。4096 行对应 64/74 行每 AIV。旧归档中的最大输入或最多 token 均在冻结记录逐项列出，单个完整窗不代表已覆盖其它实现。内置链只做一次真实三段派发，以各段独立核窗之和给出链端时间，排除 host 中间间隔，链中 Post 的前序来自同一真实流。','',
       '仅 shape 的 FP32 Sinkhorn 对大 T 可能没有时间。独立冷启动可另查显式 resident_aiv=0；4096 行和完整 262144 行均已将两种调用的可用性保存在逐端记录中。完整大形如果跨多个 UB 分块而没有发起价格，模型时间写 N/A。Q9/KV32Ki 索引测试的输入、所有输出 top-512 集合和整芯片核窗可核对。Flash 上游的压缩页与过滤 top-k，以及 Pro 的 top-1024，均需要独立运行时证据。','',
       '宿主新增实验只调整 smoke 输入大小限制，并沿用同一输出构造和数值 golden。所有 950 端点需要全量预定数值组、准确编译目标及 DRAM 配置、对应安装 ELF 的 .text、每核物理 END、每个核与官方 duration 表的完整窗口。整行或整链的 CA 只有全部合格后进入表格。队列与其他大实验共用单槽，每端点完成采集即释放，解析期间其它队列可以采集。每端预算 1.5 到 6 小时、16 到 96 GiB raw，状态与达预算原因单列。','',
       'B8 × 32Ki、N4、D5120 的全张量有 5,368,709,120 元素，超出原宿主 golden 的有符号 32 位索引。表内明确分开了完整 token、chunk 和替代合同。970DT 全域仅填模型外推。','',
       '[冻结三卡价格、各端口径、预算和来源](freeze.json) · [逐形状态、误差和真实回执](score.json) · [实时队列](state.json)']
    file=OUT/'report.tmp';file.write_text('\n'.join(lines)+'\n');file.replace(OUT/'report.md')


if __name__=='__main__':{'register':register,'run':worker,'report':render}[sys.argv[1]]()
