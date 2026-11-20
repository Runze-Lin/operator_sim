"""One-geometry/one-SKU BF16 Kimi MLA decode experiment, with snapshot prices."""
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'development/validation/sku17'))
from operator_model.matmul_placement import matmul_placement
from development.validation.matmul28 import campaign as ca
from development.validation.matmul28.drain import recover
from development.validation.matmul28.narrow import cpu_stats
from development.validation.matmul28.streamk_audit import PROVIDER

D=ROOT/'results/v0.29/matmul/decode'
RAW=D/'raw'
REG=D/'freeze.json'
BASE=ROOT/'results/v0.28/matmul'
WHEEL=ROOT/'build/wheels/v028/ascend_operator_model-0.28.0-py3-none-any.whl'
DT,PR,FUT=ca.DT,ca.PR,ca.FUT
MEASURED=(DT,PR)
SKUS=(*MEASURED,FUT)
BATCHES=(1,9,11,16)
ALL=sorted({b*q for b in BATCHES for q in (1,9)})
NEW=[9,16,81,144]
TAGS={DT:'dt', PR:'pr'}
N,K=2304,1536


def slug(m,sku):return f'{m}-{N}-{K}-{TAGS[sku]}'
def save(p,j):ca.save(p,j)
def read(p):return ca.read(p)
def digest(p):return ca.digest(p)
def now():return datetime.now(timezone.utc).isoformat()


def old_release():
    # Source trees may already be ahead. Import the packaged release in isolation.
    code="""import json,sys,time
sys.path.insert(0,sys.argv[1])
from operator_model import estimate,__version__
assert __version__=='0.28.0'
out={}
for m in json.loads(sys.argv[2]):
 rows={}
 for c in sys.argv[3:]:
  t=time.perf_counter_ns();p=estimate('matmul',[m,2304,1536],chip=c,dtype='bf16')
  rows[c]=dict(us=p.latency_us,active_aic=p.active_aic,active_aiv=p.active_aiv,query_ms=(time.perf_counter_ns()-t)/1e6)
 out[m]=rows
print(json.dumps(out))"""
    with TemporaryDirectory() as t:
        with ZipFile(WHEEL) as wheel:wheel.extractall(t)
        raw=subprocess.check_output([sys.executable,'-I','-c',code,t,json.dumps(ALL),*SKUS],cwd=t,text=True)
    return json.loads(raw)


def preregister():
    if REG.exists() or RAW.exists():raise FileExistsError('previous registration/acquisition')
    cfg=read(ROOT/'configs/models/Kimi-K3.json')['text_config']
    assert cfg['num_attention_heads']//8*(cfg['qk_nope_head_dim']+cfg['qk_rope_head_dim'])==N and cfg['q_lora_rank']==K
    with gzip.open(ca.CATALOG,'rt') as f:
        node=next(n for line in f for v in [json.loads(line)] if v['scenario']['model']=='Kimi-K3' and v['scenario']['tp']==8
                  for n in v['contracts'] if n['op']=='mla.q_b_proj')
    assert node['tensors']['weight_logical']==[N,K]
    check=re.compile(rb'\[\s*(9|16|81|144)\s*,\s*2304\s*,\s*1536\s*\]')
    counts=[0,0];seen=[]
    for root,dirs,files in os.walk(ROOT/'results'):
        dirs[:]=[d for d in dirs if d not in ('raw','profile','runs','log','log_ca','device0')]
        for name in files:
            if name.endswith('.json') and name!='trace.json':
                f=Path(root)/name;counts[0]+=1
                if check.search(f.read_bytes()):seen.append(str(f.relative_to(ROOT)))
    log_pat=re.compile(rb'\bM=(9|16|81|144) N=2304 K=1536\b')
    for root,dirs,files in os.walk(ROOT/'runs'):
        dirs[:]=[d for d in dirs if d not in ('profile','simulator','device0','log_ca','dump','log') and not d.startswith('OPPROF_')]
        for n in ('run.log','host.log'):
            if n not in files or 'matmul' not in root.lower():continue
            p=Path(root)/n;counts[1]+=1
            with p.open('rb') as f:
                if any(log_pat.search(line)for line in f):seen.append(str(p.relative_to(ROOT)))
    if seen:raise ValueError(('registered geometry or prior execution exists',seen))
    released=old_release();records=[]
    for m in ALL:
        shape=[m,N,K];pred={};oldrefs={}
        for c in SKUS:
            t=time.perf_counter_ns();p=matmul_placement(shape,chip=c);ms=(time.perf_counter_ns()-t)/1e6
            old=released[str(m)][c]
            assert max(ms,old['query_ms'])<100 and p['latency_us'] is not None
            pred[c]=dict(us=p['latency_us'],placement=p,placement_query_ms=ms,v028=old)
            if m not in NEW and c in MEASURED:
                base=BASE/(slug(m,c)+'.json');e=read(base)
                path=base if e['qualified'] else BASE/(slug(m,c)+'-drain.json')
                qualified=read(path);stage,=qualified['stages']
                assert qualified['qualified'] and qualified['correctness']==[f'correctness count={m*N} bad=0']
                assert stage['binary_sha256']==digest(PROVIDER.with_suffix('.o')) and stage['active_aic']+stage['active_aiv'] > 0
                if path!=base: assert qualified['acquisition_sha256']==digest(base)
                oldrefs[c]=dict(full_receipt=str(path.relative_to(ROOT)),sha256=digest(path),original=str(base.relative_to(ROOT)),original_sha256=digest(base))
        records.append(dict(m=m,shape=shape,role='pre-CA v0.29 placement prospect' if m in NEW else 'v0.28 known CA replay',
                            predictions=pred,previous_receipts=oldrefs,wall_s=1200,raw_mib=1024))
    model_sources=[ROOT/'operator_model'/f for f in ('matmul_placement.py','matmul_multisplit.py','matmul_splitk.py','matmul_resource.py','matmul_waves.py','matmul_v15.py','chips.py')]
    model_sources += [WHEEL,ca.CATALOG,ROOT/'configs/models/Kimi-K3.json']+[ROOT/'configs/chips'/f'{c}.json' for c in SKUS]
    acquisition=[Path(__file__),ca.BIN,ROOT/'development/validation/matmul28/runner.cpp',ROOT/'development/validation/matmul28/campaign.py',
                 ROOT/'development/validation/matmul28/drain.py',ROOT/'development/validation/sku17/campaign_collect.py',
                 ROOT/'development/validation/pro_prefill23/evidence.py',ROOT/'scripts/cann-run.sh',PROVIDER.with_suffix('.o')]
    save(REG,dict(utc=now(),model='Kimi-K3',projection='mla.q_b_proj',tp=8,model_config=dict(heads=96,per_rank_heads=12,nope=128,rope=64,q_lora_rank=1536),
                  source_node=node,contract='BF16 ND X[M,1536] @ W[1536,2304], no bias. Logical serving-equivalent operand geometry; no full vLLM launch or quantized checkpoint measured.',
                  scenarios=[dict(batch=b,query=q,m=b*q)for q in (1,9) for b in BATCHES],targets=records,
                  archives_checked=dict(manifests=counts[0],matmul_host_logs=counts[1],found_existing_targets=seen),
                  policy='One local worker across both SKUs and all output shapes; always DT then PR. Same shape M=9 reused for B9/Q1 and B1/Q9. Each new endpoint checked against whole output and exact installed ELF, official envelope and complete AIC/AIV retirement; original and supplementary drain receipts both retained. Aggregate WAPE counts distinct geometry, only paired if both ends qualified. 970DT extrapolated.',
                  admission='Repeated three-second per-physical-core idle samples. Select 32–56 continuously quiet physical cores, preferring CPUs outside in-flight large-job masks and reserving at least 16 idle cores and 32 GiB disk. No dependency on other queue completion.',
                  hypotheses='K=1536 remains below modeled half-core split-K threshold for both 950s. An ordinary kernel is predicted. The four new timing/branch pairs are independent under the frozen registration even when old/new latency prices coincide.',
                  rows_to_capture=NEW,provider_sha256=digest(PROVIDER.with_suffix('.o')),
                  model_sha256={str(p.relative_to(ROOT)):digest(p)for p in model_sources},
                  acquisition_sha256={str(p.relative_to(ROOT)):digest(p)for p in acquisition}))
    report()


def source_record(job,sku):
    m=job['m'];p=job['previous_receipts'].get(sku)
    if p:
        file=ROOT/p['full_receipt'];assert digest(file)==p['sha256']
        src=read(file);orig=read(ROOT/p['original']);return src,orig['cost'],str(file.relative_to(ROOT))
    file=D/(slug(m,sku)+'.json')
    if file.exists():
        src=read(file);return src,src['cost'],str(file.relative_to(ROOT))
    return {},{},None


def report():
    reg=read(REG);table={};groups={};pairs={}
    for e in reg['targets']:
        m=e['m'];table[m]={}
        for c in MEASURED:
            r,cost,path=source_record(e,c);ca_us=r['ca_us'] if r.get('qualified') else None
            s=r['stages'][0] if ca_us else {}
            p=e['predictions'][c]
            table[m][c]=dict(shape=e['shape'],role=e['role'],ca_us=ca_us,old_us=p['v028']['us'],placement_us=p['us'],
                             extrapolated_970_us=e['predictions'][FUT]['us'],forecast_route=p['placement']['route'],
                             route='split_k' if 'mix_aic' in s.get('kernel','') else 'ordinary' if s else None,
                             signed_pct=100*(p['us']/ca_us-1) if ca_us else None,reference=p['v028']['us'],qualified=bool(ca_us),
                             kernel=s.get('kernel'),effective_aic=s.get('active_aic'),effective_aiv=s.get('active_aiv'),
                             trace_sha256=s.get('trace_sha256'),wall_s=cost.get('seconds'),peak_mib=cost.get('peak_bytes',0)/2**20 if cost else None,
                             receipt=path,status='qualified' if ca_us else ('unqualified' if path else 'pending'),reason=r.get('reason'))
    for name,ms in [('archive',[m for m in ALL if m not in NEW]),('new',NEW),('all',ALL)]:
        groups[name]={}
        for c in MEASURED:
            q=[table[m][c]for m in ms if table[m][c]['qualified']]
            groups[name][c]=dict(qualified=len(q),registered=len(ms),**{k:dict(
                wape_pct=100*sum(abs(r[k]-r['ca_us'])for r in q)/sum(r['ca_us'] for r in q) if q else None,
                max_pct=max((100*abs(r[k]/r['ca_us']-1)for r in q),default=None)) for k in ('old_us','placement_us')})
    for m,x in table.items():
        dt,pr=x[DT],x[PR]
        pairs[m]=dict(placement_dt_over_pr=dt['placement_us']/pr['placement_us'],ca_dt_over_pr=dt['ca_us']/pr['ca_us'] if dt['qualified'] and pr['qualified'] else None)
    save(D/'score.json',dict(utc=now(),registration_sha256=digest(REG),unique_rows=table,groups=groups,dt_over_pr=pairs))
    def fmt(x):return '—' if x is None else f'{x:.3f}'
    header=['# Kimi-K3 的同宽解码投影','',
        '单卡 TP8 的 MLA 查询投影取 96/8 = 12 个头，每头 128 个非旋转维和 64 个旋转维，因此 N = 2304；低秩输入 K = 1536。模拟对象为连续 ND、BF16 的独立 aclnnMatmul。主模型校验推测 8 个 token 时 Q = 9，假设一批中每个请求都参加且无补齐，M = B × Q。','',
        '| 芯片 | 已有三形 WAPE / 最大 | 冻结四形 WAPE / 最大 | 合格/计划 |','|---|---:|---:|---:|']
    for c in MEASURED:
        line=[f"{fmt(groups[g][c]['placement_us']['wape_pct'])} / {fmt(groups[g][c]['placement_us']['max_pct'])}"for g in ('archive','new')]
        header.append(f"| {c} | {line[0]}% | {line[1]}% | {groups['archive'][c]['qualified']}/3 + {groups['new'][c]['qualified']}/4 |")
    header+=['','| B | Q | M | 950DT 旧/新/CA µs | 950PR 旧/新/CA µs | 新误差 DT / PR % | 970DT 外推 µs | 证据 DT / PR |','|---:|---:|---:|---:|---:|---:|---:|---|']
    for sc in reg['scenarios']:
        m=sc['m'];dt,pr=table[m][DT],table[m][PR]
        def triple(v):return ' / '.join(fmt(v[k])for k in ('old_us','placement_us','ca_us'))
        role='旧形回放' if m not in NEW else '本轮采前冻结'
        header.append(f"| {sc['batch']} | {sc['query']} | {m} | {triple(dt)} | {triple(pr)} | {fmt(dt['signed_pct'])} / {fmt(pr['signed_pct'])} | {fmt(dt['extrapolated_970_us'])} | {role} · {dt['status']} / {pr['status']} |")
    header+=['','M = 9 供两个真实调用场景共用，误差汇总只计一次。同宽输入的对照可以检验批量变化、对齐和分块的影响。K=1536 未跨理论切 K 门槛，所以 v0.29 的位置选择器对这些形状返回普通分块，预测与 v0.28 逐项核对。真实入口、活跃核数、设备全窗、DT/PR 加速比、采集时间与空间均在逐形 JSON 中列出。','',
        '每个新增形状在 DT、PR 各执行一次。单槽、每端点 1200 秒/1 GiB；八个新端点的累计上限 160 分钟与 8 GiB。先核验全部 M × 2304 个 BF16 输出，再核验安装库 ELF、全部活动核真实 END 与异步尾指令及官方全窗。缺测、受限与失败显示为 —，失格数据保留原原因。970DT 没有 CA。','',
        '[冻结合同与三卡预测](freeze.json) · [逐端分支、耗时、误差与退核证据](score.json) · [队列状态](state.json)']
    tmp=D/'report.md.tmp';tmp.write_text('\n'.join(header)+'\n');tmp.replace(D/'report.md')


def state(phase,**extra):
    save(D/'state.json',dict(phase=phase,utc=now(),pid=os.getpid(),registration_sha256=digest(REG),**extra))


def host(allowed):
    before=cpu_stats(); max_busy={cpu:0. for cpu in allowed}
    for _ in range(3):
        time.sleep(1);after=cpu_stats()
        for cpu in allowed:
            delta=after[cpu][0]-before[cpu][0]
            max_busy[cpu]=max(max_busy[cpu],1-(after[cpu][1]-before[cpu][1])/max(1,delta))
        before=after
    large=set()
    for name in ('prefill-status.json','long-prefill/state.json','wide-prefill/state.json'):
        f=BASE/name
        if f.exists():
            phase=read(f)
            if phase.get('phase')=='acquiring':
                try:large.update(os.sched_getaffinity(phase['pid']))
                except (ProcessLookupError,KeyError):pass
    domains=set();idle=[]
    for c in sorted(allowed,key=lambda c:(c in large,max_busy[c],c)):
        sibling=(Path('/sys/devices/system/cpu')/f'cpu{c}/topology/thread_siblings_list').read_text().strip()
        if max_busy[c]<=.30 and sibling not in domains:
            idle.append(c);domains.add(sibling)
    mask=idle[:max(0,min(56,len(idle)-16))]
    ram=int(re.search(r'^MemAvailable:\s*(\d+)',Path('/proc/meminfo').read_text(),re.M)[1])/2**20
    disk=shutil.disk_usage(D).free/2**30
    return mask,dict(eligible=len(mask)>=32 and ram>=48 and disk>=32,
        quiet_physical=len(idle),available_ram_gib=ram,available_disk_gib=disk,
        outside_large_masks=len(set(mask)-large),load_1m=os.getloadavg()[0],chosen_cpus=mask)


def publish(job,sku,folder):
    label=slug(job['m'],sku);cost=read(folder/'cost.json');reg=read(REG)
    assert cost['started_utc']>reg['utc']
    try:
        if cost['exit_code']!=0 or cost['stop_reason'] or cost['peak_bytes']>job['raw_mib']*2**20:
            raise ValueError('process, time budget or storage limit')
        initial=ca.qualify(job,sku,folder)
    except (AssertionError,ValueError,KeyError,IndexError,OSError,StopIteration)as e:
        initial=dict(qualified=False,ca_us=None,reason=repr(e))
    capture=dict(**initial,case=label,chip=sku,shape=job['shape'],cost=cost)
    path=D/(label+'-initial.json')
    if not path.exists():save(path,capture)
    else:assert read(path)==capture
    data=capture
    if not capture['qualified']:
        try:extra=recover(job,capture,folder,acquisition=path)
        except (AssertionError,ValueError,KeyError,IndexError,OSError,StopIteration)as e:
            extra=None;data={**capture,'reason':str(capture['reason'])+'; drain: '+repr(e)}
        if extra:data={**capture,**extra,'reason':None}
    if data.get('qualified'):
        s,=data['stages']
        if s['binary_sha256']!=reg['provider_sha256'] or data['correctness']!=[f"correctness count={job['m']*N} bad=0"]:
            data={**data,'qualified':False,'reason':'wrong kernel or output coverage','ca_us':None}
    save(D/(label+'.json'),dict(**data,initial_sha256=digest(path),register_sha256=digest(REG)))
    report()


def run():
    RAW.mkdir(exist_ok=True);reg=read(REG)
    with (RAW/'serial.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for source,h in reg['acquisition_sha256'].items():
            if digest(ROOT/source)!=h:raise ValueError('capture source changed: '+source)
        os.nice(12);os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
        allowed=os.sched_getaffinity(0)
        for job in (j for j in reg['targets'] if j['m'] in reg['rows_to_capture']):
            for c in MEASURED:
                label=slug(job['m'],c);folder=RAW/label
                if (D/(label+'.json')).exists():continue
                if folder.exists() and not (folder/'cost.json').exists():
                    state('interrupted raw preserved; inspect active host before recovery',case=label);return
                if not folder.exists():
                    while True:
                        mask,mem=host(allowed)
                        if mem['eligible']:break
                        state('waiting for physical CPU, RAM or storage',case=label,host=mem)
                        time.sleep(27)
                    try:
                        os.sched_setaffinity(0,mask);folder.mkdir()
                        state('acquiring',case=label,host=mem)
                        ca.measure(job,c,folder)
                    finally:os.sched_setaffinity(0,allowed)
                state('verifying full numerical output and terminal windows',case=label)
                publish(job,c,folder)
        state('finished');report()


if __name__=='__main__':
    {'register':preregister,'run':run,'report':report}[sys.argv[1]]()
