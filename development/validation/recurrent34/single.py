"""Collect one preregistered maximal dense single-chain ABI boundary on both real SKUs."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import fcntl
import importlib.util
import json
import os
import re
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/v0.34/recurrent'
RAW=ROOT/'runs/recurrent34'
REG=OUT/'freeze.json'
LOCK=ROOT/'results/v0.26/sink/raw/lock'
DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'
SHAPE=[1,8,1,1,128,256]
ATTRS=dict(layout='BSND',dtype='BF16',gate_dtype='FP32',state_dtype='FP32',state_v_first=False,use_gate_in_kernel=False,
           chunk_size=None,output_final_state=True,export_gk=False)
ARCHIVE=ROOT/'results/v0.32/replay.json'
HOST=ROOT/'build/vector-v8/KDA'
NAME='RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0'
CANN=ROOT/'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe'
DEVICE=CANN/'kernel/ascend950/ops_transformer/recurrent_kda_apt'/(NAME[:-2]+'.o')
KERNEL=CANN/'impl/ops_transformer/ascendc/recurrent_kda/op_kernel/arch35/recurrent_kda_arch35.h'
DOC=ROOT/'third_party/ops-transformer/attention/recurrent_kda/docs/aclnnRecurrentKda.md'
INPUT=HERE/'one.case'
CAPTURE=ROOT/'development/validation/large29/vector/acquire.py'
CHECK=ROOT/'development/validation/large29/cube/verify.py'
ELF=ROOT/'development/validation/recurrent_figure/figure.py'


def read(p):return json.loads(p.read_bytes())
def digest(p):return sha256(p.read_bytes()).hexdigest()
def rel(p):return str(p.relative_to(ROOT))
def utc():return datetime.now(timezone.utc).isoformat()
def write(p,d,new=False):
    if new and p.exists():raise FileExistsError(p)
    t=p.with_suffix('.'+str(os.getpid())+'.tmp')
    t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def require(test,message):
    if not test:raise ValueError(message)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def register(prefix):
    if REG.exists() or RAW.exists():raise FileExistsError('pre-existing freeze or raw')
    sys.path.insert(0,str(prefix));import operator_model
    from importlib.metadata import distribution
    require(Path(operator_model.__file__).is_relative_to(prefix) and distribution('ascend-operator-model').version=='0.34.0','installed release')
    prev=[x for x in read(ARCHIVE)['rows'] if x['op']=='kda_recurrent' and x['shape'][0]==x['shape'][2]==x['shape'][3]==1]
    require(not any(x['shape']==SHAPE for x in prev),'already recorded')
    hist={c:{str(v):sorted({r['shape'][1] for r in prev if r['chip']==c and r['shape'][5]==v}) for v in (128,256)} for c in (DT,PR)}
    pred={}
    for chip in (DT,PR,FUT):
        start=time.perf_counter_ns()
        r=operator_model.estimate('kda_recurrent',SHAPE,ATTRS,chip=chip,dtype='BF16')
        pred[chip]=dict(us=r.latency_us,query_ms=(time.perf_counter_ns()-start)/1e6,role=r.status,work=r.work,
                        aiv=r.active_aiv,aic=r.active_aic,stages=r.projection['stages_us'] if r.projection else None)
        require(r.latency_us>0 and r.active_aiv==1,'target public price')
    name,=(ROOT/'build/wheels/v034').glob('*.whl')
    sources=[Path(__file__),INPUT,DOC,KERNEL,DEVICE,HOST,ROOT/'smoke/vector_v8.cpp',ROOT/'smoke/extension_kda.cpp',
             CAPTURE,CAPTURE.with_name('common.py'),CHECK,ELF,ARCHIVE,name,ROOT/'configs/models/Kimi-K3.json']
    require(INPUT.read_text()=='kda_recurrent 0 8 256 0\n','one registered operator')
    job=dict(id='single-recurrent',op='kda_recurrent',shape=SHAPE,attrs=ATTRS,runner=rel(HOST),protocol='instruction',
             args=[str(INPUT)],wall_s=1200,raw_gib=2,prefix=['RecurrentKda'],contract='BATCH_BEGIN 0',
             golden={'attn_out':2048,'final_state':32768,'initial_state_unchanged':32768})
    write(REG,dict(utc=utc(),job=job,forecasts=pred,archive_lengths=hist,slot=rel(LOCK),sources={rel(p):digest(p) for p in sources},
        eligibility='Maximal legal B=Hq=Hv=1 dense BF16/FP32 recurrent geometry by S*K*V: S=8, K=128, V=256. Generic ABI endpoint. Kimi V=128 reaches S=8 on both devices already. Do not count this as a whole-prefill or multi-chain runtime.',
        budget='One instruction-mode launch per SKU, one shared flock per endpoint, release before checking raw. 1200s and 2GiB each, no repetition. Official physical-memory model, numeric attention/final and unmodified initial-state tensors, matching installed ELF, full one-AIV merged and shard timestamps plus vendor duration required.',
        registration_correction={'predecessor':rel(OUT/'preflight-registration.json'), 'sha256':digest(OUT/'preflight-registration.json'), 'reason':'Before any acquisition a historical-log verifier test showed the official kernel line has a timestamp prefix. Predictions, budget and qualification policy identical.'} if (OUT/'preflight-registration.json').exists() else None,
        method_role='No tuning, v0.34.0 wheel predictions fixed before either new shape was measured. 970DT extrapolation.'),True)


def inspect(job,chip,folder,cost):
    require(cost['exit_code']==0 and not cost['stop_reason'],'process or budget')
    log=(folder/'run.log').read_text(errors='replace')
    require(re.findall(r'^BATCH_(BEGIN|END) (\d+)',log,re.M)==[('BEGIN','0'),('END','0')],'single frozen case')
    require(re.findall(r'Start profiling on kernel: (\S+)',log,re.M)==[NAME],'kernel identity')
    require('initial_state_unchanged' in log and re.search(r'correctness name=initial_state_unchanged .*?bad=0 max_abs_error=0 ',log),'input state modified')
    # Reuse the merged/shard equality, END, vendor tables and all numeric groups.
    verify=load(CHECK,'recurrent34_official_check')
    q=verify.qualify(job,chip,folder,cost,{},ROOT)
    require(q['qualified'],q.get('reason','qualified'))
    st,=q['stages']
    require(st['active_aic']==0 and st['active_aiv']==1 and st['mmads']==0,'physical single-AIV recurrent')
    trace=Path(st['trace']);exp=trace.parent.parent/'dump/aicore_binary.o'
    text=load(ELF,'recurrent34_published_text').elf_text
    require(text(DEVICE)==text(exp),'published kernel text')
    q['released_text_sha256']=sha256(text(exp)).hexdigest();q['source_object']=rel(DEVICE)
    st['trace']=rel(trace)
    return q


def run():
    reg=read(REG);RAW.mkdir(exist_ok=True);os.nice(13)
    for k,h in reg['sources'].items():
        require(digest(ROOT/k)==h,'changed source '+k)
    sys.path.insert(0,str(CAPTURE.parent))
    from acquire import capture
    def state(phase,**kw):write(OUT/'state.json',dict(utc=utc(),phase=phase,pid=os.getpid(),freeze_sha256=digest(REG),**kw))
    with (RAW/'owner.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for c,tag in ((DT,'dt'),(PR,'pr')):
            dest=OUT/(tag+'.json');folder=RAW/tag
            if dest.exists():continue
            state('waiting for shared CA lane',chip=c)
            with LOCK.open('r+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if folder.exists():state('previous run preserved',chip=c);return
                if shutil.disk_usage(ROOT).free<(26*2**30):state('insufficient disk headroom',chip=c);return
                state('collecting',chip=c)
                capture(dict(reg['job'],chip=c),folder)
            state('verifying after releasing shared lane',chip=c)
            cost=read(folder/'cost.json')
            try:
                require(cost['started_utc']>reg['utc'],'late freeze')
                require(digest(INPUT)==reg['sources'][rel(INPUT)] and digest(HOST)==reg['sources'][rel(HOST)],'runner or input changed')
                require(str(INPUT) in cost['command'] and str(HOST) in cost['command'],'actual frozen invocation')
                v=inspect(reg['job'],c,folder,cost)
            except (AssertionError,ValueError,KeyError,TypeError,IndexError,StopIteration,OSError) as exc:
                v=dict(qualified=False,ca_us=None,reason=repr(exc))
            write(dest,dict(utc=utc(),chip=c,shape=SHAPE,verdict=v,cost=cost,freeze_sha256=digest(REG),forecast=reg['forecasts'][c]),True)
            report();time.sleep(2)
        state('terminal')


def report():
    reg=read(REG);lines=['# Recurrent KDA 单状态链上界','',
        '安装的 CANN 9.2.1 ops-transformer 实现与 BF16 dense BSND 接口下，一个请求只保留一个 Q 头和一个 V 头，一次 kernel 只让一个 AIV 更新状态。该接口规定每条序列最多 8 步，K=128，V 可为 128 或 256。Kimi 的真实头宽 V=128，S=1–8 已有 DT/PR 完整回放；其中 S=8 为 35.396 / 39.450 µs，不能代替 32Ki prefill 的 chunk KDA 模型。','',
        '新增 [1,8,1,1,128,256] 是同一实现允许的最大单链状态与最多串行步数，V=256 为通用库边界。初始 FP32 状态 131072 B，每条链 8 次依赖更新。已有回放中 V=256 的 DT 只采 1/2/3/4/6 步，PR 只采 1/2 步。','',
        '| 芯片 | 模型 µs | CA µs | 有符号误差 | CA 墙钟 | 口径 |','|---|---:|---:|---:|---:|---|']
    times={}
    for chip,tag in ((DT,'dt'),(PR,'pr'),(FUT,'future')):
        file=OUT/(tag+'.json');p=reg['forecasts'][chip]['us']
        ca=None;wall='N/A';role='970DT 外推' if chip==FUT else '已预注册，待完整窗'
        if file.exists():
            obj=read(file);v=obj['verdict'];require(obj['freeze_sha256']==digest(REG),'changed freeze')
            wall=f"{obj['cost']['seconds']:.1f} s"
            if v.get('qualified'):
                ca=v['ca_us'];times[chip]=ca;role='冻结后独立完整指令窗'
            else:role='终态未合格'
        lines.append(f'| {chip} | {p:.3f} | '+(f'{ca:.3f} | {100*(p/ca-1):+.2f}%' if ca else 'N/A | N/A')+f' | {wall} | {role} |')
    if DT in times and PR in times:
        lines.extend(['',f"同形 DT/PR 时间比：模型 {reg['forecasts'][DT]['us']/reg['forecasts'][PR]['us']:.4f}×；实测 {times[DT]/times[PR]:.4f}×。"])
    lines.extend(['','每端墙钟预算 20 分钟、最多 2 GiB 指令原始数据。按现有串行设备锁逐端执行。核验包括真实 SKU/存储模型、安装包 kernel .text、完整 attention 与最终状态、输入状态未修改、唯一 END、物理核分片与全核指令集合一致，以及官方逐核时长。该几何单独报告为 ABI 压力样例。','',
        '[冻结、真实输入和源 SHA](freeze.json) · [排队状态](state.json) · [CANN ABI 的已发布接口说明](https://gitcode.com/cann/ops-transformer/tree/master/attention/recurrent_kda/docs) · [真实 Kimi 单链算例](../../../docs/examples/kda_recurrent.md)',''])
    t=OUT/'report.tmp';t.write_text('\n'.join(lines));t.replace(OUT/'report.md')
if __name__=='__main__':
    if sys.argv[1]=='freeze':register(Path(sys.argv[2]).resolve())
    elif sys.argv[1]=='report': report()
    elif sys.argv[1]=='run':run()
