"""Freeze a Cast-derived service scenario and supervise one independent native quant case."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'development/validation/large20'),str(ROOT/'development/validation/sku17'),str(ROOT)]
from native import capture_vector
from operator_model import estimate
from operator_model.chips import load_chip
from operator_model.quant_resource import resource_quant
from development.workloads.prefill_shapes import Scenario, config

DT='Ascend950DT_9582'
DEST=ROOT/'results/v0.27/quant-transfer'
RAW=DEST/'raw'
PRIORITY=ROOT/'results/v0.27/memory/cast-pair/score.json'
MEM_SLOT=ROOT/'runs/memory27-followup/remaining.lock'
CTRL=ROOT/'results/v0.27/quant-channel-service.json'
HOST=ROOT/'build/smoke/large_vector'
REG=DEST/'freeze.json'
RESULT=DEST/'native.json'
STATE=DEST/'state.json'
OBJECT=ROOT/'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_nn/dynamic_mx_quant/DynamicMxQuant_bf16toe4m3_high_performance.o'
FULL_SCOPE='all 64 AIV wrapper starts to done; single dispatch; verified values and E8M0 scale; all-instruction duration unavailable'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj, exclusive=False):
    txt=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if exclusive:
        with path.open('x') as f:f.write(txt)
    else:
        tmp=path.with_name(path.name+'.tmp')
        tmp.write_text(txt)
        tmp.replace(path)


def efficiency():
    approved={'Cast [8192, 5120]','Cast [2097152, 32]','Pro indexer Cast'}
    rows=[r for r in read(CTRL)['rows'] if r['case'] in approved]
    assert {r['case'] for r in rows}==approved and len(rows)==3
    stats=[]
    for r in rows:
        c=r['channel_1us_bins']['mixed/queued_95pct']['summed_channel_counts']
        eta=(c['read']+c['write'])/c['slots']
        assert math.isclose(eta*100,r['channel_1us_bins']['mixed/queued_95pct']['payload_service_pct'])
        stats.append(dict(op=r['case'],shape=r['shape'],accepted_periods=c['slots']//1000,read_beats=c['read'],
                          write_beats=c['write'],channel_DFI_cycles=c['slots'],eta=eta,raw=r['raw'],receipt=r['receipt']))
    values=sorted(r['eta'] for r in stats)
    return stats,dict(low=values[0],mid=statistics.median(values),high=values[-1])


def scenario(rows,coef):
    chip=load_chip(DT)
    h=config('Kimi-K3')['hidden_size']
    assert h==7168
    shape=[rows,h]
    a=estimate('quant',shape,chip=DT)
    resource,_,aiv,diag=resource_quant(shape,chip)
    r,w=diag['assumed_offchip_read_bytes'],int(diag['assumed_offchip_write_bytes'])
    # Existing resource model exposes non-overlapping first issue and drain; the
    # last core's arrival spread can overlap chip-level service and is excluded.
    exposed=diag['first_launch_cycles']/(1000*chip.frequency_ghz)+diag['stages_us']['drain']
    bandwidth=4096
    times={name:max(resource,exposed+(r+w)/(1000*bandwidth*eta)) for name,eta in coef.items()}
    return dict(shape=shape,chip=DT,active_aiv=aiv,auto_us=a.latency_us,resource_us=resource,target_plan=diag['target_plan'],
                logical_read=a.work['logical_read_bytes'],logical_write=a.work['logical_write_bytes'],assumed_read=r,
                assumed_write=w,assumed_output_retained=diag['output_retained_bytes'],wire_gbs=bandwidth,exposed_us=exposed,
                eta_case_us=times,time_range_us=[times['high'],times['low']])


def register():
    DEST.mkdir(parents=True,exist_ok=True)
    stats,coef=efficiency()
    s=Scenario('Kimi-K3',32768,batch=4)
    h=config(s.model)['hidden_size']
    n=s.t*h
    shape=[s.t,h]
    assert s.t==131072 and h==7168
    if RAW.exists(): raise ValueError('pre-existing raw output')
    existing=read(ROOT/'results/v0.27/replay.json')['rows']
    assert all(r['shape']!=shape for r in existing if r['op']=='quant')
    sources=[CTRL,HOST,OBJECT,OBJECT.with_suffix('.json'),Path(__file__).resolve(),ROOT/'scripts/cann-run.sh',
             ROOT/'scripts/cann-native-sim-run.sh',ROOT/'build/capture/libsim_soc_adapter.so',ROOT/'smoke/large_vector.cpp',
             ROOT/'operator_model/quant_resource.py',ROOT/'operator_model/target_tiling.py',ROOT/'configs/models/Kimi-K3.json',
             ROOT/'configs/chips/Ascend950DT_9582.json',ROOT/'development/validation/large20/native.py',
             ROOT/'development/workloads/prefill_shapes.py']
    earlier=read(ROOT/'results/v0.20/full-quant/freeze.json')
    for p in [HOST,ROOT/'smoke/large_vector.cpp',ROOT/'build/capture/libsim_soc_adapter.so']:
        assert sha(p)==earlier['binary_and_source'][str(p.relative_to(ROOT))]
    job=dict(created_utc=now(),role='Independent B4 native wrapper after Cast-only sensitivity selection. The B8 application is retrospective.',
             efficiency_source=stats,efficiency=coef,source_rule='min / equal-shape median / max of per-shape achieved aggregate DDR payload fraction in mixed R+W 1us channel periods whose DMC queue is occupied >=95% of ticks. No Kimi latency or traffic was used in prediction.',
             formula='max(T_resource, T_first_issue+T_drain+(R+W_visible)/(1000*4096*eta)), us. R and W_visible from current 64-AIV tiling and bounded output retention. Physical source: 128 x 32-bit x 8 Gbit/s per pin.',
             retrospective_b8=scenario(262144,coef),independent_b4=scenario(s.t,coef),shape=shape,scenario=s.__dict__,
             operator_contract='aclnnDynamicMxQuant; BF16 input contiguous, block=32, E4M3FN y and E8M0 scale [rows, hidden/64,2]; Kimi mixed QKV, MXFP8 dynamic method, scale_alg=0.',
             items_to_verify=[n,n//32],raw_cap_gib=12,wall_s=4*3600,expected_aiv=64,attempts=1,
             measurement=FULL_SCOPE,participates_in_instruction_WAPE=False,
             qualification='same frozen host and adapter; RUN success + exact case and SMOKE_PASS; full numerical checks; all 64 unique AIV IDs for task0 and kernel0, both done events and wrapper timestamps; no simulator sync error; within time/disk cap.',
             comparison='Frozen interval hit or miss; signed error of all three sensitivity scenarios and auto/resource vs wrapper. No interval expansion or fitting.',
             priority_receipt=str(PRIORITY.relative_to(ROOT)),slot=str(MEM_SLOT.relative_to(ROOT)),
             hashes={str(f.relative_to(ROOT)):sha(f) for f in sources})
    save(REG,job,exclusive=True)
    write_summary(job)
    return job


def headroom():
    def sample():
        with Path('/proc/stat').open() as f:
            rows=[line.split() for line in f if re.match(r'cpu\d+ ',line)]
        return {int(k[3:]):(sum(map(int,x[:8])),int(x[3])+int(x[4])) for k,*x in rows}
    allowed=os.sched_getaffinity(0)
    a=sample();time.sleep(2);b=sample()
    cpu_usage={i:1-(b[i][1]-a[i][1])/max(1,b[i][0]-a[i][0]) for i in allowed}
    free_cpu=[]
    for i,rate in sorted(cpu_usage.items(),key=lambda j:(j[1],j[0])):
        sib=(Path('/sys/devices/system/cpu')/f'cpu{i}/topology/thread_siblings_list').read_text().strip()
        if rate<=.30 and all(sib!=domain for _,domain in free_cpu): free_cpu.append((i,sib))
    ram=int(re.search(r'^MemAvailable:\s*(\d+)',Path('/proc/meminfo').read_text(),re.M)[1])/2**20
    free=shutil.disk_usage(DEST).free/2**30
    load=os.getloadavg()[0]
    return dict(eligible=len(free_cpu)>=80 and free>16 and ram>48 and load<len(allowed)*.65,
                sampled_cpus=len(allowed),idle_physical=len(free_cpu),chosen_cpus=[i for i,_ in free_cpu[:80]],
                available_ram_gib=ram,free_disk_gib=free,load_1m=load)


def status(phase,**extra):
    save(STATE,dict(phase=phase,utc=now(),pid=os.getpid(),freeze_sha256=sha(REG),**extra))


def verify_target(j,res):
    reasons=[]
    if not res.get('qualified_core_envelope'): reasons.append(res.get('qualification_error','native wrapper or numeric gate failed'))
    if res.get('active_aiv')!=j['expected_aiv']:reasons.append('expected exact 64 AIV')
    if res.get('output_counts')!=j['items_to_verify']:reasons.append('incomplete outputs')
    log=(RAW/'run.log').read_text(errors='replace') if (RAW/'run.log').exists() else ''
    if f"case=dynamic_mx_quant rows={j['shape'][0]} cols={j['shape'][1]} input=BF16 output=E4M3FN block=32" not in log:
        reasons.append('target contract absent in log')
    if re.search(r'REPEAT_SET|SYNC ERROR|illegal instruction|deadlock',log,re.I): reasons.append('fault signal in run log')
    traced={}
    for f in (RAW/'log_ca').glob('core*_summary_log'):
        for x in re.finditer(r'core: (\d+), subcore: (\d+), start: (\d+), tick: (\d+), kernelId: (\d+), blkID: (\d+), blkDim: (\d+),.*?done',f.read_text(errors='replace')):
            core,sub,begin,end,k,bid,dim=map(int,x.groups())
            if bid in traced or sub not in (1,2) or bid!=2*core+sub-1 or k!=0 or dim!=64 or begin>=end: reasons.append('unexpected launch, duplicate id or incomplete core')
            traced[bid]=(begin,end)
    if set(traced)!=set(range(64)): reasons.append('missing whole roster')
    if res['cost']['peak_bytes']>12*2**30 or res['cost'].get('stop_reason'): reasons.append('capture budget exhausted')
    if res['cost']['seconds']>j['wall_s']: reasons.append('wall cap exceeded')
    qualified=not reasons
    us=res.get('core_envelope_us') if qualified else None
    if us is not None:
        others=j['independent_b4']; lo,hi=others['time_range_us']
        assert math.isclose(us,(max(e for b,e in traced.values())-min(b for b,e in traced.values()))/1650)
        score=dict(inside_interval=lo<=us<=hi,signed_error_pct={key:100*(value/us-1) for key,value in {
            'auto':others['auto_us'],'resource':others['resource_us'],**others['eta_case_us']}.items()})
    else: score=None
    return dict(qualified_core_envelope=qualified,core_envelope_us=us,instruction_window_us=None,
                failed_checks=reasons,score=score,capture=res,raw=str(RAW.relative_to(ROOT)),
                run_log_sha256=sha(RAW/'run.log') if (RAW/'run.log').exists() else None,
                core_sha256={f.name:sha(f) for f in sorted((RAW/'log_ca').glob('*_summary_log'))},freeze_sha256=sha(REG))


def run():
    j=read(REG)
    with (DEST/'queue.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if RESULT.exists(): return
        if RAW.exists():raise FileExistsError('prior attempt present; will not run the case again')
        try:
            while not PRIORITY.exists():
                status('queued behind serial Cast pair',wait_for=j['priority_receipt']);time.sleep(30)
            with MEM_SLOT.open('a') as common:
                status('waiting for shared CA slot')
                fcntl.flock(common,fcntl.LOCK_EX)
                while True:
                    info=headroom()
                    if info['eligible']:break
                    status('waiting for spare physical cores, RAM and disk',resource_check=info);time.sleep(30)
                for name,h in j['hashes'].items():
                    if sha(ROOT/name)!=h: raise ValueError('frozen input changed: '+name)
                os.sched_setaffinity(0,info['chosen_cpus']); os.nice(8)
                status('running one B4 native DT case',resource_check=info,raw=str(RAW.relative_to(ROOT)))
                res=capture_vector(j,RAW,HOST,['quant',*map(str,j['shape'])],j['items_to_verify'])
                receipt=verify_target(j,res)
                save(RESULT,receipt,exclusive=True)
                write_summary(j,receipt)
                status('finished',qualified=receipt['qualified_core_envelope'],cost=res['cost'],result=str(RESULT.relative_to(ROOT)))
        except Exception as exc:
            status('stopped',error=repr(exc),raw_exists=RAW.exists());raise


def write_summary(j,r=None):
    retrospective=j['retrospective_b8']; new=j['independent_b4']
    historical=read(ROOT/'results/v0.20/full-quant/native.json')['core_envelope_us']
    data=j['efficiency_source'];e=j['efficiency']
    body=f"""# 950DT 大形量化的跨算子带宽情景

冻结于 {j['created_utc']}。只使用三个 Cast 的真实 DDR 数据服务率，低、中、高取三个比例的最小值、中位数与最大值。各比例统计的是单个物理通道中队列至少 95% 周期非空、且在同一 1 µs 内确实既读又写的完整报告期。不同读写比例、地址局部性或输出驻留会使实际效率超出此范围。此区间是敏感性情景，没有统计置信水平。

| Cast 对照 | 混合满队列报告期 | DDR 数据服务率 |
|---|---:|---:|
"""
    for c in data:body+=f"| {c['op']} {c['shape'] if c['op']=='Pro indexer Cast' else ''} | {c['accepted_periods']:,} | {c['eta']*100:.3f}% |\n"
    body+="""
对总 token 数 $M$、宽度 $H=7168$，读取 $R=2MH$ 字节，生成 $W_{log}=MH+2M\\lceil H/64\\rceil$ 字节输出。资源模型预留 64 个 AIV 的双输入 tile 后，以剩余 L2 容量估计退出时可留下的脏输出 $C_{dirty}$，得到 $W=W_{log}-C_{dirty}$。情景时间为

$$T(\\eta)=\\max\\left[T_{res},T_{first}+T_{end}+\\frac{R+W}{1000B\\eta}\\right].$$

$B=4096$ GB/s 为归档 CA 引脚线速，$\\eta$ 是 Cast 的实测比例。$T_{first}$ 和 $T_{end}$ 是已有资源配置中暴露的首次发起与收尾设备时间（µs）。多核启动跨度能与总线服务重叠，未另作串行相加。$T_{res}$ 保留原计算、UB、MTE 和芯片共享的资源约束。公式的字节来自量化张量及分块，未使用量化总耗时、实测量化 DMC 字节或实测量化服务率来设定任何参数。

| Kimi BF16 shape | 来源 | 预测的 HBM 读/写 B | 默认 / 资源 µs | 高效率→低耗时 µs | 中位 µs | 低效率→高耗时 µs | 原生窗口 µs |
|---|---|---:|---:|---:|---:|---:|---:|
"""
    for x,role,us in [(retrospective,'B8 × 32 Ki 已见形回顾',historical),(new,'B4 × 32 Ki 冻结独立形',None if not r else r['core_envelope_us'])]:
        t=x['eta_case_us'];meas='N/A' if us is None else f'{us:.3f}'
        body+=f"| {x['shape']} | {role} | {x['assumed_read']:,} / {x['assumed_write']:,} | {x['auto_us']:.3f} / {x['resource_us']:.3f} | {t['high']:.3f} | {t['mid']:.3f} | {t['low']:.3f} | {meas} |\n"
    body+=f"""
B8 模型所对应的量化分支为 Kimi 混合 QKV、MXFP8、`scale_alg=0`。整批单次调用数值验证通过。其已知原生包络相对本次情景中点偏差（预测/实测−1）为 {100*(retrospective['eta_case_us']['mid']/historical-1):+.2f}%，用于解释迁移表现。B4 减小了输入与输出量，预测每核 {new['target_plan']['iterations']} 轮；硬件、张量类型和原生采集合同延续 B8。B4 启动前锁定全部预测与 SHA，出错保留原始原因，且最多采一次。标准逐指令窗均为 N/A。

单槽任务在现有两端 Cast 收尾后启动。离线采集限 4 小时及 12 GiB，检查全部 {j['items_to_verify'][0]:,} 个 FP8 输出、{j['items_to_verify'][1]:,} 个 scale 及全部 64 个 AIV。状态、PID、冻结值和终态分别在 [state.json](state.json)、[freeze.json](freeze.json) 及完成后生成的 `native.json`。
"""
    if r and r['qualified_core_envelope']:
        body+=f"\n独立 B4 的原生窗口为 **{r['core_envelope_us']:.3f} µs**；位于冻结的区间内：{r['score']['inside_interval']}。三档及对照的有符号百分比误差见 [终态回执](native.json)。\n"
    elif r:
        body+='\nB4 此次未取得合格的原生时间。详细阻断保存在 [回执](native.json)。\n'
    (DEST/'README.md').write_text(body)


def start():
    with (DEST/'spawn.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if RESULT.exists(): raise RuntimeError('already completed')
        if STATE.exists():
            state=read(STATE)
            proc=Path('/proc')/str(state['pid'])/'cmdline'
            if proc.exists() and b'quant_transfer.py' in proc.read_bytes(): raise RuntimeError('supervisor already exists')
            if RAW.exists(): raise RuntimeError('attempt started previously; will not overwrite')
        log=(DEST/'supervisor.log').open('a')
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run'],cwd=ROOT,stdin=subprocess.DEVNULL,
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        print(p.pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','start','run'])
    {'register':register,'start':start,'run':run}[p.parse_args().action]()
