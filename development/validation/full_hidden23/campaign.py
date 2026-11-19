"""Pre-register an unchanged native Cast, audit its partial predecessor, and queue one retry."""
from collections import Counter
from datetime import datetime,timezone
from hashlib import sha256
import fcntl,json,os,re,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'development/validation/large20'))
from development.validation.large20.native import qualify_vector
from development.validation.sku19.campaign import read,write,now
OUT=ROOT/'results/v0.23/full-hidden';RAW=ROOT/'runs/full-hidden23';OLD=ROOT/'runs/prefill20-native/cast_262144_5120';SMALL=ROOT/'runs/prefill20-native/cast_32_4096';REG=OUT/'registration.json';KIMI=ROOT/'runs/kimi-full-quant20';KOUT=ROOT/'results/v0.20/full-quant/native.json'
NATIVE=ROOT/'scripts/cann-native-sim-run.sh';BIN=ROOT/'build/smoke/large_cast20';SI=ROOT/'results/v0.20/large/native-freeze.json';TERM=ROOT/'results/v0.21/large/full-hidden-terminal.json';CONT=ROOT/'results/v0.21/large/full-hidden-continuation.json'

def sha(p):return sha256(p.read_bytes()).hexdigest()

def trace_facts(folder):
    text=(folder/'run.log').read_text(); maps={}
    for kind in ('start','end'):
        matches=[(int(t),int(core),int(block)) for t,core,block in re.findall(r'\[(\d+)\] \[block_'+kind+r'\]\s+: AIV, task_id=0, core_id=(\d+), block_id=(\d+)',text)]
        if len(matches)!=len({c for _,c,_ in matches}):raise ValueError('duplicate AIV '+kind)
        maps[kind]={c:(b,t) for t,c,b in matches}
    windows={}
    for f in sorted((folder/'log_ca').glob('core*_summary_log')):
        matches=re.findall(r'core: (\d+), subcore: (\d+), start: (\d+), tick: (\d+), kernelId: 0, blkID: (\d+), blkDim: (\d+), rtsqId: 2, streamId: 0, taskId: 0, subtaskId: 0, threadId: 0, done',f.read_text())
        for c,l,t,u,b,n in matches:
            c,l,t,u,b,n=map(int,(c,l,t,u,b,n));i=2*c+l-1
            assert l in (1,2) and i not in windows and u>t and maps['start'][i][0]==maps['end'][i][0]==b
            windows[i]=dict(start=t,done=u,block=b,block_dim=n,physical_core=c,subcore=l)
    assert set(windows)==set(maps['end'])<=set(maps['start']);assert len({v[0] for v in maps['start'].values()})==len(maps['start'])
    return dict(block_start=maps['start'],block_end=maps['end'],wrapper=windows,first_completed_lane_start=min(w['start'] for w in windows.values()),last_completed_lane_done=max(w['done'] for w in windows.values()),run_log_sha256=sha(folder/'run.log'),summary_sha={f.name:sha(f) for f in sorted((folder/'log_ca').glob('core*_summary_log'))},goldens=re.findall(r'^correctness count=(\d+) bad=(\d+)$',text,re.M),task_done='[TASK_DONE]' in text,smoke_pass='SMOKE_PASS' in text)

def nbytes(root):return sum(f.lstat().st_size for f in root.rglob('*') if not f.is_dir())

def cpustat():
    result={}
    for line in Path('/proc/stat').read_text().splitlines():
        parts=line.split();m=re.fullmatch(r'cpu(\d+)',parts[0]);
        if m:
            c=int(m[1]);v=list(map(int,parts[1:9]));result[c]=(sum(v),v[3]+v[4])
    return result

def register():
    if REG.exists() or RAW.exists():raise FileExistsError(REG)
    old=read(TERM)['cost'];obs=trace_facts(OLD);files=sorted(OLD.glob('log/lpddr_state*.log'));names=[p.name for p in files];small_names=sorted(f.name for f in SMALL.glob('log/lpddr_state*.log'));assert names==small_names and len(names)==128
    assert len(obs['block_start'])==64 and len(obs['wrapper'])==31 and not obs['goldens'] and not obs['smoke_pass'];lo=obs['first_completed_lane_start'];hi=obs['last_completed_lane_done'];stats_bytes=sum(f.stat().st_size for f in files)
    write(OUT/'previous-partial.json',dict(shape=[262144,5120],qualified=False,ca_us=None,core_envelope_us=None,partial_wrapper_lower_bound_us=(hi-lo)/1650,completed_lane_span_ticks=[lo,hi],different_clocks='Collector progress counter and STARS block ticks are not wrapper ticks; neither is used in this lower bound.',completed_aiv=len(obs['wrapper']),started_aiv=len(obs['block_start']),periodic_stat_bytes=stats_bytes,original_cost=old,logs=obs))
    archive=read(SI);assert archive['host_sha256']==sha(BIN);assert archive['adapter_sha256']==sha(ROOT/'build/capture/libsim_soc_adapter.so')
    a=cpustat();time.sleep(.6);b=cpustat();allowed=os.sched_getaffinity(0);ids=sorted(allowed,key=lambda c:(-((b[c][1]-a[c][1])/max(b[c][0]-a[c][0],1)),c))[:16]
    extra=[Path(__file__).resolve(),NATIVE,ROOT/'scripts/cann-run.sh',BIN,ROOT/'build/capture/libsim_soc_adapter.so',SI,TERM,CONT,ROOT/'development/validation/large20/cast.cpp',ROOT/'smoke/large_vector.cpp',ROOT/'results/v0.20/large/cast_32_4096_native.json',ROOT/'results/v0.20/full-quant/freeze.json',ROOT/'development/validation/large20/native.py']
    registration=dict(utc=now(),sku='Ascend950DT_9582',shape=[262144,5120],api='aclnnCast BF16 to FP32, all 1,342,177,280 elements checked bit-for-bit',
        previous_frozen_us={'v020':archive['jobs'][0]['prediction_us'],'v021_before_retry':read(CONT)['target_predictions_us']},
        sources={str(p.relative_to(ROOT)):sha(p) for p in extra},original_controller_names=names,control_shape=[32,4096],control_cpu_ids=ids,control_nice=19,
        control_wall_s=900,control_raw_cap_gib=2,control_scheme='Run full-output then redirected control sequentially on identical 16 lightly used CPUs at nice 19. Confirm exact 128 paths opened write-only, both all-element golden, same 64 wrapper and block identities, original archived control and <=5% envelope difference. Existing outputs untouched.',
        retry_wall_s=24*3600,redirect_raw_cap_gib=16,unmodified_raw_cap_gib=48,archive_output_gib_per_hour=old['raw_bytes']/2**30/(old['seconds']/3600),fallback='Use the original complete period logging with 48 GiB if the short non-timing-output contract does not pass. At measured 1.29 GiB/h, 24 h requires about 31 GiB; 48 GiB allows about 55% margin.',
        queue_gate='Single retry only after Kimi terminal receipt and no remaining Kimi native child; stale missing receipt allowed only after registered Kimi budget has elapsed, GNU time host completion and a second observation of no live process. Do not modify Kimi.',host_space_free_gib=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize/2**30)
    write(REG,registration)

def invoke(folder,shape,reg,*,redirect,control):
    folder.mkdir(parents=True,exist_ok=False);logdir=folder/'log';logdir.mkdir()
    if redirect:
        for name in reg['original_controller_names']:
            assert re.fullmatch(r'lpddr_state\d+\.log',name);(logdir/name).symlink_to('/dev/null')
    command=[str(NATIVE),str(folder),str(BIN),*map(str,shape)];cap=reg['control_raw_cap_gib'] if control else reg['redirect_raw_cap_gib'] if redirect else reg['unmodified_raw_cap_gib'];budget=reg['control_wall_s'] if control else reg['retry_wall_s']
    # Restrict the short comparison and give the ongoing native Kimi priority.
    if control:
        command=['nice','-n','19','taskset','-c',','.join(map(str,reg['control_cpu_ids'])),'strace','-f','-qq','--seccomp-bpf','-e','trace=open,openat,openat2','-o',str(folder/'open.trace'),*command]
    command=['/usr/bin/time','-f','%U %S','-o',str(folder/'cpu.txt'),*command]
    start=now();t0=time.monotonic();peak=0;why=None;env={**os.environ,'OMP_WAIT_POLICY':'PASSIVE','GOMP_SPINCOUNT':'0'}
    with (folder/'run.log').open('x') as out:
        p=subprocess.Popen(command,cwd=folder,stdout=out,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        write(folder/'process.json',dict(pid=p.pid,started_utc=start,cap_gib=cap,wall_s=budget))
        while p.poll() is None:
            peak=max(peak,nbytes(folder))
            if peak>cap*2**30 or time.monotonic()-t0>budget:
                why='disk cap' if peak>cap*2**30 else 'wall cap';os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(15)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                break
            time.sleep(3 if control else 20)
    cost=dict(started_utc=start,finished_utc=now(),seconds=time.monotonic()-t0,exit_code=p.returncode,stop_reason=why,peak_bytes=max(peak,nbytes(folder)),raw_bytes=nbytes(folder),cpu_text=(folder/'cpu.txt').read_text(),raw=str(folder.relative_to(ROOT)));write(folder/'cost.json',cost)
    result=qualify_vector(dict(shape=shape),folder,cost,[shape[0]*shape[1]]);result['stats_redirected']=redirect
    if result['qualified_core_envelope']:
        facts=trace_facts(folder);text=(folder/'run.log').read_text();assert len(facts['wrapper'])==len(facts['block_start'])==64 and all(i['block_dim']==64 for i in facts['wrapper'].values()) and facts['task_done'] and f'case=cast_bf16_fp32 api=aclnnCast R={shape[0]} D={shape[1]}' in text;result['evidence']=facts
    if control:
        s=(folder/'open.trace').read_text() if (folder/'open.trace').exists() else '';seen={};miss=[]
        for line in s.splitlines():
            if 'lpddr_state' not in line:continue
            m=re.search(r'"[^"\n]*\b(lpddr_state\d+\.log)"\s*,\s*([^,)]+)',line)
            if not m:miss.append(line[:140]);continue
            seen.setdefault(m[1],set()).add(m[2]);
        result['controller_open_flags']={n:sorted(v) for n,v in sorted(seen.items())};result['all_writes_only']=set(seen)==set(reg['original_controller_names']) and all(v and all('O_WRONLY' in f and 'O_RDWR' not in f for f in v) for v in seen.values()) and not miss
        result['controller_output_bytes']=sum(f.lstat().st_size for f in logdir.glob('lpddr_state*.log') if not f.is_symlink());result['state_symlinks']=sum(f.is_symlink() and os.readlink(f)=='/dev/null' for f in logdir.glob('lpddr_state*.log'));result['open_trace_sha256']=sha(folder/'open.trace') if (folder/'open.trace').exists() else None
    return result

def kimi_processes():
    live=[]
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            if (entry/'cwd').resolve(strict=True)==KIMI:live.append(int(entry.name))
        except (PermissionError,FileNotFoundError,OSError):continue
    return sorted(live)

def gate(reg):
    while True:
        live=kimi_processes();receipt=KOUT.is_file();age=time.time()-(KIMI/'run.log').stat().st_ctime
        fallback=not receipt and (KIMI/'cpu.txt').stat().st_size>0 and age>read(ROOT/'results/v0.20/full-quant/freeze.json')['wall_s']+120
        write(OUT/'queue-status.json',dict(utc=now(),state='waiting-for-Kimi',kimi_live_pids=live,kimi_terminal_receipt=receipt,kimi_registered_hours=12,kimi_registered_gib=24))
        if (receipt or fallback) and not live:
            time.sleep(10)
            if not kimi_processes():return
        time.sleep(45)

def run():
    reg=read(REG)
    for name,h in reg['sources'].items():assert sha(ROOT/name)==h,name
    RAW.mkdir(exist_ok=True)
    with (RAW/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name,red in [('control-original',False),('control-redirected',True)]:
            dest=OUT/(name+'.json')
            if dest.exists():continue
            write(OUT/'queue-status.json',dict(utc=now(),state=name));result=invoke(RAW/name,reg['control_shape'],reg,redirect=red,control=True);write(dest,result)
        a=read(OUT/'control-original.json');b=read(OUT/'control-redirected.json');orig=read(ROOT/'results/v0.20/large/cast_32_4096_native.json');ok=all(x['qualified_core_envelope'] for x in (orig,a,b)) and a['all_writes_only'] and b['all_writes_only'] and a['controller_output_bytes']>0 and b['controller_output_bytes']==0 and b['state_symlinks']==128
        if ok:ok=set(a['core_ticks'])==set(b['core_ticks'])==set(orig['core_ticks']) and max(abs(b['core_envelope_us']/q['core_envelope_us']-1) for q in (a,orig))<=.05
        write(OUT/'control-comparison.json',dict(eligible_for_redirect=ok,archived_envelope_us=orig['core_envelope_us'],baseline_us=a.get('core_envelope_us'),redirect_us=b.get('core_envelope_us'),merged_checks='Full bitwise golden, identical lane roster, read-only input/ELF/adapter checksums, all state paths write-only, <=5% window against fresh and archived full output.',applied_mode='suppress only periodic reports' if ok else 'full native logging, expanded disk budget'))
        if (OUT/'flash-retry.json').exists():return
        gate(reg);write(OUT/'queue-status.json',dict(utc=now(),state='flash-native-active',redirect=ok));result=invoke(RAW/'flash-retry',reg['shape'],reg,redirect=ok,control=False);write(OUT/'flash-retry.json',result);write(OUT/'queue-status.json',dict(utc=now(),state='complete',qualified=result['qualified_core_envelope']))

if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
