"""Fair one-endpoint shared CA queue with durable, single-attempt receipts."""
from common import *
import fcntl
import os
import shutil
import signal
import subprocess
import time
from verify import qualify


def state(phase,case=None,**data):
    save(OUT/'state.json',dict(utc=now(),phase=phase,case=case,pid=os.getpid(),**data))

def capture(j,d):
    chip=j['chip'];op=j['op'];budget=j['wall_s'];d.mkdir()
    env=dict(os.environ,OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    args=['/usr/bin/time','-f','%U %S','-o',str(d/'cpu.txt'),'bash',str(ROOT/'scripts/cann-run.sh')]
    if j['protocol']=='instruction':
        if op=='attn_res_fwd':
            args+=['bash','-c','export ASCEND_CUSTOM_OPP_PATH="$1"; export LD_LIBRARY_PATH="$1/op_api/lib:$LD_LIBRARY_PATH"; shift; exec "$@"','attn',str(VENDOR)]
        args+=['msopprof','simulator','--soc-version='+chip,'--dump=false','--launch-count=1',
               '--timeout='+str((budget+59)//60),'--output='+str(d/'profile')]
    else:
        args+=['bash',str(Path(__file__).with_name('native-entry.sh')),str(ROOT),chip]
    args+=[str(ROOT/j['runner']),*j['args']]
    start=time.monotonic();stamp=now();peak=0;stop=None
    with (d/'run.log').open('w') as log:
        proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,cwd=d,env=env,start_new_session=True)
        try:
            while proc.poll() is None:
                used=0
                for directory,_,files in os.walk(d):
                    for name in files:
                        try:used+=(Path(directory)/name).lstat().st_size
                        except FileNotFoundError:pass
                peak=max(peak,used)
                if used>j['raw_gib']*2**30 or time.monotonic()-start>budget:
                    stop='disk cap' if used>j['raw_gib']*2**30 else 'wall cap';break
                time.sleep(10)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    cpu=(d/'cpu.txt').read_text().strip().splitlines()
    cost=dict(started_utc=stamp,finished_utc=now(),seconds=time.monotonic()-start,peak_bytes=peak,exit_code=proc.returncode,
              stop_reason=stop,cpu_seconds=sum(map(float,cpu[-1].split())) if cpu and len(cpu[-1].split())==2 else None,command=args[5:])
    save(d/'cost.json',cost)

def run():
    reg=read(FREEZE)
    for f,h in reg['inputs_sha256'].items():
        if digest(ROOT/f)!=h:raise ValueError('changed frozen source '+f)
    RAW.mkdir(parents=True,exist_ok=True);os.nice(10)
    with (RAW/'owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for j in reg['jobs']:
            case=j['case'];dest=OUT/(case+'.json');d=RAW/case
            if dest.exists():continue
            result=dict(case=case,op=j['op'],shape=j['shape'],chip=j['chip'],protocol=j['protocol'],qualified=False,
                instruction_us=None,wrapper_us=None,freeze_sha256=digest(FREEZE))
            ctrl=j['control_required']
            if ctrl and not read(OUT/(ctrl+'.json')).get('qualified'):
                result.update(reason='PR native control did not qualify',physical_launch=False);save(dest,result,new=True);continue
            if not (d/'cost.json').exists():
                state('waiting for shared CA slot',case)
                with (RAW.parent/'shared.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    if d.exists():state('partial previous run preserved; review before retry',case);return
                    if shutil.disk_usage(ROOT).free<(j['raw_gib']+16)*2**30:state('disk space insufficient for registered budget',case);return
                    state('collecting',case);capture(j,d)
            cost=read(d/'cost.json');state('verifying exported raw',case)
            result.update(cost=cost,physical_launch=True,raw=rel(d))
            try:
                if cost['started_utc']<=reg['utc']:raise ValueError('late freeze')
                result.update(qualify(j,d,cost))
            except (AssertionError,OSError,ValueError,StopIteration,KeyError,TypeError,IndexError) as e:result['reason']=repr(e)
            save(dest,result,new=True)
            from report import render
            render()
            time.sleep(2)
        state('finished')
        from report import render
        render()

if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
