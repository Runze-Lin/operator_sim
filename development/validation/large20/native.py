"""Low-volume 950DT all-core CA summary acquisition with the existing SKU adapter."""
from pathlib import Path
import os
import re
import signal
import subprocess
import sys
import time
from large_cast import BIN, MODULE, DT, OUT, ROOT, digest, estimate, now, read, write
from development.validation.sku19.collect import nbytes

NATIVE=ROOT/'runs/prefill20-native'
REG=OUT/'native-freeze.json'
def freeze():
    assert not REG.exists()
    shapes=[(262144,5120),(32,4096),(262144,32)]
    jobs=[dict(shape=[t,h],prediction_us=estimate('cast_bf16_fp32',(t,h),chip=DT,timing_mode='resource').latency_us,
               wall_s=4*3600 if t*h>100_000_000 else 3600,raw_cap_gib=8) for t,h in shapes]
    write(REG,dict(utc=now(),jobs=jobs,host_sha256=digest(BIN),module_sha256=digest(MODULE),
          adapter_sha256=digest(ROOT/'build/capture/libsim_soc_adapter.so'),chip=DT,frequency_ghz=1.65,
          comparison='Archive all per-core start to done ticks from the actual CA wrapper. This envelope includes core setup; compare to all-instruction window for same shape, never replace canonical timing with host or STARS deltas.'))
def run(index):
    reg=read(REG);j=reg['jobs'][index];t,h=j['shape']
    folder=NATIVE/f'cast_{t}_{h}';folder.mkdir(parents=True,exist_ok=False)
    assert digest(BIN)==reg['host_sha256'] and digest(ROOT/'build/capture/libsim_soc_adapter.so')==reg['adapter_sha256']
    cmd=['/usr/bin/time','-f','%U %S','-o',str(folder/'cpu.txt'),'bash',str(ROOT/'scripts/cann-native-sim-run.sh'),str(folder),str(BIN),str(t),str(h)]
    start=now();t0=time.monotonic();peak=0;stop=None
    env={**os.environ,'OMP_WAIT_POLICY':'PASSIVE','GOMP_SPINCOUNT':'0'}
    with (folder/'run.log').open('w') as log:
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=folder,start_new_session=True,env=env)
        while p.poll() is None:
            peak=max(peak,nbytes(folder))
            if time.monotonic()-t0>j['wall_s'] or peak>2**30*j['raw_cap_gib']:
                stop='time cap' if time.monotonic()-t0>j['wall_s'] else 'disk cap'
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                break
            time.sleep(3)
    raw=nbytes(folder)
    cpu=(folder/'cpu.txt').read_text().splitlines()
    cost=dict(started_utc=start,finished_utc=now(),seconds=time.monotonic()-t0,raw_bytes=raw,
              peak_bytes=max(raw,peak),exit_code=p.returncode,stop_reason=stop,
              cpu_seconds=sum(map(float,cpu[-1].split())) if cpu and re.fullmatch(r'[.\d]+ [.\d]+',cpu[-1]) else None)
    res=dict(shape=j['shape'],chip=DT,cost=cost,qualified_core_envelope=False,instruction_ca_us=None)
    try:
        text=(folder/'run.log').read_text(errors='replace')
        assert cost['exit_code']==0 and not stop
        assert f'runtime_soc={DT}' in text and 'hbm3_8000M.ini' in text and 'SMOKE_PASS' in text
        assert f'correctness count={t*h} bad=0' in text
        windows={}
        for file in (folder/'log_ca').glob('core*_summary_log'):
            for core,sub,start,end in re.findall(r'core: (\d+), subcore: (\d+), start: (\d+), tick: (\d+), kernelId: \d+, blkID: \d+.*?done',file.read_text(errors='replace')):
                key=(int(core),int(sub));assert key not in windows and int(end)>int(start)
                windows[key]=(int(start),int(end))
        emitted={int(i) for i in re.findall(r'\[block_end\]\s+: AIV, task_id=0, core_id=(\d+),',text)}
        assert len(windows)==len(emitted)>0 and all(s in (1,2) for c,s in windows)
        duration=(max(y for x,y in windows.values())-min(x for x,y in windows.values()))/1650
        res.update(qualified_core_envelope=True,core_envelope_us=duration,active_aiv=len(windows),core_ticks={f'{c}.{s}':v for (c,s),v in windows.items()},
                   correct_elements=t*h,metric='all-core wrapper start to done ticks, distinct from all-instruction window')
    except (AssertionError,OSError,ValueError) as e:res['qualification_error']=repr(e)
    write(OUT/(folder.name+'_native.json'),res)
    print(folder.name,res['qualified_core_envelope'],res.get('core_envelope_us'),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:run(int(sys.argv[1]))
