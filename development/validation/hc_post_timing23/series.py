"""Write-once registration and serial capture using the existing release verifier."""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'results/v0.23/hc-timing'
RAW=OUT/'raw'
sys.path.insert(0,str(HERE.parent/'hcpost23'))
import capture as original
DT,PR=original.CHIPS
original.BUILD=RAW/'build'
ORIGINAL_REG=ROOT/'results/v0.23/hc-post/development.json'
DEVELOPMENT=[(DT,65,256),(PR,65,256),(DT,2,16448)]
# Three official model widths, 16-lane boundary, first/second split and many rows;
# choose distinct full shapes whose combined size stays within the unchanged host budget.
HOLDOUT=[(1,512),(2,1280),(4,4095),(8,4096),(7,4097),
         (3,7168),(2,8192),(1,16384),(57,192),(129,128)]
def utc(): return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_bytes())
def save(p,obj):
    path=p.with_suffix('.tmp');path.write_text(json.dumps(obj,indent=2)+'\n');path.replace(p)
def hashes(files):return {str(p.relative_to(ROOT)):original.digest(p) for p in files}
def register(role):
    path=OUT/f'{role}-register.json'
    if path.exists():raise FileExistsError(path)
    old=read(ORIGINAL_REG)
    pairs=DEVELOPMENT if role=='development' else [(chip,t,d) for t,d in HOLDOUT for chip in (DT,PR)]
    files=[HERE/'series.py',HERE/'build_host.py',HERE.parent/'hcpost23/capture.py',ORIGINAL_REG,RAW/'build/host_probe',RAW/'build/host.cpp',
           ROOT/'smoke/extension_indexer_mhc.cpp']
    if role=='holdout':
        files+=[HERE/'model.py',OUT/'parameters.json']
        from model import predict
    cases=[]
    for chip,t,d in pairs:
        case=dict(chip=chip,shape=[1,t,4,d],id=f'{role}-{chip}-t{t}-d{d}')
        if (RAW/case['id']).exists():raise FileExistsError(case['id'])
        if role=='holdout':case['predictions']={sku:predict(t,d,sku) for sku in (DT,PR,'Ascend970DT')}
        cases.append(case)
    data=dict(role=role,utc=utc(),hashes={**old['hashes'],**hashes(files)},kernel=old['kernel'],
              golden=old['golden'],raw_mib=1650,wall_s=450,max_concurrent=1,
              note='Each host run uses only the publicly registered vLLM HcPost vendor, with exact BF16 full output. Per-case hard wall/disk budget. No retries of started directories.',
              cases=cases)
    if role=='holdout':
        history={(c['tokens'],c['hidden']) for c in old['cases']}|{(t,d) for _,t,d in DEVELOPMENT}
        assert not history.intersection(HOLDOUT) and len(set(HOLDOUT))==10
    with path.open('x') as f:json.dump(data,f,indent=2);f.write('\n')

def run(role):
    reg=read(OUT/f'{role}-register.json')
    for p,sha in reg['hashes'].items():
        if original.digest(ROOT/p)!=sha:raise ValueError(f'Frozen source changed: {p}')
    with (RAW/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for case in reg['cases']:
            name=case['id'];folder=RAW/name;result=OUT/(name+'.json')
            if result.exists():continue
            if folder.exists():raise RuntimeError('Unclosed previous capture '+name)
            folder.mkdir()
            save(OUT/f'{role}-state.json',dict(status='running',case=name,started=utc()))
            _,t,_,d=case['shape'];chip=case['chip']
            launch='export ASCEND_CUSTOM_OPP_PATH="$1"; export LD_LIBRARY_PATH="$1/op_api/lib:$LD_LIBRARY_PATH"; shift; exec "$@"'
            args=['bash',str(ROOT/'scripts/cann-run.sh'),'bash','-c',launch,'hcpost',str(original.VENDOR),'msopprof','simulator',
                  '--soc-version='+chip,'--dump=false','--launch-count=1','--timeout=8','--output='+str(folder/'profile'),str(original.BUILD/'host_probe'),str(t),str(d)]
            start=time.monotonic();started=utc();reason=None
            with (folder/'run.log').open('x') as log:
                proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=folder)
                while proc.poll() is None:
                    if time.monotonic()-start>reg['wall_s'] or original.nbytes(folder)>reg['raw_mib']*2**20:
                        reason='registered wall/disk budget';os.killpg(proc.pid,signal.SIGKILL);break
                    time.sleep(2)
                status=proc.wait()
            cost=dict(exit_code=status,stop_reason=reason,started_utc=started,finished_utc=utc(),
                      seconds=time.monotonic()-start,raw_bytes=original.nbytes(folder))
            save(folder/'cost.json',cost)
            info=dict(shape=case['shape'],chip=chip,role=role,api='aclnnHcPost',qualified=False,
                      raw=str(folder.relative_to(ROOT)),cost=cost,registration_sha256=original.digest(OUT/f'{role}-register.json'))
            if not status and not reason:
                try:original.qualify(info,reg)
                except Exception as exc:info['reason']=type(exc).__name__+': '+str(exc)
            else:info['reason']=reason or f'host/profiler exit {status}'
            save(result,info)
            if not info['qualified']:
                save(OUT/f'{role}-state.json',dict(status='stopped',case=name,reason=info.get('reason')))
                raise RuntimeError(f'Ineligible {name}: {info.get("reason")}')
        save(OUT/f'{role}-state.json',dict(status='complete',finished=utc()))
if __name__=='__main__':
    action,role=sys.argv[1:];assert role in ('development','holdout')
    if action=='register':register(role)
    elif action=='run':run(role)
    else:raise ValueError(action)
