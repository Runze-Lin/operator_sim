"""Serial, one-attempt development campaign for the published direct FLA V2 ABI."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SUPPORT = HERE.parent/'chunk_recovery/fla2691'
sys.path.insert(0,str(SUPPORT/'v2'))
from probe_v2 import probe, BIN, REG as FIRST_PLAN
from verify_chrome import content, multiset_digest

OUT=ROOT/'results/kda-v23'
RAW=OUT/'raw'
REG=OUT/'development-register.json'
DT='Ascend950DT_9582'
PR='Ascend950PR_957b'
MEMORY={DT:'hbm3_8000M.ini',PR:'lpddr5_6400M_bailu.ini'}
# These are all development samples. A later holdout must use different full shapes.
CASES=[(PR,2,1,'paired control with previous DT'),(DT,64,1,'first full chunk'),
       (DT,65,1,'second chunk at one head'),(PR,65,1,'paired second-chunk anchor'),
       (DT,15,1,'short one-head reference'),(DT,15,2,'head-count contrast'),
       (PR,15,2,'paired two-head check'),(DT,63,1,'one row before chunk boundary')]

def when():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(path.read_bytes())

def save(path,value):
    probe.put(path,value)

def register():
    if REG.exists():raise FileExistsError(REG)
    old=read(FIRST_PLAN)
    code=[Path(__file__).resolve(),HERE/'qualify.py',FIRST_PLAN,probe.VENDOR/'op_api/lib/libcust_opapi.so',
          BIN,BIN.with_suffix('.cpp'),SUPPORT/'v2/verify_chrome.py',SUPPORT/'probe.py',ROOT/'scripts/cann-run.sh',
          FIRST_PLAN.parent/'verified.json',ROOT/'configs/chips'/f'{DT}.json',ROOT/'configs/chips'/f'{PR}.json']
    for name,h in old['files'].items():
        if probe.digest(ROOT/name)!=h:raise ValueError('previous source differed: '+name)
    cases=[]
    for sku,t,h,aim in CASES:
        case_id=f'{sku[:11]}-t{t}-h{h}'.lower()
        cases.append(dict(id=case_id,sku=sku,physical_memory=MEMORY[sku],shape=[1,t,h,h,128,128],
                          work_items=h*((t+63)//64),purpose=aim,oracle=dict(attention=t*h*128,aqk=t*h*64,akk=t*h*64,state_vk=h*128*128)))
    assert len({c['id'] for c in cases})==len(cases)
    save(REG,dict(utc=when(),role='mechanism development only; zero post-freeze holdouts',max_concurrency=1,
        budget_per_capture_seconds=900,budget_per_capture_bytes=3*1024**3,budget_total_capture_seconds=5400,
        reused_control=dict(sku=DT,shape=[1,2,1,1,128,128],receipt=probe.rel(FIRST_PLAN.parent/'verified.json')),
        contract=old['contract'],axes_overridden_by_case=['S','Hq','Hv','cu=[0,S]','indices=[[0,i] for i in range(ceil(S/64))]'],
        source_hashes={probe.rel(p):probe.digest(p) for p in code},published_device_text=old['released_device_text'],
        protocol='Same frozen host and release, direct V2. Verify selected SKU and its physical memory config, all four numerical groups, unmodified initial state, no CA sync errors, same three primary release families, captured ELFs, full official roster and per-core durations, physical END and merged Chrome B/E versus shard signatures on every launch. Record every auxiliary launch.',
        cases=cases))
    print(probe.digest(REG))

def capture(reg,case,folder,remaining):
    t=case['shape'][1];h=case['shape'][2]
    path=folder/'cost.json';log_path=folder/'run.log'
    if path.exists():return read(path)
    if log_path.exists():
        # Never launch again over a previously started run.
        return dict(exit_code=None,stop='interrupted previous supervisor; preserved raw attempt',seconds=None,peak_bytes=None)
    command=['bash',str(ROOT/'scripts/cann-run.sh'),'bash','-c','source "$1/bin/set_env.bash"; shift; exec "$@"',
             'fla',str(probe.VENDOR),'msopprof','simulator','--soc-version='+case['sku'],'--dump=false',
             '--launch-count=32','--timeout=15','--output='+str(folder/'profile'),str(BIN),str(t),str(h)]
    started=when();began=time.monotonic();peak=0;stop=None;budget=min(remaining,reg['budget_per_capture_seconds'])
    with log_path.open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=folder,start_new_session=True,
                               env=dict(os.environ,OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0'))
        while child.poll() is None:
            total=0
            for p in folder.rglob('*'):
                try:
                    if p.is_file():total+=p.stat().st_size
                except FileNotFoundError:pass
            peak=max(peak,total)
            if peak>reg['budget_per_capture_bytes']:stop='disk budget'
            elif time.monotonic()-began>budget:stop='wall budget'
            if stop:
                os.killpg(child.pid,signal.SIGKILL);child.wait();break
            time.sleep(2)
    cost=dict(exit_code=child.returncode,stop=stop,seconds=time.monotonic()-began,peak_bytes=peak,
              began=started,finished=when(),command=command)
    save(path,cost)
    return cost

def run():
    import qualify
    reg=read(REG)
    for name,h in reg['source_hashes'].items():
        if probe.digest(ROOT/name)!=h:raise ValueError('frozen asset differs: '+name)
    with (RAW/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        total=0;completed=[]
        for case in reg['cases']:
            verdict=OUT/(case['id']+'.json')
            if verdict.exists():
                d=read(verdict);total+=d['cost'].get('seconds') or 0;completed.append(case['id']);continue
            if total>=reg['budget_total_capture_seconds']:break
            folder=RAW/case['id'];folder.mkdir(exist_ok=True)
            save(OUT/'status.json',dict(state='running',id=case['id'],utc=when(),completed=completed,registration_sha256=probe.digest(REG)))
            cost=capture(reg,case,folder,reg['budget_total_capture_seconds']-total)
            try:result=qualify.verify(reg,case,folder,cost)
            except (ValueError,KeyError,OSError,AssertionError,IndexError,TypeError) as exc:
                result=dict(qualified=False,reason='incomplete verification: '+repr(exc))
            save(verdict,dict(case=case,cost=cost,registration_sha256=probe.digest(REG),**result))
            total+=cost.get('seconds') or 0;completed.append(case['id'])
        save(OUT/'status.json',dict(state='finished' if len(completed)==len(reg['cases']) else 'total CA budget',utc=when(),completed=completed,total_ca_seconds=total,registration_sha256=probe.digest(REG)))

if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
