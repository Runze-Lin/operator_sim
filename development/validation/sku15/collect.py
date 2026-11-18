"""Run each preregistered host once; persist costs and failed qualifications before replacements."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from campaign import ROOT, OUT, RUNS, PR, load, sha, utc, write
from qualify import qualify

def bytes_used(directory):
    n=0
    for parent,_,names in os.walk(directory):
        for filename in names:
            try: n+=os.stat(os.path.join(parent,filename)).st_size
            except FileNotFoundError: pass
    return n

def collect(c, started, budget, sizes, lock):
    binary=ROOT/c['runner'];case=c['id'];directory=RUNS/case
    if sha(binary)!=c['host_binary_sha256']:raise ValueError('host binary changed')
    directory.mkdir(exist_ok=False); tm=directory/'cpu.txt'
    cmd=['/usr/bin/time','-f','%U %S','-o',str(tm),'bash',str(ROOT/'scripts/cann-run.sh'),'msopprof','simulator','--soc-version='+PR,'--dump=true','--launch-count=10','--timeout='+str(max(2,(c['wall_limit_s']-10)//60)),'--output='+str(directory/'profile'),str(binary),c['op'],*map(str,c['cli_shape'])]
    wall=time.monotonic();when=utc(); stop=None
    with (directory/'run.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=directory,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            size=bytes_used(directory)
            with lock:
                sizes[case]=size; total=sum(sizes.values())
            if size>c['raw_limit_mib']*2**20 or total>budget['max_total_raw_mib']*2**20:stop='raw_cap'
            if time.monotonic()-wall>c['wall_limit_s'] or time.monotonic()-started>budget['max_total_seconds']:stop='wall_cap'
            if stop:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait();break
            time.sleep(2)
    seconds=time.monotonic()-wall;cpu=None
    if tm.is_file() and tm.stat().st_size:
        try: cpu=sum(map(float,tm.read_text().splitlines()[-1].split()))
        except (ValueError,IndexError): pass
    cost=dict(started_utc=when,finished_utc=utc(),seconds=seconds,cpu_seconds=cpu,raw_bytes=bytes_used(directory),exit_code=proc.returncode,stop_reason=stop,cmd=cmd[5:])
    with lock: sizes[case]=cost['raw_bytes']
    write(directory/'cost.json',cost,True)
    return cost

def run():
    budget=load(OUT/'budget.json');f=load(OUT/'freeze.json');pred={r['id']:r for r in f['predictions']}
    for p,digest in f['sha256'].items():
        if sha(ROOT/p)!=digest:raise ValueError('frozen input changed: '+p)
    started=time.monotonic();RUNS.mkdir(exist_ok=False); sizes={};lock=threading.Lock();attempts=[];rows=[];workers=budget['max_workers']
    cases=budget['cases'];ops=tuple(budget['contracts']);by={op:[r for r in cases if r['op']==op] for op in ops}
    primary=[by[op][i] for i in range(10) for op in ops];reserve={op:iter(by[op][10:]) for op in ops}
    def task(c):
        cost=collect(c,started,budget,sizes,lock);row=None;reason=None;ex=time.monotonic()
        try:
            if cost['exit_code']!=0 or cost['stop_reason']:raise ValueError('collector exit/budget: '+str(cost['exit_code'])+'/'+str(cost['stop_reason']))
            row=qualify(c,pred[c['id']],budget['contracts'][c['op']]);write(RUNS/c['id']/'qualified.json',row,True)
        except (ValueError,AssertionError,OSError,KeyError) as e: reason=type(e).__name__+': '+str(e)
        return row,dict(id=c['id'],op=c['op'],shape=c['shape'],priority=c['priority'],qualified=row is not None,reason=reason,qualification_seconds=time.monotonic()-ex,**cost)
    def batch(work):
        with ThreadPoolExecutor(workers) as pool:
            futures={pool.submit(task,c):c for c in work}
            for fu in as_completed(futures):
                r,a=fu.result();attempts.append(a)
                if r:rows.append(r)
                write(OUT/'attempts.json',sorted(attempts,key=lambda x:x['started_utc']));write(OUT/'holdout.json',sorted(rows,key=lambda x:(ops.index(x['op']),primary.index(next(c for c in primary if c['id']==x['id'])) if x['priority']=='primary' else 1000)))
                print(f"{a['id']}: {a['qualified']} {a['reason'] or ''}, CA wall={a['seconds']:.1f}s; qualified {len(rows)}/{len(primary)}",flush=True)
    batch(primary)
    while time.monotonic()-started<budget['max_total_seconds'] and sum(sizes.values())<budget['max_total_raw_mib']*2**20:
        more=[]
        for op in ops:
            need=10-sum(r['op']==op for r in rows)
            for i in range(need):
                item=next(reserve[op],None)
                if item:more.append(item)
        if not more:break
        batch(more)
    write(OUT/'campaign-cost.json',dict(wall_seconds=time.monotonic()-started,workers=workers,attempted=len(attempts),qualified=len(rows),sum_simulator_wall_seconds=sum(r['seconds'] for r in attempts),simulator_cpu_seconds=sum(r['cpu_seconds'] or 0 for r in attempts),unknown_cpu_rows=sum(r['cpu_seconds'] is None for r in attempts),raw_bytes=sum(r['raw_bytes'] for r in attempts),dt_new_runs=0))
    if len(rows)!=20 or budget['mandatory_shape'] not in [r['shape'] for r in rows]: raise SystemExit('PR coverage incomplete; see attempts.json')

if __name__=='__main__':run()
