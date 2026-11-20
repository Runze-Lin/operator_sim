"""One optional host affinity change. Never launch, pause or modify device work."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/v0.27/kda';RAW=OUT/'raw';BIN=RAW/'packed-v1'
PRO=ROOT/'results/v0.23/pro-prefill';PR=PRO/'raw/pro-b8-256ki-pr';LOG=OUT/'host-scheduling.json'

def iso():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_bytes())
def put(d):
    tmp=LOG.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(LOG)
def stat(p):
    v=p.read_text().split(') ')[-1].split();return int(v[19]),int(v[11])+int(v[12]),int(v[36])
def ids(pid):return [int(p.name) for p in (Path('/proc')/str(pid)/'task').iterdir()]
def ticks():
    result={}
    for line in Path('/proc/stat').read_text().splitlines():
        k,*v=line.split()
        if k.startswith('cpu') and k[3:].isdigit():
            t=[int(x) for x in v[:8]];result[int(k[3:])]=(sum(t),t[3])
    return result
def sibling(cpu):
    return (Path('/sys/devices/system/cpu')/f'cpu{cpu}/topology/thread_siblings_list').read_text().strip()
def pro_usage():
    result={}
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if not (proc/'cwd').resolve().is_relative_to(PR):continue
            for t in (proc/'task').iterdir():
                try:
                    _,cpu_time,cpu=stat(t/'stat');result[int(t.name)]=(cpu_time,cpu)
                except (OSError,IndexError):continue
        except (OSError,RuntimeError,ValueError): continue
    return result

def set_threads(pid, mask):
    changed=[]
    for tid in ids(pid):
        try:os.sched_setaffinity(tid,mask);changed.append(tid)
        except ProcessLookupError:continue
    return len(changed)

def main(pid):
    start=stat(Path(f'/proc/{pid}/stat'))[0];digest=hashlib.sha256(BIN.read_bytes()).hexdigest();base=os.sched_getaffinity(pid)
    assert Path(f'/proc/{pid}/exe').resolve()==BIN and read(OUT/'v1-register.json')['hashes'][str(BIN.relative_to(ROOT))]==digest and len(base)==8
    assert not LOG.exists(), 'do not repeat an enacted controller'
    report=dict(registered_utc=iso(),pid=pid,proc_start_ticks=start,elf_sha256=digest,original_cpus=sorted(base),maximum_cpus=24,extra_cpus=16,unchanged='same live host process and command, target ELF, input/output, protocol, 60 minute wall deadline and 8 GiB cap',admission='Only after Pro 256ki PR starts and active workers are seen, or its receipt is recorded. Four independent 2s CPU samples all <=35% busy. Exclude physical siblings of running Pro threads. Keep at least 48 idle CPU-equivalents after expansion; revert on observed Pro collision or excess load.',state='waiting',expanded=None)
    put(report);extra=set();allowed=os.sched_getaffinity(0)
    while Path(f'/proc/{pid}/stat').exists():
        if Path(f'/proc/{pid}/exe').resolve()!=BIN or stat(Path(f'/proc/{pid}/stat'))[0]!=start:break
        state=read(PRO/'state.json');done=(PRO/'pro-b8-256ki-pr.json').exists()
        admitted=done or (state['state']=='acquiring' and state.get('case')=='pro-b8-256ki-pr')
        report['pro_state']=state['state'];report['last_checked_utc']=iso()
        if not admitted:
            put(report);time.sleep(15);continue
        rates={c:[] for c in allowed};protected=set();before=ticks();jobs=pro_usage()
        for _ in range(4):
            time.sleep(2);after=ticks();current=pro_usage()
            for c in allowed:
                total=after[c][0]-before[c][0];idle=after[c][1]-before[c][1];rates[c].append(1-idle/total if total else 1.)
            for tid,(activity,c) in current.items():
                if tid in jobs and activity>jobs[tid][0]:protected.update((c,jobs[tid][1]))
            jobs=current;before=after
        live=Path(f'/proc/{pid}/exe');
        if not live.exists() or live.resolve()!=BIN or stat(Path(f'/proc/{pid}/stat'))[0]!=start:break
        occupied={sibling(c) for c in protected};load=os.getloadavg()[0];idle=sum(1-sum(x)/len(x) for x in rates.values());report['load_1m']=load;report['idle_cpu_equivalents']=round(idle,1)
        if extra:
            if any(sibling(c) in occupied for c in extra) or idle<48:
                affected=set_threads(pid,base);report.update(state='returned to original 8 CPUs to preserve Pro headroom',restored_utc=iso(),threads_restored=affected);put(report);return
            put(report);time.sleep(15);continue
        if (not done and not protected) or idle<64 or len(allowed)-load<64:
            put(report);time.sleep(15);continue
        excluded=occupied|{sibling(c) for c in base};selected=[]
        for c in sorted(allowed-base,key=lambda c:(max(rates[c]),sum(rates[c]),c)):
            topo=sibling(c)
            if max(rates[c])<=.35 and topo not in excluded:
                selected.append(c);excluded.add(topo)
            if len(selected)==16: break
        if len(selected)!=16:
            put(report);time.sleep(15);continue
        extra=set(selected);threads=set_threads(pid,base|extra)
        report.update(state='same process on at most 24 CPUs',expanded=dict(utc=iso(),pro_completed=done,running_pro_logical_cpus=sorted(protected),new_cpu_busy_fractions={c:rates[c] for c in selected},new_cpu_siblings={c:sibling(c) for c in selected},threads_changed=threads,actual_worker_cpus=sorted(os.sched_getaffinity(pid)),registered_pid_unchanged=stat(Path(f'/proc/{pid}/stat'))[0]==start));put(report)
        time.sleep(15)
    report.update(state='worker exited',last_checked_utc=iso());put(report)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('pid',type=int);a=p.parse_args();main(a.pid)
