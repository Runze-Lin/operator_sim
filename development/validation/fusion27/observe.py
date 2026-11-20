"""Sample only the descendants of the registered fused-CA supervisor and their scheduling counters."""
import json,os,re,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/v0.27/fusion'
ORIG=ROOT/'results/v0.26/presink';RAW=ORIG/'raw'
SUP=2701028
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def save(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def argv(p):return [os.fsdecode(x) for x in (p/'cmdline').read_bytes().split(b'\0') if x]
def children(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return []
    return sorted({int(x) for f in (p/'task').glob('*/children') for x in f.read_text().split()})
def tree(pid):
    result=[pid]
    for p in result:result.extend(c for c in children(p) if c not in result)
    return result
def snapshot(pid):
    info={}
    for item in tree(pid):
        p=Path('/proc')/str(item)
        try:
            command=argv(p);stats={}
            for t in (p/'task').iterdir():
                seq=(t/'schedstat').read_text().split()
                stats[t.name]=dict(runtime_s=int(seq[0])/1e9,runqueue_s=int(seq[1])/1e9,periods=int(seq[2]),state=(t/'stat').read_text().split(') ')[1][0])
            info[item]=dict(command=command,affinity=sorted(os.sched_getaffinity(item)),threads=stats,
                            nice=os.getpriority(os.PRIO_PROCESS,item),policy=os.sched_getscheduler(item))
        except (ProcessLookupError,FileNotFoundError):continue
    return info
def point(case):
    p=RAW/case/'run.log';s=p.read_text(errors='replace')
    samples=re.findall(r'Cycle Count:\s*(\d+)',s);finished=re.findall(r'sim time:\s*([\d.]+)s, cycle:\s*(\d+)',s)
    return dict(size=p.stat().st_size,mtime=p.stat().st_mtime,last_cycle_count=int(samples[-1]) if samples else None,finished_subunits=finished,smoke_pass='SMOKE_PASS' in s)
def measure(pid,case,seconds):
    before=utc();before_log=point(case);a=snapshot(pid);clock=time.monotonic();time.sleep(seconds);b=snapshot(pid);elapsed=time.monotonic()-clock
    results=[]
    for n,s in a.items():
        if n not in b:continue
        t=b[n];workers=[]
        for tid,old in s['threads'].items():
            if tid not in t['threads']:continue
            new=t['threads'][tid]
            workers.append(dict(tid=int(tid),state=new['state'],cpu_s=round(new['runtime_s']-old['runtime_s'],5),
                                runnable_wait_s=round(new['runqueue_s']-old['runqueue_s'],5)))
        results.append(dict(pid=n,argv=s['command'],nice=t['nice'],scheduler=t['policy'],affinity=t['affinity'],thread_count=len(workers),
                            effective_cpu=sum(x['cpu_s'] for x in workers)/elapsed,
                            effective_runnable_wait=sum(x['runnable_wait_s'] for x in workers)/elapsed,
                            thread_cpu_sum_s=sum(v['runtime_s'] for v in t['threads'].values()),tasks=workers))
    return dict(start=before,end=utc(),sample_seconds=elapsed,supervisor=pid,case=case,processes=results,
                load=os.getloadavg(),before_log=before_log,after_log=point(case))
if __name__=='__main__':
    p,case,secs,dest=sys.argv[1:];save(OUT/dest,measure(int(p),case,int(secs)))
