"""Drain the first fusion21 collector's running children before restarting it at the registered cap."""
from pathlib import Path
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
RAW=ROOT/'runs/fusion21'
OUT=ROOT/'results/v0.21/fusion'
def cmd(p):return (p/'cmdline').read_bytes().rstrip(b'\x00').split(b'\x00')
def stamp(t):return datetime.fromtimestamp(t,timezone.utc).isoformat()
def stat(p):
    # comm may contain whitespace or parentheses.
    return (p/'stat').read_text().rsplit(')',1)[1].strip().split()

def handoff():
    reg=json.loads((OUT/'freeze.json').read_text())
    old=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:
            if cmd(p)[1:]==[b'development/validation/fusion21/campaign.py',b'run'] and (p/'cwd').resolve()==ROOT:
                old.append(p)
        except (OSError,PermissionError):continue
    if len(old)!=1:raise RuntimeError(f'Expected exactly one old collector, got {old}')
    pid=int(old[0].name)
    os.kill(pid,signal.SIGSTOP)
    try:
        while stat(old[0])[0]!='T':time.sleep(.02)
        children=[]
        for task in (old[0]/'task').iterdir():
            children.extend(int(c) for c in (task/'children').read_text().split())
        captures=[]
        for cid in children:
            p=Path('/proc')/str(cid); argv=[a.decode() for a in cmd(p)]
            folder=(p/'cwd').resolve()
            if not argv:
                # Already finished (e.g. GNU time exit), no duplicate CA should start.
                continue
            if argv[0]!='/usr/bin/time' or folder.parent!=RAW or os.getpgid(cid)!=cid:
                raise RuntimeError(f'Unexpected child outside fusion21 capture: {cid} {folder}')
            start=time.time()-float(Path('/proc/uptime').read_text().split()[0])+float(stat(p)[19])/os.sysconf('SC_CLK_TCK')
            captures.append((p,folder,argv,start))
        for p,folder,argv,start in captures:
            reason=None
            while stat(p)[0]!='Z':
                if time.time()-start>=reg['per_attempt_wall_s'] or sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())>=reg['raw_mib']*2**20:
                    os.killpg(int(p.name),signal.SIGKILL)
                    reason='original wall or raw cap during controller handoff'
                time.sleep(2)
            # GNU time exit state remains available until the stopped parent reaps it.
            rc=os.waitstatus_to_exitcode(int(stat(p)[49]))
            cpu=(folder/'cpu.txt').read_text().split()
            cpu_s=sum(map(float,cpu[-2:])) if len(cpu)>=2 and all(x.replace('.','').isdigit() for x in cpu[-2:]) else None
            end=(folder/'cpu.txt').stat().st_mtime if cpu else time.time()
            cost=dict(seconds=end-start,raw_bytes=sum(f.stat().st_size for f in folder.rglob('*') if f.is_file()),
                      cpu_seconds=cpu_s,exit_code=rc,stop_reason=reason,started_utc=stamp(start),finished_utc=stamp(end),
                      command=argv[5:],timing_source='GNU time PID boot ticks to GNU time receipt mtime during queue handoff')
            (folder/'cost.json').write_text(json.dumps(cost,indent=2)+'\n')
        (OUT/'handoff.json').write_text(json.dumps(dict(utc=stamp(time.time()),drained=[f.name for p,f,a,s in captures],
                 reason='The original AttnRes collector capped msopprof at 3 min, below the frozen outer budget. Subsequent jobs pass the original registered cap to both layers.'),indent=2)+'\n')
        os.kill(pid,signal.SIGTERM)
    finally:
        os.kill(pid,signal.SIGCONT)
    os.execv(sys.executable,[sys.executable,str(ROOT/'development/validation/fusion21/campaign.py'),'run'])
if __name__=='__main__':handoff()
