"""Record a non-invasive host-only efficiency observation on the original live runner."""
import os
import time
from pathlib import Path
from watch import OUT, ROOT, INPUT, read, save, now
from reduce import CASES

def idle_counters():
    fields=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
    return sum(fields[:8]),fields[3]+fields[4]

if __name__=='__main__':
    reg=read(OUT/'monitor-freeze.json')
    pid=reg['process_chain'][-1];proc=Path(f'/proc/{pid}')
    argv=proc.joinpath('cmdline').read_bytes().split(b'\0')
    assert argv[0].endswith(b'/matmul28') and list(map(int,argv[1:4]))==reg['shape']
    hz=os.sysconf('SC_CLK_TCK')
    stat=(proc/'stat').read_text().split(') ')[-1].split()
    age=float(Path('/proc/uptime').read_text().split()[0])-int(stat[19])/hz
    cpu=(int(stat[11])+int(stat[12]))/hz
    t0,i0=idle_counters();time.sleep(2);t1,i1=idle_counters()
    previous=[]
    for reference,directory in CASES:
        raw=ROOT/directory
        c=read(raw/'cost.json');user,kernel=map(float,(raw/'cpu.txt').read_text().split())
        previous.append(dict(reference=reference,wall_s=c['seconds'],process_tree_user_s=user,process_tree_sys_s=kernel,
                             avg_host_cpus=(user+kernel)/c['seconds']))
    output=dict(utc=now(),source_pid=pid,argv=[a.decode() for a in argv if a],allowed_logical_cpus=len(os.sched_getaffinity(pid)),
                live_threads=int(stat[17]),process_wall_s=age,process_user_sys_s=cpu,avg_host_cpus_process_only=cpu/age,
                two_second_host_idle_or_iowait_fraction=(i1-i0)/(t1-t0),sample_scope='/proc monotic process age and jiffies; archived /usr/bin/time includes process tree; not a device-duration proxy or controlled host-causality test.',
                prior=previous)
    save('host-cost-observation.json',output)
