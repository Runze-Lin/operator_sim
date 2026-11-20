"""Hand the idle registered worker lock to a smaller physical CPU allocation."""
from pathlib import Path
import os
import re
import signal
import stat
import sys
import time
import campaign as base

DEST=base.D/'host-adjustment.json'


def plan():
    rows=[]
    for m in (1,11,99):
        for sku in base.MEASURED:
            label=base.slug(m,sku);folder=base.ca.RAW/label
            cost=base.read(folder/'cost.json');file=folder/'cpu.txt'
            user,system=map(float,file.read_text().split());cpu=user+system
            rows.append(dict(m=m,sku=sku,wall_s=cost['seconds'],cpu_s=cpu,
                             mean_runnable_cpu_equivalents=cpu/cost['seconds'],
                             ideal_s_at_8_cpu=cpu/8,ideal_s_at_16_cpu=cpu/16,
                             cpu_counter=str(file.relative_to(base.ROOT)),sha256=base.digest(file)))
    contract=dict(utc=base.now(),registration_sha256=base.digest(base.REG),source_sha256=base.digest(Path(__file__)),
                  measurements=rows,selection='8–16 physical cores, three 1s samples each <=40% busy; keep >=8 measured quiet physical cores outside mask. Prefer CPUs outside active large Matmul masks, nice +12. CPU seconds / cores describes ideal work conservation only; original per-case time and storage limits remain in force.',
                  applies_to='Host placement only. All dates, prices, host/profiler, raw paths, exclusive lifecycle lock, 1200s / 1GiB caps and qualification from the original registration.')
    if DEST.exists():
        old=base.read(DEST)
        assert old['source_sha256']==contract['source_sha256'] and old['registration_sha256']==contract['registration_sha256']
    else:base.save(DEST,contract)


def spare(allowed):
    # Select directly from the three CPU snapshots; no wait on unrelated queue lifecycle.
    cpus=sorted(allowed);budget={c:0. for c in cpus};before=base.cpu_stats()
    for _ in range(3):
        time.sleep(1);after=base.cpu_stats()
        for c in cpus:
            ticks=after[c][0]-before[c][0]
            budget[c]=max(budget[c],1-(after[c][1]-before[c][1])/max(1,ticks))
        before=after
    large=set()
    for name in ('prefill-status.json','long-prefill/state.json','wide-prefill/state.json'):
        p=base.BASE/name
        if p.exists():
            item=base.read(p)
            if item.get('phase')=='acquiring':
                try:large.update(os.sched_getaffinity(item['pid']))
                except (KeyError,ProcessLookupError):pass
    idle=[];domains=set()
    for c in sorted(cpus,key=lambda x:(x in large,budget[x],x)):
        d=(Path('/sys/devices/system/cpu')/f'cpu{c}/topology/thread_siblings_list').read_text().strip()
        if budget[c]<=.4 and d not in domains:idle.append(c);domains.add(d)
    selected=idle[:max(0,min(16,len(idle)-8))]
    mem=int(re.search(r'^MemAvailable:\s*(\d+)',Path('/proc/meminfo').read_text(),re.M)[1])/2**20
    space=base.shutil.disk_usage(base.D).free/2**30
    return selected,dict(eligible=len(selected)>=8 and mem>=48 and space>=32,quiet_physical=len(idle),
                         reserved_quiet_physical=len(idle)-len(selected),chosen_cpus=selected,
                         max_busy_per_selected=[round(budget[c],3)for c in selected],policy_sha256=base.digest(DEST),
                         available_ram_gib=mem,available_disk_gib=space,load_1m=os.getloadavg()[0])


def idle_handoff():
    st=base.read(base.D/'state.json')
    if st.get('phase')=='finished':return False
    pid=st['pid'];proc=Path('/proc')/str(pid)
    if not proc.exists():return True
    old=base.ROOT/'development/validation/matmul29/decode/campaign.py'
    cmd=(proc/'cmdline').read_bytes().split(b'\x00')
    if proc.stat().st_uid!=os.getuid() or proc.joinpath('cwd').resolve()!=base.ROOT or not any((base.ROOT/os.fsdecode(s)).resolve()==old for s in cmd if s):
        raise RuntimeError('not original supervisor PID')
    if st.get('phase')!='waiting for physical CPU, RAM or storage':
        return None
    with open(proc/'stat') as f:start_id=f.read().rsplit(')',1)[1].split()[19]
    if (base.RAW/st['case']).exists():return None
    # pidfd identifies the same idle worker even if a PID is later recycled.
    handle=os.pidfd_open(pid);stopped=False
    try:
        signal.pidfd_send_signal(handle,signal.SIGSTOP);stopped=True
        limit=time.monotonic()+2
        while '(stopped)' not in (proc/'status').read_text():
            if time.monotonic()>limit:raise TimeoutError('idle supervisor suspension')
            time.sleep(.015)
        after=base.read(base.D/'state.json')
        kids=(proc/'task'/str(pid)/'children').read_text().strip()
        if after['phase']!=st['phase'] or kids or (base.RAW/st['case']).exists():return None
        base.save(base.D/'host-handoff.json',dict(utc=base.now(),previous_pid=pid,previous_process_start_ticks=start_id,
                  previous_state=after,raw_case_directory_exists=False,live_children=kids,registration_sha256=base.digest(base.REG),
                  new_pid=os.getpid(),action='Retire stopped idle original supervisor; next worker must obtain the same flock. No CA was active.'))
        signal.pidfd_send_signal(handle,signal.SIGTERM)
    finally:
        if stopped:signal.pidfd_send_signal(handle,signal.SIGCONT)
        os.close(handle)
    for _ in range(100):
        if not proc.exists():return True
        time.sleep(.1)
    raise RuntimeError('original supervisor did not exit')


if __name__=='__main__':
    plan()
    ok=idle_handoff()
    while ok is None:
        time.sleep(15);ok=idle_handoff()
    if ok:
        base.host=spare
        base.run()
