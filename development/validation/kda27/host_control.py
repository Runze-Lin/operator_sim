"""Single post-timeout DT control: original CA contract, adequate host scheduling."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time

import packed_probe as probe
from affinity import sibling, ticks

ROOT, OUT = probe.ROOT, probe.OUT
DEST = OUT/'host-control'
RAW = OUT/'raw/host-control'
REG = DEST/'register.json'
DT = probe.old.DT
CAP = 1440
CPUS = 112


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slots():
    """Require four intervals of low-busy distinct physical cores; reserve >=96."""
    allowed = os.sched_getaffinity(0)
    before = ticks()
    obs = {cpu: [] for cpu in allowed}
    for _ in range(4):
        time.sleep(2)
        after = ticks()
        for cpu in allowed:
            n = after[cpu][0]-before[cpu][0]
            busy = 1-(after[cpu][1]-before[cpu][1])/n if n else 1
            obs[cpu].append(round(busy,4))
        before = after
    ordered = sorted(allowed, key=lambda c:(max(obs[c]), sum(obs[c]), c))
    selected = []; domains = set()
    for cpu in ordered:
        topo = sibling(cpu)
        if max(obs[cpu]) <= .20 and topo not in domains:
            selected.append(cpu); domains.add(topo)
        if len(selected) == CPUS: break
    idle = sum(1-sum(v)/len(v) for v in obs.values())
    load = os.getloadavg()
    eligible = len(selected)==CPUS and idle >= CPUS + 96 and len(allowed)-load[0] >= CPUS+96
    return dict(eligible=eligible, cpu_count=len(allowed), idle_equivalents=idle, load=load,
                cpu_samples={c:obs[c] for c in selected}, actual_selected_cpus=selected,
                physical_siblings={c:sibling(c) for c in selected}, utc=probe.now())


def register():
    if REG.exists(): raise FileExistsError(REG)
    parent = OUT/'v1-register.json';old=probe.read(parent)
    files=[parent, OUT/f'v1-{DT}.json', OUT/'host-scheduling.json', Path(__file__).resolve(),
           Path(probe.__file__), ROOT/'results/v0.20/full-quant/native.json',
           ROOT/'results/v0.23/full-hidden/v3/flash-retry.json',ROOT/'results/v0.23/pro-prefill/pro-b8-256ki-pr.json']
    for name,h in old['hashes'].items():
        assert sha(ROOT/name)==h,name
    prior=probe.read(files[1]); assert not prior['qualified'] and len(prior['stages'])==1 and prior['stages'][0]['kernel'].startswith('Transpose_')
    probed=slots()
    probe.save(REG,dict(utc=probe.now(),role='single DT post-timeout host scheduling control',original=parent.relative_to(ROOT).as_posix(),original_sha256=sha(parent),
       contrast='Identical branch, tensors, host executable, vendor files, standard msopprof command/flags, target and scoring; compare the prior mostly-8 logical CPU placement to 112 idle physical cores after all registered large cases completed. Different process and host conditions remain a confounder.',
       one_attempt=True,also_run_pr=False,target=DT,wall_cap_s=CAP,disk_cap_bytes=old['budget_bytes'],
       original_instrumentation_timeout_min=60,criteria=old['retention'],shape=old['shape'],cu=old['cu'],chunk_indices=old['chunk_indices'],
       readiness=dict(physical_cores=CPUS,max_busy_fraction=.20,interval_s=2,samples=4,spare_cpu_equivalents=96,minimum_free_disk_gib=16,minimum_memory_gib=40),
       initial_headroom=probed,hashes={str(f.relative_to(ROOT)):sha(f) for f in files}))


def native(folder):
    log=folder/'run.log';lines=log.read_text(errors='replace').splitlines();banner='before any reported launch';counts={}; faults=[];phases=0;seen={}
    for num,line in enumerate(lines,1):
        m=re.search(r'Start profiling on kernel: (\S+)',line)
        if m:
            name=m.group(1);ordinal=seen.get(name,0);seen[name]=ordinal+1
            banner=f'{name}#{ordinal}';counts[banner]=dict(error_count=0,first_fault_line=None,first_pc=None,launch_banner_line=num)
            phases+=name.startswith('ChunkKdaFwd_')
        if '[ERROR]' in line and ('set_flag' in line or 'wait_flag' in line or 'deadlock' in line.lower()):
            entry=counts.setdefault(banner,dict(error_count=0,first_fault_line=None,first_pc=None));entry['error_count']+=1
            if entry['first_fault_line'] is None:
                match=re.search(r'pc:(0x[\da-fA-F]+)',line);entry.update(first_fault_line=num,first_pc=match.group(1) if match else None)
                faults.append(dict(nearest_preceding_profiler_banner=banner,line=num,text=line.strip()))
    records={}
    for f in [folder/'stars_log0_0_task.dump',folder/'log_ca/stars_log0_0.dump',folder/'log_ca/stars_log0_1.dump']:
        if not f.exists(): continue
        start=[];end=[];block=[];kick=[]
        for line in f.read_text(errors='replace').splitlines():
            if '[TASK_BEGIN]' in line:start.append(line.strip())
            elif '[TASK_END]' in line:end.append(line.strip())
            elif '[block_start]' in line:block.append(line.strip())
            elif 'kickstartAivCore' in line or 'kickstartAicCore' in line:kick.append(line.strip())
        records[str(f.relative_to(folder))]=dict(sha256=sha(f),task_begin=start,task_end=end,blocks_started=len(block),first_block=block[:1],first_kick=kick[:1])
    return dict(log_sha256=sha(log),launch_banners=len(counts),target_v1_banners=phases,by_latest_banner=counts,first_faults=faults,stars=records,
                note='STARS task/core kick is an event in a different clock domain. Banners plus per-launch dumped ELF and official duration establish device identity. Any fault bars V1 timing.')


def run():
    RAW.mkdir(parents=True,exist_ok=True);reg=probe.read(REG);parent=probe.read(OUT/'v1-register.json');
    assert reg['target']==DT and not reg['also_run_pr']
    for name,h in {**reg['hashes'],**parent['hashes']}.items():
        if sha(ROOT/name)!=h:raise RuntimeError('changed frozen input: '+name)
    with (OUT/'raw/lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (RAW/'run.log').exists():raise FileExistsError('one attempt is already present')
        import shutil
        while True:
            info=slots();mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))/1024**2
            disk=shutil.disk_usage(RAW).free/1024**3
            if info['eligible'] and disk>16 and mem>40: break
            probe.save(DEST/'state.json',dict(state='waiting for idle physical cores',utc=probe.now(),headroom=info,mem_gib=mem,disk_gib=disk));time.sleep(30)
        os.sched_setaffinity(0,info['actual_selected_cpus']);os.nice(8)
        probe.save(DEST/'state.json',dict(state='collecting DT',utc=probe.now(),register_sha256=sha(REG),headroom=info,mem_gib=mem,disk_gib=disk))
        cost=probe.capture(DT,dict(budget_seconds=CAP,budget_bytes=reg['disk_cap_bytes']),RAW)
        evidence=native(RAW);probe.save(DEST/'native-evidence.json',evidence)
        result=probe.score(DT,parent,RAW,cost)
        probe.save(DEST/'dt.json',dict(register_sha256=sha(REG),**result))
        probe.save(DEST/'state.json',dict(utc=probe.now(),state='finished',qualified=result['qualified'],complete_v1_stages=len([s for s in result['stages'] if s['kernel'].startswith('ChunkKdaFwd_') and s.get('ca_us') is not None]),target_v1_banners=evidence['target_v1_banners'],sync_errors=result['sync_errors'],sample_role='DT post-timeout development; no PR sample',cost=cost,headroom=info))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('register','run','original'));a=p.parse_args()
    if a.action=='original':probe.save(DEST/'original-native-evidence.json',native(OUT/'raw'/DT))
    else:{'register':register,'run':run}[a.action]()
