"""Only replace cache25's admission policy. Keep its runner, file lock, budgets, and scorer."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/v0.26/cache-queue'
CACHE=ROOT/'results/v0.25/cache'
ORIGINAL=ROOT/'development/validation/cache25'
POLICY=OUT/'admission.json'
sys.path[:0]=[str(ORIGINAL),str(ROOT)]
from development.validation.cache25 import campaign

ALL_CPUS=sorted(os.sched_getaffinity(0))
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def write(p,d):
    temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');temp.replace(p)

def init():
    if POLICY.exists():raise FileExistsError(POLICY)
    old=read(campaign.REG);costs=[]
    for f in sorted(CACHE.glob('*traffic.json')):
        r=read(f);raw=ROOT/r['raw_trace'];cost=next((d/'cost.json' for d in raw.parents if (d/'cost.json').is_file()),None)
        if not cost:continue
        o=read(cost);costs.append(dict(trace=str(f.relative_to(ROOT)),seconds=o['seconds'],cpu_seconds=o['cpu_seconds'],raw_bytes=o['raw_bytes']))
    cfg=dict(utc=utc(),registration_sha256=campaign.digest(campaign.REG),jobs=[j['id'] for j in old['jobs']],clock_scope='host scheduler only',
             continuous_seconds=180,poll_seconds=15,load1_fraction=.63,load5_fraction=.7,min_idle_cpu_fraction=.35,min_idle_cpus=64,
             eligible_cpu_cap=112,minimum_eligible_cpus=104,minimum_average_idle_in_mask=.64,priority_nice=19,policy='SCHED_IDLE',
             free_disk_gib=128,available_ram_gib=128,dirty_writeback_gib=16,
             priority='Earlier Pro pending admission; earlier FA awaiting capacity; newly admitted Flash when Kimi ends; a scanner error or unknown active queue state defers admission.',
             historical_costs=costs,old_narrow_mask_evidence='results/v0.23/full-hidden/v3/registration.json',
             src={str(p.relative_to(ROOT)):campaign.digest(p) for p in (Path(__file__).resolve(),HERE/'handover.py')})
    write(POLICY,cfg)

def verify():
    reg=read(campaign.REG);cfg=read(POLICY)
    if campaign.digest(campaign.REG)!=cfg['registration_sha256']:raise ValueError('cache25 registration changed')
    for n,h in list(cfg['src'].items())+list(reg['sha256'].items()):
        if campaign.digest(ROOT/n)!=h:raise ValueError('frozen file changed '+n)
    if [j['id'] for j in reg['jobs']]!=cfg['jobs']:raise ValueError('queue order')
    return cfg

def ticks(ids):
    out={}
    for line in Path('/proc/stat').read_text().splitlines():
        key,*v=line.split()
        if not key.startswith('cpu') or not key[3:].isdigit():continue
        cpu=int(key[3:])
        if cpu not in ids:continue
        t=list(map(int,v[:8]));out[cpu]=(sum(t),t[3])
    return out

def fractions(a,b):
    return {c:max(0.,min(1.,(b[c][1]-v[1])/(b[c][0]-v[0]))) for c,v in a.items() if c in b and b[c][0]>v[0]}

def inspect(cfg,prev,now):
    idle=fractions(prev,now)
    mask=sorted(idle,key=lambda idx:(-idle[idx],idx))[:cfg['eligible_cpu_cap']]
    mask_idle=sum(idle[x] for x in mask)/len(mask) if mask else 0.
    n=len(ALL_CPUS);l1,l5,_=os.getloadavg()
    memory={r.split(':')[0]:int(r.split()[1])/2**20 for r in Path('/proc/meminfo').read_text().splitlines() if r.startswith(('MemAvailable:','Dirty:','Writeback:'))}
    pro=read(ROOT/'results/v0.23/pro-prefill/state.json')
    fa=read(ROOT/'results/v0.25/coverage/supplement/state.json')
    full=read(ROOT/'results/v0.23/full-hidden/v3/queue-status.json')
    kimi_alive=[i for i in full.get('kimi_live_pids',[]) if Path(f'/proc/{i}').is_dir()]
    reasons=[]
    if pro['state'] not in ('acquiring','scoring existing full trace','finished'):reasons.append('earlier Pro queued or state unknown')
    if fa['state'] not in ('acquiring','qualifying','finished','interrupted raw attempt; preserved without duplicate'):reasons.append('earlier FA queued or state unknown')
    if full['state']=='waiting-for-Kimi':
        if not kimi_alive:reasons.append('handoff to higher priority Flash due')
    elif full['state'] not in ('complete','flash-native-active'):
        reasons.append('Flash state pending/unknown')
    if l1>n*cfg['load1_fraction'] or l5>n*cfg['load5_fraction']:reasons.append('load average')
    if len(mask)<cfg['minimum_eligible_cpus']:reasons.append('too few available physical CPUs for vendor native workers')
    if sum(idle.values())<max(cfg['min_idle_cpus'],cfg['min_idle_cpu_fraction']*n) or mask_idle<cfg['minimum_average_idle_in_mask']:reasons.append('insufficient recent actual idle CPU')
    free=shutil.disk_usage(ROOT).free/2**30
    if free<cfg['free_disk_gib'] or memory['MemAvailable']<cfg['available_ram_gib'] or memory['Dirty']+memory['Writeback']>cfg['dirty_writeback_gib']:
        reasons.append('RAM/disk/writeback reserve')
    info=dict(pro=pro['state'],pro_case=pro.get('case'),fa=fa['state'],fa_case=fa.get('id'),full_hidden=full['state'],
              kimi_live=bool(kimi_alive),load_1m=l1,load_5m=l5,affinity_available=n,idle_core_equivalents=round(sum(idle.values()),1),
              selected_core_average_idle=round(mask_idle,3),free_disk_gib=round(free,1),available_memory_gib=round(memory['MemAvailable'],1),
              dirty_writeback_gib=round(memory['Dirty']+memory['Writeback'],3),reasons=reasons)
    return mask,info

def run():
    cfg=verify();orig=campaign.wait_prior;poll=cfg['poll_seconds']
    def gate(job):
        # The previous job's process mask can be narrower; CPU counts and samples
        # always refer to the original host affinity. No other process changes.
        os.sched_setaffinity(0,ALL_CPUS)
        first=None;prev=ticks(ALL_CPUS)
        while True:
            time.sleep(poll);cur=ticks(ALL_CPUS)
            try:mask,info=inspect(cfg,prev,cur)
            except (OSError,KeyError,ValueError) as e:
                mask=[];info=dict(reasons=['unreadable higher-priority state'],detail=repr(e))
            prev=cur;now=time.monotonic();ok=not info['reasons']
            if ok and first is None:first=now
            if not ok:first=None
            elapsed=now-first if first is not None else 0.
            state=dict(utc=utc(),case=job['id'],continuous_good_seconds=round(elapsed,1),required_seconds=cfg['continuous_seconds'],**info)
            if elapsed<cfg['continuous_seconds']:
                write(OUT/'state.json',dict(state='waiting for stable headroom',**state));continue
            os.sched_setaffinity(0,mask)
            os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0))
            os.setpriority(os.PRIO_PROCESS,0,19)
            if os.sched_getscheduler(0)!=os.SCHED_IDLE or set(os.sched_getaffinity(0))!=set(mask):raise RuntimeError('Cannot enforce CPU scheduling policy')
            entry=dict(state='admitted under inherited low CPU priority',cpu_affinity=sorted(mask),nice=os.getpriority(os.PRIO_PROCESS,0),
                       scheduler='SCHED_IDLE',freeze_sha256=cfg['registration_sha256'],**state)
            write(OUT/(job['id']+'-admission.json'),entry);write(OUT/'state.json',entry)
            return
    campaign.wait_prior=gate
    try:campaign.run()
    finally:campaign.wait_prior=orig
    write(OUT/'state.json',dict(utc=utc(),state='original collector returned',cache25=read(CACHE/'state.json')))

if __name__=='__main__':
    cmd=argparse.ArgumentParser();cmd.add_argument('action',choices=('init','verify','run'));act=cmd.parse_args().action
    if act=='init':init()
    elif act=='verify':verify();print('original and addendum hashes OK',flush=True)
    else:run()
