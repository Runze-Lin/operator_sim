"""One official 950 pair, same B4 slot. Optionally retain controller files as hard links."""
from collections import Counter
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import sys
from threading import Event, Thread
import time

from study import ROOT,OUT,OLD,FROZEN,DT,PR,RUNTIME,now,read,store,digest
from development.validation.sku19.collect import capture
from development.validation.rms_ordered_verify.verify import checked,events,tables,text_bytes

RAW=ROOT/'runs/quant29'
STATE=OUT/'state.json'
STARTED='mixed-service large pair'


def preserve(source,dest,stop):
    dest.mkdir(parents=True,exist_ok=True)
    while True:
        for p in (source/'log').glob('lpddr_state*.log'):
            try:os.link(p,dest/p.name)
            except (FileExistsError,FileNotFoundError):pass
        if stop.wait(4):return


def measured(j,sku,folder,cost):
    shape=j['shape'];t,w=shape
    checked(cost['exit_code']==0 and not cost['stop_reason'],'process exit or budget')
    txt=(folder/'run.log').read_text(errors='replace')
    checked(re.findall(r'^runtime_soc=(\S+)',txt,re.M)==[sku] and ('lpddr5_6400M_bailu.ini' if sku==PR else 'hbm3_8000M.ini') in txt,'runtime SKU and memory controller')
    checked(all(s in txt for s in ('SMOKE_PASS','All task success.','Profiling data parse finished.')) and not re.search(r'SMOKE_FAIL|\[ERROR\].*(?:deadlock|invalid instruction|execute_set_flag|execute_wait_flag)',txt,re.I),'runtime/parse')
    checked(re.findall(r'Start profiling on kernel: (\S+)',txt)==[j['kernel']] and f'case=dynamic_mx_quant rows={t} cols={w} input=BF16 output=E4M3FN block=32' in txt,'kernel and shape')
    checked(re.findall(r'^correctness count=(\d+) bad=(\d+)',txt,re.M)==[(str(v),'0') for v in j['output_counts']],'every output and scale byte')
    tr,=folder.glob('profile/OPPROF_*/simulator/trace.json')
    binary=tr.parents[1]/'dump/aicore_binary.o'
    checked(text_bytes(binary)==text_bytes(ROOT/j['vendor']),'released ELF text')
    tab,=tables(txt);wins={};ends=Counter();labels={};out=Counter();pipes=Counter()
    for e in events(tr):
        core=e['pid'];name=e['name'];first=round(1650*e['ts']);last=round(1650*(e['ts']+e['dur']))
        bounds=wins.setdefault(core,[first,last]);bounds[0]=min(bounds[0],first);bounds[1]=max(bounds[1],last)
        ends[core]+=name=='END';pipes[e['tid']]+=1
        if name=='END_LABEL':
            checked(core not in labels,'duplicate end label');labels[core]=last
        if name.startswith('MOV_SRC_TO_DST') and '"queue_type":"MTE3"' in e['args']['detail']:
            detail=json.loads(e['args']['detail'])
            if not(detail['is_sync'] or detail['is_wait'] or detail['is_duration_excluded']):out[core]+=detail['processed_bytes']
    checked(set(wins)==set(tab)==set(out) and len(wins)==j['active_aiv'],'roster')
    for c,(a,b) in wins.items():
        checked(re.fullmatch(r'core\d+\.veccore\d+',c) and ends[c]==1 and labels.get(c,b)==b and abs((b-a)/1650-float(tab[c][0]))<.025,'incomplete physical core '+c)
    checked(sum(out.values())==sum(j['output_counts']),'complete aggregate MTE3 byte count')
    a=min(v[0] for v in wins.values());b=max(v[1] for v in wins.values())
    return dict(qualified=True,instruction_us=(b-a)/1650,wrapper_us=None,ticks_per_us=1650,core_ticks=wins,
                active_aiv=len(wins),retired_instructions=dict(pipes),written_bytes=sum(out.values()),
                checked_elements=j['output_counts'],trace=str(tr.relative_to(ROOT)),trace_sha256=digest(tr),
                released_text_sha256=sha256(text_bytes(binary)).hexdigest(),kernel=j['kernel'])


def status(s,**fields):
    if STATE.exists():
        old=read(STATE)
        if old['state']==s and all(old.get(k)==v for k,v in fields.items()):return
    store(STATE,dict(state=s,utc=now(),freeze_sha256=digest(FROZEN),**fields))


def collect():
    reg=read(FROZEN)
    for f,h in reg['hashes'].items():checked(digest(ROOT/f)==h,'frozen source '+f)
    RAW.mkdir(exist_ok=True)
    with (RAW/'lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (OLD/'native.json').exists() and read(OLD/'state.json')['phase']!='stopped':
            status('queued after existing B4 native',existing=read(OLD/'state.json')['phase']);time.sleep(45)
        from report import publish
        publish()
        with (ROOT/'runs/memory27-followup/remaining.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for sku in reg['official']['order']:
                receipt=OUT/(sku+'.json')
                if receipt.exists():continue
                d=RAW/sku
                spec=dict(**reg['official'],active_aiv=reg['predictions']['pair'][sku]['active_aiv'])
                if not (d/'cost.json').exists():
                    if d.exists():status('interrupted attempt preserved',sku=sku);return
                    while True:
                        # Reuse the established host headroom gate without touching the B4 process.
                        from development.validation.kda27.quant_transfer import headroom
                        info=headroom()
                        if info['eligible'] and shutil.disk_usage(ROOT).free>140*2**30:break
                        status('waiting for 80 idle physical cores and storage',sku=sku);time.sleep(45)
                    status('collecting',sku=sku)
                    stop=Event();arch=Thread(target=preserve,args=(d,RAW/f'{sku}-controllers',stop),daemon=True);arch.start()
                    affinity=os.sched_getaffinity(0)
                    try:
                        os.sched_setaffinity(0,info['chosen_cpus'])
                        t,w=spec['shape'];job=dict(runner=str(RUNTIME.relative_to(ROOT)),args=['quant',str(t),str(w)],**spec['budget'])
                        capture(job,sku,d)
                    finally:os.sched_setaffinity(0,affinity);stop.set();arch.join()
                cost=read(d/'cost.json');status('verifying',sku=sku)
                result=dict(sku=sku,shape=spec['shape'],qualified=False,instruction_us=None,wrapper_us=None,cost=cost,
                    freeze_sha256=digest(FROZEN),controller_raw=str((RAW/f'{sku}-controllers').relative_to(ROOT)))
                try:
                    checked(reg['utc']<cost['started_utc'],'freeze timing');result.update(measured(spec,sku,d,cost))
                except (AssertionError,OSError,ValueError,TypeError,KeyError,IndexError) as exc:result['reason']=repr(exc)
                store(receipt,result,exclusive=True);publish()
            status('finished');publish()


if __name__=='__main__':collect()
