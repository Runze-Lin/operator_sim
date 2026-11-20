"""Two fresh channel geometries. Existing 0.28 and 0.29 workers have priority."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from study import ROOT,OUT,OLD,DT,PR,FUT,read,save,now,h
from operator_model import OperatorStream,estimate_post_history
from development.validation.post24.collect import capture,qualify,BIN,LIB
from development.validation.kda27.quant_transfer import headroom

REG=OUT/'channel-register.json'
RAW=ROOT/'runs/post30-channel'
GEOMETRIES=([13,4,3520],[19,4,2720])


def register():
    if REG.exists() or RAW.exists():raise FileExistsError(REG)
    prior_files=0;hashes=[]
    patterns=[re.compile(rb'\[\s*'+rb'\s*,\s*'.join(str(v).encode() for v in geometry)+rb'\s*\]')for geometry in GEOMETRIES]
    for base,dirs,files in os.walk(ROOT/'results'):
        dirs[:]=[d for d in dirs if d not in ('raw','profile','__pycache__','.git')]
        for name in files:
            if not name.endswith('.json')or name=='trace.json':continue
            p=Path(base)/name;data=p.read_bytes();prior_files+=1
            if any(v.search(data)for v in patterns):raise ValueError('prior geometry '+str(p))
    wanted=set(map(tuple,GEOMETRIES))
    for log in (ROOT/'runs').glob('post*/**/batch.txt'):
        for line in log.read_text().splitlines():
            if line.startswith('mhc_post ') and tuple(map(int,line.split()[2:])) in wanted:raise ValueError(str(log))
        hashes.append(h(log))
    scenarios={}
    for c in (DT,PR,FUT):
        stream=OperatorStream(c,track_post_history=True);resident=0;previous=None;rr=[]
        for i,shape in enumerate([GEOMETRIES[0],GEOMETRIES[0],GEOMETRIES[1],GEOMETRIES[1]]):
            old=stream.advance('mhc_post',shape);s=estimate_post_history(shape,chip=c);state=s.for_predecessor(resident,previous)
            if s.route!='post_channel':raise ValueError('not a channel geometry on target')
            rr.append(dict(i=i,shape=shape,previous_shape=previous,active=s.active_aiv,prior_aiv=resident,
                 role='exact-repeat' if previous==shape else 'ordered-prefix',candidate_us=state.latency_us,
                 public_ordered_us=old.latency_us,route=s.route,stages_us=dict(state.stages_us)))
            previous=shape;resident=max(resident,s.active_aiv)
        scenarios[c]=rr
    files=[LIB,BIN,ROOT/'development/validation/post24/collect.py',ROOT/'operator_model/post_history.py',ROOT/'results/v0.28/post/parameters.json',
           Path(__file__).resolve()]
    save(REG,dict(utc=now(),cohort='Supplement to v0.24 frozen channel operand prices. v0.28 failure and original gate are left intact.',
        uniqueness=dict(prior_result_files=prior_files,post_sequence_logs=len(hashes),hits=[]),holdout=GEOMETRIES,
        wait='After existing v0.28 channel endpoints and v0.29 split sequence have closed; share runs/large29/shared.lock per endpoint.',
        qualification='Each successful pair covers both full shape/contract matches on one stream, exact SKU, installed ELF and all elements/END/output bytes. Compare both new repeat shapes on each SKU to prior ordered occupancy. Separately assess >=10 distinct original-forecast+prospective geometry group per SKU WAPE<=25%, max<=50% and lower WAPE than baseline. Record failed old and new endpoints. No duplicate launches or replacement of missing v0.28 targets.',
        budget=dict(wall_s=3600,mib=4096),scenarios=scenarios,
        collection_hashes={str(f.relative_to(ROOT)):h(f)for f in files}),True)


def status(msg,**extra):
    save(OUT/'channel-state.json',dict(phase=msg,utc=now(),pid=os.getpid(),**extra))


def run():
    reg=read(REG);RAW.mkdir(exist_ok=True);os.nice(12)
    for f,sha in reg['collection_hashes'].items():
        if h(ROOT/f)!=sha:raise ValueError('changed frozen source '+f)
    with (RAW/'serial.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (OLD/'split-score.json').exists() or not read(OLD/'split-score.json').get('complete'):
            # A terminal failure need not block later cohorts forever.
            st=read(OLD/'split-state.json')
            if st.get('phase') in ('complete','finished','completed'):break
            status('waiting behind existing v0.28/29 Post acquisitions',predecessor=st['phase']);time.sleep(60)
        allowed=os.sched_getaffinity(0)
        for sku in (PR,DT):
            dest=OUT/('channel-'+sku+'.json')
            if dest.exists():continue
            d=RAW/sku
            if not (d/'cost.json').exists():
                if d.exists():status('old partial preserved',chip=sku);return
                while True:
                    m=headroom()
                    if m['eligible']:break
                    status('waiting for host headroom',chip=sku,idle_physical=m['idle_physical']);time.sleep(45)
                status('waiting for shared CA slot',chip=sku)
                with (ROOT/'runs/large29/shared.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    # Do not use stale CPU choices after a long lock wait.
                    m=headroom()
                    while not m['eligible']:
                        fcntl.flock(lock,fcntl.LOCK_UN);status('waiting for host headroom',chip=sku,idle_physical=m['idle_physical'])
                        time.sleep(45);fcntl.flock(lock,fcntl.LOCK_EX);m=headroom()
                    os.sched_setaffinity(0,m['chosen_cpus'][:64]);status('collecting',chip=sku)
                    try:capture(sku,d,reg['scenarios'][sku],reg['budget'])
                    finally:os.sched_setaffinity(0,allowed)
            cost=read(d/'cost.json');row=dict(chip=sku,cost=cost,registration_sha256=h(REG),raw=str(d.relative_to(ROOT)))
            try:
                assert cost['started_utc']>reg['utc'] and cost['exit_code']==0 and cost['stop_reason'] is None
                data=qualify(sku,d,reg['scenarios'][sku])
                for r in data:
                    if r['timing_qualified'] and (not r['active_matches'] or not r['stage']['kernel'].endswith('_5')):
                        r.update(timing_qualified=False,reason='channel route/active mismatch')
                row['rows']=data
            except (ValueError,AssertionError,IndexError,OSError,KeyError) as exc:row['reason']=repr(exc)
            save(dest,row,True)
            from report import write_report
            write_report()
        status('finished')
        from report import write_report
        write_report()
if __name__=='__main__':{'freeze':register,'run':run}[sys.argv[1]]()
