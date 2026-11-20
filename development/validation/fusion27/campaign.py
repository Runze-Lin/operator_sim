"""Score the unchanged v0.26 contracts and run only genuinely unstarted fused work, one slot."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from collections import defaultdict

from observe import ROOT,OUT,ORIG,RAW as PRIOR_RAW,read,save,utc
from admission import ALL,POLICY,ticks,capacity,set_limits

# The original guard module imports 'campaign' by its unqualified script name;
# this file is also named campaign.py. Load the original under its expected
# name before loading its verified endpoint addendum.
from importlib.util import module_from_spec,spec_from_file_location
V26=ROOT/'development/validation/presink26'
def module(name,p):
    spec=spec_from_file_location(name,p);mod=module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
orig=module('frozen_presink26',V26/'campaign.py')
sys.modules['campaign']=orig
ep=module('frozen_presink26_endpoint',V26/'resume.py')
from development.validation.fusion21.history import walk
REG=OUT/'register.json';RAW=OUT/'raw';HOLD=OUT/'receipts';ORIG_REG=ORIG/'freeze.json'
CHIPS=(orig.PR,orig.DT);FUT=orig.FUT
MAIN=(21,25,41,55,57,65,97,127);RESERVE=91

def novelty():
    hits=defaultdict(set);files=subprocess.check_output(['git','ls-files','results/**/*.json','configs/**/*.json'],cwd=ROOT,text=True).splitlines()
    for name in files:
        p=ROOT/name
        if not p.exists() or OUT in p.parents:continue
        raw=p.read_bytes()
        if b'"mhc_pre_sinkhorn"' in raw:walk(json.loads(raw),None,name,hits)
    for base in (ROOT/'runs',ROOT/'results'):
        for folder,dirs,names in os.walk(base):
            dirs[:]=[s for s in dirs if s not in {'sources','profile','simulator','npusim','dump','log','log_ca','raw-kernel','record','ca_workspace'} and not s.startswith(('PROF_','OPPROF_'))]
            if 'run.log' not in names:continue
            p=Path(folder)/'run.log'
            if p.stat().st_size>32*1024**2:continue
            for t,n,d in re.findall(r'api=aclnnMhcPreSinkhorn T=(\d+) N=(\d+) D=(\d+)',p.read_text(errors='replace')):hits[orig.OP,(int(t),int(n),int(d))].add(str(p.relative_to(ROOT)))
    known={str(s[0]):sorted(v) for (op,s),v in hits.items() if op==orig.OP and tuple(s[1:])==(4,4096)}
    for t in (*MAIN,RESERVE):
        if str(t) in known:raise ValueError(('prior shape',t,known[str(t)]))
    return known

def freeze():
    if REG.exists() or RAW.exists():raise FileExistsError(REG)
    v=read(ORIG_REG);save(OUT/'novelty.json',novelty());jobs=[]
    for j in v['jobs']:
        if j['predictions']!={c:orig.forecast(j['shape'],c) for c in (*CHIPS,FUT)}:raise ValueError('v0.26 predictions lost parity')
        jobs.append(dict(**j,cohort='v0.26 original'))
    for t in (*MAIN,RESERVE):
        s=[t,4,4096]
        for chip in CHIPS:
            prior=next(j for j in v['jobs'] if j['sku']==chip)
            jobs.append({**prior,'id':f'fused-t{t}-'+('pr' if chip==orig.PR else 'dt'),'shape':s,'args':[orig.OP,*map(str,s)],
                         'predictions':{c:orig.forecast(s,c) for c in (*CHIPS,FUT)},
                         'cohort':'reserved when any original pair fails' if t==RESERVE else 'v0.27 prospective'})
    deps=[Path(__file__).resolve(),V26/'resume.py',POLICY,OUT/'novelty.json',ROOT/'development/validation/fusion27/admission.py',
          ROOT/'development/validation/fusion27/observe.py',ORIG_REG,ORIG/'endpoint-addendum.json']
    reg=dict(utc=utc(),original_sha256=orig.sha(ORIG_REG),same_frozen_model=True,source_sha256={str(p.relative_to(ROOT)):orig.sha(p) for p in deps},
             original_sources=v['hashes'],jobs=jobs,jobs_budget='45 minutes/10 GiB per run, serial; additional (8+conditional ninth) geometry aggregate ceilings 9 host CPU-hours of elapsed acquisition time and 80 GiB raw. No duplicate job. Existing failed/partial runs retained.',
             aggregate_seconds=32400,aggregate_raw_gib=80,role='Frozen v0.26 candidate and v0.25 reference, all forecasts locked before any new geometry. Output availability cannot change the forecasts.',
             admission='same original flock; never supersede/stop an acquisition. Hold a full-size (96 CPUs when available), idle-scheduling mask under continuous headroom.',
             comparison='One invocation, three deterministic all-element goldens, same per-core full retirement (including terminal AIC DCCI) and binary SHA as v0.26. Do not switch default before ten distinct qualified geometries per SKU, WAPE <=20%, max <=40%, and improvement vs old same-set. Publish direction, paired DT/PR ratio and residual small-token regressions separately.')
    with REG.open('x') as f:json.dump(reg,f,indent=2);f.write('\n')
def qualify(j,folder):
    [s],checks=orig.stages(j,j['sku'],folder);t,n,d=j['shape']
    if s['kernel']!=j['kernel'] or s['binary_sha256']!=j['elf']:raise ValueError('ELF identity')
    if [(m[1],int(m[2])) for c in checks if (m:=re.search(r'name=(\S+) count=(\d+) bad=0',c))]!=[('hin',t*d),('hpost',t*n),('hres',t*n*n)]:raise ValueError('golden coverage')
    if f'api={j["api"]} T={t} N=4 D=4096 dtype=BF16 phi_dtype=FP32 num_iters=20 need_backward=false' not in (folder/'run.log').read_text():raise ValueError('actual shape/attributes')
    shards={p.parent.name:p for p in (ROOT/s['trace']).parent.glob('core*/trace.json')}
    if set(shards)!=set(s['windows']):raise ValueError('core roster')
    offsets={c:max(z['retire'] for z in ep.terminal(path).values())-s['windows'][c][1]*1650 for c,path in shards.items()}
    if max(offsets.values())-min(offsets.values())>3:raise ValueError('physical retirement vs official window')
    return dict(qualified=True,chip=j['sku'],shape=j['shape'],stage=s,ca_us=s['ca_us'],correctness=checks,measured_retire_offsets_cycles=offsets,
                original_source='v0.26 contracts plus registered DCCI endpoint',prediction=j['predictions'][j['sku']])
def status(msg,**fields):save(OUT/'state.json',dict(utc=utc(),state=msg,**fields))
def run():
    reg=read(REG)
    for n,h in {**reg['original_sources'],**reg['source_sha256']}.items():
        if orig.sha(ROOT/n)!=h:raise ValueError('frozen source changed '+n)
    RAW.mkdir(exist_ok=True);HOLD.mkdir(exist_ok=True)
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    with (PRIOR_RAW/'exclusive.lock').open('a') as lock:
        status('waiting for original single slot')
        fcntl.flock(lock,fcntl.LOCK_EX)
        oldstate=read(ORIG/'state.json')['state']
        if oldstate not in ('stopped at preserved failure','stopped at qualification failure','interrupted prior attempt preserved','finished'):
            status('original single slot released without a terminal state; no duplicate started',original=oldstate);return
        failed=any((ORIG/(j['id']+'.json')).exists() and not read(ORIG/(j['id']+'.json'))['qualified'] for j in reg['jobs'] if j['cohort']=='v0.26 original')
        for j in reg['jobs']:
            dest=HOLD/(j['id']+'.json');raw=RAW/j['id']
            if j['cohort'].startswith('reserved') and not failed:continue
            if dest.exists():continue
            if j['cohort']=='v0.26 original':
                if (ORIG/(j['id']+'.json')).exists():continue
                if (PRIOR_RAW/j['id']).exists():
                    status('original raw exists without terminal receipt, preserved',id=j['id']);return
            if not (raw/'cost.json').exists():
                if raw.exists():status('unscored interrupted attempt preserved',id=j['id']);return
                costs=[read(p) for p in RAW.glob('*/cost.json')]
                remaining=reg['aggregate_seconds']-sum(c['seconds'] for c in costs)
                available=reg['aggregate_raw_gib']*1024-sum(c['raw_bytes']/2**20 for c in costs)
                if available<j['raw_mib'] or remaining<j['wall_s']:status('aggregate budget exhausted',id=j['id']);return
                os.sched_setaffinity(0,ALL);p=read(POLICY);start=None;a=ticks()
                while True:
                    time.sleep(p['sample_s']);b=ticks();mask,info=capacity(a,b);a=b
                    if info['reasons']:start=None
                    elif start is None:start=time.monotonic()
                    good=time.monotonic()-start if start else 0
                    status('waiting for older tasks or capacity',id=j['id'],continuous_good_seconds=round(good),**info)
                    if good>=p['continuous_new_s']:break
                set_limits(0,mask);status('acquiring',id=j['id'],cpus=mask,**info);orig.capture(j,j['sku'],raw)
            status('qualifying',id=j['id'])
            cost=read(raw/'cost.json')
            try:
                if cost['started_utc']<reg['utc']:raise ValueError('wrong chronology')
                result=qualify(j,raw)
            except (ValueError,KeyError,AssertionError,OSError,TypeError,IndexError) as exc:
                result=dict(qualified=False,chip=j['sku'],shape=j['shape'],ca_us=None,reason=repr(exc))
            save(dest,{**result,'id':j['id'],'cohort':j['cohort'],'cost':cost,'registration_sha256':orig.sha(REG)})
            if j['cohort']=='v0.26 original' and not result['qualified']:failed=True
        status('finished')
if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
