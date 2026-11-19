"""Freeze same public v0.25 reference and distinct no-CA forecast scenarios once."""
from collections import defaultdict
import os
import re
import subprocess
import sys
from evidence import ROOT,OUT,DT,PR,FUT,read,save,sha
from model import forecast
from operator_model import estimate,load_chip
from development.validation.fusion21.history import walk
from development.validation.sku19.campaign import now

RAW=OUT/'raw';REG=OUT/'register.json';HERE=__import__('pathlib').Path(__file__).resolve().parent
BINARY=ROOT/'build/vector-v8/MHC'
VENDOR=next((ROOT/'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/mhc_sinkhorn_apt').glob('MhcSinkhorn_*.o'))
IMPL=ROOT/'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/mhc_sinkhorn/op_kernel/arch35/mhc_sinkhorn_arch35.h'
GROUPS={'first':[(67,10)],'grow':[(13,3),(46,8),(67,10),(89,12),(157,12),(241,16)],
        'boundary':[(58,4),(130,10),(173,12),(251,20)],'wide':[(257,4),(449,8),(513,6),(4097,4)]}

def novelty():
    hits=defaultdict(set)
    files=subprocess.check_output(['git','ls-files','results/**/*.json','configs/**/*.json'],cwd=ROOT,text=True).splitlines()
    for fn in files:
        f=ROOT/fn
        if not f.is_file() or OUT in f.parents:continue
        b=f.read_bytes()
        if b'"mhc_sinkhorn"' in b:walk(__import__('json').loads(b),None,fn,hits)
    # Include aborted and unregistered case folders (results and runs); skip raw simulator trees.
    for base in (ROOT/'runs',ROOT/'results'):
        for folder,dirs,fs in os.walk(base):
            dirs[:]=[s for s in dirs if s not in {'sources','profile','simulator','npusim','dump','log','record','ca_workspace'} and not s.startswith(('OPPROF_', 'PROF_'))]
            if 'run.log' not in fs:continue
            p=__import__('pathlib').Path(folder)/'run.log'
            if p.stat().st_size>32*1024**2:continue
            for t,n in re.findall(r'case=mhc_sinkhorn api=aclnnMhcSinkhorn T=(\d+) N=(\d+)',p.read_text(errors='replace')):
                hits['mhc_sinkhorn',(int(t),int(n))].add(str(p.relative_to(ROOT)))
    known={str(s[0]):sorted(ps) for (op,s),ps in hits.items() if op=='mhc_sinkhorn' and len(s)==2 and s[1]==4}
    for g in GROUPS.values():
        for t,i in g:
            if str(t) in known:raise ValueError(('registered or measured geometry already exists',t,known[str(t)]))
    return known

def freeze():
    if REG.exists() or RAW.exists():raise FileExistsError(REG)
    save(OUT/'novelty.json',novelty())
    obj={'utc':now(),'source_hashes':{},'jobs':[],'order':list(GROUPS),'role':'prospective ordered processes; old evidence all development replay',
         'contract':'same-family FP32 [T,4,4], no intermediates, physical END + vendor full output and exact ELF text',
         'budget':{'first':dict(wall_s=600,mib=1100),'grow':dict(wall_s=2200,mib=3800),'boundary':dict(wall_s=1800,mib=3500),'wide':dict(wall_s=2200,mib=5500)},
         'target':'compare 10 unique in-domain geometries per side plus 4 beyond the v0.25 T<=256 public contract. T67 I10 first vs ordered separately; count once for geometry. No default update on in-sample performance. Full H evidence and shape-only errors both required; an H-aware candidate cannot take credit for unknown H.',
         'priorities':'single active SKU/stream for this study; compact --dump=false; yield before starting a group to earlier Pro queued or constrained RAM/disk/CPU. Keep raw attempts and do not retry automatically.'}
    for group,cases in GROUPS.items():
        for chip in (DT,PR,FUT):
            profile=load_chip(chip);h=0
            for rank,(t,i) in enumerate(cases):
                s=[t,4];old=estimate('mhc_sinkhorn',s,{'num_iters':i},chip=chip);known=forecast(s,i,profile,h);unknown=forecast(s,i,profile)
                obj['jobs'].append(dict(group=group,chip=chip,index=rank,shape=s,iters=i,prior_h=h,v025_us=old.latency_us,
                                       v025_status=old.status,unknown_us=unknown['latency_us'],known=known,
                                       evidence='projection only' if chip==FUT else 'prospective CA'))
                h=max(h,known['active_aiv'])
    files=[BINARY,VENDOR,IMPL,ROOT/'smoke/vector_v8.cpp',ROOT/'smoke/extension_indexer_mhc.cpp',ROOT/'scripts/cann-run.sh',OUT/'prices.json',
           OUT/'novelty.json',ROOT/'development/validation/rms_ordered_verify/verify.py',*[HERE/s for s in ('evidence.py','model.py','calibrate.py','registration.py','collect.py','score.py')]]
    files.extend(ROOT/'configs/chips'/f'{c}.json' for c in (DT,PR,FUT))
    obj['source_hashes']={str(f.relative_to(ROOT)):sha(f) for f in files}
    preflight=OUT/'register-before-collector-preflight.json'
    if preflight.exists():
        previous=read(preflight)
        if previous['jobs']!=obj['jobs']:raise ValueError('A collector-only clarification changed frozen forecasts')
        obj['preflight_registration_sha256']=sha(preflight)
        obj['preflight_note']='Before any CA was started, the verifier was extended to the one-kernel output root used by msopprof. Predictions and jobs are identical.'
    with REG.open('x') as out:__import__('json').dump(obj,out,indent=2);out.write('\n')
if __name__=='__main__':freeze()
