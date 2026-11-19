"""Four pre-frozen stage-recurrence holdouts on one CA slot. Keep all previous evidence read-only."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from evidence import HERE,ROOT,OUT,OLD,read,sha
from model import predict

LEGACY=HERE.parent/'kda_v23'
sys.path.insert(0,str(LEGACY))
import series as ca
import qualify

RAW=OUT/'raw'
HOST=RAW/'host-kda'
GOLD=RAW/'golden'
SHAPES=((80,1),(145,2))
DT,PR=ca.DT,ca.PR

def compact(path,item):path.write_text(json.dumps(item,indent=2)+'\n')

def register():
    path=OUT/'register.json'
    if path.exists():raise FileExistsError(path)
    RAW.mkdir(exist_ok=True);GOLD.mkdir(exist_ok=True)
    old=OLD/'precision-supplement-register.json';ref=read(old)
    oldcpp=OLD/'raw/precision-host/kimi-v2-aux.cpp'
    source=oldcpp.read_text();oldgold=str(oldcpp.parent)+'/'
    if source.count(oldgold)!=1:raise ValueError('Unexpected old output/reference path')
    cpp=HOST.with_suffix('.cpp');cpp.write_text(source.replace(oldgold,str(GOLD)+'/'))
    api=ca.probe.VENDOR/'op_api';cann=ca.probe.CANN
    subprocess.run(['bash',str(ROOT/'scripts/cann-run.sh'),'g++','-std=c++17','-O2','-fPIE','-pie',str(cpp),
        '-I'+str(cann/'include'),'-I'+str(api/'include'),'-L'+str(api/'lib'),'-Wl,-rpath,'+str(api/'lib'),
        '-L'+str(cann/'lib64'),'-Wl,-rpath-link,'+str(cann/'lib64'),str(ROOT/'toolchains/envs/cann-9.2.1/lib/libstdc++.so.6'),
        '-lcust_opapi','-lopapi','-lnnopbase','-lascendcl','-ldl','-o',str(HOST)],check=True)
    old_cases=set()
    for f in OLD.glob('*.json'):
        v=read(f)
        if isinstance(v,dict) and isinstance(v.get('case'),dict):old_cases.add(tuple(v['case']['shape']))
    pins=[HERE/'model.py',HERE/'evidence.py',Path(__file__),OUT/'cross-chunk-evidence.json',ROOT/'operator_model/kda_fla_v2.py',
          ROOT/'operator_model/chips.py',ROOT/'configs/kda-fla-v2-26.9.1.json',LEGACY/'reference.py',LEGACY/'series.py',
          LEGACY/'qualify.py',LEGACY/'precise_host.hpp',old,oldcpp,cpp,HOST,ROOT/'smoke/extension_kda.cpp']
    cases=[]
    for s,h in SHAPES:
        shape=[1,s,h,h,128,128];assert tuple(shape) not in old_cases
        folder=GOLD/f'{s}-{h}'
        subprocess.run([str(ROOT/'toolchains/envs/cann-9.2.1/bin/python'),str(LEGACY/'reference.py'),str(s),str(h),str(folder)],
                       env={**os.environ,'OPENBLAS_NUM_THREADS':'1'},check=True,stdout=subprocess.DEVNULL)
        pins.extend(folder.iterdir())
        for sku in (DT,PR):
            ident=f'independent-{sku}-{s}-{h}';assert not (RAW/ident).exists() and not (OUT/(ident+'.json')).exists()
            cases.append(dict(id=ident,sku=sku,physical_memory=ca.MEMORY[sku],shape=shape,
                    oracle=dict(attention=s*h*128,aqk=s*h*64,akk=s*h*64,state_vk=h*128*128),
                    predictions={target:predict(shape,target) for target in (DT,PR,'Ascend970DT')}))
    pins += [ROOT/'configs/chips'/f'{sku}.json' for sku in (DT,PR,'Ascend970DT')]
    reg=dict(utc=ca.when(),role='prospective; all previous qualified windows including v0.23 holdouts are development replays',
             source_hashes={str(f.relative_to(ROOT)):sha(f) for f in pins},published_device_text=ref['published_device_text'],
             actual_predicted_scope='sum of exactly three non-overlapping full device kernel windows; inter-kernel gaps excluded',
             budget_per_capture_seconds=720,budget_per_capture_bytes=3*1024**3,budget_total_capture_seconds=2400,
             max_concurrency=1,host_change='Only the directory containing the published numerical reference changed; matching host source and numerical protocol pinned.',
             stop_at_first_failure=True,cases=cases)
    with path.open('x') as f:json.dump(reg,f,indent=2);f.write('\n')
    print('freeze',sha(path))

def run():
    p=OUT/'register.json';r=read(p)
    for name,digest in r['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Pin changed '+name)
    ca.BIN=HOST
    with (RAW/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for c in r['cases']:
            result=OUT/(c['id']+'.json')
            if result.exists():
                if read(result)['qualified']:continue
                raise ValueError('Previous ineligible capture')
            folder=RAW/c['id'];folder.mkdir()
            ca.save(OUT/'state.json',dict(state='running',id=c['id'],started=ca.when()))
            used=sum(read(q)['cost']['seconds'] for q in OUT.glob('independent-Ascend*.json'))
            cost=ca.capture(r,c,folder,r['budget_total_capture_seconds']-used)
            rec=qualify.verify(r,c,folder,cost)
            actual={}
            for op in ('aqk','akk'):
                arr=folder/(op+'.bf16')
                if arr.exists():actual[op]=dict(bytes=arr.stat().st_size,sha256=sha(arr))
            assert len(actual)==2 and all(actual[k]['bytes']==2*c['oracle'][k] for k in actual)
            rec.update(case=dict(id=c['id'],sku=c['sku'],shape=c['shape'],oracle=c['oracle']),cost=cost,
                       registration_sha256=sha(p),actual_aux_bf16=actual)
            ca.save(result,rec)
            if not rec['qualified']:
                ca.save(OUT/'state.json',dict(state='failed',id=c['id']));return
        ca.save(OUT/'state.json',dict(state='complete',finished=ca.when()))

if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
