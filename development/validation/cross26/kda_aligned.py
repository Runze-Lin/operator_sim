"""One pre-registered aligned large-workload check, without changing the KDA numerical thresholds."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from kda import capacity,ca,qualify,RAW,GOLD,LEGACY,HOST,DT,PR,sha,read,predict,REGISTER,OUT,ROOT

REG=OUT/'aligned-register.json'
T,H=192,4


def freeze():
    shape=[1,T,H,H,128,128]
    for p in (ROOT/'results/v0.24/kda-prefetch/raw',ROOT/'results/kda-v23/raw'):
        if any(x.name.endswith(f'-{T}-{H}') or x.name.endswith(f't{T}-h{H}') for x in p.iterdir()):
            raise FileExistsError('prior acquisition')
    parent=read(REGISTER)
    g=GOLD/f'{T}-{H}'
    if g.exists() or REG.exists():
        raise FileExistsError('input/output already exists')
    subprocess.run([str(ROOT/'toolchains/envs/cann-9.2.1/bin/python'),str(LEGACY/'reference.py'),str(T),str(H),str(g)],
                   env={**os.environ,'OPENBLAS_NUM_THREADS':'1'},check=True)
    r={k:v for k,v in parent.items() if k not in ('utc','cases','parent_sha256','registration_reason','source_hashes')}
    pins={k:v for k,v in parent['source_hashes'].items() if '/raw/golden/' not in k}
    for p in [*g.iterdir(),Path(__file__),REGISTER]:pins[str(p.relative_to(ROOT))]=sha(p)
    forecasts={s:predict(shape,s) for s in (DT,PR,'Ascend970DT')}
    r.update(utc=ca.when(),parent_sha256=sha(REGISTER),source_hashes=pins,
             registration_reason='Aligned three full chunks at four concurrent heads (768 token-head pairs), complementing failed fourth or later chunk tests at one head. Chosen after both failures. Same four mathematical checks, no refit; publish both outcomes and every previous exclusion.',
             cases=[dict(id=f'aligned26-{s}-{T}-{H}',sku=s,physical_memory=ca.MEMORY[s],shape=shape,
                         oracle=dict(attention=T*H*128,aqk=T*H*64,akk=T*H*64,state_vk=H*128*128),
                         predictions=forecasts) for s in (DT,PR)])
    with REG.open('x') as f:json.dump(r,f,indent=2);f.write('\n')


def run():
    r=read(REG)
    for k,h in r['source_hashes'].items():
        if sha(ROOT/k)!=h:raise ValueError(k)
    ca.BIN=HOST
    with (RAW/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for c in r['cases']:
            dest=OUT/(c['id']+'.json')
            if dest.exists():continue
            while not capacity(r)[1]:time.sleep(45)
            folder=RAW/c['id']
            if not (folder/'cost.json').exists():folder.mkdir()
            ca.save(OUT/'aligned-state.json',dict(state='acquiring or verifying',case=c['id'],utc=ca.when()))
            cost=ca.capture(r,c,folder,r['budget_per_capture_seconds'])
            v=qualify.verify(r,c,folder,cost)
            files={k:folder/(k+'.bf16') for k in ('aqk','akk')}
            v['actual_aux_bf16']={k:dict(bytes=f.stat().st_size,sha256=sha(f)) for k,f in files.items() if f.is_file()}
            v['qualified']=v['qualified'] and all(f.is_file() and f.stat().st_size==2*c['oracle'][k] for k,f in files.items())
            ca.save(dest,dict(**v,case={k:c[k] for k in ('id','shape','sku','oracle')},cost=cost,registration_sha256=sha(REG)))
        ca.save(OUT/'aligned-state.json',dict(state='complete',utc=ca.when()))


if __name__=='__main__':
    {'freeze':freeze,'run':run}[sys.argv[1]]()
