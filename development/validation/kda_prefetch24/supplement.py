"""Single-slot follow-on holdouts, using the original v0.24 frozen engine and host."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from capture import ca, qualify, RAW, HOST, GOLD, LEGACY, DT, PR
from evidence import HERE, ROOT, OUT, OLD, read, sha
from frozen import ARCHIVE, PINS, engine, predict

DEST = OUT / 'supplement'
REGISTER = DEST / 'register.json'
SKUS = (DT, PR, 'Ascend970DT')
SHAPES = ((63,4),(64,3),(65,4),(127,1),(128,2),(130,3),(193,2),(257,1))
FIRST_REGISTER = OUT/'register.json'


def freeze():
    if REGISTER.exists():
        raise FileExistsError(REGISTER)
    engine()  # validates archived source/params against the original registration
    parent = read(FIRST_REGISTER)
    prior = set()
    for folder in (OLD, OUT):
        for path in folder.glob('*.json'):
            r = read(path)
            if isinstance(r, dict) and isinstance(r.get('case'),dict) and 'shape' in r['case']:
                prior.add(tuple(r['case']['shape']))
            if isinstance(r,dict):
                for old in r.get('cases',[]):
                    if isinstance(old,dict) and 'shape' in old:
                        prior.add(tuple(old['shape']))
    # Current public code has advanced. The archived copies of the six old
    # resource files, including the prices, are checked by frozen.engine().
    pins = {n:v for n,v in parent['source_hashes'].items() if n not in PINS
            and '/raw/golden/' not in n}
    for name, expected in pins.items():
        if sha(ROOT/name) != expected:
            raise ValueError('Original frozen capture changed: '+name)
    for path in (FIRST_REGISTER, ARCHIVE, HERE/'frozen.py', Path(__file__), HERE/'supplement_score.py', ROOT/'scripts/cann-run.sh',
                 ca.probe.VENDOR/'op_api/lib/libcust_opapi.so'):
        pins[str(path.resolve().relative_to(ROOT))]=sha(path)
    cases = []
    for t,h in SHAPES:
        shape=[1,t,h,h,128,128]
        if tuple(shape) in prior:
            raise ValueError(f'Existing shape {shape}')
        golden = GOLD/f'{t}-{h}'
        if golden.exists():
            raise FileExistsError(golden)
        subprocess.run([str(ROOT/'toolchains/envs/cann-9.2.1/bin/python'),str(LEGACY/'reference.py'),
                        str(t),str(h),str(golden)],env={**os.environ,'OPENBLAS_NUM_THREADS':'1'},
                       check=True,stdout=subprocess.DEVNULL)
        for path in golden.iterdir():
            pins[str(path.relative_to(ROOT))]=sha(path)
        for sku in SKUS[:2]:
            ident=f'supplement-{sku}-{t}-{h}'
            if (RAW/ident).exists() or (DEST/(ident+'.json')).exists():
                raise FileExistsError(ident)
            cases.append(dict(id=ident,sku=sku,physical_memory=ca.MEMORY[sku],shape=shape,
                              oracle=dict(attention=t*h*128,aqk=t*h*64,akk=t*h*64,state_vk=h*128*128),
                              predictions={target:predict(shape,target) for target in SKUS}))
    reg=dict(utc=ca.when(),parent_registration_sha256=sha(FIRST_REGISTER),
             role='additional frozen independent shapes; first two pairs were known when selecting new geometry',
             source_hashes=pins,published_device_text=parent['published_device_text'],
             actual_predicted_scope=parent['actual_predicted_scope'],max_concurrency=1,
             budget_per_capture_seconds=720,budget_per_capture_bytes=3*1024**3,
             budget_total_capture_seconds=9600,min_free_bytes=64*1024**3,min_available_ram_bytes=16*1024**3,
             max_load_per_online_cpu=.92,prior_known_shapes=sorted(prior),
             queue_precedence=['v0.24 Post collector complete','v0.24 RMS last PR receipt present',
                               'pre-existing v0.23 Pro 128Ki PR acquisition has started or finished'],
             protocol='Same ELF, four mixed-sign numerical groups and official whole-core windows. Preserve every failure; do not rerun raw attempts.',
             cases=cases)
    DEST.mkdir(exist_ok=True)
    with REGISTER.open('x') as f:
        json.dump(reg,f,indent=2);f.write('\n')
    print('freeze',sha(REGISTER))


def status(state, **fields):
    ca.save(DEST/'state.json', dict(state=state,utc=ca.when(),registration_sha256=sha(REGISTER),**fields))


def capacity(reg):
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    free=shutil.disk_usage(RAW).free
    load=os.getloadavg()[0]/(os.cpu_count() or 1)
    post=ROOT/'results/v0.24/post/status.json'
    rms=ROOT/'results/v0.24/rms-prefix/many-rows-Ascend950PR_957b.json'
    pro=ROOT/'results/v0.23/pro-prefill'
    def observed(path):
        try:return read(path)
        except (FileNotFoundError,json.JSONDecodeError):return {}
    first=observed(pro/'state.json')
    post_done=observed(post).get('state')=='complete'
    pro_started=(pro/'pro-b8-128ki-pr.json').exists() or (
        first.get('case')=='pro-b8-128ki-pr' and first.get('state') in ('acquiring','validating full trace'))
    return dict(available_ram_bytes=available,free_bytes=free,load_per_cpu=load,
                post_complete=post_done,rms_final_receipt=rms.exists(),pro_128ki_pr_started=pro_started),(
        post_done and rms.exists() and pro_started and available>=reg['min_available_ram_bytes']
        and free>=reg['min_free_bytes'] and load<=reg['max_load_per_online_cpu'])


def run():
    reg=read(REGISTER)
    engine()
    for name,digest in reg['source_hashes'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Frozen capture source changed: '+name)
    ca.BIN=HOST
    with (RAW/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        completed=[]
        seconds=0
        for c in reg['cases']:
            result=DEST/(c['id']+'.json')
            if result.exists():
                old=read(result)
                if old['registration_sha256']!=sha(REGISTER):raise ValueError(result)
                seconds+=old['cost']['seconds'] or 0
                completed.append(c['id'])
                continue
            folder=RAW/c['id']
            if (folder/'run.log').exists() and not (folder/'cost.json').exists():
                status('interrupted raw attempt; no duplicate launch',id=c['id'],completed=completed)
                return
            if not (folder/'cost.json').exists():
                while True:
                    headroom,ready = capacity(reg)
                    if ready:break
                    status('waiting for host capacity',id=c['id'],completed=completed,**headroom)
                    time.sleep(45)
                if seconds>=reg['budget_total_capture_seconds']:
                    status('CA budget reached',id=c['id'],completed=completed);return
            folder.mkdir(exist_ok=True)
            status('running',id=c['id'],completed=completed)
            cost=ca.capture(reg,c,folder,reg['budget_total_capture_seconds']-seconds)
            try:
                rec=qualify.verify(reg,c,folder,cost)
                aux={}
                for name in ('aqk','akk'):
                    arr=folder/(name+'.bf16')
                    if arr.exists():
                        aux[name]=dict(bytes=arr.stat().st_size,sha256=sha(arr))
                sizes_ok=len(aux)==2 and all(aux[k]['bytes']==2*c['oracle'][k] for k in aux)
                rec['actual_aux_bf16']=aux
                rec['qualified']=rec['qualified'] and sizes_ok
                if not sizes_ok:rec['reason']='Incomplete full auxiliary BF16 outputs'
            except (ValueError,KeyError,AssertionError,TypeError,OSError,IndexError) as exc:
                rec=dict(qualified=False,reason=repr(exc),golden_qualified=False)
            rec.update(cost=cost,case={k:c[k] for k in ('id','sku','shape','oracle')},registration_sha256=sha(REGISTER))
            ca.save(result,rec)
            seconds+=cost['seconds'] or 0
            completed.append(c['id'])
            from supplement_score import publish
            publish()
        status('complete',completed=completed,total_ca_seconds=seconds)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['freeze','run','publish'])
    args=parser.parse_args()
    if args.action=='publish':
        from supplement_score import publish
        publish()
    else:
        {'freeze':freeze,'run':run}[args.action]()
