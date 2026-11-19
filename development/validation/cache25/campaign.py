"""Register or run two previously unseen equal-payload shapes after existing large CA jobs."""
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
from math import prod
import os
from pathlib import Path
import shutil
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'results/v0.25/cache'
RAW=OUT/'raw'
REG=OUT/'freeze.json'
sys.path.insert(0,str(ROOT))
from shadow import predict
from development.validation.sku19.collect import capture
from campaign_collect import stages
from descriptors import measure

DT='Ascend950DT_9582'
PR='Ascend950PR_957b'
FUTURE='Ascend970DT'
CASES={'cast_bf16_fp32':([851968,32],'build/smoke/large_cast20','aclnnCast'),
       'sigmoid_fp32':([1703936,12],'build/smoke/sigmoid20','aclnnSigmoid')}


def read(p):return json.loads(p.read_bytes())
def utc():return datetime.now(timezone.utc).isoformat()
def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)
def digest(path):
    h=sha256()
    with path.open('rb') as src:
        for data in iter(lambda:src.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()
def state(label,**args):save(OUT/'state.json',{'utc':utc(),'state':label,**args})


def register():
    if REG.exists():raise FileExistsError(REG)
    catalog=read(ROOT/'results/v0.25/coverage/audit.json')['rows']
    jobs=[]
    for op,(shape,host,api) in CASES.items():
        n=prod(shape)
        if any(r['op']==op and prod(r['shape'])==n for r in catalog):raise ValueError('previously measured flattened extent')
        pattern='*'+'_'.join(map(str,shape))+'*'
        if any(any(folder.glob(pattern)) for folder in [ROOT/'runs/vector21',ROOT/'runs/sigmoid20',ROOT/'runs/prefill20',ROOT/'results/v0.23/pro-prefill/raw',RAW]):
            raise ValueError('previous raw job')
        for chip in (PR,DT):
            library=next(v for v in catalog if v['op']==op and v['chip']==chip and '≥262144 行' in v['slices'])
            reference=next(r for r in (OUT.glob(op+'-*'+chip+'-traffic.json')))
            frozen=read(reference)
            assert library['kernel']==frozen['kernel']
            fore={sku:predict(op,shape,sku) for sku in (DT,PR,FUTURE)}
            jobs.append(dict(id=op+'-'+('pr' if chip==PR else 'dt'),sku=chip,op=op,shape=shape,runner=host,args=list(map(str,shape)),api=api,
                 kernels=1,nchecks=1,expected_elements=n,working_set_MiB=n*(6 if op.startswith('cast') else 8)/2**20,
                 kernel=frozen['kernel'],elf=frozen['binary_sha256'],predictions=fore,
                 wall_s=5400,raw_mib=16384,receipt=reference.name,receipt_sha256=digest(reference)))
    assert {j['working_set_MiB'] for j in jobs}=={156}
    sources=list(HERE.glob('*.py'))+[ROOT/h for _,h,_ in CASES.values()]+[ROOT/p for p in (
      'operator_model/cast_resource.py','operator_model/sigmoid_resource.py','operator_model/vector_memory.py','operator_model/rms_resource.py',
      'scripts/cann-run.sh','development/validation/sku19/collect.py','development/validation/sku17/campaign_collect.py',
      'development/validation/pro_prefill23/evidence.py','development/validation/large20/cast.cpp','development/validation/sigmoid20/smoke.cpp')]+[ROOT/'configs/chips'/(s+'.json') for s in (DT,PR,FUTURE)]
    save(REG,dict(utc=utc(),jobs=jobs,sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
         role='Prospective cross-operator mechanism check, no model refit. Previous model and two frozen scenarios are scored equally; identical 156 MiB logical working set, 2:1 vs 1:1 writes/read.',
         qualification='Actual target RAM backend + runtime SKU + expected ELF + shape + all outputs + every active core duration table and actual per-core END, continuous GM payload and earliest/latest instruction window. No replacement by wrapper or wall.',
         optional_traffic='Mirror native controller summaries before profiler cleanup where available. Missing physical counters do not imply zero writes. Record coverage separately.',
         bounds='4 serial jobs; each 90 min and 16 GiB wall/raw; existing Pro 256k, Flash full-hidden retry and Kimi native job have priority; 970DT prediction only.'))
    print(digest(REG))


def mirror(folder,stop,errors):
    dest=folder/'controller-snapshots'
    while not stop.is_set():
        try:
            base=folder/'log'
            if base.is_dir():
                for src in base.glob('*'):
                    if not src.is_file() or src.suffix!='.log':continue
                    if not any(k in src.name.lower() for k in ('state','stat','memory','ddr','dmc','cache','sllc')):continue
                    dest.mkdir(exist_ok=True)
                    size=src.stat().st_size;out=dest/src.name
                    if not size or (out.exists() and out.stat().st_size>=size):continue
                    if size>64*1024**2:continue
                    tmp=out.with_suffix('.partial')
                    with src.open('rb') as f,tmp.open('wb') as w:
                        for data in iter(lambda:f.read(65536),b''):w.write(data)
                    if tmp.stat().st_size>=size:tmp.replace(out)
        except FileNotFoundError:pass
        except OSError as error:errors.append(str(error));return
        stop.wait(3)


def wait_prior(j):
    while True:
        pro=read(ROOT/'results/v0.23/pro-prefill/state.json')['state']
        full=read(ROOT/'results/v0.23/full-hidden/v3/queue-status.json')['state']
        # This existing full-hidden supervisor already serializes behind Kimi native.
        enough=shutil.disk_usage(OUT).free>32*2**30
        if pro=='finished' and full=='complete' and enough:return
        state('awaiting existing large queues',case=j['id'],pro=pro,full_hidden=full,space_ok=enough)
        time.sleep(45)


def run():
    manifest=read(REG)
    RAW.mkdir(exist_ok=True)
    with (RAW/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for filename,h in manifest['sha256'].items():
            if digest(ROOT/filename)!=h:raise ValueError('frozen contract changed: '+filename)
        os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
        for j in manifest['jobs']:
            dest=OUT/(j['id']+'-result.json');folder=RAW/j['id']
            if dest.exists():
                if not read(dest)['qualified']:state('preserved failed attempt',case=j['id']);return
                continue
            if not (folder/'cost.json').exists():
                if folder.exists():state('preserved incomplete previous attempt',case=j['id']);return
                wait_prior(j);folder.mkdir()
                state('acquiring',case=j['id'])
                end=threading.Event();errors=[];monitor=threading.Thread(target=mirror,args=(folder,end,errors))
                monitor.start()
                try:capture(j,j['sku'],folder)
                finally:end.set();monitor.join()
                if errors:save(folder/'mirror-errors.json',errors)
            state('validating complete output',case=j['id'])
            cost=read(folder/'cost.json');n=j['expected_elements'];cast=j['op'].startswith('cast');t,h=j['shape']
            try:
                got,correct=stages(j,j['sku'],folder);stage,=got
                if stage['kernel']!=j['kernel'] or stage['binary_sha256']!=j['elf']:raise ValueError('different ELF or name')
                if correct!=['correctness count='+str(n)+' bad=0'+('' if cast else ' atol=2e-6')]:raise ValueError('incorrect golden protocol')
                tokens=f'R={t} D={h}' if cast else f'T={t} H={h}'
                if f"case={j['op']} api={j['api']} {tokens}" not in (folder/'run.log').read_text():raise ValueError('incorrect launched contract')
                r=dict(shape=j['shape'],chip=j['sku'],qualified=True,ca_us=stage['ca_us'],stage=stage,correctness=correct)
                traffic=measure(r)
                save(OUT/(j['id']+'-prospective-traffic.json'),traffic)
                r['forecast']=j['predictions'][j['sku']]
                r['error_pct']={k:100*(v['us']/r['ca_us']-1) for k,v in r['forecast'].items()}
                r['controller_snapshots']=[x.name for x in sorted((folder/'controller-snapshots').glob('*.log'))]
            except (ValueError,KeyError,AssertionError,OSError,IndexError) as exc:r=dict(shape=j['shape'],chip=j['sku'],qualified=False,ca_us=None,reason=repr(exc))
            r.update(freeze_sha256=digest(REG),cost=cost);save(dest,r)
            if not r['qualified']:state('failed; preserved and stopped',case=j['id']);return
        state('finished',cases=[j['id'] for j in manifest['jobs']])


if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
