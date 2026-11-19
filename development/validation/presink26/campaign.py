"""Freeze and validate the fused inference ABI. Run one official simulator process at a time."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/v0.26/presink';REG=OUT/'freeze.json';RAW=OUT/'raw'
sys.path.insert(0,str(ROOT))
from operator_model import load_chip
from operator_model.mhc_sink_resource import fused as old
from operator_model.mhc_pre_sink_resource import fused as candidate
from development.validation.sku19.collect import capture
from campaign_collect import stages
sys.path.insert(0,str(ROOT/'development/validation/pro_prefill23'))
from evidence import endpoint

DT='Ascend950DT_9582';PR='Ascend950PR_957b';FUT='Ascend970DT'
OP='mhc_pre_sinkhorn';ATTRS={'num_iters':20,'need_backward':False}
HOST='build/smoke/extension_indexer_mhc'

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_bytes())
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2)+'\n');tmp.replace(p)
def state(label,**v):save(OUT/'state.json',dict(utc=now(),state=label,**v))

def forecast(shape,chip):
    target=load_chip(chip)
    before=old(shape,target,ATTRS);after=candidate(shape,target,ATTRS)
    return dict(v025_us=before[0],candidate_us=after[0],target_plan=after[-1]['target_plan'],stages_us=after[-1]['stages_us'],schedule=after[-1]['schedule'])

def register():
    if REG.exists():raise FileExistsError(REG)
    data=load(ROOT/'results/v0.25/coverage/audit.json')['rows']; jobs=[]
    for t in (72,113):
        shape=[t,4,4096]
        if any(r['op']==OP and r['shape']==shape for r in data):raise ValueError('duplicate shape')
        for sku in (PR,DT):
            id=f'fused-t{t}-'+('pr' if sku==PR else 'dt'); pattern=f'{OP}_{t}_4_4096_{sku}'
            if any(p.exists() for p in [RAW/id,ROOT/'runs/fusion21'/pattern,ROOT/'runs/sku17/extension'/pattern]):raise ValueError('prior raw')
            control=load(ROOT/f'results/v0.22/recovered/mhc_pre_sinkhorn_80_4_4096_{sku}.json')['observation']
            jobs.append(dict(id=id,sku=sku,op=OP,shape=shape,attrs=ATTRS,api='aclnnMhcPreSinkhorn',dtype='bf16',runner=HOST,args=[OP,str(t),'4','4096'],
                             kernels=1,nchecks=3,wall_s=2700,raw_mib=10240,kernel=control['kernel'],elf=control['binary_sha256'],
                             predictions={s:forecast(shape,s) for s in (DT,PR,FUT)}))
    sources=[Path(__file__),HERE/'prices.py',ROOT/'operator_model/mhc_pre_sink_resource.py',ROOT/'operator_model/mhc_sink_resource.py',OUT/'archive.json',OUT/'prices.json',
             ROOT/HOST,ROOT/'smoke/extension_indexer_mhc.cpp',ROOT/'results/v0.8/fusion/model.json',ROOT/'development/validation/pro_prefill23/evidence.py',
             ROOT/'development/validation/sku19/collect.py',ROOT/'development/validation/sku17/campaign_collect.py',ROOT/'scripts/cann-run.sh',
             *(ROOT/'configs/chips'/f'{s}.json' for s in (PR,DT,FUT))]
    save(REG,dict(utc=now(),jobs=jobs,hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},
        role='Two unseen token counts after freezing DT-only phase calibrations and the previous resource baseline. T72: two local post rows on each SKU. T113: two DT / three PR post rows. 970DT projection only.',
        budget='One process, nice +10, 16 logical CPUs, require load <0.70 of available logical CPUs and 25 GiB free. Per run 45 min / 10 GiB. No retries automatically.',
        gates='Expected SKU, memory model, one actual MIX ELF, three complete all-element goldens, official roster and duration table with all core instruction ends; fixed launch input attributes. No timing for missing absolute start/end.'))

def run():
    manifest=load(REG);RAW.mkdir(exist_ok=True)
    with (RAW/'exclusive.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for path,exp in manifest['hashes'].items():
            if sha(ROOT/path)!=exp:raise ValueError('frozen source has changed: '+path)
        allcpu=os.sched_getaffinity(0);available=len(allcpu)
        selected=sorted(allcpu)[-16:]
        os.sched_setaffinity(0,selected);os.nice(10);os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
        import shutil
        for j in manifest['jobs']:
            dest=OUT/(j['id']+'.json');folder=RAW/j['id']
            if dest.exists():
                if not load(dest)['qualified']:state('stopped at preserved failure',case=j['id']);return
                continue
            if not (folder/'cost.json').exists():
                if folder.exists():state('interrupted prior attempt preserved',case=j['id']);return
                while os.getloadavg()[0]>.70*available or shutil.disk_usage(ROOT).free<25*2**30:
                    state('waiting for resource headroom',case=j['id'],load_1m=os.getloadavg()[0],host_cpus=available);time.sleep(40)
                state('acquiring',case=j['id'],start=now(),cpus=selected);capture(j,j['sku'],folder)
            state('validating existing trace',case=j['id'])
            try:
                [s],checks=stages(j,j['sku'],folder)
                if s['kernel']!=j['kernel'] or s['binary_sha256']!=j['elf']:raise ValueError('ELF identity')
                t,n,d=j['shape']
                if [(m[1],int(m[2])) for c in checks if (m:=re.search(r'name=(\S+) count=(\d+) bad=0',c))]!=[('hin',t*d),('hpost',t*n),('hres',t*n*n)]:raise ValueError('golden coverage')
                if f'api={j["api"]} T={t} N=4 D=4096 dtype=BF16 phi_dtype=FP32 num_iters=20 need_backward=false' not in (folder/'run.log').read_text():raise ValueError('input attributes')
                shards={p.parent.name:p for p in (ROOT/s['trace']).parent.glob('core*/trace.json')}
                if set(s['windows'])!=set(shards):raise ValueError('core roster')
                offs={c:max(z['retire']for z in endpoint(p).values())-s['windows'][c][1]*1650 for c,p in shards.items()}
                if max(offs.values())-min(offs.values())>3:raise ValueError('actual retirement / official bounds differ')
                pred=j['predictions'][j['sku']]
                r=dict(qualified=True,stage=s,ca_us=s['ca_us'],correctness=checks,terminal_offset_spread_cycles=max(offs.values())-min(offs.values()),
                       measured_retire_offsets_cycles=offs,prediction=pred,absolute_relative_error={name:abs(pred[name]/s['ca_us']-1) for name in ('candidate_us','v025_us')})
            except (ValueError,KeyError,OSError,IndexError,AssertionError) as exc:r=dict(qualified=False,ca_us=None,reason=repr(exc))
            r.update(shape=j['shape'],chip=j['sku'],op=OP,role='post-freeze independent',cost=load(folder/'cost.json'),freeze_sha256=sha(REG));save(dest,r)
            if not r['qualified']:state('stopped at qualification failure',case=j['id']);return
        state('finished',cases=[j['id'] for j in manifest['jobs']])

if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
