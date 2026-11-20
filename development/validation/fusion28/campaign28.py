"""Four novel N4 geometries, paired cold/reuse, immutable single-slot forward collection."""
import fcntl
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from schedule28 import estimate,load_chip,OUT,ROOT
from operator_model import estimate as public
from development.validation.sku19.campaign import now
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'development/validation/sink26'))
import registration as prior
from collect import atomic,capture,qualify
from evidence import sha,read,DT,PR,FUT
REG=OUT/'register.json'
RAW=OUT/'raw'
GROUPS={'first':[(69,10)],'sequence':[(137,10),(69,10),(1153,6),(2257,4),(1153,6)]}

def register():
    if REG.exists() or RAW.exists():raise FileExistsError('registration/attempt already exists')
    # The original audit scans all versioned registries and raw host invocations,
    # including nonqualifying cases. Only its set of target geometries changes.
    prior.GROUPS=GROUPS
    history=prior.novelty()
    for row in read(ROOT/'results/v0.26/sink/register.json')['jobs']:
        t=str(row['shape'][0]);history.setdefault(t,[])
        if 'results/v0.26/sink/register.json' not in history[t]:history[t].append('results/v0.26/sink/register.json')
    targets={t for cases in GROUPS.values() for t,_ in cases}
    if any(str(t) in history for t in targets):raise ValueError('novelty failure in registries or host logs')
    atomic(OUT/'novelty.json',dict(matched=[],tokens=sorted(targets),known=history))
    register=dict(utc=now(),order=list(GROUPS),jobs=[],role='Four new geometries per SKU; each repeated case is one geometry and multiple independent-of-fitting windows. No runtime fitting.',
                  budget={'first':dict(wall_s=600,mib=1100),'sequence':dict(wall_s=1800,mib=5500)},
                  resource_policy='One live process at nice 15. New process requires one-minute load under 50% of schedulable CPUs, available memory >=96GiB, free disk >=120GiB. Older CA work keeps higher priority. No automatic retry of interrupted acquisition.',
                  eligibility='same provider MhcSinkhorn_8997... FP32 [T,4,4], released ELF text, exact scope, all elements golden, all physical AIV retirements and 64T output bytes, CPU and wall separate',
                  target='Score geometry-wise unknown-H WAPE/max on each SKU (25%/50%). Show known-H and unknown G1 vs multi-G; every single cycle-state repeat kept. Check first/steady issue and active cores. Fewer than ten distinct novel geometries cannot promote an all-shape default.')
    for group,seq in GROUPS.items():
        for chip in (DT,PR,FUT):
            h=0;profile=load_chip(chip)
            for i,(t,it) in enumerate(seq):
                s=[t,4];a=estimate(s,it,profile);k=estimate(s,it,profile,h,h);stage=next(iter(k['states'].values()))
                old=public('mhc_sinkhorn',s,{'num_iters':it},chip=chip)
                register['jobs'].append(dict(group=group,index=i,chip=chip,shape=s,iters=it,prior_h=h,
                  old_unknown_us=old.latency_us,old_status=old.status,old_known_us=stage['legacy_known_us'],point_us=a['point_us'],
                  range_us=a['range_us'],relative_minimax_pct=a['model_uncertainty_pct'],mean_us=a['mean_us'],harmonic_us=a['harmonic_us'],
                  conditional=stage,status='architecture extrapolation' if chip==FUT else 'prospective whole CA'))
                h=max(h,stage['active'])
    sources=[HERE/x for x in ('landmarks.py','calibrate.py','schedule28.py','campaign28.py')]
    sources.extend([OUT/'prices.json',OUT/'novelty.json',ROOT/'operator_model/sink_history.py',ROOT/'results/v0.26/sink/prices.json',prior.BINARY,prior.VENDOR,prior.IMPL,ROOT/'scripts/cann-run.sh',ROOT/'smoke/vector_v8.cpp',ROOT/'smoke/extension_indexer_mhc.cpp',ROOT/'development/validation/rms_ordered_verify/verify.py',ROOT/'development/validation/sink26/collect.py',ROOT/'development/validation/sink26/registration.py'])
    sources.extend(ROOT/'configs/chips'/f'{s}.json' for s in (DT,PR,FUT))
    register['source_sha256']={str(p.relative_to(ROOT)):sha(p) for p in sources}
    with REG.open('x') as f:json.dump(register,f,indent=2,allow_nan=False);f.write('\n')

def run():
    reg=read(REG);assert all(sha(ROOT/p)==s for p,s in reg['source_sha256'].items())
    RAW.mkdir(exist_ok=True);(OUT/'receipts').mkdir(exist_ok=True)
    def status(state,**fields):atomic(OUT/'state.json',dict(state=state,utc=now(),registration_sha256=sha(REG),**fields))
    with (RAW/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for group in reg['order']:
            for chip in (DT,PR):
                label=group+'-'+chip
                d=RAW/label;dest=OUT/'receipts'/(label+'.json')
                if dest.exists():continue
                jobs=[j for j in reg['jobs'] if j['group']==group and j['chip']==chip]
                if not (d/'cost.json').exists():
                    if d.exists():status('interrupted attempt preserved',case=label);return
                    while True:
                        load=os.getloadavg()[0]
                        ram=int(re.search(r'^MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text(),re.M)[1])/2**20
                        if load < .5*len(os.sched_getaffinity(0)) and shutil.disk_usage(ROOT).free>120*2**30 and ram>96:break
                        status('waiting for headroom',case=label,load=load,available_gib=ram);time.sleep(45)
                    status('acquiring',case=label);capture(jobs,d,reg['budget'][group])
                cost=read(d/'cost.json');status('verifying',case=label)
                out=dict(chip=chip,group=group,cost=cost,rows=[],register_sha256=sha(REG))
                try:
                    assert cost['started_utc']>reg['utc']
                    out['rows']=qualify(jobs,d,cost)
                except (OSError,AssertionError,ValueError,TypeError,IndexError,KeyError) as exc:out['reason']=repr(exc)
                atomic(dest,out);print(label,sum(x['qualified'] for x in out['rows']),flush=True)
                from score28 import score
                score()
        status('finished')
if __name__=='__main__':
    {'freeze':register,'collect':run}[sys.argv[1]]()
