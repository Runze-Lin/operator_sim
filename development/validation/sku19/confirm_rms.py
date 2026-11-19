"""An independent, paired second freeze after the first RMS ranking still failed."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import CHIPS,DT,PR,HOST,OUT,forecast,forecast_old,records,read,write,digest,now
from development.validation.sku19.collect import capture, qualify, make_pair, summarize

DEST=OUT.parent/'rms-confirmation'
RAW=ROOT/'runs/sku19-rms-confirmation'
SHAPES=[(3,6080),(7,7360),(18,4992),(38,2624),(57,2304),(69,3328),(103,2944),(229,3968),(367,5312),(495,6912)]
BUDGET_GIB=18
LOCK=threading.Lock()


def register():
    assert not (DEST/'registration.json').exists() and not RAW.exists()
    known={tuple(s) for s in read(ROOT/'results/v0.18/sku/excluded-history.json')['shapes']['rmsnorm']}
    for version in ('v0.18','v0.19'):
        for path in (ROOT/'results'/version).rglob('*.json'):
            for op,shape in records(read(path)):
                if op=='rmsnorm':known.add(shape)
    assert len(set(SHAPES))==10 and not known.intersection(SHAPES)
    caps=[.4,.5,.6,.65,.8,.8,.9,1.4,2.6,4.]
    jobs=[]
    for (r,c),cap in zip(SHAPES,caps):
        runner=HOST['rmsnorm']
        jobs.append(dict(id=f'rmsnorm_{r}_{c}',op='rmsnorm',shape=[r,c],args=['rmsnorm',str(r),str(c)],
            attrs={},dtype='bf16',role='holdout',regime='small' if r<32 else 'boundary' if r<=128 else 'large',
            runner=runner,runner_sha256=digest(ROOT/runner),wall_s=660 if r>=200 else 240,raw_mib=int(cap*1024),kernels=1,nchecks=2,batch=None))
    write(DEST/'registration.json',dict(utc=now(),jobs=jobs,known_shape_count=len(known),concurrency=3,total_retained_gib=BUDGET_GIB,
        purpose='Second RMS cross-chip direction holdout after the first v0.19 campaign. Only pre-v0.19 traces allowed for stage price estimation. All release costs retained.'))


def seal():
    assert not (DEST/'freeze.json').exists() and not RAW.exists()
    jobs=read(DEST/'registration.json')['jobs']
    new=forecast(jobs)
    old=forecast_old(jobs)
    refs=[ROOT/'operator_model'/k for k in ('rms_resource.py','rms_target_cost.py','backend.py','rms_v12.py','chips.py')]
    refs += list((ROOT/'configs/chips').glob('*.json')) + [Path(__file__),ROOT/HOST['rmsnorm'],DEST/'registration.json']
    for job,a,b in zip(jobs,new['rows'],old['rows']):
        assert a['id']==b['id']==job['id']
        o=dict(op=job['op'],shape=job['shape'],attrs={},indices=[0])
        for side in CHIPS:
            for prefix,source in (('',a),('old_',b)):
                val=source['targets'][side]
                assert val['us'] and val['query_ms']<100 and val['aiv']>0
                for key in ('us','aic','aiv','status','query_ms'):
                    o[prefix+side+'_'+key]=val[key]
        job['outputs']=[o]
    write(DEST/'freeze.json',dict(utc=now(),refs={str(p.relative_to(ROOT)):digest(p) for p in refs},jobs=jobs,v018=old,v019_final=new))


def archive(jobs):
    events=[];qualified=[];pairs=[]
    for j in jobs:
        both=[]
        for chip in (DT,PR):
            folder=RAW/(j['id']+'_'+chip)
            if (folder/'outcome.json').exists():events.append(read(folder/'outcome.json'))
            if (folder/'qualified.json').exists():
                r=read(folder/'qualified.json');both.append(r);qualified.append(r)
        if len(both)==2:
            pair=make_pair(j,*both);pair['sample_role']='holdout';pairs.append(pair)
    write(DEST/'pairs.json',pairs);write(DEST/'records.json',qualified);write(DEST/'costs.json',events)
    if events:write(DEST/'completion.json',dict(**summarize(events),paired=len(pairs)))


def run():
    frozen=read(DEST/'freeze.json')
    assert all(digest(ROOT/p)==sha for p,sha in frozen['refs'].items())
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    def one(pair):
        j,chip=pair;folder=RAW/(j['id']+'_'+chip)
        if (folder/'outcome.json').exists():return
        with LOCK:
            spent=sum(r['raw_bytes'] for r in (read(p) for p in RAW.glob('*/cost.json')))
            if spent > BUDGET_GIB*2**30:raise RuntimeError('confirmation storage cap reached')
        cost=capture(j,chip,folder)
        result=dict(id=j['id'],chip=chip,op=j['op'],shape=j['shape'],**cost)
        try:
            r,=qualify(j,chip,folder)
            contract=[s for s in (folder/'run.log').read_text().splitlines() if s.startswith('case=')]
            assert len(contract)==1 and cost['started_utc']>frozen['utc']
            r.update(backend='official paired second RMS freeze, dump=false',dtype='bf16',runner_sha256=j['runner_sha256'],
                     input_contract_log=contract,host_args=j['args'],run_log=str((folder/'run.log').relative_to(ROOT)))
            result.update(eligible=True,reason=None)
            write(folder/'qualified.json',r)
        except (ValueError,AssertionError,KeyError,OSError,IndexError) as error:
            result.update(eligible=False,reason=repr(error))
        with LOCK:
            write(folder/'outcome.json',result);archive(frozen['jobs'])
            print(j['id'],chip,result['reason'] or 'qualified',flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(one,((j,c) for j in frozen['jobs'] for c in (DT,PR))))

if __name__=='__main__':
    {'register':register,'freeze':seal,'run':run}[sys.argv[1]]()
