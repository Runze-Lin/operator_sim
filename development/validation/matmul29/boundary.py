"""One new 3585-row numerical and timing test per SKU after the original prefill slot."""
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time
from validate import ROOT, OUT, REG, DT, PR, PARENT, read, save, digest, scoring
from development.validation.matmul28 import campaign
from development.validation.matmul28.drain import recover
from development.validation.matmul28.multi.verify import geometry


def collect():
    reg=read(REG); assert all(digest(ROOT/p)==h for p,h in reg['sources'].items())
    job,=[r for r in reg['rows'] if r['group']=='boundary']
    job={**job,'predictions':{c:{'us':p['latency_us']} for c,p in job['predictions'].items()},'wall_s':1500,'raw_mib':2048}
    raw=ROOT/'runs/matmul29-boundary';raw.mkdir(exist_ok=True)
    def state(phase,**data):save(OUT/'boundary-state.json',dict(phase=phase,utc=campaign.now(),pid=os.getpid(),**data))
    os.nice(8);os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,cpus[len(cpus)//2:][:96])
    with (raw/'worker.lock').open('a') as own, (ROOT/'runs/matmul28/prefill.lock').open('a') as slot:
        fcntl.flock(own,fcntl.LOCK_EX | fcntl.LOCK_NB)
        state('waiting for original prefill slot');fcntl.flock(slot,fcntl.LOCK_EX)
        if read(PARENT/'prefill-status.json')['phase']!='finished':
            state('original prefill exited before completion');return
        for c,tag in ((DT,'dt'),(PR,'pr')):
            slug='-'.join(map(str,job['shape']))+'-'+tag;folder=raw/slug;dst=OUT/(slug+'.json')
            if dst.exists():continue
            if folder.exists() and not (folder/'cost.json').exists():state('interrupted raw retained',case=slug);return
            if not folder.exists():
                while shutil.disk_usage(ROOT).free<20*2**30:state('waiting for storage',case=slug);time.sleep(60)
                folder.mkdir();state('acquiring',case=slug);campaign.measure(job,c,folder)
            cost=read(folder/'cost.json');assert cost['started_utc']>reg['utc']
            try:initial=campaign.qualify(job,c,folder)
            except (ValueError,KeyError,AssertionError,OSError,IndexError) as e:initial=dict(qualified=False,reason=repr(e))
            source=OUT/(slug+'-initial.json')
            acq=dict(initial,case=slug,chip=c,shape=job['shape'],cost=cost)
            save(source,acq)
            try:record=recover(job,acq,folder,acquisition=source) if not acq['qualified'] else acq
            except (ValueError,KeyError,AssertionError,OSError,IndexError) as e:record=dict(qualified=False,reason=repr(e))
            record=dict(record or acq,chip=c,shape=job['shape'],cost=cost,register_sha256=digest(REG),initial_sha256=digest(source))
            if record.get('qualified'):
                st,=record['stages'];assert st['binary_sha256']==reg['provider_sha256']
                if 'mix_aic' in st['kernel']:
                    try:record['observed_geometry']=geometry(st,job['shape'])
                    except (ValueError,KeyError,AssertionError,OSError,IndexError) as e:record['geometry_unresolved']=repr(e)
            save(dst,record);scoring()
        state('finished')

if __name__=='__main__':collect()
