"""Authorize the already frozen T91 pair after a supplemental main geometry failed its endpoint gate."""
import fcntl
import os
import sys
import time
from pathlib import Path
from resume import work
from admission import ALL,POLICY,ticks,capacity,set_limits
from observe import OUT,ORIG,RAW,ROOT,read,save,utc

ADD=OUT/'reserve-activation.json'
def register():
    if ADD.exists():raise FileExistsError(ADD)
    reg=read(work.REG);jobs=[j for j in reg['jobs'] if j['cohort'].startswith('reserved')]
    if len(jobs)!=2 or any(j['shape']!=[91,4,4096] for j in jobs):raise ValueError('wrong reserve')
    if any((d/j['id']).exists() or (work.HOLD/(j['id']+'.json')).exists() for j in jobs for d in (RAW,work.RAW)):
        raise ValueError('reserve already started')
    failure=work.HOLD/'fused-t127-dt.json'
    if read(failure)['qualified'] or read(OUT/'state.json')['state']!='finished':raise ValueError('no terminal failure')
    save(ADD,dict(utc=utc(),original_registration_sha256=work.orig.sha(work.REG),failure_sha256=work.orig.sha(failure),
         source_sha256=work.orig.sha(Path(__file__)),jobs=jobs,
         reason='T127 DT failed the endpoint comparison and is retained. The original condition enabled T91 only after a failure among T72/T113, so the new-main failure did not activate it. Before any T91 CA existed, extend only activation to that failure. Both SKU forecasts and the exact host, ELF, outputs and window gates were preregistered, and remain unmodified.',
         resources='Original single slot, 120 seconds of headroom, 45 min/10 GiB per acquisition, within original aggregate ceiling.'))
def run():
    reg=read(work.REG);add=read(ADD)
    if work.orig.sha(Path(__file__))!=add['source_sha256'] or work.orig.sha(work.REG)!=add['original_registration_sha256']:raise ValueError('activation changed')
    for name,digest in {**reg['source_sha256'],**reg['original_sources']}.items():
        if work.orig.sha(ROOT/name)!=digest:raise ValueError('frozen dependency changed '+name)
    with (RAW/'exclusive.lock').open('a') as fd:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
        for j in add['jobs']:
            case=j['id'];p=work.RAW/case;receipt=work.HOLD/(case+'.json')
            if receipt.exists():continue
            if not (p/'cost.json').exists():
                if p.exists() or (RAW/case).exists():raise ValueError('partial raw preserved: '+case)
                costs=[read(c) for c in work.RAW.glob('*/cost.json')]
                if reg['aggregate_seconds']-sum(c['seconds'] for c in costs)<j['wall_s'] or reg['aggregate_raw_gib']*1024-sum(c['raw_bytes']/2**20 for c in costs)<j['raw_mib']:
                    work.status('reserve budget exhausted',id=case);return
                os.sched_setaffinity(0,ALL);before=ticks();start=None
                while True:
                    time.sleep(read(POLICY)['sample_s']);after=ticks();mask,info=capacity(before,after);before=after
                    if info['reasons']:start=None
                    elif start is None:start=time.monotonic()
                    good=0 if start is None else time.monotonic()-start
                    work.status('reserve waiting for headroom',id=case,good_seconds=round(good),**info)
                    if good>=read(POLICY)['continuous_new_s']:break
                set_limits(0,mask);work.status('reserve acquiring',id=case,affinity=mask,**info);work.orig.capture(j,j['sku'],p)
            work.status('qualifying reserve',id=case)
            cost=read(p/'cost.json')
            if cost['started_utc']<=add['utc']:raise ValueError('not prospective')
            try:r=work.qualify(j,p)
            except (ValueError,KeyError,AssertionError,OSError,TypeError,IndexError) as e:
                r=dict(qualified=False,ca_us=None,shape=j['shape'],chip=j['sku'],reason=repr(e))
            save(receipt,{**r,'id':case,'cohort':j['cohort'],'cost':cost,'registration_sha256':work.orig.sha(work.REG),
                          'activation_sha256':work.orig.sha(ADD)})
        work.status('finished',reserve_activation_sha256=work.orig.sha(ADD))
if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
