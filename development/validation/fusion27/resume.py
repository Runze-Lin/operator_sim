"""Resume a completed raw trace after the legacy standalone importer lost its scorer namespace."""
from importlib import util
import fcntl
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
spec=util.spec_from_file_location('fusion27_campaign',HERE/'campaign.py')
work=util.module_from_spec(spec)
spec.loader.exec_module(work)
from observe import ROOT,OUT,ORIG,RAW as PRIOR,save,read,utc
from admission import ALL,POLICY,ticks,capacity,set_limits

ADD=OUT/'scoring-resume.json'

def register():
    if ADD.exists(): raise FileExistsError(ADD)
    case='fused-t72-pr';cost=read(PRIOR/case/'cost.json')
    assert (cost['exit_code'],cost['stop_reason'])==(0,None) and not (ORIG/(case+'.json')).exists()
    assert read(ORIG/'state.json')['state']=='validating existing trace'
    original=ROOT/'development/validation/presink26/campaign.py'
    save(ADD,dict(utc=utc(),file_sha256=work.orig.sha(Path(__file__)),registration_sha256=work.orig.sha(work.REG),
         original_phase_source_sha256=work.orig.sha(original),first_cost_sha256=work.orig.sha(PRIOR/case/'cost.json'),
         correction='The original resume imports campaign_collect through a partially initialized module named campaign. Its stages globals lack read, math, PR and sha. The parent exited after cost exit=0 and before scoring. The same scorer loaded via the v0.27 namespace has all four. Re-open completed raw under the original exclusive lock; never re-run or change price, budget, golden, identity, or all-core retirement conditions.',
         original_exit_state=read(ORIG/'state.json'),initial_continuation_exit=read(OUT/'state.json')))

def run():
    a=read(ADD);reg=read(work.REG)
    if a['file_sha256']!=work.orig.sha(Path(__file__)) or a['registration_sha256']!=work.orig.sha(work.REG):
        raise ValueError('addendum changed')
    for name,expected in {**reg['source_sha256'],**reg['original_sources']}.items():
        if work.orig.sha(ROOT/name)!=expected:raise ValueError('frozen dependency changed: '+name)
    assert all(n in work.orig.stages.__globals__ for n in ('read','PR','sha','math'))
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    with (PRIOR/'exclusive.lock').open('a') as fd:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        failure=False
        for j in reg['jobs']:
            if j['cohort'].startswith('reserved') and not failure: continue
            case=j['id'];out=work.HOLD/(case+'.json');old=ORIG/(case+'.json');is_old=j['cohort']=='v0.26 original'
            if out.exists() and old.exists():raise ValueError('two receipts for '+case)
            if out.exists() or old.exists():
                rr=read(out if out.exists() else old)
                if is_old and not rr['qualified']:failure=True
                continue
            prior=PRIOR/case;future=work.RAW/case
            if prior.exists() and future.exists():raise ValueError('two raw invocations for '+case)
            folder=prior if prior.exists() else future
            if not (folder/'cost.json').exists():
                if folder.exists():
                    work.status('unscored interrupted raw preserved',id=case);return
                costs=[read(p) for p in work.RAW.glob('*/cost.json')]
                remain=reg['aggregate_seconds']-sum(x['seconds'] for x in costs)
                left=reg['aggregate_raw_gib']*1024-sum(x['raw_bytes']/2**20 for x in costs)
                if remain<j['wall_s'] or left<j['raw_mib']:
                    work.status('aggregate budget exhausted',id=case);return
                os.sched_setaffinity(0,ALL);policy=read(POLICY);t0=None;prev=ticks()
                while True:
                    time.sleep(policy['sample_s']);cur=ticks();mask,info=capacity(prev,cur);prev=cur
                    if info['reasons']:t0=None
                    elif t0 is None:t0=time.monotonic()
                    good=time.monotonic()-t0 if t0 is not None else 0
                    work.status('waiting for older tasks or capacity',id=case,continuous_good_seconds=round(good),**info)
                    if good>=policy['continuous_new_s']:break
                set_limits(0,mask)
                work.status('acquiring',id=case,cpus=mask,**info);work.orig.capture(j,j['sku'],folder)
            work.status('qualifying existing raw',id=case,raw=str(folder.relative_to(ROOT)))
            cost=read(folder/'cost.json')
            cutoff=read(ORIG/'freeze.json')['utc'] if is_old else reg['utc']
            if cost['started_utc']<=cutoff:raise ValueError('not a prospective observation')
            try: row=work.qualify(j,folder)
            except (ValueError,KeyError,AssertionError,OSError,TypeError,IndexError) as e:
                row=dict(qualified=False,ca_us=None,shape=j['shape'],chip=j['sku'],reason=repr(e))
            save(out,{**row,'id':case,'cohort':j['cohort'],'cost':cost,'registration_sha256':work.orig.sha(work.REG),
                      'applicable_predate_registration_sha256':reg['original_sha256'] if is_old else work.orig.sha(work.REG),
                      'scorer_addendum_sha256':work.orig.sha(ADD)})
            if is_old and not row['qualified']:failure=True
        work.status('finished')
if __name__=='__main__':
    {'register':register,'run':run}[sys.argv[1]]()
