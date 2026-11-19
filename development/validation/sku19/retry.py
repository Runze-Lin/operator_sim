"""Retry only frozen contracts whose original trace cannot establish the all-core window."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import OUT, DT, PR, read, write, digest, now
from development.validation.sku19.collect import capture,qualify
from development.validation.sku19.recover_windows import parse

REG=OUT/'retry-registration.json'
DIR=ROOT/'runs/sku19-retry'


def register():
    assert not REG.exists()
    freeze=read(OUT/'freeze.json')
    initial=read(OUT/'costs.json')
    assert len(initial)==44
    have={(r['case_id'],r['chip']) for r in read(OUT/'final-records.json')}
    jobs=[]
    for j in freeze['jobs']:
        for chip in (DT,PR):
            if (j['id'],chip) in have:continue
            if (j['id'],chip)==('rmsnorm_2048_5120',PR) and (OUT.parent/'prefill/chunk-retry-registration.json').exists():
                continue  # independently registered single retry of the large PR chunk
            previous=next(r for r in initial if (r['id'],r['chip'])==(j['id'],chip))
            # Give real numerical readback and vendor export time, with fewer
            # concurrent workers. Preserve the original executable and inputs.
            job=dict(j,wall_s=min(2700,max(600,2*j['wall_s'])))
            jobs.append(dict(chip=chip,job=job,previous_reason=previous['reason']))
    write(REG,dict(utc=now(),jobs=jobs,freeze_sha256=digest(OUT/'freeze.json'),
          policy='Same shape and predictions; one sequential CA at a time, passive OpenMP waits, original raw cap. Only after complete-window recovery failed.'))
    print('same-contract retries',len(jobs))


def run():
    reg=read(REG)
    assert reg['freeze_sha256']==digest(OUT/'freeze.json')
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    for x in reg['jobs']:
        j=x['job'];chip=x['chip'];folder=DIR/(j['id']+'_'+chip)
        if (folder/'outcome.json').exists():continue
        cost=capture(j,chip,folder)
        outcome=dict(id=j['id'],chip=chip,op=j['op'],shape=j['shape'],**cost)
        try:
            record,=qualify(j,chip,folder)
            record.update(backend='official full-window retry',dtype='bf16',host_args=j['args'],runner_sha256=j['runner_sha256'])
        except (AssertionError,IndexError,KeyError,OSError,ValueError) as err:
            try:record=parse(j,chip,folder)
            except (AssertionError,IndexError,KeyError,OSError,ValueError) as inner:
                record=None
                outcome.update(eligible=False,reason=str(err)+' / '+str(inner))
        if record:
            record['run_log']=str((folder/'run.log').relative_to(ROOT))
            write(folder/'qualified.json',record)
            outcome.update(eligible=True,reason=None)
        write(folder/'outcome.json',outcome)
        print(j['id'],chip,outcome['eligible'],flush=True)


if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
