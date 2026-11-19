"""Retry unsuccessful frozen router jobs with unchanged predictions and kernel contract."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from router_validate import ROOT,OUT,RAW,capture,stages,digest,write

base=OUT/'router-holdout-freeze.json'
original=OUT/'router-holdout.json'
registration=OUT/'router-recovery-freeze.json'


def register():
    if registration.exists():
        raise ValueError('Recovery registration already exists')
    freeze=json.loads(base.read_text())
    jobs={tuple(j['shape']):j for j in freeze['jobs']}
    failed=[r for r in json.loads(original.read_text()) if not r['qualified']]
    previously_recovered=json.loads((OUT/'router-window-recovery.json').read_text())['recovered']
    qualified={(r['sku'],tuple(r['shape'])) for r in previously_recovered}
    failed=[r for r in failed if (r['sku'],tuple(r['shape'])) not in qualified]
    entries=[]
    for r in failed:
        j=dict(jobs[tuple(r['shape'])])
        j.update(wall_s=480,raw_mib=max(512,j['raw_mib']))
        entries.append(dict(job=j,chip=r['sku'],original_reason=r['reason']))
    write(registration,dict(utc=datetime.now(timezone.utc).isoformat(),
                            prior_sha256=digest(original),freeze_sha256=digest(base),jobs=entries,
                            max_workers=1,description='Retry identical shape, executable, and inputs; original costs retained.'))


def collect():
    reg=json.loads(registration.read_text())
    assert digest(base)==reg['freeze_sha256'] and digest(original)==reg['prior_sha256']
    rows=[]
    for pair in reg['jobs']:
        j,chip=pair['job'],pair['chip']
        d=RAW/(j['id']+'_'+chip+'-retry')
        cost=json.loads((d/'cost.json').read_text()) if (d/'cost.json').exists() else capture(j,chip,d,reg)
        row=dict(shape=j['shape'],sku=chip,cost=cost,role='same frozen holdout contract retry')
        try:
            ks,checks=stages(j,chip,d)
            row.update(qualified=True,correctness=checks,kernel=ks[0],predictions=j['predictions'])
        except (ValueError,IndexError,AssertionError) as err:
            row.update(qualified=False,reason=str(err))
        rows.append(row)
        write(OUT/'router-recovery.json',rows)
        print(j['id'],chip,row['qualified'],flush=True)

if __name__=='__main__':
    if sys.argv[1:] == ['register']:
        register()
    elif sys.argv[1:]==['collect']:
        collect()
    else: raise SystemExit('expected register or collect')
