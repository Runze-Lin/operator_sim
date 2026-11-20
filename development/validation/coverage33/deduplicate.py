"""Register two skipped launches. Preserve pre-existing v0.29 source/freeze and raw."""
from hashlib import sha256
from pathlib import Path
import json
import os
import tempfile
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
RMS=ROOT/'results/v0.33/large'
OUT=ROOT/'results/v0.29/large/vector'
RAW=ROOT/'runs/large29/vector'
FREEZE=OUT/'freeze.json'
DT,PR='Ascend950DT_9582','Ascend950PR_957b'

def read(path):
    return json.loads(path.read_bytes())

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def disposition(job,external):
    r=read(external); c=job['chip']
    require=(r['qualified'] is True and r['role']=='revalidated archive replay' and
             r['shape']==job['shape']==[8192,5120] and r['chip']==c and c in (DT,PR) and
             job['op']=='rmsnorm' and job['attrs']=={} and job['protocol']=='instruction' and
             job['runner']=='build/smoke/large_vector' and job['args']==['rmsnorm','8192','5120'] and
             r['api']=='aclnnRmsNorm' and r['golden']==['correctness count=41943040 bad=0','correctness count=8192 bad=0'])
    if not require or r['registration_sha256']!=digest(RMS/'freeze.json'):
        raise ValueError('Incorrect archived evidence or stale registration')
    if (RAW/job['case']).exists() or read(OUT/'state.json').get('case')==job['case']:
        raise ValueError('Physical acquisition started or waiting on this case')
    # Keep standalone timing empty so neither old scorer will treat it as a
    # v0.29 observation. Exact complete elapsed time belongs to the hashed evidence.
    return dict(case=job['case'],op=job['op'],shape=job['shape'],chip=c,protocol='instruction',
        status='duplicate_archive_skip',qualified=False,physical_launch=False,instruction_us=None,wrapper_us=None,
        ca_us=None,cost=None,reason='Same runner, shape, attrs, SKU and kernel already have a complete archival device window. No launch, no new independent timing.',
        role='deduplication only; reference is historical replay',freeze_sha256=digest(FREEZE),
        registered_prediction_us=job['prices'][c]['public_us'],
        external_qualified_receipt=str(external.relative_to(ROOT)),external_receipt_sha256=digest(external),
        external_trace_sha256=r['trace_sha256'],issued_utc=datetime.now(timezone.utc).isoformat())

def publish():
    reg=read(FREEZE)
    jobs=[j for j in reg['jobs'] if j['shape']==[8192,5120] and j['op']=='rmsnorm' and j['protocol']=='instruction']
    if len(jobs)!=2 or {j['chip'] for j in jobs}!={DT,PR}:
        raise ValueError('Wrong frozen slots')
    for j in jobs:
        tag='dt' if j['chip']==DT else 'pr';external=RMS/f'rms-8192-5120-{tag}.json';dest=OUT/(j['case']+'.json')
        record=disposition(j,external)
        if dest.exists():
            old=read(dest)
            if any(old.get(k)!=v for k,v in record.items() if k!='issued_utc'):
                raise FileExistsError(dest)
            continue
        with tempfile.NamedTemporaryFile(mode='w',dir=OUT,delete=False) as f:
            temp=Path(f.name)
            json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
        try:
            os.link(temp,dest)  # Never replace somebody else's completed receipt.
        finally:
            temp.unlink()
        print(dest.relative_to(ROOT))
if __name__=='__main__': publish()
