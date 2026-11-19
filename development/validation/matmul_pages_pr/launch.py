"""One PR CA attempt. Persistent execution, immutable prelaunch contract."""
from datetime import datetime,timezone
from pathlib import Path
import json
import os
import subprocess
import sys
from freeze import ROOT, OUT, PR, digest

sys.path.insert(0,str(ROOT/'development/validation/sku17'))
from campaign_collect import capture,qualify


def write(name,data):
    p=OUT/(name+'.json');tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p)


if __name__=='__main__':
    reg=json.loads((OUT/'freeze.json').read_text());job=reg['job']
    for path,fingerprint in reg['sources'].items():
        assert digest(ROOT/path)==fingerprint,path
    slot=ROOT/'results/v0.21/chunk/kimi-k3-bf16g-64x12-dt-compact-Ascend950DT_9582.json'
    previous=json.loads(slot.read_text())
    assert previous.get('exit') is not None or previous.get('stop'),previous
    raw=OUT/'raw'
    if raw.exists() or (OUT/'admission.json').exists():
        raise FileExistsError('one launch maximum')
    with (OUT/'admission.json').open('x') as f:
        f.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
            prior_completed_slot=slot.relative_to(ROOT).as_posix(),prior_receipt_sha256=digest(slot),
            load_1_5_15=os.getloadavg(),freeze_sha256=digest(OUT/'freeze.json'),launches=1),indent=2)+'\n')
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    try:
        cost=capture(job,PR,raw,reg)
        (obs,)=qualify(job,PR,raw)
        if not any(f'elements={job["shape"][0]*job["shape"][1]} bad=0' in c for c in obs['correctness']):
            raise ValueError('expected numerical comparison over every M×N output')
        write('observation',dict(qualified=True,observation=obs,freeze_sha256=digest(OUT/'freeze.json')))
    except Exception as exc:
        write('observation',dict(qualified=False,reason=repr(exc),cost=json.loads((raw/'cost.json').read_text()) if (raw/'cost.json').exists() else None))
        raise
    decoder=Path(__file__).with_name('publish.py')
    subprocess.run([str(ROOT/'toolchains/envs/cann-9.2.1/bin/python'),str(decoder)],cwd=ROOT,check=True)
