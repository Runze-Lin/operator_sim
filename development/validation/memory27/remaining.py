"""Run each untouched 156-MiB case under its original data and timing contract."""
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.27/memory/remaining'
sys.path[:0]=[str(ROOT/'development/validation/cache25'),str(ROOT)]
from development.validation.cache25 import campaign
from forward import score


def main():
    OUT.mkdir(exist_ok=True)
    original=campaign.REG
    contract=campaign.read(original)
    case='cast_bf16_fp32-pr'
    failed=campaign.OUT/(case+'-result.json')
    assert not campaign.read(failed)['qualified'] and 'missing END' in campaign.read(failed)['reason']
    receipts={j['id']:campaign.OUT/(j['id']+'-result.json') for j in contract['jobs']}
    host=ROOT/'runs/memory27-followup'
    host.mkdir(exist_ok=True)
    with (host/'remaining.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.nice(10)
        # Original Pro, Kimi and Flash jobs retain first claim on the machine.
        for j in contract['jobs']:
            if j['id']==case or receipts[j['id']].exists():
                continue
            folder=campaign.RAW/j['id']
            if folder.exists() and not (folder/'cost.json').exists():
                raise RuntimeError('existing partial attempt: '+str(folder))
            reg=OUT/(j['id']+'-registration.json')
            if not reg.exists():
                data=dict(contract,jobs=[j],original_freeze_sha256=campaign.digest(original),
                          earlier_failure_sha256=sha256(failed.read_bytes()).hexdigest(),
                          activation_utc=datetime.now(timezone.utc).isoformat(),
                          scope='Previously registered untouched job. Single slot and original per-case budgets, qualification and predictions. Prior failed receipt is immutable.')
                campaign.save(reg,data)
            expected=campaign.read(reg)
            assert expected['jobs']==[j] and expected['sha256']==contract['sha256']
            assert expected['original_freeze_sha256']==campaign.digest(original)
            campaign.REG=reg
            while os.getloadavg()[0]>len(os.sched_getaffinity(0))*.7:
                time.sleep(30)
            campaign.run()
            if not receipts[j['id']].exists():
                raise RuntimeError('no terminal receipt: '+j['id'])
            score()
        campaign.save(OUT/'summary.json',{j:campaign.read(p)['qualified'] for j,p in receipts.items() if p.exists()})


if __name__=='__main__':
    main()
