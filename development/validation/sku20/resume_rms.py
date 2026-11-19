"""Finish the v0.20 RMS cohort if its original collector leaves during a long capture."""
import fcntl
import os
from pathlib import Path
import time

from rms_campaign import OUT, RAW, read, run, write
from development.validation.large20.recover_cost import recover_cost


def live_collector():
    script=Path(__file__).with_name('rms_campaign.py').resolve()
    for proc in Path('/proc').iterdir():
        try:
            if not proc.name.isdigit() or proc.stat().st_uid!=os.getuid():
                continue
            args=(proc/'cmdline').read_bytes().split(b'\0')
            if len(args)>=3 and args[2]==b'run' and args[1].endswith(b'rms_campaign.py'):
                if ((proc/'cwd').resolve()/os.fsdecode(args[1])).resolve()==script:
                    return True
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            continue
    return False


def resume():
    with (RAW/'.recovery.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while live_collector():
            time.sleep(45)
        jobs=read(OUT/'freeze.json')['jobs']
        for job in jobs:
            for folder in RAW.glob(job['id']+'_Ascend*'):
                if (folder/'run.log').exists() and not (folder/'cost.json').exists():
                    cost=recover_cost(folder,job['wall_s'],job['raw_mib']*2**20)
                    write(folder/'cost.json',cost)
        run()   # Original qualification and freeze; skips every prior terminal outcome.


if __name__=='__main__':
    resume()
