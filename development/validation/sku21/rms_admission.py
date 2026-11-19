"""Run the frozen standalone RMS queue on the first released pair of CA slots."""
import fcntl
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import CHIPS,read,now,write
from development.validation.sku21.rms_campaign import run


def main():
    guard_path=ROOT/'runs/.rms21-admission.lock'
    with guard_path.open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        total=2*len(read(ROOT/'results/v0.21/dispatch/freeze.json')['jobs'])
        previous=ROOT/'runs/sku20-rms'
        while True:
            ended=all((previous/('rmsnorm_32768_1280_'+CHIPS[s])/'outcome.json').is_file() for s in ('dt','pr'))
            if ended:
                trigger='both wide RMS processes finished'
                break
            if len(list((ROOT/'results/v0.21/dispatch/receipts').glob('*.json')))==total:
                trigger='original 40 Addmm/Post captures finished; one of the released slots'
                break
            time.sleep(30)
        write(ROOT/'results/v0.21/rms/admission.json',dict(utc=now(),trigger=trigger,
            concurrency=1,freeze='results/v0.21/rms/freeze.json'))
        run()


if __name__=='__main__':
    main()
