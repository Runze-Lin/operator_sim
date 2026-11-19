"""Publish atomic measurement snapshots while independent captures and retries run."""
from pathlib import Path
import time
from score import OUT, report, read

if __name__=='__main__':
    previous=None
    while True:
        n=(len(list((OUT/'receipts').glob('*.json'))),len(list((OUT/'retries').glob('*.json'))))
        if n!=previous:
            report();previous=n
        reg=OUT/'retry-registration.json'
        if n[0]==60 and reg.is_file() and len(read(reg)['cases'])==n[1]:
            break
        time.sleep(20)
