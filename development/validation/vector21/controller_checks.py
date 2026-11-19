"""Retain existing DT controller logs for the four long registered jobs only."""
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.vector21.campaign import C,DT,OUT,RAW,read,write
sys.path.insert(0,str(ROOT/'development/validation/large20'))
from dram import counters


def monitor():
    jobs=[j for j in read(OUT/'freeze.json')['jobs'] if j['shape'][0]>=393216]
    summaries=OUT/'controller-checks.json'
    rows=read(summaries) if summaries.exists() else []
    while len(rows)<len(jobs):
        measured=OUT/'holdout.json'
        completed={r['id']:r for r in read(measured) if r['chip']==DT} if measured.exists() else {}
        for job in jobs:
            if any(r['id']==job['id'] for r in rows):
                continue
            name=job['id']+'_'+DT
            dest=ROOT/'runs/vector21-controllers'/name/'log'
            for p in (RAW/name/'log').glob('lpddr_state*.log'):
                dest.mkdir(parents=True,exist_ok=True)
                try:
                    os.link(p,dest/p.name)
                except (FileExistsError, FileNotFoundError):
                    pass
            if job['id'] in completed:
                evidence=counters(dest.parent) if completed[job['id']]['qualified'] else None
                diag=job['targets'][DT]['diagnostics']
                rows.append(dict(id=job['id'],op=job['op'],shape=job['shape'],chip=DT,
                    forecast_external_bytes=diag['modeled_external_bytes'],
                    qualified_window=completed[job['id']]['qualified'],counters=evidence,
                    note='Final reports from 128 channels; includes metadata. Null when diagnostics unavailable.'))
                write(summaries,rows)
        if len(rows)<len(jobs):
            time.sleep(15)
if __name__=='__main__':
    monitor()
