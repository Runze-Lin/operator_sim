"""Collect the completion record for an already-running wave after wrapper loss."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw',type=Path)
    p.add_argument('--watcher-exit-code',type=int,required=True)
    p.add_argument('--case',choices=('pr','dt'),required=True)
    a=p.parse_args()
    raw=a.raw.resolve()
    log=raw/'run.log'
    start=int(subprocess.check_output(['stat','-c','%W',log]).strip())
    if not start:
        raise RuntimeError('Missing log creation time')
    golden_seen=None
    last=''
    historical=log.read_text()
    if 'V112_WAVE_GOLDEN_PASS' in historical:
        raise RuntimeError('Need separate provenance for an already completed device')
    initial=time.time()
    collection_budget={'pr':25,'dt':38}[a.case]*60
    while True:
        last=log.read_text()
        now=time.time()
        if golden_seen is None and 'V112_WAVE_GOLDEN_PASS' in last:
            golden_seen=now
        finished='All task success.' in last and 'Profiling data parse finished.' in last
        if finished and list((raw/'profile').glob('OPPROF_*/simulator/core*.*core*/trace.json')):
            break
        if (golden_seen is None and now-start>collection_budget+120) or (golden_seen and now-golden_seen>20*60):
            break
        time.sleep(3)
    end=log.stat().st_mtime
    completed='All task success.' in last and 'Profiling data parse finished.' in last
    source=Path(__file__).with_name('v112_wave_probe.py')
    result=dict(case=a.case,sku='Ascend950PR_957b' if a.case=='pr' else 'Ascend950DT_9582',
                start_utc=datetime.fromtimestamp(start,timezone.utc).isoformat(),
                first_adopted_utc=datetime.fromtimestamp(initial,timezone.utc).isoformat(),
                last_log_write_utc=datetime.fromtimestamp(end,timezone.utc).isoformat(),
                wall_s=round(end-start,3),golden_observed_s=round(golden_seen-start,3) if golden_seen else None,
                rc=None,original_monitor_exit_code=a.watcher_exit_code,watchdog=not completed,
                profiler_final_log_seen=completed,probe_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source='Original cost monitor terminated; CA and msOpProf continued in their existing process group. Simulator return code unobserved. Duration uses log birth (1s precision) to last modification. Adopted poll interval 3s.')
    path=raw/'cost.json'
    with path.open('x') as out:
        json.dump(result,out,indent=2)
        out.write('\n')

if __name__=='__main__':
    main()
