"""Continue the existing full-hidden Cast process; never start a second CA kernel."""
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import os
import re
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT/'development/validation/large20')]
from development.validation.sku19.campaign import CHIPS, read, write, digest, now
from recover_cost import recover_cost, timer
from native import qualify_vector
from operator_model import estimate

RAW = ROOT/'runs/prefill20-native/cast_262144_5120'
OUT = ROOT/'results/v0.21/large'
REG = OUT/'full-hidden-continuation.json'
NATIVE = ROOT/'results/v0.20/large/native-freeze.json'
ORIGINAL = NATIVE.parent/(RAW.name+'_native.json')
SUP = ROOT/'runs/prefill20'


def register():
    if REG.exists() or ORIGINAL.exists():
        raise FileExistsError('Terminal receipt or continuation already exists')
    OUT.mkdir(exist_ok=True)
    shape = [262144,5120]
    original = next(j for j in read(NATIVE)['jobs'] if j['shape']==shape)
    counts = re.findall(r'Cycle Count: (\d+)',(RAW/'run.log').read_text(errors='replace'))
    assert len(counts)>2 and int(counts[-1])>int(counts[-2])
    target = {s:estimate('cast_bf16_fp32',shape,chip=c,dtype='bf16').latency_us for s,c in CHIPS.items()}
    write(REG,dict(utc=now(),shape=shape,original_registration=digest(NATIVE),original=original,
        target_predictions_us=target,cycles_at_decision=int(counts[-1]),
        added_wall_s=9000,total_wall_s=23400,raw_cap_gib=original['raw_cap_gib'],
        reason='The original CA advances about 10k ticks per log step. Its multi-GiB transfer is still in progress before the initial four-hour limit. Continue this identical process with a bounded additional 2.5 h. Same 8 GiB disk cap, original input, kernel, numerical checks, and CA wrapper endpoints. The v0.20 forecast was frozen before launch; v0.21 is a supplemental prediction made during capture.'))


def assert_other_lanes_finished():
    official = read(NATIVE.parent/'registration.json')['jobs']
    pro = read(NATIVE.parent/'pro-freeze.json')['job']
    for job in [*official,pro]:
        for side in ('dt','pr'):
            path=NATIVE.parent/(job['id']+'_'+CHIPS[side]+'.json')
            if not path.is_file():
                raise RuntimeError('Original supervisor still owns other work: '+str(path))


def run():
    registration=read(REG)
    assert digest(NATIVE)==registration['original_registration']
    assert_other_lanes_finished()
    if ORIGINAL.is_file():
        return
    group=timer(RAW)
    if group is None:
        raise RuntimeError('No existing CA process group at the exact target directory')
    oldpid=int((SUP/'supervisor.pid').read_text())
    proc=Path('/proc')/str(oldpid)
    if proc.exists():
        args=(proc/'cmdline').read_bytes().split(b'\0')
        old_script=(ROOT/'development/validation/large20/supervise.py').resolve()
        if (proc.stat().st_uid!=os.getuid() or len(args)<3 or args[2]!=b'serve' or
                ((proc/'cwd').resolve()/os.fsdecode(args[1])).resolve()!=old_script or
                oldpid in (group,os.getpgrp())):
            raise RuntimeError('Original supervisor identity changed')
        # All other lanes have terminal receipts and the GNU time/CA process group
        # is independent. Hand over only the old Python log/cost monitor.
        os.kill(oldpid,signal.SIGTERM)
    with (SUP/'supervise.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        if ORIGINAL.exists():
            return
        if timer(RAW)!=group:
            raise RuntimeError('Original CA session changed during handoff')
        write(OUT/'native-monitor.json',dict(utc=now(),same_ca_process=True,original_registration=digest(NATIVE),
                    original_monitor=oldpid,ca_group=group,unchanged_disk_gib=registration['raw_cap_gib']))
        cap=registration['total_wall_s']
        cost=recover_cost(RAW,cap,registration['raw_cap_gib']*2**30)
        cost.update(original_wall_s=registration['original']['wall_s'],continued_wall_s=cap,
            original_cap_exceeded=cost['seconds']>registration['original']['wall_s'],
            continuation=str(REG.relative_to(ROOT)))
        t,h=registration['shape']
        result=qualify_vector(registration,RAW,cost,[t*h])
        write(OUT/'full-hidden-terminal.json',result)
        # Wake the already frozen quant collector waiting for this terminal record.
        write(ORIGINAL,result)


if __name__=='__main__':
    {'register':register,'run':run}[sys.argv[1]]()
