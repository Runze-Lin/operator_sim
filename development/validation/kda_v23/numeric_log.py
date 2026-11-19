"""Recover this campaign's four buffered golden-result lines after simulator teardown.

The OS core is matched to the exact registered user, executable, capture time,
failed host log and unique output counts. Never save process memory or other logs.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from series import BIN, OUT, RAW, REG, read, save, probe

PAT = re.compile(rb'^correctness name=(attention|aqk|akk|state_vk) count=(\d+) bad=(\d+) max_abs_error=([eE0-9.+-]+) atol=([eE0-9.+-]+) rtol=([eE0-9.+-]+) hard_limit=([eE0-9.+-]+)$',re.M)
# First line can immediately follow unrelated memory without an actual newline.
FIRST = re.compile(PAT.pattern.lstrip(b'^'),re.M)


def read_core(path):
    found=set()
    carry=b''
    with subprocess.Popen(['lz4','-dc',str(path)],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL) as proc:
        while block:=proc.stdout.read(2**21):
            data=carry+block
            # Keep the line terminator so a bare chunk boundary cannot yield a partial line.
            found.update(m[0] for m in FIRST.finditer(data) if m.end()<len(data))
            carry=data[-384:]
        if proc.wait():return []
    return sorted(found)


def scan():
    plan=read(REG)
    result=OUT/'numeric-detail'
    result.mkdir(exist_ok=True)
    for path in Path('/var/lib/systemd/coredump').glob(f'core.{BIN.name}.{os.getuid()}.*.lz4'):
        try:
            get=lambda k:os.getxattr(path,'user.coredump.'+k).decode()
            if get('exe')!=str(BIN):continue
            pid=int(get('pid'))
            crash=int(get('timestamp'))/1e6
            for case in plan['cases']:
                dest=result/(case['id']+'.json')
                base=RAW/case['id']
                if dest.exists() or not (base/'cost.json').exists():continue
                cost=read(base/'cost.json')
                if not datetime.fromisoformat(cost['began']).timestamp()<=crash+1 or crash>datetime.fromisoformat(cost['finished']).timestamp():continue
                log=(base/'run.log').read_text(errors='replace')
                if 'SMOKE_FAIL one or more V2 outputs differ' not in log or 'Child process killed by signal 11' not in log:continue
                rows=read_core(path)
                groups={}
                for line in rows:
                    m=PAT.fullmatch(line)
                    name,num,bad,err,atol,rtol,limit=[f.decode() for f in m.groups()]
                    groups[name]=dict(count=int(num),bad=int(bad),max_abs_error=float(err),atol=float(atol),rtol=float(rtol),hard_limit=float(limit))
                if len(rows)!=4 or {k:g['count'] for k,g in groups.items()}!=case['oracle']:continue
                save(dest,dict(id=case['id'],sku=case['sku'],shape=case['shape'],numerics_pass=False,role='diagnose original failure; no requalification',
                     source='Only the four uniquely formatted numerical messages in the original executable stdout buffer were extracted from its OS crash dump.',
                     compressed_core_sha256=probe.digest(path),uid=os.getuid(),pid=pid,crash_time_unix_s=crash,
                     executable_sha256=probe.digest(BIN),capture_log_sha256=probe.digest(base/'run.log'),groups=groups,messages=[s.decode() for s in rows]))
        except (FileNotFoundError,PermissionError,OSError):
            continue


if __name__=='__main__':
    watch='--watch' in sys.argv
    while True:
        scan()
        if not watch or read(OUT/'status.json')['state']!='running':break
        time.sleep(12)
