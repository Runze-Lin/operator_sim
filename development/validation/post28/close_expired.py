"""Stop an escaped simulator child only when its own registered Post case has expired."""
import json
import os
from pathlib import Path
import signal
import time
ROOT=Path(__file__).resolve().parents[3]
DST=ROOT/'results/v0.28/post'
RAW=ROOT/'runs/post28'
EXE=ROOT/'build/vector-v8/MHC'


def reap():
    actions=[]
    for f in (DST/'receipts').glob('*.json'):
        j=json.loads(f.read_text())
        if j['cost'].get('stop_reason') not in ('wall cap','raw cap'):continue
        case=(ROOT/j['raw']).resolve()
        assert case.parent==RAW and case.name==f.stem
        for d in Path('/proc').glob('[0-9]*'):
            try:
                if (d/'cwd').resolve(strict=True)!=case or (d/'exe').resolve(strict=True)!=EXE:continue
                args=(d/'cmdline').read_bytes().strip(b'\x00').split(b'\x00')
                if not args or args[-1] != os.fsencode(case/'batch.txt'):continue
                pid=int(d.name)
                if os.stat(d).st_uid!=os.getuid():continue
                os.kill(pid,signal.SIGTERM);time.sleep(.5)
                if d.exists() and (d/'exe').resolve()==EXE and (d/'cwd').resolve()==case:
                    os.kill(pid,signal.SIGKILL)
                actions.append(dict(case=case.name,pid=pid,receipt=str(f.relative_to(ROOT)),when=time.time()))
            except (OSError,ValueError):continue
    if actions:
        out=DST/'closed-expired-workers.json'
        prior=json.loads(out.read_text()) if out.exists() else []
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(prior+actions,indent=2)+'\n');tmp.replace(out)
    return actions


if __name__=='__main__':
    import sys
    while True:
        print(reap(),flush=True)
        if 'watch' not in sys.argv or json.loads((DST/'state.json').read_text())['phase']=='complete':break
        time.sleep(5)
