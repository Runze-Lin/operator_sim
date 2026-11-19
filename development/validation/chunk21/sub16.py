"""One existing host, minimum positive KDA tail; stopped at the first CA synchronization error."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from probe import ROOT, RUN, ART, sha, save

if __name__=='__main__':
    reg=ART/'sub16-registration.json'
    c=json.loads(reg.read_text())
    if any(sha(ROOT/f)!=h for f,h in c['files'].items()):
        raise ValueError('registered file changed')
    path=RUN/'sub16'/c['chip']
    path.mkdir(exist_ok=False)
    cmd=['bash',str(ROOT/'scripts/cann-run.sh'),'env','LD_PRELOAD='+str(ROOT/'build/capture/libsim_soc_adapter.so'),
         'OPERATOR_SIM_TARGET_SOC='+c['chip'],'CAMODEL_CONFIG_PATH='+str(ROOT/'configs/camodel-single-thread.json'),
         'npusim','record',str(ROOT/'build/smoke/extension_kda'),'-s','Ascend950DT','-o',str(path/'profile'),'-u','chunk 1 128']
    start=time.monotonic()
    log=path/'run.log'
    stop=None
    with log.open('w') as out:
        proc=subprocess.Popen(cmd,start_new_session=True,cwd=path,stdout=out,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(2)
            text=log.read_text(errors='replace')
            if 'already has same set_flag!' in text: stop='duplicate SET_FLAG'
            elif time.monotonic()-start>c['budget_seconds']: stop='wall budget'
            elif sum(f.stat().st_size for f in path.rglob('*') if f.is_file())>c['budget_bytes']: stop='storage budget'
            if stop:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
                break
    text=log.read_text(errors='replace')
    save(ART/f'sub16-{c["chip"]}.json',dict(chip=c['chip'],command=cmd,backend='npusim compact',
         actual_runtime=re.findall(r'runtime_soc=(\w+)',text),elapsed_s=time.monotonic()-start,exit=proc.returncode,stop=stop,
         checks=[v for v in text.splitlines() if 'correctness name=' in v],success='SMOKE_PASS' in text and proc.returncode==0 and not stop,
         duplicate_set_pcs=re.findall(r'already has same set_flag! pc:(0x[0-9a-f]+)',text)[:12],
         log_sha256=sha(log),reg_sha256=sha(reg)))
