"""Repeat the BF16 gate host on the DT compact path, limited to one simulator worker."""
from pathlib import Path
import json
import os
import re
import signal
import subprocess
import sys
import time
import datetime
from probe import ART, ROOT, RUN, save, sha

chip='Ascend950DT_9582'
tag='kimi-k3-bf16g-64x12-dt-compact'
reg=ART/f'{tag}-registration.json'
source=ART/'kimi-k3-bf16g-64x12-registration.json'
adapter=ROOT/'build/capture/libsim_soc_adapter.so'
budget=ROOT/'configs/camodel-single-thread.json'
config=json.loads(source.read_text())
if sys.argv[1]=='register':
    if reg.exists():
        raise FileExistsError(reg)
    config.update(chip=chip,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  backend='npusim record; compact DT developer diagnostic with single host simulator worker',
                  parent_contract_sha256=sha(source))
    config['hashes'].update({str(p.relative_to(ROOT)):sha(p) for p in (adapter,budget,Path(__file__).resolve())})
    save(reg,config)
elif sys.argv[1]=='collect':
    spec=json.loads(reg.read_text())
    if any(sha(ROOT/p)!=h for p,h in spec['hashes'].items()):
        raise ValueError('registered hashes changed')
    path=RUN/tag/chip
    path.mkdir(parents=True,exist_ok=False)
    log=path/'run.log'
    cmd=['bash',str(ROOT/'scripts/cann-run.sh'),'env',f'LD_PRELOAD={adapter}',
         f'CAMODEL_CONFIG_PATH={budget}',f'OPERATOR_SIM_TARGET_SOC={chip}','npusim','record',
         str(ROOT/'build/smoke/chunk21-kimi'),'-s','Ascend950DT','-o',str(path/'profile'),'-u','64 12']
    start=time.monotonic()
    stop=None
    with log.open('w') as stream:
        child=subprocess.Popen(cmd,cwd=path,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            time.sleep(2)
            text=log.read_text(errors='replace')
            if 'execute_set_flag already has same set_flag' in text: stop='duplicate SET_FLAG'
            elif time.monotonic()-start>spec['budget_seconds']: stop='wall budget'
            elif sum(f.stat().st_size for f in path.rglob('*') if f.is_file())>spec['budget_bytes']: stop='storage budget'
            if stop:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()
                break
    text=log.read_text(errors='replace')
    save(ART/f'{tag}-{chip}.json',dict(chip=chip,command=cmd,registration_sha256=sha(reg),log_sha256=sha(log),
        seconds=time.monotonic()-start,exit=child.returncode,stop=stop,sku_evidence=re.findall(r'(?:runtime_soc|SIM_SKU_ADAPTER target)=([\w]+)',text),
        thread_count=re.findall(r'(?:esl_top_thread_cnt = |thread count: )(\d+)',text),
        duplicate_set_pcs=re.findall(r'already has same set_flag! pc:(0x[0-9a-f]+)',text)[:12],
        success='SMOKE_PASS' in text and child.returncode==0 and stop is None))
