"""Collect one changed official MIX template. Keep its full numerical/timing verdict separate."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/chunk-recovery'
RAW=ROOT/'runs/chunk-recovery'
TAG='generic-fp16-c128-3x16'
CHIP='Ascend950DT_9582'
reg=OUT/f'{TAG}-registration.json'
paths=[RAW/'fp16',Path(__file__).resolve(),Path(__file__).with_name('fp16.cpp'),
       ROOT/'smoke/extension_kda.cpp',ROOT/'configs/camodel-single-thread.json',ROOT/'build/capture/libsim_soc_adapter.so']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

if sys.argv[1]=='register':
    counts=dict(runtime_gate_dtype='fp32',qkv_dtype='fp16',beta_dtype='fp32',layout='BNSD',state=False,
                output_final_state=False,export_gk=False,safe_gate=False,use_gate_in_kernel=False,
                chunk_size=128,shape=[1,3,1,1,16,16],chip=CHIP)
    with reg.open('x') as f:
        json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),role='development; zero latency forecast',
            contract=counts,budget_seconds=1500,budget_bytes=256*2**20,flags='log first error; continue to terminal numerical checks unless no forward progress for 300s after an error',
            files={str(p.relative_to(ROOT)):sha(p) for p in paths}),f,indent=2)
        f.write('\n')
else:
    config=json.loads(reg.read_text())
    if any(sha(ROOT/p)!=v for p,v in config['files'].items()): raise ValueError('changed source')
    case=RAW/TAG;case.mkdir(exist_ok=False)
    log=case/'run.log'
    args=['bash',str(ROOT/'scripts/cann-run.sh'),'env',f'OPERATOR_SIM_TARGET_SOC={CHIP}',
          'CAMODEL_CONFIG_PATH='+str(paths[4]),'LD_PRELOAD='+str(paths[5]),'npusim','record',
          str(paths[0]),'-s','Ascend950DT','-o',str(case/'profile')]
    start=time.monotonic(); stop=None; last_error=None; last_progress=None
    with log.open('w') as out:
        child=subprocess.Popen(args,stdout=out,stderr=subprocess.STDOUT,cwd=case,start_new_session=True)
        while child.poll() is None:
            time.sleep(3)
            text=log.read_text(errors='replace')
            evidence=[l for l in text.splitlines() if '[user]' in l or '[phase]' in l]
            if 'already has same set_flag' in text and evidence and evidence[-1]!=last_error:
                last_progress=time.monotonic();last_error=evidence[-1]
            if time.monotonic()-start>config['budget_seconds']:stop='wall budget'
            elif sum(p.stat().st_size for p in case.rglob('*') if p.is_file())>config['budget_bytes']:stop='storage budget'
            elif last_progress and time.monotonic()-last_progress>300:stop='no further application/simulator event after duplicate SET for 300s'
            if stop:
                os.killpg(child.pid,signal.SIGKILL);child.wait();break
    text=log.read_text(errors='replace')
    errors=re.findall(r'already has same set_flag! pc:(0x[0-9a-f]+)',text)
    numerical='SMOKE_PASS' in text and all('check='+k+' ' in text for k in ('output','aqk','akk'))
    windows=[]
    for p in sorted(case.glob('profile/*/record/*.db')):
        with sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True) as db:
            info=dict(db.execute('select KernelInfoAttr,KernelInfoVal from KernelInfo'))
            count,lo,hi=db.execute('select count(*), min(ExecInstrTickStart), max(ExecInstrTickEnd) from ExecutedInstructions').fetchone()
        windows.append(dict(db=str(p.relative_to(ROOT)),db_sha256=sha(p),rows=count,kernel_info=info,
                            verified=bool(info and count and info.get('instrs_exec_num')==count and info.get('exec_time_ticks')==hi-lo)))
    sku_ok='runtime_soc='+CHIP in text and 'SIM_SKU_ADAPTER target='+CHIP+' physical_model=Ascend950DT' in text
    eligible=sku_ok and numerical and child.returncode==0 and not stop and not errors and bool(windows) and all(w['verified'] for w in windows)
    (OUT/f'{TAG}-receipt.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,cmd=args,exit=child.returncode,
          stop=stop,log_sha256=sha(log),sku_verified=sku_ok,checks=[l for l in text.splitlines() if 'check=' in l],
          numeric=numerical,duplicate_count=len(errors),first_faults=errors[:8],windows=windows,timing_qualified=eligible,
          ca_us=sum(w['kernel_info']['exec_time_ns'] for w in windows)/1000 if eligible else None),indent=2)+'\n')
