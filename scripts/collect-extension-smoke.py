#!/usr/bin/env python3
"""At most one tiny invocation per new dense primitive; no timing fitting."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import resource
import signal
import sqlite3
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.npusim import extract_compact

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--op',choices=['rmsnorm','addmm'],required=True);args=parser.parse_args()
    params=['rmsnorm','1','128'] if args.op=='rmsnorm' else ['addmm','2','16','16','bias']
    folder=ROOT/'runs/extensions/smoke950'/args.op;folder.mkdir(parents=True,exist_ok=False)
    binary=ROOT/'build/smoke/extension_norm_addmm'
    command=['bash',str(ROOT/'scripts/cann-run.sh'),'env','LD_PRELOAD='+str(ROOT/'build/capture/libsim_soc_adapter.so'),
        'OPERATOR_SIM_TARGET_SOC=Ascend950DT_9582','npusim','record',str(binary),'-s','Ascend950DT','-o',str(folder/'profile'),'-u',' '.join(params)]
    record={'op':args.op,'arguments':params,'purpose':'library invocation / full-output correctness / dispatch discovery; not latency calibration',
        'soc_version':'Ascend950DT_9582','cann_version':'9.2.1','started_utc':datetime.now(timezone.utc).isoformat(),
        'host_binary_sha256':sha256(binary),'host_source_sha256':sha256(ROOT/'smoke/extension_norm_addmm.cpp'),
        'sku_adapter_sha256':sha256(ROOT/'build/capture/libsim_soc_adapter.so'),'command':command,'timeout_seconds':120}
    (folder/'input.json').write_text(json.dumps(record,indent=2)+'\n')
    before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.perf_counter();timeout=False
    env=os.environ.copy();env.pop('CAMODEL_CONFIG_PATH',None)
    with (folder/'run.log').open('w') as log:
        child=subprocess.Popen(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:child.wait(timeout=120)
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    after=resource.getrusage(resource.RUSAGE_CHILDREN);log=(folder/'run.log').read_text(errors='replace')
    record.update(status='failed',returncode=124 if timeout else child.returncode,timed_out=timeout,
        wall_seconds=time.perf_counter()-start,cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        cpu_accounting='lower_bound_unreaped_descendants' if timeout else 'reaped_child_processes',
        correctness=re.findall(r'correctness count=[^\r\n]+',log),log_sha256=sha256(folder/'run.log'))
    expected=2 if args.op=='rmsnorm' else 1
    complete=not timeout and child.returncode==0 and 'SMOKE_PASS' in log and 'Simulation SUCCESS' in log and 'runtime_soc=Ascend950DT_9582' in log
    complete=complete and len(record['correctness'])==expected and all('bad=0' in s for s in record['correctness'])
    dbs=list((folder/'profile').glob('npusim_*/record/*.db'));record['kernel_db_count']=len(dbs)
    if complete and len(dbs)==1:
        try:
            record['measurement']=extract_compact(folder/'profile',folder/'run.log')
            record['measurement']['correctness']=record['correctness'];record['status']='complete_single_kernel'
        except (ValueError,OSError,KeyError) as exc:record['error']=str(exc)
    elif complete and len(dbs)>1:
        record['kernels']=[]
        for path in dbs:
            with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
                if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('Corrupt DB')
                record['kernels'].append({'info':dict(db.execute('SELECT * FROM KernelInfo')),'db_sha256':sha256(path)})
        record['status']='complete_multiple_kernels';record['reference_us']=None
        record['scope_limit']='Per-kernel DB clocks are not summed into whole ACLNN latency without cross-launch alignment.'
    record['raw_bytes']=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    record['finished_utc']=datetime.now(timezone.utc).isoformat()
    (folder/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('command','measurement','kernels')},indent=2))
    if 'measurement' in record:print(record['measurement']['kernel'],record['measurement']['reference_us'])

if __name__=='__main__':main()
