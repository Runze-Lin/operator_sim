#!/usr/bin/env python3
"""Serial bounded installed-library CA collection, with frozen holdout admission."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import sqlite3
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.npusim_v7 import extract_run


def size(root):
    total=0
    for p in root.rglob('*'):
        try:
            if p.is_file():total+=p.stat().st_size
        except FileNotFoundError:pass
    return total

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--case',required=True);args=parser.parse_args()
    manifest=ROOT/args.manifest;spec=json.loads(manifest.read_text());case=next(c for c in spec['cases'] if c['id']==args.case)
    run=ROOT/'runs/model-v7';run.mkdir(exist_ok=True)
    with (run/'collection.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=[json.loads(p.read_text()) for p in run.glob('*/result.json')]
        if sum(r['wall_seconds'] for r in previous)+case['timeout_seconds']>3600:raise ValueError('60 minute total CA budget exhausted')
        if sum(r['raw_bytes'] for r in previous)>4*2**30:raise ValueError('4GiB retained budget exhausted')
        identity={'soc_version':'Ascend950DT_9582','cann_version':'9.2.1','binary_sha256':sha256(ROOT/spec['binary']),
            'source_sha256':sha256(ROOT/spec['source']),'manifest_sha256':sha256(manifest),
            'adapter_sha256':sha256(ROOT/'build/capture/libsim_soc_adapter.so'),'collector_sha256':sha256(Path(__file__)),
            'extractor_sha256':sha256(ROOT/'operator_model/npusim_v7.py')}
        if case['split']=='holdout':
            fp=ROOT/'results/v0.7/frozen-predictions.json';frozen=json.loads(fp.read_text())
            fc=next(c for c in frozen['predictions'] if c['id']==case['id'])
            if fc['case']!=case:raise ValueError('Case changed after freeze')
            for path,digest in frozen['artifact_sha256'].items():
                if sha256(ROOT/path)!=digest:raise ValueError('Frozen artifact changed: '+path)
            identity.update(model_sha256=frozen['model_sha256'],freeze_sha256=sha256(fp))
        folder=run/case['id'];folder.mkdir(exist_ok=False)
        command=['bash',str(ROOT/'scripts/cann-run.sh'),'env','LD_PRELOAD='+str(ROOT/'build/capture/libsim_soc_adapter.so'),
            'OPERATOR_SIM_TARGET_SOC=Ascend950DT_9582','npusim','record',str(ROOT/spec['binary']),'-s','Ascend950DT','-o',str(folder/'profile'),'-u',' '.join(map(str,case['arguments']))]
        record={'case':case,'identity':identity,'manifest':args.manifest,'command':command,'started_utc':datetime.now(timezone.utc).isoformat()}
        (folder/'input.json').write_text(json.dumps(record,indent=2)+'\n')
        before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.perf_counter();stop=None
        env=os.environ.copy();env.pop('CAMODEL_CONFIG_PATH',None)
        with (folder/'run.log').open('w') as log:
            child=subprocess.Popen(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                if time.perf_counter()-start>=case['timeout_seconds']:stop='wall budget'
                if size(folder)>768*2**20:stop='transient output budget'
                recent=(folder/'run.log').read_text(errors='replace')
                if '[ERROR]' in recent and any(x in recent for x in ('execute_set_flag already has same set_flag','execute_wait_flag','deadlock')):stop='CA synchronization error; timing unqualified'
                if stop:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                time.sleep(1)
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        record.update(status='failed',returncode=124 if stop else child.returncode,timed_out=stop=='wall budget',stop_reason=stop,
            wall_seconds=time.perf_counter()-start,cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
            cpu_accounting='lower_bound_unreaped_descendants' if stop else 'reaped_child_processes',finished_utc=datetime.now(timezone.utc).isoformat(),log_sha256=sha256(folder/'run.log'))
        t=time.perf_counter()
        if not stop and child.returncode==0:
            try:record['measurement']=extract_run(folder/'profile',folder/'run.log',case['correctness_groups']);record['status']='ok'
            except (ValueError,OSError,KeyError,sqlite3.Error) as exc:record['error']=str(exc)
        record['extraction_seconds']=time.perf_counter()-t;record['raw_bytes']=size(folder)
        (folder/'result.json').write_text(json.dumps(record,indent=2)+'\n')
        print(case['id'],record['status'],record.get('measurement',{}).get('reference_us'),record['wall_seconds'],flush=True)
        if record['status']!='ok':print(record.get('error',record.get('stop_reason')),flush=True)

if __name__=='__main__':
    main()
