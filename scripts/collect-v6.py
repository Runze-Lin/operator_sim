#!/usr/bin/env python3
"""Bounded, frozen quant holdouts using the already-qualified compact collector."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.npusim import extract_compact


def directory_bytes(folder):
    total=0
    for p in folder.rglob('*'):
        try:
            if p.is_file():total+=p.stat().st_size
        except FileNotFoundError:
            pass  # npusim removes temporary logs while finalizing its DB.
    return total


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',required=True);args=parser.parse_args()
    frozen_path=ROOT/'results/v0.6/frozen-predictions.json';frozen=json.loads(frozen_path.read_text())
    if sha256(ROOT/'results/v0.6/model.json')!=frozen['model_sha256']:raise ValueError('Changed frozen model')
    for file,digest in frozen['artifact_sha256'].items():
        if sha256(ROOT/file)!=digest:raise ValueError('Changed frozen artifact: '+file)
    spec=json.loads((ROOT/'configs/sampling-v6.json').read_text())
    case=next((c for c in spec['cases'] if c['id']==args.case),None)
    if case is None:raise ValueError('Case not in frozen manifest')
    root=ROOT/'runs/model-v6';root.mkdir(exist_ok=True)
    previous=[json.loads(f.read_text()) for f in root.glob('*/result.json')]
    wall_left=spec['total_wall_budget_seconds']-sum(r['wall_seconds'] for r in previous)
    cpu_used=sum(r['cpu_seconds'] for r in previous);raw_used=sum(r['raw_bytes'] for r in previous)
    if wall_left<case['timeout_seconds'] or cpu_used>=spec['total_recorded_cpu_budget_seconds'] or raw_used>=spec['total_retained_raw_budget_bytes']:
        raise SystemExit('Collection budget exhausted; no automatic retry or budget expansion')
    folder=root/case['id'];folder.mkdir(exist_ok=False)
    command=['bash',str(ROOT/'scripts/cann-run.sh'),'env',
        'LD_PRELOAD='+str(ROOT/'build/capture/libsim_soc_adapter.so'),'OPERATOR_SIM_TARGET_SOC=Ascend950DT_9582',
        'npusim','record',str(ROOT/'build/smoke/three_ops'),'-s','Ascend950DT','-o',str(folder/'profile'),'-u',case['op']+' '+' '.join(map(str,case['shape']))]
    started=datetime.now(timezone.utc).isoformat()
    if started<=frozen['frozen_utc']:raise ValueError('Capture must start after freeze')
    identity={'soc_version':'Ascend950DT_9582','cann_version':'9.2.1','model_sha256':frozen['model_sha256'],
        'frozen_predictions_sha256':sha256(frozen_path),'host_binary_sha256':sha256(ROOT/'build/smoke/three_ops'),
        'sku_adapter_sha256':sha256(ROOT/'build/capture/libsim_soc_adapter.so')}
    record={'case':case,'identity':identity,'command':command,'started_utc':started,'timeout_seconds':case['timeout_seconds']}
    (folder/'input.json').write_text(json.dumps(record,indent=2)+'\n')
    before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.perf_counter();timeout=False;reason=None
    env=os.environ.copy()
    # No inherited experimental CA configuration.
    env.pop('CAMODEL_CONFIG_PATH',None)
    with (folder/'run.log').open('w') as output:
        child=subprocess.Popen(command,cwd=folder,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            elapsed=time.perf_counter()-start
            # Bound transient native logs as well as the final DB. This differs from retained-byte accounting.
            size=directory_bytes(folder)
            if elapsed>=case['timeout_seconds'] or size>2*spec['total_retained_raw_budget_bytes']:
                timeout=True;reason='wall budget' if elapsed>=case['timeout_seconds'] else 'transient output budget'
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                break
            time.sleep(min(2,max(.01,case['timeout_seconds']-elapsed)))
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    record.update(returncode=124 if timeout else child.returncode,status='failed',timed_out=timeout,stop_reason=reason,
        wall_seconds=time.perf_counter()-start,cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        cpu_accounting='lower_bound_unreaped_descendants' if timeout else 'reaped_child_processes',
        finished_utc=datetime.now(timezone.utc).isoformat(),log_sha256=sha256(folder/'run.log'))
    t=time.perf_counter()
    if record['returncode']==0:
        try:
            record['measurement']=extract_compact(folder/'profile',folder/'run.log')
            if not record['measurement']['correctness'] or not all('bad=0' in x for x in record['measurement']['correctness']):
                raise ValueError('Missing complete numerical validation')
            import sqlite3
            with sqlite3.connect(Path(record['measurement']['db_path']).resolve().as_uri()+'?mode=ro',uri=True) as db:
                counts=list(db.execute("SELECT ExecInstrCoreId,CoreTypeId,SUM(substr(ExecInstrName,1,3)='RV_') FROM ExecutedInstructions GROUP BY ExecInstrCoreId,CoreTypeId"))
            record['measurement']['max_core_rv_count']=max(c[2] for c in counts)
            record['status']='ok' 
        except (ValueError,OSError,KeyError) as exc:record['error']=str(exc)
    record['extraction_seconds']=time.perf_counter()-t
    record['raw_bytes']=directory_bytes(folder)
    (folder/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('command','measurement')},indent=2))
    if record['status']=='ok':print('CA_REFERENCE_US',record['measurement']['reference_us'])

if __name__=='__main__':main()
