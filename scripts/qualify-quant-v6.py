#!/usr/bin/env python3
"""Recover a quant log-format mismatch offline; preserve original collector records.

The frozen shared compact extractor recognizes `correctness elements=...` (FA),
whereas quant prints `quant_elements=... scale_elements=... comparison=exact_bytes`.
Only this exact extraction failure is admissible. No simulation or fitting occurs.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import sqlite3
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256


def qualify(path):
    start=time.perf_counter();r=json.loads(path.read_text());c=r['case']
    if c['op']!='quant' or r['status']!='failed' or r.get('error')!='Missing complete numerical validation':
        raise ValueError('Not the known quant log-format mismatch')
    if r['returncode']!=0 or r['timed_out'] or 'measurement' not in r:raise ValueError('Incomplete CA')
    log_path=path.parent/'run.log'
    if sha256(log_path)!=r['log_sha256']:raise ValueError('Log changed')
    log=log_path.read_text(errors='strict')
    if 'SMOKE_PASS' not in log or 'SMOKE_FAIL' in log or 'Simulation SUCCESS' not in log:
        raise ValueError('Simulation or numeric check did not finish')
    matches=re.findall(r'correctness quant_elements=(\d+) scale_elements=(\d+) bad=(\d+) comparison=exact_bytes',log)
    expected=c['shape'][0]*c['shape'][1]
    if len(matches)!=1 or tuple(map(int,matches[0]))!=(expected,expected//32,0):
        raise ValueError('Incomplete or failed exact-byte correctness')
    m=r['measurement'];db_path=Path(m['db_path'])
    if sha256(db_path)!=m['db_sha256']:raise ValueError('DB changed')
    with sqlite3.connect(db_path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        counts=list(db.execute("SELECT ExecInstrCoreId,CoreTypeId,SUM(substr(ExecInstrName,1,3)='RV_') FROM ExecutedInstructions GROUP BY ExecInstrCoreId,CoreTypeId"))
    m['max_core_rv_count']=max(c[2] for c in counts)
    m['correctness']=['correctness quant_elements=%s scale_elements=%s bad=%s comparison=exact_bytes'%matches[0]]
    r.update(status='ok',collector_status='failed',collector_error=r.pop('error'),
        recovery={'method':'offline exact quant output/scale log validation; original result untouched',
            'source_result_sha256':sha256(path),'qualifier_sha256':sha256(Path(__file__)),
            'qualified_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-start})
    return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);args=p.parse_args()
    folder=ROOT/'runs/model-v6'/args.case;target=folder/'qualified-result.json'
    if target.exists():raise ValueError('Already qualified; preserve provenance')
    r=qualify(folder/'result.json');target.write_text(json.dumps(r,indent=2)+'\n')
    print(args.case,r['measurement']['reference_us'],r['measurement']['max_core_rv_count'])
if __name__=='__main__':main()
