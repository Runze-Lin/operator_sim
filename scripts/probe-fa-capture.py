#!/usr/bin/env python3
"""Bounded capture experiments; preserve every attempt, never change model parameters."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.ca import extract,sha256
from operator_model.npusim import extract_compact

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--kv',type=int,required=True)
 ap.add_argument('--q',type=int,default=9);ap.add_argument('--d',type=int,default=64);ap.add_argument('--batch',type=int,default=1)
 ap.add_argument('--native-helper',action='store_true')
 ap.add_argument('--budget-seconds',type=int,default=120,help='Native/npusim wall budget, explicit bounded experiment')
 ap.add_argument('--sku-adapter',action='store_true');ap.add_argument('--local-sku-adapter',action='store_true')
 ap.add_argument('--backend',choices=['default','legacy','npusim','native'],default='default')
 ap.add_argument('--dump',choices=['true','false'],default='false');ap.add_argument('--config');args=ap.parse_args()
 if not 30<=args.budget_seconds<=600:raise ValueError('Budget must be 30..600 seconds')
 if not args.name.replace('-','').replace('_','').isalnum():raise ValueError('Invalid run name')
 folder=ROOT/'runs/long-fa'/args.name
 if folder.exists():raise SystemExit('Existing attempt: do not overwrite or auto-retry')
 folder.mkdir();shape=[args.q,args.kv,args.d,args.batch,1]
 cmd=['bash',str(ROOT/('scripts/cann-legacy-sim-run.sh' if args.backend=='legacy' else 'scripts/cann-run.sh')),'msopprof','simulator','--soc-version=Ascend950DT_9582','--dump='+args.dump,
      '--launch-count=10','--timeout=2','--output='+str(folder/'profile'),str(ROOT/'build/smoke/three_ops'),'fa',*map(str,shape)]
 if args.backend=='npusim':
  cmd=['bash',str(ROOT/'scripts/cann-run.sh'),'npusim','record',str(ROOT/'build/smoke/three_ops'),'-s','Ascend950DT','-o',str(folder/'profile'),'-u','fa '+' '.join(map(str,shape))]
 if args.backend=='native':
  cmd=['bash',str(ROOT/'scripts/cann-native-sim-run.sh'),str(folder),str(ROOT/'build/smoke/three_ops'),'fa',*map(str,shape)]
 env=os.environ.copy()
 if args.native_helper:env['OPERATOR_CAPTURE_HELPER']='1'
 if args.backend=='npusim':env['CAMODEL_SOC_VERSION']='Ascend950DT_9582'

 if args.sku_adapter:
  cmd=cmd[:2]+['env','LD_PRELOAD='+str(ROOT/'toolchains/cann/9.2.1/cann-9.2.1/tools/msopprof/lib64/libmsopprof_injection.so'),'IS_SIMULATOR_ENV=1']+cmd[2:]
 if args.local_sku_adapter:
  cmd=cmd[:2]+['env','LD_PRELOAD='+str(ROOT/'build/capture/libsim_soc_adapter.so'),'OPERATOR_SIM_TARGET_SOC=Ascend950DT_9582']+cmd[2:]
 if args.config:env['CAMODEL_CONFIG_PATH']=str(Path(args.config).resolve())
 uses_adapter=args.local_sku_adapter or args.backend=='native'
 inp={'shape':shape,'backend':args.backend,'native_helper':args.native_helper,'target_soc_override':'Ascend950DT_9582' if uses_adapter else env.get('CAMODEL_SOC_VERSION'),'sku_adapter':args.sku_adapter,'local_sku_adapter':uses_adapter,'local_adapter_sha256':sha256(ROOT/'build/capture/libsim_soc_adapter.so') if uses_adapter else None,'command':cmd,'dump':args.dump,'config_path':env.get('CAMODEL_CONFIG_PATH'),
      'config_sha256':sha256(Path(args.config)) if args.config else None,'binary_sha256':sha256(ROOT/'build/smoke/three_ops'),
      'model_sha256':sha256(ROOT/'results/v0.4/model.json'),'outer_timeout_seconds':args.budget_seconds if args.backend in ('npusim','native') else 180,'started_utc':datetime.now(timezone.utc).isoformat()}
 (folder/'input.json').write_text(json.dumps(inp,indent=2)+'\n');before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.perf_counter()
 with (folder/'run.log').open('w') as log:
  child=subprocess.Popen(cmd,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  outer_timeout=False
  try:code=child.wait(timeout=inp['outer_timeout_seconds'])
  except subprocess.TimeoutExpired:
   outer_timeout=True;os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   code=124
 after=resource.getrusage(resource.RUSAGE_CHILDREN);wall=time.perf_counter()-start
 text=(folder/'run.log').read_text(errors='replace')
 result={**inp,'returncode':code,'wall_seconds':wall,'cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
         'outer_timeout':outer_timeout,'profiler_timeout':'timeout has reached' in text,'smoke_pass':'SMOKE_PASS' in text,
         'status':'failed','finished_utc':datetime.now(timezone.utc).isoformat(),'log_sha256':sha256(folder/'run.log'),'config_sha256_after':sha256(Path(args.config)) if args.config else None}
 if code==0 and args.backend in ('npusim','native'):
  result['status']='correctness_only' if 'SMOKE_PASS' in text and 'runtime_soc=Ascend950DT_9582' in text else 'failed'
  if args.backend=='npusim' or args.native_helper:
   try:
    result['measurement']=extract_compact(folder/'profile',folder/'run.log') if args.backend=='npusim' else extract_compact(folder,folder/'run.log',native_returncode=code)
    result['status']='ok'
   except (ValueError,OSError,KeyError) as exc:result['status']='failed';result['error']=str(exc)
 elif code==0:
  try:result['measurement']=extract(folder/'profile',folder/'run.log');result['status']='ok'
  except (ValueError,OSError,KeyError) as exc:result['error']=str(exc)
 result['raw_bytes']=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
 result['cpu_accounting']='lower_bound_unreaped_descendants' if args.backend=='npusim' and outer_timeout else 'reaped_child_processes'
 (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='measurement'},indent=2))
 if result['status']=='ok':print('REFERENCE_US',result['measurement']['reference_us'])
if __name__=='__main__':main()
