"""Freeze seven additional quant and FA holdouts each, then collect complete CA references."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.fast_query import Predictor
from operator_model.model_v6 import Case
from operator_model.npusim_v7 import extract_db

OUT=ROOT/'results/v0.8/core'
FROZEN=OUT/'frozen.json'
def save(path,obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n')
    tmp.replace(path)

def freeze():
    if FROZEN.exists():raise ValueError('already frozen')
    shapes={'quant':[(1,192),(7,320),(17,1024),(36,3072),(128,5120),(576,7168),(1024,6144)],
            'fa':[(1,64,64,1,1),(4,256,128,1,1),(9,512,64,1,1),(17,384,64,1,2),(32,512,64,1,1),(64,320,128,1,1),(9,4096,64,1,1)]}
    old=json.loads((ROOT/'results/v0.7/evaluation.json').read_text())['rows']
    model=Predictor(ROOT/'results/v0.6/model.json')
    rows=[]
    for op,dims in shapes.items():
        for shape in dims:
            assert not any(r['op']==op and tuple(r['shape'])==shape for r in old)
            case=Case(op,shape)
            start=time.perf_counter()
            pred=model.predict_us(case,use_cache=False)
            rows.append(dict(op=op,shape=shape,id=op+'_'+'_'.join(map(str,shape)),predicted_us=pred,query_s=time.perf_counter()-start))
    files=['scripts/core-v8.py','build/smoke/three_ops','build/capture/libsim_soc_adapter.so','results/v0.6/model.json','scripts/cann-run.sh','operator_model/npusim_v7.py','operator_model/fast_query.py','operator_model/model.py']+[f'operator_model/model_v{i}.py' for i in range(2,7)]
    save(FROZEN,dict(rows=rows,utc=datetime.now(timezone.utc).isoformat(),timeout_seconds=240,max_case_bytes=1024**3,sha256={p:sha256(ROOT/p) for p in files}))

def collect():
    f=json.loads(FROZEN.read_text())
    if any(sha256(ROOT/p)!=h for p,h in f['sha256'].items()):raise ValueError('freeze mismatch')
    for c in f['rows']:
        record=OUT/(c['id']+'.json')
        if record.exists():continue
        d=ROOT/'runs/v8core'/c['id']
        d.mkdir(parents=True,exist_ok=False)
        args=' '.join([c['op'],*map(str,c['shape'])])
        cmd=['bash',str(ROOT/'scripts/cann-run.sh'),'env','LD_PRELOAD='+str(ROOT/'build/capture/libsim_soc_adapter.so'),'OPERATOR_SIM_TARGET_SOC=Ascend950DT_9582','npusim','record',str(ROOT/'build/smoke/three_ops'),'-s','Ascend950DT','-o',str(d/'profile'),'-u',args]
        r=dict(c,status='failed',freeze_sha256=sha256(FROZEN),command=cmd,started=datetime.now(timezone.utc).isoformat())
        env=os.environ.copy();env.pop('CAMODEL_CONFIG_PATH',None)
        start=time.monotonic()
        with (d/'run.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=d,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
            while proc.poll() is None:
                try:size=sum(p.stat().st_size for p in d.rglob('*') if p.is_file())
                except FileNotFoundError:continue
                if size>f['max_case_bytes'] or time.monotonic()-start>f['timeout_seconds']:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(5)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    r['error']='collection budget';break
                time.sleep(1)
        r['wall_seconds']=time.monotonic()-start
        log=(d/'run.log').read_text()
        checks=[l.split('correctness ')[-1] for l in log.splitlines() if 'correctness ' in l]
        dbs=list(d.rglob('*.db'))
        if proc.returncode==0 and 'Simulation SUCCESS' in log and 'SMOKE_PASS' in log and 'runtime_soc=Ascend950DT_9582' in log and len(dbs)==1 and checks and all(' bad=0 ' in l for l in checks):
            r['measurement']=extract_db(dbs[0]);r['reference_us']=r['measurement']['reference_us'];r['correctness']=checks;r['status']='ok'
        save(record,r);print(c['id'],r['status'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','collect']);args=parser.parse_args()
    (freeze if args.mode=='freeze' else collect)()
