"""Repeat failed infrastructure captures in private binary directories with the original predictions."""
from pathlib import Path
import argparse, json, os, shutil, subprocess, sys, signal, time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.npusim_v7 import extract_db

def main(extra=False):
    out=ROOT/'results/v0.8/core'
    f=json.loads((out/'frozen.json').read_text())
    assert all(sha256(ROOT/p)==h for p,h in f['sha256'].items())
    cases=f['rows']
    freeze_path=out/'frozen.json'
    if extra:
        from operator_model.predictor import predict
        freeze_path=out/'fa-reserve-freeze.json'
        if not freeze_path.exists():
            from datetime import datetime, timezone
            shapes=[[2,80,64,1,1],[1,224,64,1,1],[4,160,64,1,1],[8,128,128,1,1]]
            cases=[dict(op='fa',id='fa_reserve_'+str(i),shape=s,predicted_us=predict('fa',s)['predicted_us']) for i,s in enumerate(shapes)]
            freeze_path.write_text(json.dumps(dict(rows=cases,utc=datetime.now(timezone.utc).isoformat(),basis_sha256=sha256(out/'frozen.json')),indent=2)+'\n')
        cases=json.loads(freeze_path.read_text())['rows']
    for c in cases:
        first=out/(c['id']+'.json')
        if not extra and (not first.exists() or json.loads(first.read_text())['status']=='ok'):continue
        record=out/(c['id']+'-retry.json')
        if record.exists():continue
        dest=ROOT/'runs/v8core'/record.stem;dest.mkdir()
        binary=dest/'three_ops'
        shutil.copy2(ROOT/'build/smoke/three_ops',binary)
        assert sha256(binary)==f['sha256']['build/smoke/three_ops']
        cmd=['bash',str(ROOT/'scripts/cann-run.sh'),'env','LD_PRELOAD='+str(ROOT/'build/capture/libsim_soc_adapter.so'),'OPERATOR_SIM_TARGET_SOC=Ascend950DT_9582','npusim','record',str(binary),'-s','Ascend950DT','-o',str(dest/'profile'),'-u',' '.join([c['op'],*map(str,c['shape'])])]
        start=time.monotonic()
        env=os.environ.copy();env.pop('CAMODEL_CONFIG_PATH',None)
        with (dest/'run.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=dest,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
            while proc.poll() is None:
                try:size=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())
                except FileNotFoundError:continue
                if time.monotonic()-start>f['timeout_seconds'] or size>f['max_case_bytes']:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait();break
                time.sleep(1)
        s=(dest/'run.log').read_text();r=dict(c,status='failed',command=cmd,wall_seconds=time.monotonic()-start,freeze_sha256=sha256(freeze_path))
        checks=[l.split('correctness ')[-1] for l in s.splitlines() if 'correctness ' in l]
        dbs=list(dest.rglob('*.db'))
        if proc.returncode==0 and 'Simulation SUCCESS' in s and 'SMOKE_PASS' in s and 'runtime_soc=Ascend950DT_9582' in s and len(dbs)==1 and checks and all(' bad=0 ' in l for l in checks):
            r['measurement']=extract_db(dbs[0]);r['reference_us']=r['measurement']['reference_us'];r['correctness']=checks;r['status']='ok'
        record.write_text(json.dumps(r,indent=2)+'\n');print(c['id'],r['status'],flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reserve',action='store_true')
    main(parser.parse_args().reserve)
