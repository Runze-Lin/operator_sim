"""Repeat only launches with an unusable output DB, preserve cold/resident dispatch."""
from pathlib import Path
from datetime import datetime,timezone
import json,re,time,os,signal,shutil,subprocess,concurrent.futures,sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));OUT=ROOT/'results/v0.12/rms'; RUN=ROOT/'runs/v12rms'
from operator_model.ca import sha256
from operator_model.npusim_v7 import extract_db
from development.validation.scaling_v11 import audit_db

def write(name,x): (OUT/name).write_text(json.dumps(x,indent=2)+'\n')
def run(i):
    f=json.loads((OUT/'freeze.json').read_text());assert all(sha256(ROOT/p)==d for p,d in f['sha256'].items());c=f['cases'][i];folder=RUN/f'recover-{i}';folder.mkdir(exist_ok=False)
    shapes=([[64,768]] if c['prior_aiv'] else [])+[c['shape']];assert c['prior_aiv'] in (0,64); shapes=[tuple(x) for x in shapes]; binary=folder/'NORM';shutil.copy2(ROOT/'build/vector-v8/NORM',binary); args=folder/'inputs.txt';args.write_text(''.join(f'rmsnorm {j} {t} {d} 0\n' for j,(t,d) in enumerate(shapes)))
    cmd=['bash',str(ROOT/'scripts/cann-run.sh'),'env','LD_PRELOAD='+str(ROOT/'build/capture/libsim_soc_adapter.so'),'OPERATOR_SIM_TARGET_SOC=Ascend950DT_9582','npusim','record',str(binary),'-s','Ascend950DT','-o',str(folder/'profile'),'-u',str(args)]
    start=time.monotonic()
    with (folder/'run.log').open('w') as fh:
        proc=subprocess.Popen(cmd,stdout=fh,stderr=subprocess.STDOUT,start_new_session=True,cwd=folder)
        while proc.poll() is None:
            if time.monotonic()-start>f['per_batch_seconds'] or sum(p.stat().st_size for p in folder.rglob('*.db'))>f['disk_budget_gib']*2**30:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise RuntimeError('Recovery budget exhausted')
            time.sleep(1)
    duration=time.monotonic()-start;write('recovery-cost-'+str(i)+'.json',dict(name=folder.name,wall_s=duration,launches=len(shapes),conditioning_controls=len(shapes)-1));assert proc.returncode==0
    rows=qualify(folder,shapes,duration);r=rows[-1];assert all(x['audit']['passed'] for x in rows);write('recovered-'+str(i)+'.json',dict(controls=rows[:-1],target={**c,**r,'predicted_us':c['prediction']['predicted_us']}));print(i,r['reference_us'],duration,flush=True)

def qualify(folder,shapes,wall_s,subset=None):
    text=(folder/'run.log').read_text();assert 'Simulation SUCCESS' in text and 'SMOKE_PASS' in text and 'SMOKE_FAIL' not in text
    raw,=folder.rglob('npusim.log');body=raw.read_text(errors='replace');assert re.findall(r'runtime_soc=(\S+)',body)==['Ascend950DT_9582'] and not re.search(r'\[ERROR\].*(?:execute_set_flag|execute_wait_flag|deadlock|invalid instruction)',body)
    dbs=sorted(folder.rglob('*.db'));assert len(dbs)==len(shapes);rows=[]
    for j,((t,d),p) in enumerate(zip(shapes,dbs)):
        if subset is not None and j not in subset:continue
        seg,=re.findall(rf'BATCH_BEGIN {j}\n(.*?)BATCH_END {j}\n',body,re.S);assert len(set(re.findall(r'\[TASK_DONE\].*?stream_id:(\d+), task_id:(\d+) TASK_DONE',seg)))==1
        gold=re.findall(r'correctness [^\r\n]+',seg); assert [int(re.search(r'count=(\d+)',x)[1]) for x in gold]==[t*d,t] and all(' bad=0 ' in x for x in gold)
        audit=audit_db('rmsnorm',[t,d],p);assert audit['passed'],audit;k=extract_db(p); assert k['helper_info']['launch_id']==str(j) or int(k['helper_info']['launch_id'])==j
        rows.append(dict(shape=[t,d],batch=folder.name,batch_wall_s=wall_s,source=str(p.relative_to(ROOT)),db_sha256=k['db_sha256'],reference_us=k['reference_us'],kernel=k['kernel'],active_aic=k['active_aic'],active_aiv=k['active_aiv'],max_core_rv_count=k['max_core_rv_count'],correctness=gold,audit=audit))
    return rows
if __name__=='__main__':
    # Verified prior to retry: two empty DBs plus both successor DBs show excess output byte counts.
    ids=[6,7,8,9]; contract=dict(utc=datetime.now(timezone.utc).isoformat(),ids=ids,reason=['invalid=1242 written=0','OUT=7604320 expected=3934080','invalid=1098 written=0','OUT=10963528 expected=6293504'],unchanged_model_freeze_sha256=sha256(OUT/'freeze.json'),per_process_seconds=600,per_process_gib=1.5,controls={'6':[64,768],'7':[64,768],'9':[64,768]},max_new_launches=7)
    assert not (OUT/'recovery-contract.json').exists();write('recovery-contract.json',contract)
    with concurrent.futures.ThreadPoolExecutor(4) as ex: print(list(ex.map(run,ids)))
