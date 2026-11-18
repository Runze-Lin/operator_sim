"""Run one frozen vector-family batch in an isolated CA process."""
from pathlib import Path
import sys,json,subprocess,time,os,re,signal,sqlite3
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.npusim_v7 import extract_db
OUT=ROOT/'results/v0.8/vector'

def main(op,begin,end):
    frozen=json.load(open(OUT/'freeze.json'))
    for p,d in frozen['sha256'].items():
        if sha256(ROOT/p)!=d:raise ValueError('changed frozen artifact '+p)
    cases=[c for c in frozen['cases'] if c['op']==op][begin:end]
    if not cases:raise ValueError('empty batch')
    folder=ROOT/'runs/v8vec'/f'{op}-{begin}-{end}'
    folder.mkdir(parents=True,exist_ok=False)
    lst=folder/'inputs.txt';lst.write_text(''.join(f"{op} {i} "+' '.join(map(str,c['args']))+'\n' for i,c in enumerate(cases)))
    cmd=['bash',str(ROOT/'scripts/cann-run.sh'),'env','LD_PRELOAD='+str(ROOT/'build/capture/libsim_soc_adapter.so'),'OPERATOR_SIM_TARGET_SOC=Ascend950DT_9582','npusim','record',str(ROOT/'build/vector-v8'/cases[0]['kind']),'-s','Ascend950DT','-o',str(folder/'profile'),'-u',str(lst)]
    start=time.monotonic()
    with (folder/'run.log').open('w') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=folder,start_new_session=True)
        while proc.poll() is None:
            if time.monotonic()-start>frozen['per_batch_seconds'] or sum(p.stat().st_size for p in folder.rglob('*.db'))>frozen['disk_budget_gib']*2**30:
                os.killpg(proc.pid,signal.SIGTERM);break
            time.sleep(2)
    duration=time.monotonic()-start
    log=(folder/'run.log').read_text(errors='replace')
    assert proc.returncode==0 and 'Simulation SUCCESS' in log and 'SMOKE_PASS' in log,'batch incomplete'
    raw,=folder.rglob('npusim.log');body=raw.read_text(errors='replace')
    assert re.findall(r'runtime_soc=(\S+)',body)==['Ascend950DT_9582']
    assert not re.search(r'\[ERROR\].*(?:execute_set_flag|execute_wait_flag|deadlock|invalid instruction)',body)
    dbs=sorted(folder.rglob('*.db'));assert len(dbs)==len(cases)
    rows=[]
    for i,(c,path) in enumerate(zip(cases,dbs)):
        segment,=re.findall(rf'BATCH_BEGIN {i}\n(.*?)BATCH_END {i}\n',body,re.S)
        tasks=set(re.findall(r'\[TASK_DONE\].*?stream_id:(\d+), task_id:(\d+) TASK_DONE',segment));assert len(tasks)==1
        checks=re.findall(r'correctness [^\r\n]+',segment);assert len(checks)==c['groups'] and all(re.search(r'\bbad=0\b',s) and re.search(r'\bcount=[1-9]\d*\b',s) for s in checks)
        k=extract_db(path);assert int(k['helper_info']['launch_id'])==i
        rows.append({**c,'role':'validation','reference_us':k['reference_us'],'predicted_us':c['prediction']['predicted_us'],'kernel':k['kernel'],'active_aic':k['active_aic'],'active_aiv':k['active_aiv'],'source':str(path.relative_to(ROOT)),'correctness':checks,'db_sha256':k['db_sha256'],'batch_wall_s':duration,'batch':folder.name,'max_core_rv_count':k['max_core_rv_count']})
    (OUT/(folder.name+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
    print(folder.name,len(rows),'validated',duration,flush=True)

if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
