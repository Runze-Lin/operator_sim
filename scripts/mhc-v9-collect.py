"""Run independent frozen fusion cases in isolated directories."""
import shutil,importlib.util,argparse,concurrent.futures,json,sys,subprocess,os,time,signal,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.npusim_v7 import extract_run
OUT=ROOT/'results/v0.9/mhc'
def run(c,freeze):
    folder=ROOT/'runs/v9mhc'/c['id']
    if (folder/'result.json').exists():return
    folder.mkdir(parents=True,exist_ok=False)
    binary='extension_indexer_mhc'
    copied=folder/binary
    shutil.copy2(ROOT/"build/smoke"/binary,copied)
    assert hashlib.sha256(copied.read_bytes()).hexdigest()==freeze["sha256"]["build/smoke/"+binary]
    command=['bash',str(ROOT/'scripts/cann-run.sh'),'env','LD_PRELOAD='+str(ROOT/'build/capture/libsim_soc_adapter.so'),'OPERATOR_SIM_TARGET_SOC=Ascend950DT_9582','npusim','record',str(copied),'-s','Ascend950DT','-o',str(folder/'profile'),'-u',' '.join(map(str,c['arguments']))]
    start=time.monotonic();error=None;measurement=None
    with (folder/'run.log').open('w') as log:
        env=os.environ.copy();env.pop('CAMODEL_CONFIG_PATH',None)
        proc=subprocess.Popen(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            time.sleep(2)
            if time.monotonic()-start>freeze['timeout_s']:error='wall budget'
            try:size=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
            except FileNotFoundError:continue
            if size>freeze['case_limit_mib']*2**20:error='output budget'
            if error:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=8)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                break
        if proc.returncode:error=error or 'caller exit '+str(proc.returncode)
    if not error:
        try:
            import re
            measurement=extract_run(folder/'profile',folder/'run.log',9)
            raw=next(folder.glob('profile/npusim_*/npusim.log')).read_text()
            sections=re.split(r'correctness name=(?:hin|chain_sinkhorn|chain_post) count=',raw)
            tasks=[set(re.findall(r'\[TASK_DONE\].*stream_id:(\d+), task_id:(\d+) TASK_DONE',part)) for part in sections]
            if len(tasks)!=4 or any(len(s)!=1 for s in tasks[:3]) or len(set.union(*tasks[:3]))!=3 or tasks[3]:
                raise ValueError('Could not attribute all three synchronized stages')
            if [int(k['helper_info']['launch_id']) for k in measurement['kernels']]!=[0,1,2]:
                raise ValueError('Unexpected DB order')
        except Exception as exc:error=str(exc)
    result={'case':c,'error':error,'wall_seconds':time.monotonic()-start,'measurement':measurement if not error else None}
    (folder/'result.json').write_text(json.dumps(result)+'\n')
    print(c['id'],error or 'OK',round(result['wall_seconds'],1),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args();freeze=json.loads((OUT/'freeze.json').read_text())
    for name,h in freeze['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    primary=[c for c in freeze['cases'] if not c['reserve']]
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:list(pool.map(lambda c:run(c,freeze),primary))
    def success():
        return sum(json.loads(p.read_text())['error'] is None for p in (ROOT/'runs/v9mhc').glob('*/result.json'))
    for c in freeze['cases']:
        if c['reserve'] and success()<10:run(c,freeze)
