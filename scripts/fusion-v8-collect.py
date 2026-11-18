"""Run independent frozen fusion cases in isolated directories."""
import argparse,concurrent.futures,json,sys,subprocess,os,time,signal,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.npusim_v7 import extract_run
OUT=ROOT/'results/v0.8/fusion'
def run(c,freeze):
    folder=ROOT/'runs/v8cv'/c['id']
    if (folder/'result.json').exists():return
    folder.mkdir(parents=True,exist_ok=False)
    binary='extension_moe' if c['op']=='moe_ffn' else 'extension_indexer_mhc'
    command=['bash',str(ROOT/'scripts/cann-run.sh'),'env','LD_PRELOAD='+str(ROOT/'build/capture/libsim_soc_adapter.so'),'OPERATOR_SIM_TARGET_SOC=Ascend950DT_9582','npusim','record',str(ROOT/'build/smoke'/binary),'-s','Ascend950DT','-o',str(folder/'profile'),'-u',' '.join(map(str,c['arguments']))]
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
            measurement=extract_run(folder/'profile',folder/'run.log',c['groups'])
            expected={'moe_ffn':2,'mhc':3}.get(c['op'],1)
            if len(measurement['kernels'])!=expected:raise ValueError('unknown dispatch')
            if c['op']=='mhc' and [k['kernel'] for k in measurement['kernels']][0]!='entry_kernel_2':raise ValueError('unverified Pre stage identity')
        except Exception as exc:error=str(exc)
    result={'case':c,'error':error,'wall_seconds':time.monotonic()-start,'measurement':measurement if not error else None}
    (folder/'result.json').write_text(json.dumps(result)+'\n')
    print(c['id'],error or 'OK',round(result['wall_seconds'],1),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,choices=(1,2),default=2);args=ap.parse_args()
    freeze=json.loads((OUT/'freeze.json').read_text())
    for p,d in freeze['sha256'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=d:raise ValueError('Frozen artifact changed: '+p)
    # Round robin across families to uncover dispatch problems early.
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:list(pool.map(lambda c:run(c,freeze),sorted(freeze['cases'],key=lambda c:c['id'].split('-')[-1])))
