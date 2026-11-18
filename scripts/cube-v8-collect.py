#!/usr/bin/env python3
"""Bounded, one fresh process per CA shape. Never repeat a validation attempt."""
import concurrent.futures,os,signal,time,re,sys,subprocess,json
from datetime import datetime,timezone
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from operator_model.ca import sha256
from operator_model.npusim_v7 import extract_run
out=root/'results/v0.8/cube';runs=root/'runs/v8cube'
frozen=json.loads((out/'freeze.json').read_text());assert all(sha256(root/f)==h for f,h in frozen['artifact_sha256'].items())

def run(c):
    op=c['op'];s=c['shape'];directory=runs/c['id'];record=directory/'result.json'
    if directory.exists():
        if record.exists():return json.loads(record.read_text())
        raise RuntimeError('Existing incomplete attempt '+c['id'])
    directory.mkdir();start=time.monotonic();stamp=datetime.now(timezone.utc).isoformat();log=directory/'run.log'
    binary={'addmm':'extension_norm_addmm','matmul':'three_ops','kda_chunk':'extension_kda'}[op]
    args=[op,*map(str,s)]+(['bias'] if op=='addmm' else []) if op!='kda_chunk' else ['chunk',str(s[1]),str(s[-1])]
    cmd=['bash',str(root/'scripts/cann-run.sh'),'env','LD_PRELOAD='+str(root/'build/capture/libsim_soc_adapter.so'),'OPERATOR_SIM_TARGET_SOC=Ascend950DT_9582','npusim','record',str(root/'build/smoke'/binary),'-s','Ascend950DT','-o',str(directory/'profile'),'-u',' '.join(args)]
    x={**c,'started_utc':stamp,'command':cmd,'status':'failed','reference_us':None}
    with log.open('w') as f:
        env=os.environ.copy();env.pop('CAMODEL_CONFIG_PATH',None)
        child=subprocess.Popen(cmd,cwd=directory,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        while child.poll() is None:
            text=log.read_text(errors='replace');reason=None
            if time.monotonic()-start>c['timeout_s']:reason='wall budget'
            if re.search(r'\[ERROR\].*(?:execute_set_flag|execute_wait_flag|deadlock|invalid instruction)',text):reason='CA synchronization or instruction error'
            if sum(p.stat().st_size for p in directory.rglob('*.db') if p.is_file())>768*1024**2:reason='database size budget'
            if reason:
                x['error']=reason;os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                break
            time.sleep(1)
    x.update(wall_s=time.monotonic()-start,returncode=child.returncode)
    if child.returncode==0 and 'error' not in x:
        try:
            # three_ops labels the same all-output element count as "elements".
            txt=log.read_text();normalized=directory/'numeric-count.log';normalized.write_text(txt.replace('correctness elements=','correctness count='));m=extract_run(directory/'profile',normalized,6 if op=='kda_chunk' else 1)
            x.update(status='ok',reference_us=m['reference_us'],kernels=[{key:k[key] for key in ('kernel','reference_us','active_aic','active_aiv','max_core_mmad','db_sha256','db_bytes')} for k in m['kernels']],correctness=m['correctness'])
        except Exception as exc:x['error']=str(exc)
    x.update(log_sha256=sha256(log),raw_bytes=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()));record.write_text(json.dumps(x,indent=2)+'\n');print(c['id'],x['status'],x['reference_us'],x.get('error'),flush=True);return x

if __name__=='__main__':
    cases=frozen['cases'];kda=[c for c in cases if c['op']=='kda_chunk'];other=[c for c in cases if c['op']!='kda_chunk']
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        probe=pool.submit(run,kda[0]);futures=[pool.submit(run,c) for c in other]
        if probe.result()['status']=='ok':futures.extend(pool.submit(run,c) for c in kda[1:])
        for f in futures:f.result()
