"""Official full-window CA with raw text duplication disabled, bounded and resumable."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Lock
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import DT, PR, OUT, RAW, digest, read, write, now
# Prior validation and join functions retain their exact unit, core and numeric checks.
sys.path[:0] = [str(ROOT/'development/validation/sku17'),str(ROOT/'development/validation/sku18')]
from campaign_collect import qualify, nbytes
from development.validation.sku18.collect import make_pair, summarize


def capture(job,chip,folder):
    folder.mkdir(parents=True, exist_ok=True)
    started=now()
    cmd=['/usr/bin/time','-f','%U %S','-o',str(folder/'cpu.txt'),'bash',str(ROOT/'scripts/cann-run.sh'),
         'msopprof','simulator','--soc-version='+chip,'--dump=false','--launch-count=1',
         '--timeout='+str((job['wall_s']+59)//60),'--output='+str(folder/'profile'),
         str(ROOT/job['runner']),*job['args']]
    start=time.monotonic()
    reason=None
    peak=0
    with (folder/'run.log').open('w') as out:
        proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,cwd=folder,start_new_session=True)
        try:
            while proc.poll() is None:
                peak=max(peak,nbytes(folder))
                if time.monotonic()-start>job['wall_s']:reason='wall cap'
                if peak>job['raw_mib']*2**20:reason='raw cap'
                if reason:
                    os.killpg(proc.pid,signal.SIGKILL)
                    break
                time.sleep(2)
        finally:
            if proc.poll() is None and reason is None:os.killpg(proc.pid,signal.SIGTERM)
            proc.wait()
    raw=nbytes(folder)
    cpu=(folder/'cpu.txt').read_text().splitlines()
    val=cpu[-1].split() if cpu else []
    cost=dict(started_utc=started,finished_utc=now(),seconds=time.monotonic()-start,
              raw_bytes=raw,peak_bytes=max(peak,raw),cpu_seconds=sum(map(float,val)) if len(val)==2 else None,
              exit_code=proc.returncode,stop_reason=reason,command=cmd[5:])
    write(folder/'cost.json',cost)
    return cost


def publish(jobs):
    pairs=[];costs=[];records=[]
    for j in jobs:
        both=[]
        for chip in (DT,PR):
            d=RAW/(j['id']+'_'+chip)
            if not (d/'outcome.json').exists():continue
            o=read(d/'outcome.json')
            costs.append(o)
            if o['eligible']:
                r=read(d/'qualified.json');records.append(r);both.append(r)
        if len(both)==2:
            if j['outputs'][0]['old_dt_us'] is not None:
                row=make_pair(j,*both)
            else:
                dt,pr=both
                vals={k:v for k,v in j['outputs'][0].items() if k!='indices'}
                row=dict(**vals,id=j['id'],regime=j['regime'],dtype=j['dtype'],dt=dt,pr=pr,
                         dt_ca_us=dt['ca_us'],pr_ca_us=pr['ca_us'],speedup_ca=dt['ca_us']/pr['ca_us'],
                         speedup_pred=vals['dt_us']/vals['pr_us'],speedup_old=None)
            row.update(sample_role=j['role'])
            if j['role']!='holdout':row['role']='supplemental complete prefill chunk'
            pairs.append(row)
    write(OUT/'pairs.json',pairs)
    write(OUT/'costs.json',costs)
    write(OUT/'records.json',records)
    if costs:write(OUT/'cost-summary.json',dict(**summarize(costs),pairs=len(pairs)))


def main():
    reg=read(OUT/'registration.json');freeze=read(OUT/'freeze.json')
    assert all(digest(ROOT/name)==v for name,v in freeze['refs'].items())
    start=time.monotonic();lock=Lock();reserved=0
    RAW.mkdir(parents=True,exist_ok=True)
    jobs=freeze['jobs']
    # Interleave long chunk work with the shorter post-freeze validations.
    chunks=[j for j in jobs if j['role']=='prefill_chunk']
    main_jobs=[j for j in jobs if j['role']=='holdout']
    work=[(j,c) for j in main_jobs for c in (DT,PR)]
    for i,(job,chip) in enumerate((j,c) for j in chunks for c in (DT,PR)):
        work.insert(1+i*8,(job,chip))
    def run(arg):
        nonlocal reserved
        j,chip=arg
        d=RAW/(j['id']+'_'+chip)
        if (d/'outcome.json').exists():return
        with lock:
            spent=sum(o['raw_bytes'] for o in (read(p) for p in RAW.glob('*/cost.json')))
            cap=j['raw_mib']*2**20
            if spent+reserved+cap>reg['max_retained_gib']*2**30 or time.monotonic()-start+j['wall_s']>reg['total_wall_s']:
                raise RuntimeError('campaign budget exhausted')
            reserved+=cap
        cost=read(d/'cost.json') if (d/'cost.json').exists() else capture(j,chip,d)
        result=dict(id=j['id'],chip=chip,op=j['op'],shape=j['shape'],**cost)
        try:
            r,=qualify(j,chip,d)
            contract=[s for s in (d/'run.log').read_text(errors='replace').splitlines() if s.startswith('case=')]
            assert len(contract)==1
            r.update(backend='msopprof official all-core trace dump=false',dtype='bf16',runner_sha256=j['runner_sha256'],
                     input_contract_log=contract,host_args=j['args'],run_log=str((d/'run.log').relative_to(ROOT)))
            write(d/'qualified.json',r)
            result.update(eligible=True,reason=None)
        except (ValueError,AssertionError,KeyError,IndexError,OSError) as exc:
            result.update(eligible=False,reason=repr(exc))
        with lock:
            write(d/'outcome.json',result);reserved-=cap
            publish(jobs)
            print(j['id'],chip,result['reason'] or 'qualified',flush=True)
    with ThreadPoolExecutor(reg['concurrency']) as pool:
        list(pool.map(run,work))
    publish(jobs)

if __name__=='__main__':main()
