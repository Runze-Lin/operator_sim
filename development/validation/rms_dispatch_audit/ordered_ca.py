"""Two serial RMSNorm batch processes, one per SKU; preserve dispatch order and goldens."""
import fcntl
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import CHIPS,digest,now,read,write
from development.validation.sku19.collect import nbytes
from campaign_collect import stages
from development.validation.large20.recover_cost import recover_cost
from development.validation.rms_dispatch_audit.extract import OUT

RAW=ROOT/'runs/rms-ordered-audit'
REG=OUT/'freeze.json'
RUNNER=ROOT/'build/vector-v8/NORM'


def capture(chip,folder,jobs,plan):
    folder.mkdir(exist_ok=True)
    path=folder/'cost.json'
    cap=plan['raw_mib_per_batch']*2**20
    budget=plan['wall_per_batch_s']
    if path.exists():
        return read(path)
    if (folder/'run.log').exists():
        result=recover_cost(folder,budget,cap)
    else:
        batch=folder/'inputs.txt'
        batch.write_text(''.join(f"rmsnorm {r['i']} {r['shape'][0]} {r['shape'][1]} 0\n" for r in jobs))
        command=['/usr/bin/time','-f','%U %S','-o',str(folder/'cpu.txt'),'bash',str(ROOT/'scripts/cann-run.sh'),
            'msopprof','simulator','--soc-version='+chip,'--dump=false','--launch-count='+str(len(jobs)),
            '--timeout='+str(budget//60),'--output='+str(folder/'profile'),str(RUNNER),str(batch)]
        began=now();tick=time.monotonic();peak=0;why=None
        with (folder/'run.log').open('x') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=folder,start_new_session=True)
            try:
                while proc.poll() is None:
                    peak=max(peak,nbytes(folder))
                    if peak>cap or time.monotonic()-tick>budget:
                        why='raw cap' if peak>cap else 'wall cap'
                        os.killpg(proc.pid,signal.SIGKILL)
                        break
                    time.sleep(2)
            finally:
                if proc.poll() is None and why is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                proc.wait()
        result=dict(started_utc=began,finished_utc=now(),seconds=time.monotonic()-tick,exit_code=proc.returncode,
                    peak_bytes=max(peak,nbytes(folder)),raw_bytes=nbytes(folder),stop_reason=why,command=command[5:])
    write(path,result)
    return result


def qualify(chip,folder,jobs):
    log=(folder/'run.log').read_text(errors='replace')
    spans=re.findall(r'^BATCH_BEGIN (\d+)\s*$\n(.*?)^BATCH_END \1\s*$',log,re.M|re.S)
    if [int(k) for k,_ in spans]!=list(range(len(jobs))):
        raise ValueError('Missing complete host launch')
    records,all_gold=stages(dict(op='rmsnorm',kernels=len(jobs),nchecks=2*len(jobs)),chip,folder)
    result=[]
    for pred,stage,(i,span) in zip(jobs,records,spans):
        tokens,width=pred['shape']
        observed=re.findall(r'^correctness count=(\d+) bad=0\b',span,re.M)
        if (list(map(int,observed))!=[tokens*width,tokens] or
                f'case=rmsnorm api=aclnnRmsNorm R={tokens} C={width} dtype=BF16' not in span or
                'RmsNorm' not in stage['kernel'] or stage['active_aic']!=0):
            raise ValueError('Golden, shape or kernel mismatch at '+i)
        result.append(dict(**pred,stage=stage,correctness=all_gold[2*int(i):2*int(i)+2],
                           active_count_matches=stage['active_aiv']==pred['active_aiv']))
    if len({r['stage']['trace'] for r in result})!=len(jobs):
        raise ValueError('Repeated device window')
    return result


def main():
    plan=read(REG)
    for name,sha in plan['files'].items():
        if digest(ROOT/name)!=sha:
            raise RuntimeError('Frozen file changed: '+name)
    RAW.mkdir(exist_ok=True)
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    with (RAW/'collector.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for chip in (CHIPS['pr'],CHIPS['dt']):
            target=OUT/('ordered-'+chip+'.json')
            if target.is_file():
                continue
            jobs=[r for r in plan['rows'] if r['protocol']=='ordered new sequence' and r['chip']==chip]
            folder=RAW/chip
            # Use one freed slot after the previous wide-RMS pair. The count of
            # concurrent extra simulations is bounded independently of case size.
            prev=ROOT/'runs/sku20-rms'/('rmsnorm_32768_1280_'+CHIPS['pr'])/'outcome.json'
            retries=ROOT/'results/v0.21/dispatch/retries'
            while not prev.exists() and len(list(retries.glob('*.json')))<2:
                time.sleep(45)
            write(OUT/'ordered-status.json',dict(utc=now(),state='running',chip=chip))
            cost=capture(chip,folder,jobs,plan)
            try:
                observations=qualify(chip,folder,jobs)
                outcome=dict(qualified=True,rows=observations)
            except (ValueError,IndexError,KeyError,AssertionError,OSError) as exc:
                outcome=dict(qualified=False,reason=repr(exc))
            write(target,dict(chip=chip,cost=cost,**outcome))
        write(OUT/'ordered-status.json',dict(utc=now(),state='finished'))


if __name__=='__main__':
    main()
