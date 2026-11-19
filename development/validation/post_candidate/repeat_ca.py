"""Queue one official two-dispatch Post process. Each iteration validates its own tensor."""
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
from development.validation.sku19.campaign import DT,PR,digest,now,read,write
from development.validation.sku19.collect import nbytes
from campaign_collect import stages
from operator_model import estimate

OUT=ROOT/'results/post-candidate'
RAW=ROOT/'runs/post-candidate-repeat'
REG=OUT/'paired-freeze.json'
SHAPE=[3,4,5376]
RUNNER='build/vector-v8/MHC'
BATCH='mhc_post 0 3 4 5376\nmhc_post 1 3 4 5376\n'
WALL=3600
MIB=1536


def register():
    if REG.exists() or RAW.exists():
        raise FileExistsError('Repeat campaign already registered/started')
    old=read(OUT/'prospective.json')
    obs=read(ROOT/'results/v0.17/sku/records.json')
    kernel_hashes={r['stages'][0]['binary_sha256'] for r in obs if r['op']=='mhc_post' and r.get('stages')
                   and r['stages'][0]['kernel'].endswith('_5')}
    assert len(kernel_hashes)==1
    cpus=len(os.sched_getaffinity(0))
    freeze=dict(utc=now(),shape=SHAPE,order=[PR,DT],runner=RUNNER,runner_sha256=digest(ROOT/RUNNER),
                source_sha256=digest(Path(__file__)),candidate_source=digest(OUT/'prospective.json'),
                public_model_sha256=digest(ROOT/'operator_model/post_resource.py'),
                runner_sources={p:digest(ROOT/p) for p in ['smoke/vector_v8.cpp','smoke/extension_indexer_mhc.cpp']},
                expected_kernel_sha256=next(iter(kernel_hashes)),batch_text=BATCH,
                checks=['same native stream and single process (one host aclrtCreateStream)',
                        'Full output golden and official all-core trace per ordinal, same Post ELF for both.',
                        'Second ordinal is previous same-shape Post executed; cache contents not assumed or observed.'],
                wall_s_per_process=WALL,raw_mib_per_process=MIB,concurrency=1,
                admission=dict(cpus=cpus,load1_limit=max(1,cpus-32),load5_limit=max(1,cpus-32),
                               poll_s=60,rule='Leave 32 logical CPUs of headroom for the default CA worker count; both SKUs wait their turn. DT only after a qualified PR window pair.'),
                targets={})
    for chip in (PR,DT):
        alt=next(r for r in old['jobs'] if r['shape']==SHAPE and r['chip']==chip)
        attrs=dict(has_h_res=True)
        e=estimate('mhc_post',SHAPE,attrs,chip=chip)
        d=e.projection['timing_diagnostics']
        modes={name:estimate('mhc_post',SHAPE,{**attrs,'resident_aiv':h},chip=chip).latency_us
               for name,h in [('first',0),('reuse',e.active_aiv)]}
        modes['unknown']=e.latency_us
        freeze['targets'][chip]=dict(v021_us=modes,v021_all_state_us=d['context_predictions_us'],
            candidate_us={k:alt[k+'_us'] for k in ('first','reuse','shape_only')},prior_registered_shape=alt,
            first_resident_aiv=0,previous_issued_aiv=e.active_aiv)
    write(REG,freeze)


def capture(chip,d,reg):
    batch=d/'cases.txt'
    batch.write_text(reg['batch_text'])
    cmd=['/usr/bin/time','-f','%U %S','-o',str(d/'cpu.txt'),'bash',str(ROOT/'scripts/cann-run.sh'),
         'msopprof','simulator','--soc-version='+chip,'--dump=false','--launch-count=2',
         '--timeout='+str(WALL//60),'--output='+str(d/'profile'),str(ROOT/RUNNER),str(batch)]
    env=dict(os.environ,OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    start=now();tick=time.monotonic();peak=0;reason=None
    with (d/'run.log').open('x') as log:
        proc=subprocess.Popen(cmd,cwd=d,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        try:
            while proc.poll() is None:
                peak=max(nbytes(d),peak)
                if time.monotonic()-tick>WALL or peak>MIB*2**20:
                    reason='wall cap' if time.monotonic()-tick>WALL else 'raw cap'
                    os.killpg(proc.pid,signal.SIGKILL)
                    break
                time.sleep(2)
        finally:
            if proc.poll() is None and reason is None:os.killpg(proc.pid,signal.SIGTERM)
            proc.wait()
    cpu=(d/'cpu.txt').read_text().splitlines()
    times=cpu[-1].split() if cpu else []
    result=dict(started_utc=start,finished_utc=now(),seconds=time.monotonic()-tick,
                peak_bytes=max(peak,nbytes(d)),cpu_seconds=sum(map(float,times)) if len(times)==2 else None,
                exit_code=proc.returncode,stop_reason=reason,command=cmd[5:])
    write(d/'cost.json',result)
    return result


def qualify(chip,d,reg):
    records,check=stages(dict(op='mhc_post',kernels=2,nchecks=2),chip,d)
    assert len(check)==len(records)==2,'must have both reference tensors and both official device windows'
    log=(d/'run.log').read_text(errors='replace')
    batches=re.findall(r'^BATCH_BEGIN (\d+)\s*$\n(.*?)^BATCH_END \1\s*$',log,re.M|re.S)
    assert [idx for idx,_ in batches]==['0','1'],'host batch order differs'
    rows=[]
    for (idx,scope),s,g in zip(batches,records,check):
        assert scope.count('case=mhc_post api=aclnnMhcPost T=3 N=4 D=5376 dtype=BF16')==1
        assert scope.count(g)==1 and 'correctness name=output count=64512 bad=0 ' in g
        assert s['binary_sha256']==reg['expected_kernel_sha256'] and s['kernel'].endswith('_5') and s['active_aic']==0
        assert s['active_aiv']==reg['targets'][chip]['previous_issued_aiv']
        condition=('first','reuse')[int(idx)]
        pred=reg['targets'][chip];rows.append(dict(ordinal=int(idx),condition=condition,correctness=g,stage=s,
              current_us=pred['v021_us'][condition],current_unknown_us=pred['v021_us']['unknown'],
              candidate_us=pred['candidate_us'][condition],candidate_unknown_us=pred['candidate_us']['shape_only']))
    assert records[0]['trace']!=records[1]['trace'], 'duplicate timing scope'
    return rows


def run():
    r=read(REG);RAW.mkdir(exist_ok=True)
    if digest(ROOT/RUNNER)!=r['runner_sha256'] or digest(Path(__file__))!=r['source_sha256']:
        raise RuntimeError('Registered runner/collector changed')
    queue=OUT/'paired-queue.json'
    with (RAW/'campaign.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for chip in r['order']:
            out=OUT/f'paired-{chip}.json'
            if out.exists():
                if not read(out)['qualified']:break
                continue
            while True:
                loads=os.getloadavg();a=r['admission']
                if loads[0]<=a['load1_limit'] and loads[1]<=a['load5_limit']:break
                write(queue,dict(timestamp=now(),state='waiting for CPU headroom',next_chip=chip,
                                 load1=loads[0],load5=loads[1],max_load=a['load1_limit']))
                time.sleep(a['poll_s'])
            d=RAW/chip;d.mkdir(exist_ok=True)
            if (d/'run.log').exists() and not (d/'cost.json').exists():
                raise RuntimeError(f'Preserve interrupted acquisition for review: {d}')
            write(queue,dict(timestamp=now(),state='running',chip=chip,load1=loads[0],load5=loads[1]))
            c=read(d/'cost.json') if (d/'cost.json').exists() else capture(chip,d,r)
            try:
                rows=qualify(chip,d,r)
                result=dict(qualified=True,rows=rows)
            except (ValueError,OSError,AssertionError,IndexError,KeyError) as exc:
                result=dict(qualified=False,reason=repr(exc))
            write(out,dict(chip=chip,shape=SHAPE,cost=c,**result))
            if not result['qualified']:
                write(queue,dict(timestamp=now(),state='unqualified',chip=chip,reason=result['reason']))
                break
        else:write(queue,dict(timestamp=now(),state='finished',completed=r['order']))

if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
