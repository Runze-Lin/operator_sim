"""Freeze and run alternating RMSNorm and fused-bias Addmm on one ordered ACL stream."""
import argparse
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model import OperatorStream, estimate
from development.validation.sku19.collect import nbytes

OUT=ROOT/'results/v0.25/mixed-stream'
RAW=ROOT/'runs/mixed-history25'
HOST=ROOT/'build/mixed-stream25/host'
SKUS=('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT')
DEV={'paired':[(16,64,256),(2,2944),(2,2944),(16,64,256),(2,2944),(16,64,256),(4,608),(4,608)]}
# Each target follows exactly one registered bias Matmul. Its cold/expansion/reuse
# state is determined by earlier RMS launches, independently of measured times.
NEW={
    'expand':[(16,64,256),(1,2352),(16,64,256),(3,4848),(16,64,256),(9,1552),(16,64,256),(33,2256)],
    'reuse':[(48,2528),(32,128,128),(2,3248),(32,128,128),(6,5072),(32,128,128),(21,1776)],
    'longer':[(64,2096),(16,64,256),(32,2480),(16,64,256),(81,1712),(16,64,256),(385,3376)]
}


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def register(role):
    f=OUT/(role+'-freeze.json')
    if f.exists():
        raise FileExistsError(f)
    groups=DEV if role=='development' else NEW
    rows=[]
    for chip in SKUS:
        for name,sequence in groups.items():
            high=0
            stream=OperatorStream(chip)
            for i,shape in enumerate(sequence):
                op='rmsnorm' if len(shape)==2 else 'addmm'
                attrs={} if op=='rmsnorm' else {'self_shape':[shape[1]],'alpha':1,'beta':1}
                default=stream.advance(op,shape,attrs)
                condition=estimate(op,shape,{**attrs,'resident_aiv':high} if op=='rmsnorm' else attrs,chip=chip)
                assert default.latency_us is not None and condition.latency_us is not None
                rows.append(dict(op=op,chip=chip,group=name,i=i,shape=shape,previous_aiv=high,
                                 target=op=='rmsnorm' and i>0 and len(sequence[i-1])==3,
                                 public_us=default.latency_us,preserve_us=condition.latency_us,
                                 active_aic=condition.active_aic,active_aiv=condition.active_aiv))
                if op=='rmsnorm':
                    high=max(high,condition.active_aiv)
    if role=='independent':
        targets={tuple(r['shape']) for r in rows if r['target']}
        assert len(targets)==10
        for path in ('results/v0.22/evaluation.json','results/v0.22/rms-unknown/audit.json','results/rms-unknown23/freeze.json','results/v0.24/rms-prefix/freeze.json'):
            seen={tuple(r['shape']) for r in json.loads((ROOT/path).read_text())['rows'] if r.get('op','rmsnorm')=='rmsnorm'}
            assert not targets&seen,(path,targets&seen)
    files=[HOST,Path(__file__), Path(__file__).with_name('host.cpp'),ROOT/'smoke/extension_norm_addmm.cpp']
    write(f,dict(utc=now(),role=role,groups=list(groups),chips=SKUS[:2],rows=rows,
                 registered={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},
                 wall_s=1500,raw_mib=5120,slots=1,
                 decision='Preserve same-family RMS coverage only across a verified AIC-only intervening launch. Reject if hardware, binary, output, activity mask or full physical window fails; test first/expanded/reused states separately. No timing fitted. 970 is a scenario.'))


def run(role):
    f=OUT/(role+'-freeze.json')
    plan=json.loads(f.read_text())
    for p,h in plan['registered'].items():
        if sha256((ROOT/p).read_bytes()).hexdigest()!=h:
            raise ValueError('Frozen file changed '+p)
    RAW.mkdir(exist_ok=True)
    with (RAW/'lock').open('a') as lk:
        fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for group in plan['groups']:
            for chip in plan['chips']:
                stem=role+'-'+group+'-'+chip
                dest=OUT/(stem+'-cost.json')
                if dest.is_file():
                    continue
                folder=RAW/stem
                folder.mkdir()
                rows=[r for r in plan['rows'] if r['chip']==chip and r['group']==group]
                inp=folder/'inputs.txt'
                inp.write_text(''.join(r['op']+' '+' '.join(map(str,r['shape']+([0] if r['op']=='rmsnorm' else [])))+'\n' for r in rows))
                cmd=['bash',str(ROOT/'scripts/cann-run.sh'),'msopprof','simulator','--soc-version='+chip,'--dump=false',
                     '--launch-count='+str(len(rows)),'--timeout='+str(plan['wall_s']//60),'--output='+str(folder/'profile'),str(HOST),str(inp)]
                start=now(); tick=time.monotonic(); peak=0;stop=None
                write(OUT/(role+'-status.json'),dict(state='running',batch=stem,started=start))
                with (folder/'run.log').open('x') as log:
                    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=folder,start_new_session=True,
                                          env={**os.environ,'GOMP_SPINCOUNT':'0','OMP_WAIT_POLICY':'PASSIVE'})
                    try:
                        while proc.poll() is None:
                            peak=max(peak,nbytes(folder))
                            if peak>plan['raw_mib']*2**20 or time.monotonic()-tick>plan['wall_s']:
                                stop='raw limit' if peak>plan['raw_mib']*2**20 else 'wall limit'
                                os.killpg(proc.pid,signal.SIGKILL);break
                            time.sleep(2)
                    finally:
                        if proc.poll() is None and not stop:
                            os.killpg(proc.pid,signal.SIGTERM)
                        proc.wait()
                write(dest,dict(started_utc=start,finished_utc=now(),wall_s=time.monotonic()-tick,exit_code=proc.returncode,
                                stop=stop,bytes=nbytes(folder),peak_bytes=peak,file=str(folder.relative_to(ROOT)),
                                plan_sha256=sha256(f.read_bytes()).hexdigest(),inputs_sha256=sha256(inp.read_bytes()).hexdigest()))
        write(OUT/(role+'-status.json'),dict(state='finished',time=now()))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['register','run'])
    parser.add_argument('role',choices=['development','independent'])
    a=parser.parse_args()
    (register if a.action=='register' else run)(a.role)
