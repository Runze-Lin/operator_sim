"""One new paired indexer geometry, serialized with the existing large CA campaign."""
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.large29.fusion import fusion_campaign as runner
from operator_model import estimate

OUT=ROOT/'results/v0.34/indexer'
RAW=ROOT/'runs/indexer34'
REG=OUT/'freeze.json'
SOURCE=runner.REG
CHIPS=runner.SKUS
SHAPE=[1,9,8192,8,128]


def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_bytes())
def sig(p):return sha256(p.read_bytes()).hexdigest()
def save(p,v):
    tmp=p.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)


def freeze():
    original=read(SOURCE)
    example,=[j for j in original['jobs'] if j['op']=='lightning_indexer']
    assert example['shape']==[1,9,32768,8,128] and not RAW.exists()
    seen=[r['shape'] for r in read(OUT/'physical.json')]
    assert SHAPE not in seen
    # Search real dispatches, rather than inferring acquisition from a price file.
    visited=0
    for folder,dirs,files in os.walk(ROOT/'runs'):
        dirs[:]=[s for s in dirs if s not in ('profile','simulator','log','log_ca','build','dump','ca_workspace','record','sources')]
        if 'run.log' in files:
            visited+=1
            assert not re.search(rb'^case=lightning_indexer .*\bQ=9 L=8192 G=8 D=128\b',(Path(folder)/'run.log').read_bytes(),re.M),folder
    sources=[Path(__file__),ROOT/'operator_model/indexer_pipeline.py',OUT/'prices.json',ROOT/example['runner'],SOURCE,
             ROOT/'development/validation/large29/fusion/fusion_campaign.py',ROOT/'development/validation/sku17/campaign_collect.py']
    jobs=[]
    for shape,role in [(SHAPE,'new, before acquisition'),(example['shape'],'previously registered, predicate depends on original acquisition time')]:
        prices={}
        for c in CHIPS:
            est=estimate('lightning_indexer',shape,chip=c)
            prices[c]=dict(us=est.latency_us,plan=est.projection['target_plan'],stages_us=est.projection['stages_us'])
        jobs.append(dict(shape=shape,role=role,prices=prices))
    job=dict(example,id='compressed-8ki',shape=SHAPE,index=runner.arguments('lightning_indexer',SHAPE),
             expected_numerics={'topk_set':9*512},outputs=[{'op':'lightning_indexer'}],wall_s=7200,raw_mib=8192,
             role='B1 Q9 right-aligned dense BF16 top512, 8192 available keys; comparable size to 32Ki per request at 4:1 compression, no quantized/paged dispatch claimed')
    assert not REG.exists()
    with REG.open('x') as f:json.dump(dict(utc=now(),round='v0.34',job=job,forecasts=jobs,checked_logs=visited,
        lock=str(runner.LOCK.relative_to(ROOT)),sources={str(p.relative_to(ROOT)):sig(p) for p in sources}),f,indent=2);f.write('\n')


def run():
    reg=read(REG);job=reg['job'];old=read(SOURCE);RAW.mkdir(exist_ok=True)
    assert all(sig(ROOT/k)==v for k,v in reg['sources'].items())
    def status(phase,chip=None):save(OUT/'state.json',dict(utc=now(),phase=phase,chip=chip,pid=os.getpid(),freeze_sha256=sig(REG)))
    os.nice(12);os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    with (RAW/'owner.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for chip in CHIPS[:2]:
            endpoint=OUT/(chip+'.json');folder=RAW/chip
            if endpoint.exists():continue
            if not (folder/'cost.json').exists():
                if folder.exists():status('interrupted raw kept for inspection',chip);return
                status('waiting for original large CA slot',chip)
                with runner.LOCK.open('a') as slot:
                    fcntl.flock(slot,fcntl.LOCK_EX);status('acquiring',chip);runner.launch(job,chip,folder)
            cost=read(folder/'cost.json');status('verifying',chip)
            try:
                if cost['started_utc']<=reg['utc'] or cost['exit_code'] or cost['stop_reason']:raise ValueError('process or budget')
                log=(folder/'run.log').read_text();assert 'api=aclnnLightningIndexer B=1 Q=9 L=8192 G=8 D=128 dtype=BF16 topk=512 sparse_mode=3 return_values=false' in log
                verdict=runner.verify(job,chip,folder,old)
            except (OSError,ValueError,RuntimeError,AssertionError,KeyError,IndexError,StopIteration) as err:
                verdict=dict(qualified=False,reason=repr(err),ca_us=None)
            save(endpoint,dict(chip=chip,shape=SHAPE,freeze_sha256=sig(REG),cost=cost,verdict=verdict))
        status('terminal')

if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
