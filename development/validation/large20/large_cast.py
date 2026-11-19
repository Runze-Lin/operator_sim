"""Full CANN Cast on exact rank-local prefill shapes, one dispatch per process."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import CHIPS,DT,PR,digest,read,write,now
from development.validation.sku19.collect import capture
from campaign_collect import stages
from operator_model import estimate
from time import perf_counter

OUT=ROOT/'results/v0.20/large'
RAW=ROOT/'runs/prefill20'
BIN=ROOT/'build/smoke/large_cast20'
MODULE=ROOT/'operator_model/cast_resource.py'
SHAPES=[(262144,32),(1048576,32),(2097152,32),(8192,5120),(4096,7168)]
def freeze():
    assert not (OUT/'registration.json').exists()
    jobs=[]
    for t,h in SHAPES:
        targets={}
        for c in CHIPS.values():
            a=perf_counter()
            e=estimate('cast_bf16_fp32',(t,h),chip=c,timing_mode='resource')
            targets[c]=dict(us=e.latency_us,active_aiv=e.active_aiv,plan=e.projection['target_plan'],
                            status=e.status,query_ms=(perf_counter()-a)*1000)
        jobs.append(dict(id=f'cast_{t}_{h}',op='cast_bf16_fp32',dtype='bf16',shape=[t,h],
            args=list(map(str,(t,h))),batch=False,kernels=1,nchecks=1,runner=str(BIN.relative_to(ROOT)),
            wall_s=10800,raw_mib=24576,targets=targets))
    write(OUT/'registration.json',dict(utc=now(), jobs=jobs, chips=[DT,PR], max_workers=2,
        resource_module=digest(MODULE), runner_sha256=digest(BIN),
        provenance='Flash DeepseekV41Indexer ReplicatedLinear(5120,32).project_weights().float(), or post-norm BF16 hidden [tokens,5120]/[tokens,7168]. Shape-only CA at installed aclnnCast entry under eager, pre-scaling contract.',
        historical_version='v0.19 Cast schedule, unchanged for v0.20; predictions frozen before CA',max_raw_per_process_gib=24))
def qualify_one(job,chip):
    name=job['id']+'_'+chip
    folder=RAW/name;out=OUT/(name+'.json')
    if out.exists():return
    c=read(folder/'cost.json') if (folder/'cost.json').exists() else capture(job,chip,folder)
    result=dict(shape=job['shape'],chip=chip,cost=c,qualified=False,ca_us=None)
    try:
        s,correct=stages(job,chip,folder)
        text=(folder/'run.log').read_text(errors='replace')
        t,h=job['shape']
        assert f'case=cast_bf16_fp32 api=aclnnCast R={t} D={h}' in text
        assert len(s)==1 and s[0]['kernel'].startswith('Cast_') and f'correctness count={t*h} bad=0' in text
        result.update(qualified=True,ca_us=s[0]['ca_us'],stage=s[0],correctness=correct)
    except (AssertionError,ValueError,OSError,IndexError,KeyError) as exc:
        result['reason']=repr(exc)
    write(out,result)
    print(name,result['qualified'],result.get('ca_us'),flush=True)

def collect():
    reg=read(OUT/'registration.json')
    assert digest(BIN)==reg['runner_sha256']
    # Collection consumes frozen predictions; a later model version does not change the CA.
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    def one(task):
        qualify_one(*task)
    # Interleave SKUs; only the two current processes allocate trace storage.
    with ThreadPoolExecutor(max_workers=reg['max_workers']) as pool:
        list(pool.map(one,((j,c) for j in reg['jobs'] for c in reg['chips'])))
if __name__=='__main__':
    {'freeze':freeze,'collect':collect}[sys.argv[1]]()
