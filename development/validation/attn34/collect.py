"""Reserve/collect the smallest two-multirow real Kimi contracts; one shared slot."""
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
import fcntl,json,os,shutil,sys,time
R=Path(__file__).resolve().parents[3];O=R/'results/v0.34/attn';RAW=R/'runs/attn34'
sys.path.insert(0,str(R/'development/validation/large29/vector'))
from common import DT,PR,FUT,PREFILL,HOST,VENDOR,read,digest
from acquire import capture
from verify import qualify
from operator_model import estimate
F=O/'freeze.json';LOCK=R/'results/v0.26/sink/raw/lock'
OLD=R/'results/v0.29/large/vector/freeze.json'
def rel(p):return str(p.relative_to(R))
def utc():return datetime.now(timezone.utc).isoformat()
def record(p,d,new=False):
    if new and p.exists():raise FileExistsError(p)
    tmp=p.with_suffix('.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def freeze():
    if F.exists() or RAW.exists():raise FileExistsError(F)
    sources=[R/'operator_model'/n for n in ('attn_res_resident.py','attention_residual_timing.py','attention_residual.py','backend.py','rms_resource.py')]
    src=R/'runs/workloads19/sources/full/csrc/attention/attn_res_fwd'
    sources += [src/'op_kernel/arch35/attn_res_fwd_resident.h',src/'op_host/attn_res_fwd_tiling.cpp',HOST,HOST.with_suffix('.cpp'),
                R/'configs/models/Kimi-K3.json',O/'phase-calibration.json',OLD,Path(__file__)]
    sources += [R/'configs/chips'/(c+'.json') for c in (DT,PR,FUT)]
    sources += [VENDOR/p for p in read(R/'results/v0.19/workloads/attn-res-build.json')['vendor']]
    orig=read(OLD);targets=[x for x in orig['jobs'] if x['op']=='attn_res_fwd'];assert len(targets)==2
    assert all(not (OLD.parent/(x['case']+'.json')).exists() and not (R/'runs/large29/vector'/x['case']).exists() for x in targets)
    jobs=[]
    for t,n,new in [(65,1,True),(121,2,True),(512,1,False)]:
        s=[t,7168,8,n];prices={}
        for c in (DT,PR,FUT):
            start=time.perf_counter_ns();r=estimate('attn_res_fwd',s,PREFILL,chip=c)
            assert r.status=='extrapolated' and r.latency_us and estimate('attn_res_fwd',s,PREFILL,chip=c,timing_mode='calibrated').latency_us is None
            prices[c]=dict(us=r.latency_us,query_with_calibrated_check_ms=(time.perf_counter_ns()-start)/1e6,plan=r.projection['target_plan'],
                           dma_read_bytes=r.projection['timing_diagnostics']['dma_read_bytes'],dma_write_bytes=r.work['logical_write_bytes'],
                           stages=r.projection['stages_us'],status=r.status)
        jobs.append(dict(op='attn_res_fwd',shape=s,prices=prices,attrs=PREFILL,protocol='instruction',args=[str(t),str(n)],
                         runner=rel(HOST),new=new,wall_s=7200 if t==65 else 10800,raw_gib=20 if t==65 else 28,
                         existing=[rel(OLD.parent/(x['case']+'.json')) for x in targets] if not new else []))
    record(F,dict(utc=utc(),jobs=jobs,release_phase='source-frozen before acquisition, sibling stages only, no 40021 latency labels',
        roots=read(R/'results/v0.19/workloads/attn-res-build.json')['revision'],source_sha256={rel(p):digest(p) for p in sources},
        slot=rel(LOCK),policy='65/1 and 121/2 on both real devices, one attempt per SKU; release serial host flock after each endpoint. 512/1 reuses the running large29 queue without relaunch. msopprof dump=false. All aligned outputs, unchanged bank, installed ELF, physical MTE counts and all launched kernels required. 970DT projections only.'),True)
def run():
    reg=read(F);RAW.mkdir(exist_ok=True);os.nice(12)
    for p,h in reg['source_sha256'].items():
        if digest(R/p)!=h:raise ValueError('changed since registration '+p)
    def state(phase,**extra):record(O/'state.json',dict(utc=utc(),pid=os.getpid(),phase=phase,freeze_sha256=digest(F),**extra))
    with (RAW/'owner.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for j in (x for x in reg['jobs'] if x['new']):
            for sku,tag in ((DT,'dt'),(PR,'pr')):
                t,_,_,n=j['shape'];stem=f'{t}-{n}-{tag}';path=O/(stem+'.json');base=RAW/stem
                if path.exists():continue
                job={**j,'chip':sku,'case':stem}
                state('waiting for existing shared standalone slot',case=stem)
                with LOCK.open('r+') as slot:
                    fcntl.flock(slot,fcntl.LOCK_EX)
                    if base.exists():
                        state('partial acquisition preserved',case=stem);return
                    if shutil.disk_usage(R).free<(j['raw_gib']+24)*2**30:
                        state('insufficient headroom',case=stem);return
                    state('collecting',case=stem);capture(job,base)
                state('verifying after releasing shared slot',case=stem)
                cost=read(base/'cost.json')
                try:
                    if cost['started_utc']<=reg['utc'] or cost['exit_code'] or cost['stop_reason']:raise ValueError('process or budget')
                    result=qualify(job,base,cost)
                    if not result['qualified']:raise ValueError('not complete')
                    from evidence import check_payloads
                    result.update(check_payloads(job,result))
                except (AssertionError,ValueError,IndexError,StopIteration,KeyError,OSError) as e:
                    result=dict(qualified=False,failed=repr(e),instruction_us=None)
                record(path,dict(chip=sku,shape=j['shape'],case=stem,freeze_sha256=digest(F),cost=cost,verdict=result),True)
                time.sleep(1)
        state('terminal')
if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
