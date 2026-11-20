"""A new two-SKU prospective Kimi projection after the existing decode lane retires."""
from collections import Counter
import fcntl,gzip,json,os,re,shutil,sys,time
from pathlib import Path
from freeze import ROOT,OUT,DT,PR,FUT,digest,old
from operator_model import estimate,load_chip
from development.validation.matmul28 import campaign as ca
from development.validation.matmul28.drain import recover
from geometry import scan,compact

TARGET=OUT/'kimi-head';REG=TARGET/'freeze.json';RAW=ROOT/'runs/matmul32/kimi-head'
LOCK=ROOT/'results/v0.29/matmul/decode/raw/serial.lock'

def register():
    previous=ca.read(OUT/'freeze.json');assert all(digest(ROOT/s)==h for s,h in previous['sha256'].items())
    model=ca.read(ROOT/'configs/models/Kimi-K3.json')['text_config'];heads=model['num_attention_heads'];tp=heads; n=(model['v_head_dim']+model['qk_nope_head_dim'])*(heads//tp); k=model['kv_lora_rank']; batch=8;per=[4096]*7+[4088];shape=[sum(per),n,k]
    assert shape==[32760,256,512] and tp==96
    # Confirm the projection in both the workload builder and the previous TP8 manifest.
    first=ca.read(ROOT/'results/v0.31/matmul/kimi-prefill/freeze.json');assert first['shape'][1:]==[n*(tp//8),k] and first['node']=='mla.kv_b_proj'
    def has(x):
        if isinstance(x,dict):return x.get('shape')==shape or any(has(v) for v in x.values())
        return isinstance(x,list) and any(has(v) for v in x)
    registrations=[]
    for pattern in ('results/v0.2[89]/matmul/**/*freeze.json','results/v0.3[01]/matmul*/**/*freeze.json','results/v0.32/matmul/freeze.json'):
        for path in ROOT.glob(pattern):
            assert not has(ca.read(path));registrations.append(str(path.relative_to(ROOT)))
    assert not any(r['shape']==shape for r in ca.read(OUT/'score.json')['rows'])
    matches=[]
    for path in (ROOT/'runs').glob('matmul*/**/run.log'):
        with path.open(errors='replace') as f:
            if any('case=matmul api=aclnnMatmul M=32760 N=256 K=512 ' in ln for ln in f):matches.append(str(path))
    if matches:raise ValueError('Duplicate physical workload: '+str(matches))
    preds={}
    for chip in (DT,PR,FUT):
        r=estimate('matmul',shape,chip=chip);p=r.projection['target_plan'];assert p['branch']=='b_resident_l1'
        preds[chip]=dict(us=r.latency_us,old_us=old(shape,load_chip(chip))[0],plan=p,active_aic=r.active_aic,active_aiv=r.active_aiv)
    files=list(previous['sha256'])+['development/validation/matmul32/forward.py','development/validation/matmul32/geometry.py','configs/models/Kimi-K3.json','development/workloads/prefill_shapes.py','development/validation/matmul28/campaign.py','development/validation/matmul28/drain.py','build/smoke/matmul28','results/v0.31/matmul/kimi-prefill/freeze.json']
    TARGET.mkdir(parents=True,exist_ok=True)
    with REG.open('x') as f:json.dump(dict(utc=ca.now(),model='Kimi-K3',node='mla.kv_b_proj',shape=shape,source_model_heads=heads,tp=tp,local_heads=heads//tp,dtype='bf16',per_request_unpadded_query=per,requests=batch,requests_context=32768,scenario='Seven requests submit a complete 4096 chunk; the eighth has a final remainder of 4088. No rank padding. TP96 is an isolated MLA attention sharding scenario; does not claim whole-network support or actual vLLM weight format.',predictions=preds,wall_s=7200,raw_mib=8192,installed_matmul_elf=first['installed_matmul_elf'],dependencies={s:digest(ROOT/s) for s in files},dedup_prior_registrations=sorted(set(registrations)),policy='One run per actual SKU, serially in the retired Kimi decode CA slot, full affinity nice+12; no unlimited retries. Check all outputs and all AIC retirements; separately compare each physical MMAD size, B address reuse, A bytes and complete output address union.'),f,indent=2);f.write('\n')

def physical(s,shape,plan):
    m,n,k=shape;mm=Counter();dst=[];addr=[];abytes=0;p_core=[]
    for c in s['windows']:
        sha,h,fix,reads=scan((ROOT/s['trace']).parent/c/'trace.json');mm.update(h)
        b=[r for r in reads if r[1:]==(k,n,2*k*n)];addr.extend(r[0] for r in b);avol=sum(r[3] for r in reads)-sum(r[3] for r in b);abytes+=avol
        dst.extend((a,a+2*rows*cols) for a,rows,cols in fix);p_core.append(dict(core=c,sha256=sha,b_loads=len(b),mmad=compact(h)))
    p=plan;am,an,ak=p['tile'];ex=Counter({(am,an,ak): (p['waves']-1)*p['active_aic']*p['iterations']})
    for h,nc,count in p['tail_wave_histogram']:ex[h,nc,ak]+=count*p['iterations']
    dst.sort();checks=dict(histogram=mm==ex,mmad_total=sum(mm.values())==s['mmad']==p['mmads_total'],aic=s['active_aic']==p['active_aic'],aiv=s['active_aiv']==p['active_aiv'],weight_residency=len(set(addr))==1 and all(r['b_loads']==1 for r in p_core),a_volume=abytes==2*m*k,
        y=bool(dst) and dst[-1][1]-dst[0][0]==2*m*n and all(l==t for (_,t),(l,_) in zip(dst,dst[1:])))
    return dict(checks=checks,actual_mmad=compact(mm),frozen_mmad=compact(ex),cores=p_core)

def run():
    j=ca.read(REG);assert all(digest(ROOT/s)==h for s,h in j['dependencies'].items())
    RAW.mkdir(parents=True,exist_ok=True);os.nice(12);os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    def state(phase,**k):ca.save(TARGET/'state.json',dict(utc=ca.now(),pid=os.getpid(),phase=phase,cpus=len(os.sched_getaffinity(0)),**k))
    state('waiting for previous Kimi serial lane')
    with LOCK.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        for sku,tag in ((DT,'dt'),(PR,'pr')):
            raw=RAW/tag;dst=TARGET/(tag+'.json')
            if dst.exists():continue
            if raw.exists():state('preserved previous raw; not retried',chip=sku);return
            while shutil.disk_usage(ROOT).free<32*2**30:state('waiting for disk');time.sleep(30)
            raw.mkdir();state('acquiring',chip=sku);ca.measure(j,sku,raw)
            cost=ca.read(raw/'cost.json')
            try:
                if cost['exit_code'] or cost['stop_reason']:raise ValueError('budget/process failure')
                try:r=ca.qualify(j,sku,raw)
                except (ValueError,AssertionError,IndexError,OSError,KeyError) as exc:
                    initial=TARGET/(tag+'-initial.json');r=dict(qualified=False,ca_us=None,case=initial.stem,chip=sku,shape=j['shape'],reason=repr(exc));ca.save(initial,r)
                    r=recover(j,r,raw,acquisition=initial) or r
                if r['qualified']:
                    stage,=r['stages']
                    if stage['binary_sha256']!=j['installed_matmul_elf']:raise ValueError('provider ELF mismatch')
                    try:r['physical_audit']=physical(stage,j['shape'],j['predictions'][sku]['plan'])
                    except (KeyError,ValueError,AssertionError,OSError) as exc:r['physical_audit']=dict(error=repr(exc))
            except (KeyError,ValueError,AssertionError,OSError) as exc:r=dict(qualified=False,ca_us=None,reason=repr(exc))
            ca.save(dst,dict(r,chip=sku,shape=j['shape'],cost=cost,freeze_sha256=digest(REG)))
        state('finished')

if __name__=='__main__':{'freeze':register,'run':run}[sys.argv[1]]()
