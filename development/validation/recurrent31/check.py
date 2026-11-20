"""Freeze two multi-chain forecasts. Run one new matched B6 control pair serially.

B128 is owned by large29. This program only reads its completed receipts.
"""
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]; HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'development/validation/large29/cube'),str(ROOT/'development/validation/recurrent_figure')]
from campaign import BIN, capture, digest, qualify, save
from verify import events
from figure import elf_text
from operator_model.kda_recurrent_chains import estimate_recurrent_chains

RES=ROOT/'results/v0.31/recurrent'; RAW=ROOT/'runs/recurrent31'; OLD=ROOT/'results/v0.29/large/cube'; OLDRAW=ROOT/'runs/large29/cube'
SKUS=['Ascend950DT_9582','Ascend950PR_957b','Ascend970DT']; HDR=ROOT/'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/recurrent_kda/op_kernel/arch35/recurrent_kda_arch35.h'
PACK=ROOT/'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/recurrent_kda_apt'
HOSTREF='14d022b1a4c3df263bf697534e3c25c7c992ff5d'; HOSTPATH='attention/recurrent_kda/op_host/recurrent_kda_tiling_processor.h'
MODEL=ROOT/'operator_model/kda_recurrent_chains.py'; FREEZE=RES/'freeze.json'
def read(p): return json.loads(p.read_bytes())
def utc():return datetime.now(timezone.utc).isoformat()

def calibration(chip):
    folder=ROOT/'runs/sku17/extension'/('kda_recurrent_1_1_1_1_128_128_'+chip); receipt,=read(folder/'strict.json'); shard,=folder.glob('profile/*/simulator/core0.veccore0/trace.json')
    ev=[x for x in events(shard) if x['ph']=='X']; lo=min(round(x['ts']*1650) for x in ev); hi=max(round((x['ts']+x['dur'])*1650) for x in ev)
    dma=sorted((round(x['ts']*1650)-lo,round(x['dur']*1650),d['processed_bytes'],d['instr_type']) for x in ev if x['name']=='MOV_SRC_TO_DST_ALIGNv2' for d in [json.loads(x['args']['detail'])]);
    get=lambda n,t:[(start,dur) for start,dur,nbytes,pipe in dma if (nbytes,pipe)==(n,t)]
    meta,=get(224,'MTE2'); beta,=get(4,'MTE2'); st=get(32768,'MTE2'); end=max(sum(d) for d in get(32768,'MTE3'))
    return dict(ca_us=receipt['ca_us'],kernel=receipt['kernel'],core_count=receipt['active_aiv'],metadata_end_tick=sum(meta),first_beta_tick=beta[0],first_state_read_descriptor_ticks=st[0][1],last_store_to_end_ticks=hi-lo-end,source=str(shard.relative_to(ROOT)),sha256=digest(shard),receipt=str((folder/'strict.json').relative_to(ROOT)),receipt_sha256=digest(folder/'strict.json'))

def freeze():
    base=next(j for j in read(OLD/'freeze.json')['jobs'] if j['id']=='recurrent'); reg={}
    for b,owner in [(6,'new matched control'),(128,'existing large29 queue; do not relaunch')]:
        j={k:base[k] for k in ('host','prefix','gib','hours','ops')}; j.update(id='recurrent',shape=[b,1,12,12,128,128],attrs={},args=[b,1,12],golden={'attention':b*12*128,'state':b*12*128*128},contract=f'case=kda_recurrent api=aclnnRecurrentKda B={b} S=1 Hq=12 Hv=12')
        if b==6: j.update(hours=1,gib=3)
        preds={}
        for sku in SKUS:
            times=[]
            for _ in range(21):
                start=time.perf_counter_ns();p=estimate_recurrent_chains(j['shape'],chip=sku);times.append((time.perf_counter_ns()-start)/1e6)
            preds[sku]=dict(**p,first_query_ms=times[0],warm_p50_ms=statistics.median(times[1:]),warm_max_ms=max(times[1:]))
        folder=OLDRAW if b==128 else RAW
        if any((folder/f'recurrent-{tag}'/'run.log').exists() for tag in ('dt','pr')):raise ValueError('Collection began before this freeze')
        reg[str(b)]=dict(owner=owner,job=j,predictions=preds)
    files=[MODEL,HERE/'check.py',BIN/'recurrent',HDR,OLD/'freeze.json',ROOT/'development/validation/large29/cube/recurrent.cpp',ROOT/'operator_model/transfer.py',ROOT/'operator_model/vector_v8.py',ROOT/'results/v0.8/vector/profile.json',*[ROOT/'configs/chips'/(k+'.json') for k in SKUS]]
    host_source=subprocess.check_output(['git','-C',str(ROOT/'third_party/ops-transformer'),'show',HOSTREF+':'+HOSTPATH]);
    old={sku:calibration(sku) for sku in SKUS[:2]};assert [old[SKUS[0]][k] for k in ('metadata_end_tick','first_beta_tick','first_state_read_descriptor_ticks','last_store_to_end_ticks')]==[1218,5650,887,10]
    d=dict(utc=utc(),instrument='official msopprof full device instruction window; dense BSND S1 equal heads BF16 FP32 KV; 970 extrapolation only',old_single_chain=old,
           reference_host_source=dict(commit=HOSTREF,path=HOSTPATH,sha256=hashlib.sha256(host_source).hexdigest(),role='open-source host tiling comparison; installed op_host binary is authoritative'),
           budget='B6 dual endpoints at most one attempt, up to 1 h and 3 GiB each. Shares large29 device slot. B128 dual endpoints read only; large29 owns original budget and numerical oracle.',files={str(p.relative_to(ROOT)):digest(p) for p in files},cases=reg)
    with FREEZE.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')

def capture_control():
    if 'kda_recurrent · dense multi-chain' in read(ROOT/'results/v0.33/acceptance-scope.json')['excluded_error_rows']:
        return
    RAW.mkdir(exist_ok=True,parents=True);reg=read(FREEZE)
    if digest(BIN/'recurrent')!=reg['files'][str((BIN/'recurrent').relative_to(ROOT))]:raise ValueError('host changed')
    j=reg['cases']['6']['job'];os.nice(12);os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    with (RAW/'owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for sku,tag in zip(SKUS,('dt','pr')):
            folder=RAW/f'recurrent-{tag}'; cost=folder/'cost.json';folder.mkdir(exist_ok=True)
            state=lambda phase:save(RES/'state.json',dict(phase=phase,sku=sku,pid=os.getpid(),utc=utc(),freeze_sha256=digest(FREEZE)))
            if not cost.is_file():
                if (folder/'run.log').exists():raise RuntimeError('Existing acquisition without final cost, inspect before retrying')
                state('waiting for large29 shared slot')
                with (OLDRAW.parent/'shared.lock').open('a') as slot:
                    fcntl.flock(slot,fcntl.LOCK_EX);state('acquiring serial B6 control');capture(j,sku,folder)
            target=RES/f'control-{tag}.json'
            if not target.is_file():
                state('checking numerical, trace roster, ELF and per-core beta count');cost_row=read(cost)
                try:
                    verdict=qualify(j,sku,folder,cost_row,{},ROOT)
                    if verdict['qualified']:
                        s,=verdict['stages'];installed=PACK/(s['kernel'].rsplit('_',1)[0]+'.o');sim=Path(s['trace']).parent
                        if elf_text(installed)!=elf_text(sim.parent/'dump/aicore_binary.o'):raise ValueError('kernel .text mismatch')
                        hist={};betas={}
                        for p in sorted(sim.glob('core*/trace.json')):
                            xs=[]
                            for e in events(p):
                                if e['ph']!='X' or e['name']!='MOV_SRC_TO_DST_ALIGNv2':continue
                                d=json.loads(e['args']['detail'])
                                if d['instr_type']=='MTE2' and d['processed_bytes']==4:xs.append(round(e['ts']*1650))
                            hist[len(xs)]=hist.get(len(xs),0)+1
                            if len(xs)>1:betas[p.parent.name]=[b-a for a,b in zip(sorted(xs),sorted(xs)[1:])]
                        verdict.update(chain_histogram=hist,repeat_beta_intervals_ticks=betas,installed_kernel_text_sha256=hashlib.sha256(elf_text(installed)).hexdigest())
                        if sum(n*count for n,count in hist.items())!=72:raise ValueError('wrong state chain count')
                except (OSError,ValueError,KeyError,StopIteration,IndexError,AssertionError) as e:verdict=dict(qualified=False,ca_us=None,reason=repr(e))
                save(target,dict(freeze_sha256=digest(FREEZE),cost=cost_row,verdict=verdict))
            score()
        state('new controls complete; B128 monitored by large29')

def score():
    reg=read(FREEZE);rows=[]
    for b,c in reg['cases'].items():
        for sku,tag in zip(SKUS,['dt','pr','970']):
            p=c['predictions'][sku];rc=(RES/f'control-{tag}.json' if b=='6' else OLD/f'recurrent-{tag}.json')
            observed=read(rc) if tag!='970' and rc.exists() else {};v=observed.get('verdict',{});ok=v.get('qualified') is True;ca=v['ca_us'] if ok else None
            rows.append(dict(b=int(b),chip=sku,chains=p['chains'],rounds=p['chains_per_busiest_aiv'],predicted_us=p['latency_us'],cold_repeat_baseline_us=p['repeated_cold_windows_us'],measured_ca_us=ca,signed_error_pct=100*(p['latency_us']/ca-1) if ca else None,status='measured' if ok else v.get('reason') or ('future scenario' if tag=='970' else 'awaiting complete qualified trace'),receipt=str(rc.relative_to(ROOT)) if observed else None))
    save(RES/'score.json',dict(freeze_sha256=digest(FREEZE),rows=rows))
if __name__=='__main__':{'freeze':freeze,'run':capture_control,'score':score}[sys.argv[1]]()
