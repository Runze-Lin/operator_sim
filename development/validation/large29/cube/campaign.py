"""Freeze matched dual-SKU large geometries and run one capped endpoint per shared slot."""
from datetime import datetime,timezone
from math import ceil
from pathlib import Path
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate
from operator_model.chips import load_chip
from operator_model.kimi_kda import plan_kimi_kda_prefill
from verify import digest,qualify

OUT=ROOT/'results/v0.29/large/cube'
RAW=ROOT/'runs/large29/cube'
BIN=RAW/'bin'
VENDOR=ROOT/'results/chunk-recovery/fla2691/raw/fla_npu/opp/vendors/fla_npu_transformer'
CHIPS=('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT')
V2_OLD=ROOT/'results/v0.26/kda/aligned-register.json'


def read(p):return json.loads(p.read_bytes())
def stamp():return datetime.now(timezone.utc).isoformat()
def save(p,v):
    p.parent.mkdir(exist_ok=True,parents=True);q=p.with_suffix('.tmp')
    q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)


def prediction(op,shape,attrs,sku):
    start=time.perf_counter_ns()
    est=estimate(op,shape,attrs,chip=sku)
    chip=load_chip(sku)
    s=dict(us=est.latency_us,status=est.status,reason=est.reason,aic=est.active_aic,aiv=est.active_aiv,
           work=est.work,projection=est.projection)
    if op=='kda_recurrent':
        chains=shape[0]*shape[3];active=min(chains,chip.vector_cores);waves=ceil(chains/active)
        single=estimate(op,[1,shape[1],1,1,128,128],attrs,chip=sku)
        double_state_bytes=2*4*chains*128*128
        s['independent_state_scenario']=dict(chains=chains,active_aiv=active,max_chains_per_aiv=waves,steps_per_chain=shape[1],
            cold_state_io_us=double_state_bytes/(1000*chip.hbm_bandwidth_gbs),single_chain_us=single.latency_us,
            scenario_us=max(waves*single.latency_us,double_state_bytes/(1000*chip.hbm_bandwidth_gbs)),
            assumption='One AIV per B/H task; charge complete single-chain observations per wave including its startup. Scenario only, warm-start/port contention unidentified.')
    s['query_ms']=(time.perf_counter_ns()-start)/1e6
    return s


def specifications():
    pro=read(ROOT/'configs/models/DeepSeek-V4-Pro.json')
    flash=read(ROOT/'configs/models/DeepSeek-V4.1-Flash.json')['text_config']
    kimi=read(ROOT/'configs/models/Kimi-K3.json')['text_config']['linear_attn_config']
    t=8*32768*pro['num_experts_per_tok']//pro['n_routed_experts'];k=pro['hidden_size'];f=pro['moe_intermediate_size']//8
    b=128;h=kimi['num_heads']//8;d=kimi['head_dim']
    assert (t,k,f,h,d)==(4096,7168,384,12,128)
    n=flash['index_head_dim'];qk=flash['head_dim'];m=8*32768
    jobs=[dict(id='moe',shape=[t,k,f,1],ops=['grouped_matmul_swiglu_quant_v2','gmm2','moe_ffn'],attrs={'expert_counts':[t]},args=['moe_ffn',t,k,f,1,t],
           host='moe',hours=8,gib=48,prefix=['GroupedMatmulSwigluQuantV2','GroupedMatmul'],contract=f'case=moe_ffn M={t} K={k} F={f} E=1 expert_counts={t} valid_rows={t}',
           golden={'gmm1_output':t*f,'gmm1_mx_scale':t*f//32,'gmm2_output':t*k},
           geometry='Pro B8×32Ki, top6, balanced EP384, one of 384 experts per rank, expert TP8 => 4096 assigned tokens, hidden=7168, local F=384. Dense signed MXFP8/E8M0 and BF16 down, ND groupListType=1. Native model expert weights FP4: matched dimensions only; excludes router/communication.'),
          dict(id='recurrent',shape=[b,1,h,h,d,d],ops=['kda_recurrent'],attrs={},args=[b,1,h],host='recurrent',hours=2,gib=20,
               prefix=['RecurrentKda'],contract=f'case=kda_recurrent api=aclnnRecurrentKda B={b} S=1 Hq={h} Hv={h}',
               golden={'attention':b*h*d,'state':b*h*d*d},geometry='Kimi TP8 ordinary decode B128, S=1, 96/8=12 local heads, D=128; dense BSND BF16 QKV and FP32 gate/state, KV order. Paged state gather and fused serving gate are outside. 1536 independent state chains.'),
          dict(id='addmm',shape=[m,n,qk],ops=['addmm'],attrs={'self_shape':[n],'alpha':1,'beta':1},args=['addmm',m,n,qk,'bias'],host='addmm',hours=6,gib=32,prefix=['MatMulV3'],
               contract=f'case=addmm api=aclnnAddmm M={m} N={n} K={qk} self=bias',golden={'output':m*n},
               geometry='Flash B8×32Ki gives rank indexer.wk dimensions [262144,128,512]. Addmm is a separate explicit BF16 bias contract with that geometry, not evidence for an actual bias node in Flash.'),
          dict(id='kda-v2',shape=[1,1024,h,h,d,d],ops=['kda_chunk'],attrs={'implementation':'fla_26_9_1_v2'},args=[1024,h],host='v2',hours=3,gib=40,
               prefix=['ChunkKdaFwdPrepare','ChunkFwdH','ChunkKdaFwdFinalize'],contract='api=aclnnChunkKdaFwdV2 tokens=1024 heads=12',
               golden={'attention':1024*h*d,'aqk':1024*h*64,'akk':1024*h*64,'state_vk':h*d*d},
               geometry='Kimi rank TP8 H12, single 1024-token prefill subcall, 16 chunks per head, 192 chunk-head tasks. V2 direct ABI forced. Release auto-selector would choose V1 at this size. FP32 state in VK order and BF16 auxiliary outputs.')]
    return jobs


def register():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'freeze.json').exists():raise FileExistsError('already frozen')
    jobs=specifications()
    paths=[*HERE.glob('*.py'),*HERE.glob('*.cpp'),*HERE.glob('*.hpp'),*(BIN/n for n in ('moe','recurrent','addmm','v2','golden-check','moe.cpp','addmm.cpp','v2.cpp')),
           *(RAW/'golden/1024-12').glob('*'),V2_OLD,ROOT/'development/validation/kda_v23/reference.py',
           ROOT/'results/v0.29/replay.json',ROOT/'results/v0.29/kda/report.md',ROOT/'smoke/extension_moe.cpp',ROOT/'smoke/extension_kda.cpp',ROOT/'smoke/extension_norm_addmm.cpp']
    paths+=[ROOT/'configs/models'/(name+'.json') for name in ('DeepSeek-V4.1-Flash','DeepSeek-V4-Pro','Kimi-K3')]
    paths+=[ROOT/'configs/chips'/(name+'.json') for name in CHIPS]
    paths+=[ROOT/'operator_model'/n for n in ('backend.py','extensions/kda_moe.py','kda_fla_v2.py','addmm_v9.py')]
    history=read(ROOT/'results/v0.29/replay.json')['rows']
    for j in jobs:
        for op in j['ops']:
            if any(r['op']==op and r['shape']==j['shape'] and r.get('attrs',{}).get('implementation')==j['attrs'].get('implementation') for r in history):
                raise ValueError('shape already qualified in standard replay')
        j['predictions']={op:{chip:prediction(op,j['shape'],j['attrs'],chip) for chip in CHIPS} for op in j['ops']}
    old={}
    for op in ('addmm','grouped_matmul_swiglu_quant_v2','gmm2','moe_ffn','kda_recurrent'):
        old[op]={}
        for chip in CHIPS[:2]:
            rs=[r for r in history if r['op']==op and r['chip']==chip]
            hi=max(rs,key=lambda r:__import__('math').prod(r['shape']))
            old[op][chip]={k:hi[k] for k in ('shape','attrs','ca_us','kernel','trace_sha256')}
    old['kda_chunk_v2']={}
    for chip in CHIPS[:2]:
        name=ROOT/'results/v0.26/kda'/f'aligned26-{chip}-192-4.json';r=read(name)
        assert r['qualified'];old['kda_chunk_v2'][chip]=dict(shape=[1,192,4,4,128,128],path=str(name.relative_to(ROOT)),sha256=digest(name),ca_us=r['serialized_device_us'],scope='largest H*T in v0.26 aligned development after four independent numerical checks')
    full={s:{sku:estimate('kda_chunk',[1,s,12,12,128,128],{'implementation':'fla_26_9_1_v2'},chip=sku).latency_us for sku in CHIPS} for s in (32768,131072,262144)}
    rec=dict(utc=stamp(),jobs=jobs,old=old,large_kda_v2_single_request_projection_only=full,
             v1=dict(full_kernel_us=None,qualification='Repeated SET and incomplete true-end evidence in official packed implementation; 950DT and 950PR not timed. No duplicate CA.',diagnostic='results/v0.29/kda/report.md'),
             admission='All dual endpoints have one attempt; serialized shared flock wraps each single capture. Keep unresolved, numerical failures and identity mismatch. 970 target-only clocks. Full model embeddings/FP4 MoE, actual request mixture and dispatch not asserted.',
             source_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths})
    with (OUT/'freeze.json').open('x') as f:json.dump(rec,f,indent=2);f.write('\n')


def footprint(p):
    total=0
    for root,dirs,files in os.walk(p):
        for f in files:
            try:total+=os.stat(os.path.join(root,f)).st_size
            except FileNotFoundError:pass
    return total


def capture(j,sku,folder):
    args=['msopprof','simulator','--soc-version='+sku,'--dump=false','--timeout='+str(60*j['hours']),'--output='+str(folder/'profile'),str(BIN/j['host']),*map(str,j['args'])]
    if j['id']=='kda-v2':
        args=['bash','-c','source "$1/bin/set_env.bash"; shift; exec "$@"','fla',str(VENDOR),*args]
    cmd=['bash',str(ROOT/'scripts/cann-run.sh'),*args]; t=time.monotonic();start=stamp();peak=0;reason=None
    with (folder/'run.log').open('x') as f:
        p=subprocess.Popen(cmd,cwd=folder,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while p.poll() is None:
                peak=max(peak,footprint(folder))
                if peak>j['gib']*2**30:reason='disk cap'
                if time.monotonic()-t>j['hours']*3600:reason='wall cap'
                if reason:os.killpg(p.pid,signal.SIGKILL);break
                time.sleep(4)
        finally:
            if p.poll() is None and not reason:os.killpg(p.pid,signal.SIGTERM)
            p.wait()
    c=dict(begin=start,end=stamp(),seconds=time.monotonic()-t,peak_bytes=max(peak,footprint(folder)),stop_reason=reason,exit_code=p.returncode,command=cmd)
    save(folder/'cost.json',c)
    return c


def run():
    reg=read(OUT/'freeze.json')
    for j in reg['jobs']:
        for p in (BIN/j['host'],):
            if digest(p)!=reg['source_sha256'][str(p.relative_to(ROOT))]:raise ValueError('frozen binary mismatch')
    for p in (RAW/'golden/1024-12').glob('*'):
        if digest(p)!=reg['source_sha256'][str(p.relative_to(ROOT))]:raise ValueError('frozen oracle mismatch')
    os.nice(10);os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    with (RAW/'owner.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for j in reg['jobs']:
            for sku in CHIPS[:2]:
                slug=j['id']+('-dt' if sku==CHIPS[0] else '-pr');folder=RAW/slug;target=OUT/(slug+'.json')
                if target.exists():continue
                def status(phase):save(OUT/'state.json',dict(phase=phase,endpoint=slug,utc=stamp(),pid=os.getpid(),freeze_sha256=digest(OUT/'freeze.json')))
                if (folder/'run.log').exists() and not (folder/'cost.json').exists():
                    raise RuntimeError('Existing launch without final cost, review live host before resuming: '+slug)
                if not (folder/'cost.json').exists():
                    status('waiting shared slot')
                    with (RAW.parent/'shared.lock').open('a') as slot:
                        fcntl.flock(slot,fcntl.LOCK_EX)
                        while shutil.disk_usage(ROOT).free<(2*j['gib']+128)*2**30:status('waiting for disk');time.sleep(60)
                        folder.mkdir(exist_ok=True);status('acquiring');cost=capture(j,sku,folder)
                else:cost=read(folder/'cost.json')
                status('qualifying after releasing shared slot')
                try:verdict=qualify(j,sku,folder,cost,read(V2_OLD),ROOT)
                except (OSError,KeyError,AssertionError,StopIteration,ValueError,IndexError) as e:verdict=dict(qualified=False,ca_us=None,reason=repr(e))
                rows=[]
                for idx,op in enumerate(j['ops']):
                    pred=j['predictions'][op][sku]
                    ca=(verdict['stages'][idx]['ca_us'] if j['id']=='moe' and idx<2 else verdict['ca_us']) if verdict['qualified'] else None
                    us=pred['us']
                    rows.append(dict(op=op,ca_us=ca,prediction_us=us,signed_error_pct=100*(us/ca-1) if us and ca else None,
                                     future_extrapolated_us=j['predictions'][op][CHIPS[2]]['us'],state_scenario=pred.get('independent_state_scenario')))
                save(target,dict(utc=stamp(),sku=sku,shape=j['shape'],cost=cost,verdict=verdict,operators=rows,freeze_sha256=digest(OUT/'freeze.json')))
        save(OUT/'state.json',dict(phase='finished',pid=os.getpid(),utc=stamp()))


if __name__=='__main__':
    {'register':register,'run':run}[sys.argv[1]]()
