"""A release-identity, four-phase V1 diagnostic. No test kernel is compiled."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'results/v0.27/kda'
RAW=OUT/'raw'
FIX=RAW/'fixtures'
BIN=RAW/'packed-v1'
sys.path.insert(0,str(HERE.parent/'kda_v1_24'))
import investigate as old
from golden import make, bf
import numpy as np


def now(): return datetime.now(timezone.utc).isoformat()

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def read(path): return json.loads(path.read_bytes())


def fixtures():
    # Two independent one-token sequences. Negating both the state and V in
    # sequence 1 negates O/VK exactly, leaving the QK and KK equations intact.
    make(1,1,FIX/'base')
    d=np.arange(128); h=1
    q=bf(((7*d+3)%29-14)/64);k=bf(((11*d+5)%31-15)/128);v=bf(((5*d+7)%23-11)/2048)
    g=bf(-5+((3*d)%11-5)/8)
    for n,a in [('q',q),('k',k),('v',v),('gate',g)]:
        data=np.stack([a,-a if n=='v' else a]);(data.view('<u4')>>16).astype('<u2').tofile(FIX/(n+'.bf16'))
    np.array([3/16,3/16],'<f4').tofile(FIX/'beta.f32');(((d%7)-3)/32).astype('<f4').tofile(FIX/'dt_bias.f32')
    state=((np.arange(128**2)*13+1)%19-9).reshape(128,128).T/8192
    np.stack([state,-state]).astype('<f4').tofile(FIX/'initial.f32')
    for n in ['attention','aqk','akk','state_vk']:
        a=np.fromfile(FIX/'base'/(n+'.bin'),dtype='<f8');np.concatenate([a,-a if n in ('attention','state_vk') else a]).tofile(FIX/(n+'.bin'))


def register():
    reg=OUT/'v1-register.json'
    if reg.exists(): raise FileExistsError(reg)
    RAW.mkdir(exist_ok=True,parents=True);FIX.mkdir(exist_ok=True);fixtures()
    v=old.probe.VENDOR; cann=old.probe.CANN; api=v/'op_api';cpp=HERE/'packed_v1.cpp'
    cmd=['bash',str(ROOT/'scripts/cann-run.sh'),'g++','-std=c++17','-O2','-fPIE','-pie',str(cpp),'-I'+str(cann/'include'),'-I'+str(api/'include'),'-L'+str(api/'lib'),'-Wl,-rpath,'+str(api/'lib'),'-L'+str(cann/'lib64'),'-Wl,-rpath-link,'+str(cann/'lib64'),str(ROOT/'toolchains/envs/cann-9.2.1/lib/libstdc++.so.6'),'-lcust_opapi','-lopapi','-lnnopbase','-lascendcl','-ldl','-o',str(BIN)]
    subprocess.run(cmd,check=True)
    prior=read(old.REG); base=ROOT/'results/chunk-recovery/fla2691/raw'
    extra=[BIN,cpp,Path(__file__).resolve(), HERE.parent/'kda_v1_24/golden.py', HERE.parent/'kda_v1_24/investigate.py', ROOT/'smoke/extension_kda.cpp', ROOT/'development/validation/kda_v23/precise_host.hpp',api/'lib/libcust_opapi.so',base/'stable_chunk_kda_fwd.cpp',base/'aclnn_chunk_kda_fwd.cpp',old.REG]+[p for p in FIX.iterdir() if p.is_file()]
    save(reg,dict(utc=now(),api='aclnnChunkKdaFwd',release=prior['release_asset'],published_kda_text=prior['published_kda_text'],builtin_text=prior['builtin_text'],dtype='BF16 Q/K/V/G, FP32 beta/initial VK',shape=[1,2,1,1,128,128],cu=[0,1,2],chunk_indices=[0,0,1,0],work_items=2,primary='four physical V1 phases, BF16 varlen, packed independent nonzero opposite VK/V',control='v0.24: one length-two sequence, same B=1 physical descriptor and heads',oracle=dict(attention=256,aqk=128,akk=128,state_vk=32768),budget_seconds=3600,budget_bytes=8*1024**3,cpus=8,skus=[old.DT,old.PR],pr_condition='Only after qualified DT',retention='all four native V1 phase windows, all auxiliary windows and output arrays; no V2 family; reject any sync or nonfinite/numeric errors',hashes={str(p.relative_to(ROOT)):sha(p) for p in extra}))

def capture(sku,reg,p):
    logfile=p/'run.log';costpath=p/'cost.json'
    if costpath.exists():return read(costpath)
    if logfile.exists():raise RuntimeError('incomplete prior attempt; refusing duplicate')
    cmd=['bash',str(ROOT/'scripts/cann-run.sh'),'bash','-c','source "$1/bin/set_env.bash"; shift; exec "$@"','fla',str(old.probe.VENDOR),'msopprof','simulator','--soc-version='+sku,'--dump=false','--launch-count=24','--timeout=60','--output='+str(p/'profile'),str(BIN),str(FIX)]
    st=time.monotonic();utc=now();peak=0;stop=None
    with logfile.open('x') as log:
        child=subprocess.Popen(cmd,cwd=p,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',OPENBLAS_NUM_THREADS='1'))
        while child.poll() is None:
            size=0
            for f in p.rglob('*'):
                try:
                    if f.is_file(): size+=f.stat().st_size
                except FileNotFoundError: pass
            peak=max(peak,size)
            if peak>reg['budget_bytes'] or time.monotonic()-st>reg['budget_seconds']:
                stop='byte or wall limit';os.killpg(child.pid,signal.SIGKILL);child.wait();break
            time.sleep(2)
    r=dict(started=utc,finished=now(),seconds=time.monotonic()-st,peak_bytes=peak,stop=stop,exit_code=child.returncode,command=cmd);save(costpath,r);return r

def score(sku,reg,p,cost):
    case=dict(sku=sku,physical_memory=old.MEMORY[sku],oracle=reg['oracle'])
    observed=old.qualify_v1(reg,case,p,cost)
    stages=observed['stages'];v1=[s for s in stages if s['kernel'].startswith('ChunkKdaFwd_')]
    identity=len(v1)==4 and all(s['published_names'] and s['device_text_sha256'] not in reg['builtin_text'].values() for s in v1) and not any(any(n in s['kernel'] for n in ('FwdPrepare','FwdH_','FwdFinalize')) for s in stages)
    # Inspect every simulator [ERROR], including non-flag device errors.
    errors=[line for line in (p/'run.log').read_text(errors='replace').splitlines() if '[ERROR]' in line]
    good=bool(identity and not errors and not observed['sync_errors'] and all(observed[k] for k in ('actual_sku_verified','published_host','process_success','golden','full_windows','chronological')))
    for k in ('qualified','published_monolithic_elf','main_ca_us','serialized_ca_us'):observed.pop(k)
    outputs={f.name:sha(f) for f in p.iterdir() if f.suffix in ('.f32','.bf16')}
    # Attribute errors to the profiler's reported physical kernels for diagnosis only.
    current='before first launch';counts={}
    for line in (p/'run.log').read_text(errors='replace').splitlines():
        m=re.search(r'Start profiling on kernel: (\S+)',line)
        if m:
            current=f'{len(counts)}:{m.group(1)}';counts[current]=0
        elif '[ERROR]' in line:counts[current]=counts.get(current,0)+1
    return dict(**observed,qualified=good,four_release_v1_phases=identity,formal_v1_kernel_us=sum(x['ca_us'] for x in v1) if good else None,formal_all_kernels_us=sum(x['ca_us'] for x in stages) if good else None,error_count=len(errors),error_lines_by_latest_launch_banner=counts,error_examples=errors[:3],output_hashes=outputs,cost=cost)

def run():
    reg=read(OUT/'v1-register.json'); sha_reg=sha(OUT/'v1-register.json')
    if any(sha(ROOT/f)!=h for f,h in reg['hashes'].items()):raise RuntimeError('frozen asset mismatch')
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[-reg['cpus']:]);os.nice(12)
    with (RAW/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for sku in reg['skus']:
            receipt=OUT/('v1-'+sku+'.json');p=RAW/sku;p.mkdir(exist_ok=True)
            if receipt.exists(): continue
            if sku==old.PR and not read(OUT/('v1-'+old.DT+'.json'))['qualified']:
                save(receipt,dict(qualified=False,skipped='DT did not qualify',observed_us=None,register_sha256=sha_reg));continue
            save(OUT/'v1-state.json',dict(state='acquiring',sku=sku,utc=now(),register_sha256=sha_reg,cpus=sorted(os.sched_getaffinity(0))))
            result=score(sku,reg,p,capture(sku,reg,p));save(receipt,dict(register_sha256=sha_reg,**result))
        save(OUT/'v1-state.json',dict(state='finished',utc=now(),register_sha256=sha_reg))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('register','run'));args=parser.parse_args();{'register':register,'run':run}[args.action]()
