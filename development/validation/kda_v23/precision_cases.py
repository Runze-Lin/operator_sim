"""Register and run mixed-sign, corrected-oracle probes, separately from old attempts."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import qualify
import series as base
from series import probe, DT, PR, MEMORY, OUT, ROOT, HERE, RAW, REG, read, save
from probe_v2 import replace_once
from features import write as write_features

NEW=RAW/'precision-host'
HOST=NEW/'kimi-v2-aux'
VENDOR_ORACLE='tests/atk/chunk_kda_fwd_prepare/executor_chunk_kda_fwd_prepare.py'
FORMS={'development':[(65,1)],'holdout':[(3,1),(5,2)]}

def regfile(role):return OUT/('precision-'+role+'-register.json')

def build():
    NEW.mkdir(parents=True,exist_ok=True)
    source=base.BIN.with_suffix('.cpp').read_text()
    inc=HERE/'precise_host.hpp'
    source=replace_once(source,'void prefill(aclrtStream stream,',f'#include "{inc}"\nvoid prefill(aclrtStream stream,')
    source=replace_once(source,'    try {\n        int tokens=', '    std::setvbuf(stdout,nullptr,_IOLBF,0);\n    try {\n        int tokens=')
    orig='    checkGroup(decode(QK.read()),aqk,true,"aqk");\n    checkGroup(decode(KK.read()),akk,true,"akk");'
    new=r'''    const std::string root=std::string("GOLDEN_ROOT/")+std::to_string(tokens)+"-"+std::to_string(heads)+"/";
    auto aqkActual=QK.read(), akkActual=KK.read();
    writeActual("aqk",aqkActual);
    writeActual("akk",akkActual);
    auto aqk64=decode(aqkActual), akk64=decode(akkActual);
    auto aqkRef=readPreparedGolden(root+"aqk.bin",aqk.size());
    auto akkRef=readPreparedGolden(root+"akk.bin",akk.size());
    auxiliaryDiagnostic("aqk","mathematical",aqk64,aqk,tokens);
    auxiliaryDiagnostic("akk","mathematical",akk64,akk,tokens);
    auxiliaryDiagnostic("aqk","published",aqk64,aqkRef,tokens);
    auxiliaryDiagnostic("akk","published",akk64,akkRef,tokens);
    checkGroup(aqk64,aqkRef,true,"aqk");
    checkGroup(akk64,akkRef,true,"akk");'''.replace('GOLDEN_ROOT',str(NEW))
    source=replace_once(source,orig,new)
    cpp=HOST.with_suffix('.cpp');cpp.write_text(source)
    api=probe.VENDOR/'op_api'; cann=probe.CANN
    subprocess.run(['bash',str(ROOT/'scripts/cann-run.sh'),'g++','-std=c++17','-O2','-fPIE','-pie',str(cpp),
        '-I'+str(cann/'include'),'-I'+str(api/'include'),'-L'+str(api/'lib'),'-Wl,-rpath,'+str(api/'lib'),'-L'+str(cann/'lib64'),
        '-Wl,-rpath-link,'+str(cann/'lib64'),str(ROOT/'toolchains/envs/cann-9.2.1/lib/libstdc++.so.6'),
        '-lcust_opapi','-lopapi','-lnnopbase','-lascendcl','-ldl','-o',str(HOST)],check=True)

def register(role,pins):
    dst=regfile(role)
    if dst.exists():raise FileExistsError(dst)
    primary=read(REG)
    if role=='development':build()
    if not HOST.is_file():raise FileNotFoundError(HOST)
    if role=='holdout' and not pins:raise ValueError('frozen prediction file required for holdout')
    template=read(probe.RAW/'fla-source-tree.json')
    source=RAW/'vendor-prepare-oracle.py'
    b=source.read_bytes();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    assert [f['sha'] for f in template['tree'] if f['path']==VENDOR_ORACLE]==[blob]
    files=[Path(__file__).resolve(),HERE/'qualify.py',HERE/'reference.py',HERE/'precise_host.hpp',HOST.with_suffix('.cpp'),HOST,
           source,REG,base.BIN,base.BIN.with_suffix('.cpp'),*(Path(p).resolve() for p in pins)]
    cases=[]
    for t,h in FORMS[role]:
        folder=NEW/f'{t}-{h}'
        cp=subprocess.run([str(ROOT/'toolchains/envs/cann-9.2.1/bin/python'),str(HERE/'reference.py'),str(t),str(h),str(folder)],
                           env=dict(os.environ,OPENBLAS_NUM_THREADS='1'),check=True)
        files+=list(folder.iterdir())
        for sku in (DT,PR):
            ident=f"{role}-{sku[:11].lower()}-t{t}-h{h}"
            case=dict(id=ident,sku=sku,shape=[1,t,h,h,128,128],work_items=((t+63)//64)*h,physical_memory=MEMORY[sku],
                 numerical_reference='Independent rounded Prepare Aqk/Akk, original full mathematical recurrence for attention and state. All original per-output thresholds preserved.',
                 oracle=dict(attention=t*h*128,aqk=t*h*64,akk=t*h*64,state_vk=h*128*128))
            if role=='holdout':
                assert all((c['shape'][1],c['shape'][3])!=(t,h) for c in primary['cases'])
                assert (t,h) not in FORMS['development']
            cases.append(case)
    if role=='holdout':
        receipt=read(regfile('development'));files.append(regfile('development'))
        for c in receipt['cases']:
            p=OUT/(c['id']+'.json')
            if not p.exists() or not read(p).get('qualified'):raise ValueError('corrected development priors have not qualified')
            files.append(p)
    save(dst,dict(utc=base.when(),role=role,contract=primary['contract'],shape_fields='case.shape=[B,T,Hq,Hv,K,V]; cu=[0,T], chunk_indices=[(0,i)] for i=0..ceil(T/64)-1',
        public_reference=dict(repository='https://github.com/flashserve/flash-linear-attention-npu',commit=template['sha'],path=VENDOR_ORACLE,blob=blob),
        protocol='Input generation, mixed signs, host ABI, descriptors and device ELFs unchanged. Store actual auxiliary BF16 outputs; keep coordinate and deviation against math and published reference. Actual all four numerics must pass the original strict thresholds. Each launch must satisfy the primary exact-ELF/roster/B-E checks. Stop the new series at the first failure.',
        primary_register_sha256=probe.digest(REG),source_hashes={probe.rel(p):probe.digest(p) for p in files},
        frozen_prediction_files=[probe.rel(Path(p).resolve()) for p in pins],budget_per_capture_seconds=900,budget_per_capture_bytes=3*1024**3,
        maximum_new_cases=len(cases),max_concurrency=1,published_device_text=primary['published_device_text'],cases=cases))
    print(probe.digest(dst),flush=True)

def run(role):
    p=regfile(role);r=read(p);step=OUT/('precision-'+role+'-state.json')
    for f,h in r['source_hashes'].items():
        if probe.digest(ROOT/f)!=h:raise ValueError('hash changed '+f)
    # Same advisory lock used by the original serial supervisor.
    with (RAW/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert read(OUT/'status.json')['state']=='finished'
        # capture belongs to series; the command and all tracing limits remain identical.
        base.BIN=HOST;done=[]
        for case in r['cases']:
            out=OUT/(case['id']+'.json')
            if out.exists():result=read(out)
            else:
                folder=RAW/case['id'];folder.mkdir(exist_ok=True)
                save(step,dict(state='running',id=case['id'],completed=done))
                cost=base.capture(r,case,folder,900)
                result=qualify.verify(r,case,folder,cost)
                log=(folder/'run.log').read_text(errors='replace')
                details=[l for l in log.splitlines() if l.startswith('auxiliary name=')]
                actual={}
                for n in ('aqk','akk'):
                    f=folder/(n+'.bf16')
                    if f.exists():actual[n]=dict(bytes=f.stat().st_size,sha256=probe.digest(f),path=probe.rel(f))
                extras=len(details)==4 and len(actual)==2 and all(d['bytes']==2*case['oracle'][n] for n,d in actual.items())
                result['qualified']=bool(result['qualified'] and extras)
                if not result['qualified']:result['serialized_device_us']=result['main_stages_us']=result['device_span_us']=None
                result.update(case=case,cost=cost,registration_sha256=probe.digest(p),auxiliary_diagnostics=details,actual_aux_bf16=actual)
                save(out,result)
            if not result['qualified']:
                save(step,dict(state='stopped on ineligible case',id=case['id'],completed=done));return
            write_features(out);done.append(case['id'])
        save(step,dict(state='finished',completed=done))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['register','run']);parser.add_argument('role',choices=FORMS)
    parser.add_argument('pins',nargs='*');args=parser.parse_args()
    if args.mode=='register':register(args.role,args.pins)
    else:run(args.role)
