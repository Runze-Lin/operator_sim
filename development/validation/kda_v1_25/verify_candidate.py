"""One formal 26.10-release fused probe; enable the PR pair only after qualified DT."""
import fcntl, hashlib,re,subprocess,sys
from pathlib import Path
from packages import HERE,ROOT,OUT,RAW,OLD,read,save,when,prior,probe,vendor

PKG=OUT/'packages.json'
REG=OUT/'registration.json'
PREV=ROOT/'results/v0.24/kda-v1'
SKU={'dt':'Ascend950DT_9582','pr':'Ascend950PR_957b'}
BINARY=RAW/'fla26100-v1'
VENDOR=vendor('release')

def host():
    # Reuse the exact source, arrays, checks, and diagnostic printing of v0.24.
    src=PREV/'raw/kimi-v1-default.cpp'
    api=VENDOR/'op_api';cann=probe.CANN
    flags=['bash',str(ROOT/'scripts/cann-run.sh'),'g++','-std=c++17','-O2','-fPIE','-pie',str(src),
           '-I'+str(api/'include'),'-I'+str(cann/'include'),'-I'+str(cann/'include/aclnnop'),'-L'+str(api/'lib'),'-Wl,-rpath,'+str(api/'lib'),
           '-L'+str(cann/'lib64'),'-Wl,-rpath-link,'+str(cann/'lib64'),str(ROOT/'toolchains/envs/cann-9.2.1/lib/libstdc++.so.6'),
           '-lcust_opapi','-lopapi','-lnnopbase','-lascendcl','-ldl','-o',str(BINARY)]
    subprocess.run(flags,check=True);return flags

def register():
    if REG.exists():raise FileExistsError(REG)
    packages=read(PKG);release=packages['versions']['release']
    if release['old_unique_texts_in_common'] or release['unique_text_count']!=4 or not packages['full_set_of_release_feature_elf_text_equal']:
        raise ValueError('package comparison does not warrant the selected single release only')
    code=host();prior_plan=read(PREV/'register.json');refs=[PREV/'raw/oracles/2-1'/f'{n}.bin' for n in ('attention','aqk','akk','state_vk')]
    paths=[Path(__file__).resolve(),HERE/'packages.py',PREV/'register.json',PREV/'raw/kimi-v1-default.cpp',BINARY,PKG,
        ROOT/'smoke/extension_kda.cpp',HERE.parent/'kda_v23/precise_host.hpp',HERE.parent/'kda_v1_24/investigate.py',
        *refs,VENDOR/'op_api/lib/libcust_opapi.so',VENDOR/'op_api/include/aclnnop/aclnn_chunk_kda_fwd.h',
        VENDOR/'bin/set_env.bash',VENDOR/'version.info',RAW/'release/fla_npu/ops/ascendc/_aclnn_ctypes.py',
        *VENDOR.glob('op_impl/ai_core/tbe/kernel/ascend950/chunk_kda_fwd/*.o')]
    if any(probe.digest(f)!=prior_plan['source_sha256'][probe.rel(f)] for f in refs+[PREV/'raw/kimi-v1-default.cpp']):raise ValueError('source/goldens drift')
    cases=[c for c in prior_plan['cases'] if c['metadata']=='varlen' and c['shape'][1:3]==[2,1]]
    assert [c['sku'] for c in cases]==list(SKU.values())
    save(REG,dict(utc=when(),role='development, different published release. No replacement of pinned Kimi 26.9.1 data.',
        cases=cases,published_kda_text=release['wheel_elfs'],builtin_text=read(OLD)['cann_builtin_kda_text'],
        contract='B=1,T=2,Hq=Hv=1,K=V=128; cu=[0,2], idx=[0,0]; raw gate and QKV BF16, other fixtures, host sources and math/V1 reference goldens byte-identical to v0.24. New header/lib first; both public symbols must bind to isolated 26.10 wheel. Exactly one released fused ELF plus complete auxiliary windows required.',
        gate='Run DT only. PR may start only if DT passes the registered four goldens, sync, whole-process, release identity, and every physical window. No identical feature repeat.',
        budget_seconds=1200,budget_bytes=3*2**30,max_concurrency=1,wheel=release['asset'],build=code,
        sources={probe.rel(p):probe.digest(p) for p in paths}))
    print(probe.digest(REG),flush=True)

def run():
    reg=read(REG)
    for p,h in reg['sources'].items():
        if probe.digest(ROOT/p)!=h:raise ValueError('freeze mismatch: '+p)
    # Reuse v0.24 accounting/checks with process-local bindings. The earlier module
    # receives absolute candidate inputs and does not write previous case directories.
    prior.BIN=BINARY
    probe.VENDOR=VENDOR
    files=[]
    with (RAW/'acquisition.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for i,c in enumerate(reg['cases']):
            dest=OUT/(c['id']+'.json');folder=RAW/c['id'];folder.mkdir(exist_ok=True)
            if not dest.exists():
                save(OUT/'status.json',dict(utc=when(),state='running',active=c['id'],finalized=files,reg_sha256=probe.digest(REG)))
                if i and not read(OUT/(files[0]+'.json'))['qualified']:
                    res=dict(qualified=False,skipped='DT qualification precondition failed',cost=None)
                else:
                    cost=prior.capture(reg,c,folder)
                    res=dict(cost=cost,**prior.qualify_v1(reg,c,folder,cost))
                save(dest,dict(case=c,registration_sha256=probe.digest(REG),**res))
            files.append(c['id'])
        save(OUT/'status.json',dict(utc=when(),state='finished',finalized=files,reg_sha256=probe.digest(REG)))

if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
