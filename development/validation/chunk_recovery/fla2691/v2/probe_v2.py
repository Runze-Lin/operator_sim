"""Single forced V2 ABI call with Kimi arithmetic, same wheel, separate diagnosis."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import probe

OUT=probe.OUT/'v2'
RAW=OUT/'raw'
REG=OUT/'register.json'
LOG=RAW/'run.log'
COST=RAW/'cost.json'
BIN=RAW/'kimi-v2'
STAGE_FOLDERS=('chunk_kda_fwd_prepare','chunk_fwd_h','chunk_kda_fwd_finalize')


def replace_once(source, old, new):
    if source.count(old)!=1:raise ValueError('source anchor: '+old)
    return source.replace(old,new)


def register():
    if REG.exists():raise FileExistsError(REG)
    src=probe.SOURCE.read_text()
    src=replace_once(src,'"../../smoke/extension_kda.cpp"','"'+str(probe.ROOT/'smoke/extension_kda.cpp')+'"')
    src='#include "'+str(probe.VENDOR/'op_api/include/aclnnop/aclnn_chunk_kda_fwd_v2.h')+'"\n#include <dlfcn.h>\n'+src
    src=replace_once(src,'ACL_CHECK(aclnnChunkKdaFwdGetWorkspaceSize(Q.tensor','ACL_CHECK(aclnnChunkKdaFwdV2GetWorkspaceSize(Q.tensor')
    src=replace_once(src,'scale,chunk,true,lower,true,true,O.tensor','scale,chunk,true,lower,true,true,1e-6,false,false,false,true,O.tensor')
    src=replace_once(src,'nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,&bytes,&executor)',
                         'nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,&bytes,&executor)')
    src=replace_once(src,'ACL_CHECK(aclnnChunkKdaFwd(workspace.ptr','ACL_CHECK(aclnnChunkKdaFwdV2(workspace.ptr')
    src=replace_once(src,'api=aclnnChunkKdaFwd tokens','api=aclnnChunkKdaFwdV2 tokens')
    src=replace_once(src,'        ACL_CHECK(aclInit(nullptr));','''        for (const char* name : {"aclnnChunkKdaFwdV2", "aclnnChunkKdaFwdV2GetWorkspaceSize"}) {
            Dl_info provider{};
            void* address = dlsym(RTLD_DEFAULT, name);
            const char* wanted = std::getenv("FLA_NPU_OP_API_LIB");
            if (!address || !dladdr(address, &provider) || !wanted ||
                std::string(wanted) != provider.dli_fname) throw std::runtime_error("wrong FLA host library");
            std::printf("bound_api=%s library=%s\\n", name, provider.dli_fname);
        }
        ACL_CHECK(aclInit(nullptr));''')
    src=replace_once(src,'    compare(decode(O.read())','''    bool outputsMatch = true;
    auto checkGroup = [&](const std::vector<double>& got, const std::vector<double>& truth, bool bf, const char* label) {
        try { compare(got, truth, bf, label); } catch (const std::runtime_error&) { outputsMatch = false; }
    };
    checkGroup(decode(O.read())''')
    for old in ('compare(decode(QK.read())','compare(decode(KK.read())','compare({f.begin(),f.end()}'):
        src=replace_once(src,old,old.replace('compare','checkGroup'))
    src=replace_once(src,'std::puts("input_vk_unchanged=1");','std::puts("input_vk_unchanged=1");\n    if (!outputsMatch) throw std::runtime_error("one or more V2 outputs differ");')
    generated=RAW/'kimi-v2.cpp'
    generated.write_text(src)
    api=probe.VENDOR/'op_api'; cann=probe.CANN
    subprocess.run(['bash',str(probe.ROOT/'scripts/cann-run.sh'),'g++','-std=c++17','-O2','-fPIE','-pie',str(generated),
        '-I'+str(cann/'include'),'-I'+str(api/'include'),'-L'+str(api/'lib'),'-Wl,-rpath,'+str(api/'lib'),'-L'+str(cann/'lib64'),
        '-Wl,-rpath-link,'+str(cann/'lib64'),str(probe.ROOT/'toolchains/envs/cann-9.2.1/lib/libstdc++.so.6'),
        '-lcust_opapi','-lopapi','-lnnopbase','-lascendcl','-ldl','-o',str(BIN)],check=True)
    src_paths={name:probe.RAW/name for name in ('aclnn_chunk_kda_fwd.cpp','chunk_kda_fwd_v2.cpp','stable_chunk_kda_fwd.cpp')}
    tree=json.loads((probe.RAW/'fla-source-tree.json').read_bytes()); checked_src={}
    import hashlib
    for name,path in src_paths.items():
        [row]=[f for f in tree['tree'] if f['path'].endswith('/'+name)]
        buf=path.read_bytes()
        oid=hashlib.sha1(b'blob '+str(len(buf)).encode()+b'\0'+buf).hexdigest()
        if oid!=row['sha']:raise ValueError('official host source Git object mismatch')
        checked_src[name]=dict(commit=tree['sha'],path=row['path'],blob=oid)
    matchers={family:probe.text_digests((probe.PUBLISHED/family).glob('*.o')) for family in STAGE_FOLDERS}
    if not all(matchers.values()):raise ValueError('missing published stage')
    files=[Path(__file__).resolve(),HERE/'score_v2.py',probe.REGISTER,probe.SOURCE,probe.ROOT/'smoke/extension_kda.cpp',generated,BIN,
           api/'lib/libcust_opapi.so',api/'include/aclnnop/aclnn_chunk_kda_fwd_v2.h',*src_paths.values()]
    probe.put(REG,dict(utc=datetime.now(timezone.utc).isoformat(),maximum_ca_runs=1,chip=probe.CHIP,budget_seconds=1200,budget_bytes=3*1024**3,
        role='direct published V2 API mechanism diagnostic, never a default small-Kimi sample; no timing model is fitted',
        contract=dict(B=1,S=2,Hq=1,Hv=1,K=128,V=128,chunk=64,layout='BSND',raw_gate='BF16',qkv='BF16',beta='FP32',scale=128**-.5,
                      safe_gate=True,lower_bound=-5.,use_gate_in_kernel=True,state_v_first=True,initial_state='nonzero FP32 VK',epsilon=1e-6,
                      use_qk_l2norm_in_kernel=False,use_beta_sigmoid_in_kernel=False,allow_neg_eigval=False,use_exp2=True,cu=[0,2],indices=[0,0],
                      optional_return=['final_state_vk','aqk','akk'],direct_api='aclnnChunkKdaFwdV2',default_wrapper_choice='aclnnChunkKdaFwd',work_items=1),
        oracle_elements=dict(attention=256,aqk=128,akk=128,state_vk=16384),
        expected_stages=list(STAGE_FOLDERS),released_device_text=matchers,verified_host_source=checked_src,
        checks='Four independent golden groups with per-group mismatch counts even on failure, initial state unchanged; exact DT, wheel host symbol address and all three physical device ELF texts, true per-core END, checksum agreement between per-core and aggregate, each official duration, serial whole-call windows, no simulator sync faults.',
        files={probe.rel(f):probe.digest(f) for f in files}))


def run():
    plan=json.loads(REG.read_bytes())
    if any(probe.digest(probe.ROOT/name)!=h for name,h in plan['files'].items()):raise ValueError('registered bytes changed')
    args=['bash',str(probe.ROOT/'scripts/cann-run.sh'),'bash','-c',
          'source "$1/bin/set_env.bash"; shift; exec "$@"','fla',str(probe.VENDOR),'msopprof','simulator',
          '--soc-version='+probe.CHIP,'--dump=false','--launch-count=32','--timeout=20','--output='+str(RAW/'profile'),
          str(BIN),'2','1']
    start=time.monotonic(); peak=0;stop=None
    with LOG.open('x') as log:
        proc=subprocess.Popen(args,cwd=RAW,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0'),start_new_session=True)
        while proc.poll() is None:
            samples=[]
            for path in RAW.rglob('*'):
                try:
                    if path.is_file():samples.append(path.stat().st_size)
                except FileNotFoundError:pass
            peak=max(peak,sum(samples))
            if peak>plan['budget_bytes']:stop='disk budget'
            if time.monotonic()-start>plan['budget_seconds']:stop='wall budget'
            if stop:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait();break
            time.sleep(2)
    probe.put(COST,dict(seconds=time.monotonic()-start,peak_bytes=peak,exit_code=proc.returncode,stop=stop,command=args))

if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
