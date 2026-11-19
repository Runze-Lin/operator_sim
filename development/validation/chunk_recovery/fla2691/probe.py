"""One small CA launch of the pinned Kimi A5 FLA package, outside the CANN installation."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'results/chunk-recovery/fla2691'
RAW = OUT / 'raw'
VENDOR = RAW / 'fla_npu/opp/vendors/fla_npu_transformer'
PUBLISHED = VENDOR / 'op_impl/ai_core/tbe/kernel/ascend950'
CANN = ROOT / 'toolchains/cann/9.2.1/cann-9.2.1'
REGISTER = OUT / 'register.json'
CHIP = 'Ascend950DT_9582'
RUNNER = RAW / 'kimi-fla'
SOURCE = ROOT / 'build/smoke/chunk21-kimi.cpp'
CAPTURE = RAW / 'capture'
COST = CAPTURE / 'cost.json'
sys.path.insert(0, str(ROOT / 'development/validation/rms_ordered_verify'))
from verify import digest, text_bytes


def put(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def rel(p):
    return str(p.relative_to(ROOT))


def text_digests(files):
    return {p.name: hashlib.sha256(text_bytes(p)).hexdigest() for p in files}


def register():
    if REGISTER.exists():
        raise FileExistsError(REGISTER)
    api = VENDOR/'op_api/lib/libcust_opapi.so'
    release = json.loads((RAW/'release.json').read_bytes())
    [asset] = [a for a in release['assets'] if '_a5' in a['name'] and 'aarch64' in a['name']]
    whl = RAW/'fla-a5.whl'
    if 'sha256:'+digest(whl)!=asset['digest'] or whl.stat().st_size!=asset['size']:
        raise ValueError('release wheel digest/size mismatch')
    # Preserve all four original goldens. Report the actual bound function address as well.
    source = SOURCE.read_text().replace('"../../smoke/extension_kda.cpp"', '"'+str(ROOT/'smoke/extension_kda.cpp')+'"')
    anchor = 'int tokens=std::stoi(argv[1])'
    assert source.count(anchor) == 1
    source = '#include <dlfcn.h>\n' + source.replace(anchor, '''Dl_info info{};
        if (!dladdr(reinterpret_cast<void*>(&aclnnChunkKdaFwdGetWorkspaceSize), &info)) return 3;
        std::printf("bound_chunk_api=%s\\n", info.dli_fname);
        int tokens=std::stoi(argv[1])''')
    generated = RAW/'kimi-fla.cpp'
    generated.write_text(source)
    subprocess.run(['bash',str(ROOT/'scripts/cann-run.sh'),'g++','-std=c++17','-O2',str(generated),
                    '-I'+str(CANN/'include'),'-L'+str(api.parent),'-Wl,-rpath,'+str(api.parent),
                    '-L'+str(CANN/'lib64'),'-Wl,-rpath-link,'+str(CANN/'lib64'),
                    str(ROOT/'toolchains/envs/cann-9.2.1/lib/libstdc++.so.6'),'-lcust_opapi','-lopapi','-lnnopbase','-lascendcl','-ldl','-o',str(RUNNER)],check=True)
    candidates=text_digests((PUBLISHED/'chunk_kda_fwd').glob('*.o'))
    builtin=text_digests((CANN/'opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/chunk_kda_fwd').glob('*.o'))
    if len(candidates)!=8 or len(builtin)!=8 or set(candidates.values())&set(builtin.values()):
        raise ValueError('could not establish separate shipped ELF')
    files = [Path(__file__).resolve(), whl, SOURCE, ROOT/'smoke/extension_kda.cpp', generated, RUNNER, api,
             RAW/'fla_npu/ops/ascendc/_aclnn_ctypes.py', RAW/'fla_npu/ops/ascendc/_kda_policy.py',
             ROOT/'runs/workloads19/sources/full/Dockerfile.a5',ROOT/'runs/workloads19/sources/full/vllm_ascend/ops/kda.py',
             ROOT/'runs/workloads19/sources/full/vllm_ascend/ops/kimi_kda.py', ROOT/'scripts/cann-run.sh']
    put(REGISTER,dict(utc=datetime.now(timezone.utc).isoformat(),role='development, separate release FLA kernel and contract',chip=CHIP,
        release_asset={k:asset[k] for k in ('name','size','digest','browser_download_url','id')}, wheel_compiler=(VENDOR/'version.info').read_text().strip(),
        contract=dict(shape=[1,64,2,2,128,128],gate_dtype='bf16',qkv_dtype='bf16',beta_dtype='fp32',chunk=64,
            safe_gate=True,lower_bound=-5.,use_gate_in_kernel=True,state_v_first=True,cu=[0,64],indices=[0,0],
            check_all_outputs=['attention','aqk','akk','state_vk'],check_initial_state_unchanged=True,
            heads='two-head reduced physical shard; same K/V widths as Kimi K3',work_items=2,force_v2=False,selected_api='aclnnChunkKdaFwd'),
        budget_seconds=900,budget_bytes=2**30,maximum_ca_runs=1,
        eligibility='all four CPU oracle outputs, input VK preserved, numeric SMOKE_PASS, FLA host library and exact match of captured KDA .text to release, all serial device launches complete, positive instruction windows and all-core duration checks',
        published_kda_text=candidates,cann_builtin_kda_text=builtin,files={rel(p):digest(p) for p in files}))


def run():
    plan=json.loads(REGISTER.read_bytes())
    if any(digest(ROOT/p)!=value for p,value in plan['files'].items()):
        raise ValueError('frozen source or release mismatch')
    CAPTURE.mkdir(exist_ok=False)
    # This environment is limited to the child; the project's default CANN OPP stays intact.
    command=['bash',str(ROOT/'scripts/cann-run.sh'),'bash','-c',
             'source "$1/bin/set_env.bash"; shift; exec "$@"','fla',str(VENDOR),'msopprof','simulator',
             '--soc-version='+CHIP,'--dump=false','--launch-count=24','--timeout=15',
             '--output='+str(CAPTURE/'profile'),str(RUNNER),'64','2']
    start=time.monotonic(); peak=0; reason=None
    with (CAPTURE/'run.log').open('x') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=CAPTURE,start_new_session=True,
                              env=dict(os.environ,OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0'))
        while proc.poll() is None:
            sizes=[]
            for f in CAPTURE.rglob('*'):
                try:
                    if f.is_file():sizes.append(f.stat().st_size)
                except FileNotFoundError:
                    pass
            peak=max(peak,sum(sizes))
            if time.monotonic()-start>plan['budget_seconds']:reason='wall budget'
            if peak>plan['budget_bytes']:reason='disk budget'
            if reason:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait();break
            time.sleep(2)
    put(COST,dict(exit_code=proc.returncode,stop_reason=reason,seconds=time.monotonic()-start,peak_bytes=peak,command=command))

if __name__=='__main__':
    {'register':register,'run':run}[sys.argv[1]]()
