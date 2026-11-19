"""One bounded test of the Kimi-K3 BF16 projection gate, using the existing host oracle."""
import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from probe import ART, ROOT, RUN, save, sha

DIR = Path(__file__).resolve().parent
HOST = ROOT / 'build/smoke/chunk21-kimi'
GENERATED = ROOT / 'build/smoke/chunk21-kimi.cpp'
# Keep the already registered FP32 control verbatim. Apply only the tensor-contract
# differences to its host oracle; no device code is built here.
def build():
    source = (DIR/'kimi.cpp').read_text()
    changes = {
        '../../../smoke/extension_kda.cpp':'../../smoke/extension_kda.cpp',
        'constexpr double lower = -0.03125;':'constexpr double lower = -5.0;',
        'std::vector<float> raw(elements), beta':'std::vector<uint16_t> raw(elements);\n    std::vector<float> beta',
        'raw[i] = ((t+3*d+h)%11-5)/8.f;':'raw[i] = bf16(-5.f + ((t+3*d+h)%11-5)/8.f);',
        'double x = raw[i]+bias':'double x = fp32(raw[i])+bias',
        'Tensor<float> G({1,tokens,heads,dim},ACL_FLOAT,raw), B':'Tensor<uint16_t> G({1,tokens,heads,dim},ACL_BF16,raw);\n    Tensor<float> B',
    }
    for old,new in changes.items():
        if source.count(old) != 1:
            raise ValueError(old)
        source = source.replace(old,new)
    GENERATED.write_text(source)
    subprocess.run(['bash',str(ROOT/'scripts/cann-run.sh'),'bash','-c',
        'g++ -std=c++17 -O2 -Wall -Wextra "$1" -I"$ASCEND_HOME_PATH/include" -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath-link,"$ASCEND_HOME_PATH/lib64" -lopapi -lnnopbase -lascendcl -o "$2"',
        'build',str(GENERATED),str(HOST)],check=True)


CHIP='Ascend950PR_957b'
TAG='kimi-k3-bf16g-64x12'
FILES=(ROOT/'configs/models/Kimi-K3.json',DIR/'kimi.cpp',DIR/'probe.py',DIR/'production_probe.py',
       ROOT/'smoke/extension_kda.cpp',GENERATED,HOST)
reg=ART/f'{TAG}-registration.json'
if __name__=='__main__' and sys.argv[1]=='register':
    if reg.exists():
        raise FileExistsError(reg)
    build()
    save(reg,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), role='development, predictions unavailable',
                  chip=CHIP,shape=[1,64,12,12,128,128],gate_dtype='bf16',dtype='bf16',beta_dtype='fp32',
                  safe_gate=True,lower_bound=-5.0,use_gate_in_kernel=True,state_v_first=True,initial_state=True,dt_bias=True,
                  cu_seqlens=[0,64],chunk_indices=[0,0],chunk_size=64,export_gk=False,output_final_state=True,
                  budget_seconds=1800,budget_bytes=2**30,stop_on='first duplicate SET_FLAG, ineligible for full timing',
                  hashes={str(p.relative_to(ROOT)):sha(p) for p in FILES}))
elif __name__=='__main__' and sys.argv[1]=='collect':
    conf=json.loads(reg.read_text())
    if any(sha(ROOT/p)!=h for p,h in conf['hashes'].items()):
        raise ValueError('registered source mismatch')
    dest=RUN/TAG/CHIP
    dest.mkdir(parents=True,exist_ok=False)
    log=dest/'run.log'
    cmd=['bash',str(ROOT/'scripts/cann-run.sh'),'msopprof','simulator',f'--soc-version={CHIP}',
         '--launch-count=20','--timeout=29','--dump=true',f'--output={dest}/profile',str(HOST),'64','12']
    start=time.monotonic()
    stop=None
    with log.open('w') as stream:
        child=subprocess.Popen(cmd,cwd=dest,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            time.sleep(2)
            text=log.read_text(errors='replace')
            if 'execute_set_flag already has same set_flag' in text:
                stop='duplicate SET_FLAG'
            elif time.monotonic()-start>conf['budget_seconds']:
                stop='wall budget'
            elif sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())>conf['budget_bytes']:
                stop='storage budget'
            if stop:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()
                break
    text=log.read_text(errors='replace')
    save(ART/f'{TAG}-{CHIP}.json',dict(chip=CHIP,command=cmd,registration_sha256=sha(reg),log_sha256=sha(log),
         actual_runtime=re.findall('runtime_soc=(\\w+)',text),seconds=time.monotonic()-start,
         kernels=re.findall('Start profiling on kernel: (\\w+)',text), stop=stop,exit=child.returncode,
         duplicate_set_pcs=re.findall('already has same set_flag! pc:(0x[0-9a-f]+)',text)[:12],
         numerical_checks=[line for line in text.splitlines() if 'correctness name=' in line or 'input_vk_unchanged=1' in line],
         success='SMOKE_PASS' in text and child.returncode==0 and stop is None))
