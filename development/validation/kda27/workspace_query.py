"""Query the real A5 FLA host library; no phase-2 launch or payload allocation."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import os, signal
import subprocess,sys,time

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).parent;OUT=ROOT/'results/v0.27/kda';RAW=OUT/'raw/workspace';RAW.mkdir(exist_ok=True)
CANN=ROOT/'toolchains/cann/9.2.1/cann-9.2.1';V=ROOT/'results/chunk-recovery/fla2691/raw/fla_npu/opp/vendors/fla_npu_transformer';BIN=RAW/'ws';LIB=V/'op_api/lib/libcust_opapi.so';PY=ROOT/'toolchains/envs/cann-9.2.1';CPP=HERE/'workspace_query.cpp'

def main():
    subprocess.run(['bash',str(ROOT/'scripts/cann-run.sh'),'g++','-std=c++17','-O2',str(CPP),'-I'+str(CANN/'include'),'-I'+str(V/'op_api/include'),'-I'+str(CANN/'include/aclnnop'),'-L'+str(V/'op_api/lib'),'-Wl,-rpath,'+str(V/'op_api/lib'),'-L'+str(CANN/'lib64'),'-Wl,-rpath-link,'+str(CANN/'lib64'),str(PY/'lib/libstdc++.so.6'),'-lcust_opapi','-lopapi','-lnnopbase','-lascendcl','-ldl','-o',str(BIN)],check=True)
    rows=[]
    for seq,b,h in [(2,1,1),(4096,8,12),(32768,8,12),(131072,8,12),(262144,8,12),(524288,8,12)]:
        w=RAW/f'descriptor-{seq}-{b}-{h}';w.mkdir(exist_ok=True)
        cmd=['bash',str(ROOT/'scripts/cann-native-sim-run.sh'),str(w),'bash','-c','source "$1/bin/set_env.bash"; shift; exec "$@"','fla',str(V),str(BIN),str(seq),str(b),str(h)]
        t=time.monotonic()
        with (w/'log.txt').open('wb') as out:
            p=subprocess.Popen(cmd,cwd=w,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,
                               env=dict(os.environ,ASCEND_SLOG_PRINT_TO_STDOUT='1',ASCEND_GLOBAL_LOG_LEVEL='1'))
            try:p.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait()
        log=(w/'log.txt').read_bytes()
        m=re.search(rb'query_status=(-?\d+) executor=(\d+) workspace=(\d+)',log);status,executor,workspace=map(int,m.groups()) if m else (None,None,None)
        rows.append(dict(per_request=seq,batch=b,heads=h,return_code=p.returncode,status=status,executor=executor,workspace_bytes=workspace,provider_verified=bool(f'provider={LIB}'.encode() in log),elapsed_s=time.monotonic()-t,log_sha256=hashlib.sha256(log).hexdigest(),runtime_errors=[s.decode(errors='replace') for s in log.splitlines() if b'[ERROR]' in s and b'null pointer' in s][:2],signals=[s.decode(errors='replace') for s in log.splitlines() if s.startswith((b'symbol=',b'soc=',b'query_status='))]))
        if len(rows)==1 and p.returncode:break
    dest=OUT/'workspace-query.json';dest.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),contract='FLA A5 26.9.1 V2 host workspace-size query only; 950DT native simulator bootstrap; contiguous null-data aclTensor descriptors; no payload/HBM allocation, no target kernel phase-2 launch; the runtime may perform bootstrap work',library_sha256=hashlib.sha256(LIB.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(CPP.read_bytes()).hexdigest(),initial_bootstrap_control='An earlier aclInit + aclrtSetDevice single-shape attempt timed out after 90 seconds with no workspace receipt. This run uses descriptor-only GetSocName and no explicit device init.',rows=rows),indent=2)+'\n')
if __name__=='__main__':main()
